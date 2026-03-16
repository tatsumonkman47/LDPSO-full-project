#!/usr/bin/env python
"""
Split-job training script.

Each lap is split into 3 PBS jobs:
  1. gen_train  (4 GPU, single node) — generate synthetic training data
  2. gen_test   (4 GPU, single node) — generate synthetic test data
  3. train      (4 GPU, single node) — load pre-generated data, run training

gen_train and gen_test run in PARALLEL on separate nodes, cutting generation
wall-clock time roughly in half compared to the sequential version.

DAG per lap:
    [prev checkpoint] ──→ gen_train ──┐
                       └→ gen_test  ──┤→ train → [checkpoint]
"""

# Core Libraries
import sys
import os
from pathlib import Path

# Add local priors package to path (before conda environment version)
_local_priors_path = Path(__file__).resolve().parent.parent.parent
if str(_local_priors_path) not in sys.path:
    sys.path.insert(0, str(_local_priors_path))

import inox                  # type: ignore
import inox.nn as nn         # type: ignore
from inox import random as inox_random # type: ignore
import jax                   # type: ignore
import numpy as np           # type: ignore
import optax                 # type: ignore
import jax.numpy as jnp      # type: ignore
import pickle
import csv
import uuid

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

from dawgz import job, schedule

from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
from priors.data import prefetch
from priors.image import random_flip, random_hue, random_saturation, to_pil, flatten
from priors.common import dump_module, ppca, fit_moments, load_module
from priors.optim import Adam, EMA

from functools import partial
from tqdm import trange
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from utils import make_model, sample, measure, PATH
import zarr  # type: ignore
import time


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_gpu_memory(label: str = ""):
    """Print GPU memory usage and peak for all devices."""
    for i, device in enumerate(jax.devices()):
        try:
            stats = device.memory_stats()
            used_gb = stats['bytes_in_use'] / 1e9
            peak_gb = stats.get('peak_bytes_in_use', 0) / 1e9
            limit_gb = stats.get('bytes_limit', 0) / 1e9
            print(f"  GPU {i}: {used_gb:.2f} GB used, {peak_gb:.2f} GB peak, {limit_gb:.1f} GB total"
                  f" ({100*used_gb/limit_gb:.0f}% used, {100*peak_gb/limit_gb:.0f}% peak)" if limit_gb > 0
                  else f"  GPU {i}: {used_gb:.2f} GB used, {peak_gb:.2f} GB peak")
        except Exception as e:
            print(f"  GPU {i}: memory stats unavailable ({e})")
    if label:
        print(f"  [{label}]", flush=True)


def zarr_batch_iterator(array, batch_size, indices=None, drop_last_batch=True):
    N = array.shape[0]
    if indices is None:
        indices = np.arange(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if drop_last_batch and (end - start) < batch_size:
            break
        yield array[indices[start:end]]


def convert_omegaconf_to_native(obj):
    """Recursively convert OmegaConf objects to Python native types."""
    if OmegaConf.is_list(obj) or isinstance(obj, list):
        return tuple(convert_omegaconf_to_native(x) for x in obj)
    elif OmegaConf.is_dict(obj) or isinstance(obj, dict):
        return {k: convert_omegaconf_to_native(v) for k, v in obj.items()}
    else:
        return obj


def zarr_generate(model, dataset, rng, batch_size, shape, num_gpus, **kwargs):
    """Generate outputs for a dataset (Zarr or dict of arrays) in batches."""
    original_training = getattr(model, 'training', True)
    model.train(False)
    N = dataset['y'][:23340].shape[0]

    def sample_batch(model_fn, y_batch, A_batch, key_batch):
        return sample(model_fn, y_batch, A_batch, key_batch, **kwargs)

    print("Starting generation of dataset with", N, "samples in batches of", batch_size)
    print(f"dataset['y'] shape: {dataset['y'].shape}, dataset['A'] shape: {dataset['A'].shape}")

    # Pre-compile by running on a small batch
    _ = sample_batch(model, dataset['y'][:4], dataset['A'][:4], jax.random.split(rng.split())[0])

    # Make batch size a multiple of GPU count for better utilization
    adjusted_batch_size = (batch_size // num_gpus) * num_gpus
    if adjusted_batch_size != batch_size:
        print(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} for GPU efficiency")
        batch_size = adjusted_batch_size

    xs = []
    for start in (bar := trange(0, N, batch_size, desc="Generating batches", ncols=88)):
        end = min(start + batch_size, N)
        current_batch_size = end - start
        if current_batch_size % 4 != 0:
            adjusted_batch_size = (current_batch_size // 4) * 4
            if adjusted_batch_size == 0:
                continue
            end = start + adjusted_batch_size
        y_batch = dataset['y'][start:end]
        A_batch = dataset['A'][start:end]
        batch_key = rng.split()
        y_batch = jax.device_put(y_batch)
        A_batch = jax.device_put(A_batch)
        x_batch = sample_batch(model, y_batch, A_batch, batch_key)
        xs.append(np.asarray(x_batch))

    xs = np.concatenate(xs, axis=0)
    model.train(original_training)
    return {'x': xs}


def _get_runpath(runid: str) -> Path:
    """Build the run directory path. Must match DummyRun naming convention."""
    return PATH / f'runs/test_run_{runid}_{runid}'


def _setup_jax():
    """Common JAX configuration for all job types."""
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    jax.config.update('jax_compilation_cache_dir', '/tmp')
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)


def _load_or_fit_model(cfg, lap, runpath, src, main_rng, init_rng, sde):
    """Load previous checkpoint (lap>0) or fit Gaussian prior (lap==0).

    Returns (previous_model, use_tikhonov, H, W, C).
    """
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    use_tikhonov = cfg.generate.get('use_tikhonov', True)

    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/test", mode="r")

    # Get shape from a small sample
    y_sample = testset_yA['y'][:1]
    _, H, W, C = y_sample.shape

    if lap > 0:
        checkpoint_path = runpath / f'checkpoint_lap{lap-1:02d}.pkl'
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        converted_model_cfg = convert_omegaconf_to_native(cfg.model)
        model = make_model(key=init_rng.split(), in_channels=C, out_channels=C, **converted_model_cfg)
        model.mu_x = checkpoint_data['mu_x']
        if checkpoint_data.get('cov_x') is not None:
            model.cov_x = checkpoint_data['cov_x']
        static_part, _ = model.partition()
        model = static_part(checkpoint_data['params'])
        model.train(False)
        previous = model
        print(f"[{time.strftime('%X')}] Model loaded successfully from lap {lap-1}")
        if lap > 1 and cfg.generate.get('tikhonov_warmup_only', True):
            use_tikhonov = False
    else:
        fit_samples = cfg.training.fit_samples
        y_fit, A_fit = trainset_yA['y'][:fit_samples], trainset_yA['A'][:fit_samples]
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        B, H, W, C = y_fit.shape
        D = H * W * C
        t1a = time.time()
        mu_x, cov_x = fit_moments(
            features=D,
            rank=cfg.fit_moments.rank,
            shard=True,
            A=inox.tree.Partial(measure, A_fit, H=H, W=W, C=C),
            y=flatten(y_fit),
            cov_y=cfg.training.cov_y,
            sampler='ddim',
            sde=sde,
            steps=cfg.fit_moments.steps,
            maxiter=cfg.training.fit_moments_maxiter,
            key=main_rng.split(),
            method=cfg.training.fit_moments_method,
        )
        jax.debug.print(f"[{time.strftime('%X')}] fit_moments completed in {time.time() - t1a:.2f} seconds")
        print_gpu_memory("after fit_moments")
        del y_fit, A_fit
        previous = GaussianDenoiser(mu_x, cov_x)

    # Move model to devices
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    return previous, use_tikhonov, H, W, C


# ══════════════════════════════════════════════════════════════════════════════
# Job 1 & 2: generate_data  (runs on its own 4-GPU node)
# ══════════════════════════════════════════════════════════════════════════════

def generate_data(cfg: DictConfig, runid: str, lap: int, src: str, split: str):
    """Generate synthetic data for a single split ('train' or 'test').

    Saves the result as a zarr store at:
        <runpath>/generated_lap{lap:02d}_{split}.zarr
    """
    start_time = time.time()
    _setup_jax()

    runpath = _get_runpath(runid)
    runpath.mkdir(parents=True, exist_ok=True)

    print(f"=" * 80)
    print(f"Generating {split} data — Lap {lap}")
    print(f"Run ID: {runid}")
    print(f"Run Path: {runpath}")
    print(f"JAX devices: {jax.devices()}")
    print(f"=" * 80, flush=True)

    base_seed = hash((runpath, lap)) % 2**16
    main_rng = inox.random.PRNG(base_seed)
    init_rng = inox.random.PRNG(main_rng.split())
    # Consume the same number of RNG splits as train() does so seeds stay
    # deterministic regardless of execution order.  train() creates:
    #   init_rng, dropout_rng, sampling_rng  (3 splits from main_rng)
    _ = main_rng.split()  # dropout_rng
    _ = main_rng.split()  # sampling_rng

    sde = VESDE(**cfg.sde)

    previous, use_tikhonov, H, W, C = _load_or_fit_model(
        cfg, lap, runpath, src, main_rng, init_rng, sde,
    )

    num_gpus = len(jax.devices())
    dataset_yA = zarr.open_group(f"{src}/{split}", mode="r")

    t_gen = time.time()
    result = zarr_generate(
        model=previous,
        dataset=dataset_yA,
        rng=main_rng,
        batch_size=cfg.generate.batch_size,
        shape=(H, W, C),
        num_gpus=num_gpus,
        shard=True,
        sampler=cfg.generate.sampler,
        sde=sde,
        steps=cfg.generate.discrete,
        maxiter=cfg.generate.diff_maxiter,
        verbose=cfg.generate.verbose,
        method=cfg.generate.method,
        cov_y=cfg.training.cov_y,
        tikhonov_min_reg=cfg.generate.tikhonov_min_reg,
        tikhonov_base_reg=cfg.generate.tikhonov_base_reg,
        use_tikhonov=use_tikhonov,
    )
    print(f"[{time.strftime('%X')}] Generated {split} set in {time.time() - t_gen:.2f} seconds")
    print_gpu_memory(f"after {split} generation")

    # Save to zarr so the training job can load it (with fast LZ4 compression)
    gen_path = runpath / f'generated_lap{lap:02d}_{split}.zarr'
    z = zarr.open(str(gen_path), mode='w')
    compressor = zarr.Blosc(cname='lz4', clevel=1, shuffle=zarr.Blosc.BITSHUFFLE)
    z.create_dataset('x', data=result['x'], chunks=(100, H, W, C),
                     compressor=compressor, overwrite=True)

    print(f"[{time.strftime('%X')}] Saved generated {split} data ({result['x'].shape}) "
          f"to {gen_path}")
    # Report compressed size
    raw_bytes = result['x'].nbytes
    stored_bytes = z['x'].nbytes_stored if hasattr(z['x'], 'nbytes_stored') else raw_bytes
    print(f"  Raw: {raw_bytes/1e9:.2f} GB, Stored: {stored_bytes/1e9:.2f} GB, "
          f"Ratio: {raw_bytes/max(stored_bytes,1):.1f}x")
    print(f"Total generation time: {time.time() - start_time:.2f} seconds")


# ══════════════════════════════════════════════════════════════════════════════
# Job 3: train_only  (loads pre-generated data, runs training)
# ══════════════════════════════════════════════════════════════════════════════

def train_only(cfg: DictConfig, runid: str, lap: int, src: str):
    """Training step that loads pre-generated zarr data instead of generating inline."""
    start_time = time.time()
    _setup_jax()

    runpath = _get_runpath(runid)
    runpath.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_file = runpath / f'training_log_lap{lap:02d}.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'timestamp'])

    print(f"=" * 80)
    print(f"Training — Lap {lap}")
    print(f"Run ID: {runid}")
    print(f"Run Path: {runpath}")
    print(f"JAX devices: {jax.devices()}")
    print(f"=" * 80, flush=True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    base_seed = hash((runpath, lap)) % 2**16
    main_rng = inox.random.PRNG(base_seed)
    init_rng = inox.random.PRNG(main_rng.split())
    dropout_rng = inox.random.PRNG(main_rng.split())
    sampling_rng = inox.random.PRNG(main_rng.split())

    sde = VESDE(**cfg.sde)

    # ── Load source datasets (for shape info and validation samples) ──
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/test", mode="r")

    y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C

    # ── Load pre-generated data from zarr ──
    train_gen_path = runpath / f'generated_lap{lap:02d}_train.zarr'
    test_gen_path = runpath / f'generated_lap{lap:02d}_test.zarr'

    print(f"Loading pre-generated train data from {train_gen_path}")
    print(f"Loading pre-generated test  data from {test_gen_path}")

    trainset = {'x': zarr.open(str(train_gen_path), mode='r')['x']}
    testset = {'x': zarr.open(str(test_gen_path), mode='r')['x']}
    print(f"  train shape: {trainset['x'].shape}, test shape: {testset['x'].shape}")

    # ── PPCA ──
    t4 = time.time()
    ppca_samples = cfg.training.ppca_samples
    x_fit = trainset['x'][:ppca_samples]
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=cfg.training.ppca_rank, key=main_rng.split())
    del x_fit
    print(f"[{time.strftime('%X')}] PPCA fit in {time.time() - t4:.2f} seconds")
    print_gpu_memory("after PPCA")

    # ── Model initialisation ──
    t5 = time.time()
    if lap > 0:
        checkpoint_path = runpath / f'checkpoint_lap{lap-1:02d}.pkl'
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        converted_model_cfg = convert_omegaconf_to_native(cfg.model)
        model = make_model(key=init_rng.split(), in_channels=C, out_channels=C, **converted_model_cfg)
        model.mu_x = checkpoint_data['mu_x']
        if checkpoint_data.get('cov_x') is not None:
            model.cov_x = checkpoint_data['cov_x']
        static_part, _ = model.partition()
        model = static_part(checkpoint_data['params'])
    else:
        model = make_model(key=main_rng.split(), in_channels=C, out_channels=C, **cfg.model)
    model.train(True)
    print(f"[{time.strftime('%X')}] Model initialized in {time.time() - t5:.2f} seconds")

    model.mu_x = mu_x
    if cfg.training.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif cfg.training.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif cfg.training.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif cfg.training.heuristic == 'cov_x':
        model.cov_x = cov_x

    static, params, others = model.partition(nn.Parameter)
    objective = DenoiserLoss(sde=sde)
    steps = cfg.training.epochs * len(trainset_yA) // cfg.training.batch_size
    optimizer = Adam(steps=steps, **cfg.optimizer)
    opt_state = optimizer.init(params)
    ema = EMA(decay=cfg.training.ema_decay)
    avrg = params
    avrg, params, others, opt_state = jax.device_put(
        (avrg, params, others, opt_state), replicated
    )

    @jax.jit
    @jax.vmap
    def augment(x, key):
        keys = jax.random.split(key, 3)
        x = random_flip(x, keys[0], axis=-2)
        x = random_hue(x, keys[1], delta=1e-2)
        x = random_saturation(x, keys[2], lower=0.95, upper=1.05)
        return x

    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)
        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])
        return objective(static(params, others), x, z, t, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)
        return loss, avrg, params, opt_state

    for i, device in enumerate(jax.devices()):
        memory_info = device.memory_stats()
        print(f"GPU {i}: {memory_info['bytes_in_use'] / 1e9:.1f}GB used")
    print_gpu_memory("before training loop")
    print(f"[{time.strftime('%X')}] Setup complete. Total setup time: "
          f"{time.time() - start_time:.2f} seconds")

    # ── Training loop ──
    for epoch in (bar := trange(cfg.training.epochs, ncols=88)):
        epoch_start = time.time()
        N = trainset['x'].shape[0]
        shuffle_seed = base_seed + lap * cfg.training.epochs + epoch
        indices = np.random.RandomState(shuffle_seed).permutation(N)
        losses = []

        for x_batch in prefetch(zarr_batch_iterator(
            trainset['x'], cfg.training.batch_size,
            indices=indices, drop_last_batch=True,
        )):
            assert x_batch is not None, "x_batch is None!"
            x_batch = jax.device_put(x_batch, distributed)
            x_batch = flatten(x_batch)
            loss, avrg, params, opt_state = sgd_step(
                avrg, params, others, opt_state, x_batch, key=main_rng.split()
            )
            losses.append(loss)
        loss_train = np.stack(losses).mean()

        val_start = time.time()
        losses = []
        for x_batch in prefetch(zarr_batch_iterator(
            testset['x'], cfg.training.batch_size, drop_last_batch=True,
        )):
            x_batch = jax.device_put(x_batch, distributed)
            x_batch = flatten(x_batch)
            loss = ell(avrg, others, x_batch, key=main_rng.split())
            losses.append(loss)
        loss_val = np.stack(losses).mean()
        val_time = time.time() - val_start
        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        if (epoch + 1) % cfg.training.sample_interval == 0:
            t4b = time.time()
            sample_start = time.time()
            model = static(avrg, others)
            model.train(False)
            x = sample(
                model=model,
                y=y_eval,
                A=A_eval,
                key=main_rng.split(),
                shard=True,
                sampler=cfg.generate.sampler,
                steps=cfg.generate.discrete,
                maxiter=cfg.generate.diff_maxiter,
            )
            model.train(True)
            num = x.shape[0]
            cols = int(np.sqrt(num))
            rows = num // cols
            x = x.reshape(rows, cols, H, W, C)

            pil_images = to_pil(x, zoom=4)
            if isinstance(pil_images, list):
                for i, img in enumerate(pil_images):
                    img.save(runpath / f'sample_epoch{epoch:04d}_ch{i}.png')
            else:
                pil_images.save(runpath / f'sample_epoch{epoch:04d}.png')

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss_train, time.time()])
            print(f"Epoch {epoch}: train/loss = {loss_train:.6f}", flush=True)
            print(f"[{time.strftime('%X')}] Generated example images in "
                  f"{time.time() - t4b:.2f} seconds")
            print(f"[{time.strftime('%X')}] Epoch {epoch+1}: "
                  f"train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, "
                  f"epoch_time={time.time() - epoch_start:.2f}s, "
                  f"val_time={val_time:.2f}s, "
                  f"sample_time={time.time() - sample_start:.2f}s")
            print_gpu_memory(f"epoch {epoch+1}")
        else:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss_train, time.time()])
            print(f"[{time.strftime('%X')}] Epoch {epoch+1}: "
                  f"train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, "
                  f"epoch_time={time.time() - epoch_start:.2f}s, "
                  f"val_time={val_time:.2f}s")

    # ── Save checkpoint ──
    t_save = time.time()
    model = static(avrg, others)
    model.train(False)

    static_part, params_part = model.partition()
    checkpoint_data = {
        'params': params_part,
        'others': others,
        'mu_x': model.mu_x,
        'cov_x': getattr(model, 'cov_x', None),
        'lap': lap,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }
    ckpt_path = runpath / f'checkpoint_lap{lap:02d}.pkl'
    ckpt_tmp = runpath / f'checkpoint_lap{lap:02d}.pkl.tmp'
    # Write to temp file first, then rename for atomicity
    with open(ckpt_tmp, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    ckpt_tmp.rename(ckpt_path)
    print(f"[{time.strftime('%X')}] Saved checkpoint in {time.time() - t_save:.2f} seconds")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")


# ══════════════════════════════════════════════════════════════════════════════
# Main: build the DAG and submit
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    # Hydra changes cwd to an output dir — restore the original
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    # Resolve config once so OmegaConf interpolations don't cause issues in pickled jobs
    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # PBS settings
    pbs_cfg = cfg_resolved.pbs
    src = cfg_resolved.data.src

    # --- Resume support ---
    resume_from_lap = cfg_resolved.get("resume_from_lap", None)
    resume_runid = cfg_resolved.get("resume_runid", None)
    runid_file = PATH / 'current_runid.txt'

    if resume_from_lap is not None:
        if resume_runid is not None:
            runid = str(resume_runid)
            runid_file.write_text(runid)
            print(f"Resuming from lap {resume_from_lap} with explicit Run ID: {runid}")
        elif runid_file.exists():
            runid = runid_file.read_text().strip()
            print(f"Resuming from lap {resume_from_lap} with existing Run ID: {runid}")
        else:
            raise RuntimeError(
                "resume_from_lap is set but no resume_runid was provided and "
                f"{runid_file} does not exist. Cannot determine which run to resume."
            )
    else:
        runid = str(uuid.uuid4())[:8]
        runid_file.write_text(runid)
        print(f"Fresh start — Run ID: {runid}")

    print(f"Source: {src}")
    print(f"Max laps: {pbs_cfg.max_jobs}")

    # Environment commands for PBS compute nodes
    env_commands = [
        "source /usr/share/modules/init/bash",
        "module use -a /swbuild/analytix/tools/modulefiles",
        "module load miniconda3/v4",
        "export CONDA_ENVS_PATH=/home3/tmonkman/swbuild3/.conda/envs/",
        "export CONDA_PKGS_DIRS=/home3/tmonkman/swbuild3/.conda/pkgs/",
        "source activate jax_base || { echo 'Failed to activate conda env'; exit 1; }",
        "export OMP_NUM_THREADS=64",
        "export PYTHONUNBUFFERED=1",
        "export WANDB_MODE=offline",
        "export WANDB_API_KEY='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'",
        "cat $PBS_NODEFILE 2>/dev/null || true",
        f"cd {Path(__file__).resolve().parent} || exit 1",
    ]

    # ── Postcondition helpers ──
    runpath = _get_runpath(runid)

    def _checkpoint_exists_for_lap(lap_idx: int) -> bool:
        """True iff the checkpoint for this lap already exists and is valid."""
        ckpt = runpath / f"checkpoint_lap{lap_idx:02d}.pkl"
        if not ckpt.exists():
            return False
        try:
            with open(ckpt, 'rb') as f:
                data = pickle.load(f)
            # Verify essential keys are present
            return 'params' in data and 'lap' in data and data['lap'] == lap_idx
        except Exception:
            return False

    def _generated_data_exists(lap_idx: int, split: str) -> bool:
        """True iff the generated zarr store for this lap/split exists and is valid."""
        gen_path = runpath / f'generated_lap{lap_idx:02d}_{split}.zarr'
        if not gen_path.exists():
            return False
        try:
            z = zarr.open(str(gen_path), mode='r')
            return 'x' in z and z['x'].shape[0] > 0
        except Exception:
            return False

    # ── Build the DAG ──
    num_laps = pbs_cfg.max_jobs
    all_jobs = []       # flat list of all jobs for schedule()
    prev_train_job = None  # the training job from the previous lap

    # Separate walltimes for generation (longer) and training (shorter)
    walltime_gen = cfg_resolved.pbs.get("walltime_gen", "14:00:00")
    walltime_train = cfg_resolved.pbs.get("walltime_train", "12:00:00")

    for lap in range(num_laps):
        # --- gen_train job ---
        gen_train_fn = partial(
            generate_data, cfg=cfg_resolved, runid=runid,
            lap=lap, src=src, split="train",
        )
        j_gen_train = job(
            gen_train_fn,
            name=f"{pbs_cfg.name}_lap{lap:02d}_gen_train",
            ncpus=pbs_cfg.ncpus,
            ngpus=pbs_cfg.ngpus,
            model=pbs_cfg.model,
            mem=pbs_cfg.mem,
            walltime=walltime_gen,
            place=pbs_cfg.place,
            queue=pbs_cfg.queue,
        )
        j_gen_train.ensure(lambda _l=lap: _generated_data_exists(_l, "train"))

        # --- gen_test job ---
        gen_test_fn = partial(
            generate_data, cfg=cfg_resolved, runid=runid,
            lap=lap, src=src, split="test",
        )
        j_gen_test = job(
            gen_test_fn,
            name=f"{pbs_cfg.name}_lap{lap:02d}_gen_test",
            ncpus=pbs_cfg.ncpus,
            ngpus=pbs_cfg.ngpus,
            model=pbs_cfg.model,
            mem=pbs_cfg.mem,
            walltime=walltime_gen,
            place=pbs_cfg.place,
            queue=pbs_cfg.queue,
        )
        j_gen_test.ensure(lambda _l=lap: _generated_data_exists(_l, "test"))

        # --- train job ---
        train_fn = partial(
            train_only, cfg=cfg_resolved, runid=runid,
            lap=lap, src=src,
        )
        j_train = job(
            train_fn,
            name=f"{pbs_cfg.name}_lap{lap:02d}_train",
            ncpus=pbs_cfg.ncpus,
            ngpus=pbs_cfg.ngpus,
            model=pbs_cfg.model,
            mem=pbs_cfg.mem,
            walltime=walltime_train,
            place=pbs_cfg.place,
            queue=pbs_cfg.queue,
        )
        j_train.ensure(lambda _l=lap: _checkpoint_exists_for_lap(_l))

        # --- Dependencies ---
        # gen_train and gen_test both depend on previous lap's training
        if prev_train_job is not None:
            j_gen_train.after(prev_train_job, status="success")
            j_gen_test.after(prev_train_job, status="success")

        # train depends on BOTH generation jobs completing
        j_train.after(j_gen_train, status="success")
        j_train.after(j_gen_test, status="success")

        prev_train_job = j_train
        all_jobs.extend([j_gen_train, j_gen_test, j_train])

    # Optionally limit the number of jobs for testing
    if pbs_cfg.dry_run:
        all_jobs = all_jobs[:3]  # first lap only (gen_train, gen_test, train)
    elif pbs_cfg.dry_run_scheduler:
        all_jobs = all_jobs[:pbs_cfg.max_jobs * 3]

    print(f"\nSubmitting {len(all_jobs)} jobs ({len(all_jobs)//3} laps × 3 jobs/lap) via PBS...")
    print(f"DAG structure per lap: gen_train ∥ gen_test → train\n")
    for i, j in enumerate(all_jobs):
        deps = list(j.dependencies.keys())
        dep_names = [str(d) for d in deps] if deps else ["none"]
        print(f"  [{i:3d}] {j.name} (depends on: {', '.join(dep_names)})")

    # Submit — prune=True skips jobs whose postconditions are already satisfied
    schedule(
        *all_jobs,
        backend="pbs",
        name=f"Training_{pbs_cfg.name}_{runid}",
        interpreter="python -u",
        env=env_commands,
        prune=True,
    )

    print(f"\nAll jobs submitted successfully.")


if __name__ == '__main__':
    main()