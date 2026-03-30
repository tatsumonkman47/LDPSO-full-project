#!/usr/bin/env python
"""
Split-job training script (Slurm / Torch cluster version).

Each lap is split into 3 Slurm jobs:
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
import wandb                 # type: ignore  # Weights & Biases logging
import jax.numpy as jnp      # type: ignore
import pickle
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

    # ── Initialize wandb ──
    if lap == 0:
        run = wandb.init(
            project=cfg.wandb.project,
            id=runid,
            resume='never',
            dir=PATH,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=None,
            tags=['split_job_training', f'lap_{lap}'] + list(cfg.wandb.get('tags', [])),
        )
        auto_name = run.name
        custom_name = f'{auto_name}_{cfg.slurm.name}_{runid}'
        run.name = custom_name
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            id=runid,
            resume='must',
            dir=PATH,
            tags=[f'lap_{lap}'],
        )

    runpath = _get_runpath(runid)
    runpath.mkdir(parents=True, exist_ok=True)

    print(f"=" * 80)
    print(f"Training — Lap {lap}")
    print(f"Run ID: {runid}")
    print(f"Run Path: {runpath}")
    print(f"wandb run: {run.name} ({run.id})")
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
            # Save images locally as well
            if isinstance(pil_images, list):
                for i, img in enumerate(pil_images):
                    img.save(runpath / f'sample_epoch{epoch:04d}_ch{i}.png')
            else:
                pil_images.save(runpath / f'sample_epoch{epoch:04d}.png')

            # Log to wandb with sample images
            log_dict = {
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'sample_time': time.time() - sample_start,
                'lap': lap,
                'global_epoch': lap * cfg.training.epochs + epoch,
            }
            if isinstance(pil_images, list):
                for i, img in enumerate(pil_images):
                    log_dict[f'samples_channel_{i}'] = wandb.Image(img)
            else:
                log_dict['samples'] = wandb.Image(pil_images)
            run.log(log_dict)

            print(f"[{time.strftime('%X')}] Generated example images in "
                  f"{time.time() - t4b:.2f} seconds")
            print(f"[{time.strftime('%X')}] Epoch {epoch+1}: "
                  f"train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, "
                  f"epoch_time={time.time() - epoch_start:.2f}s, "
                  f"val_time={val_time:.2f}s, "
                  f"sample_time={time.time() - sample_start:.2f}s")
            print_gpu_memory(f"epoch {epoch+1}")
        else:
            # Log to wandb without sample images
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'lap': lap,
                'global_epoch': lap * cfg.training.epochs + epoch,
            })
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

    run.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Main: build the DAG and submit
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    # Hydra changes cwd to an output dir — restore the original
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    # Ensure Slurm binaries are on PATH (Torch cluster keeps them in /opt/slurm/bin)
    slurm_bin = "/opt/slurm/bin"
    if slurm_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = slurm_bin + ":" + os.environ.get("PATH", "")

    # Unset SLURM_CONF if it points to a non-existent file (e.g. the slurmd
    # conf-cache path that only exists on compute nodes).  This lets sbatch
    # fall back to the default search path and find /run/slurm/conf/slurm.conf.
    slurm_conf = os.environ.get("SLURM_CONF", "")
    if slurm_conf and not os.path.isfile(slurm_conf):
        print(f"WARNING: SLURM_CONF={slurm_conf} does not exist — unsetting it")
        del os.environ["SLURM_CONF"]

    # Log in to wandb on the login node (before submitting jobs)
    wandb.login()

    # Resolve config once so OmegaConf interpolations don't cause issues in pickled jobs
    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Slurm settings
    slurm_cfg = cfg_resolved.slurm
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
    print(f"Max laps: {slurm_cfg.max_jobs}")

    # Environment commands for Slurm compute nodes
    # The singularity container is handled by the dawgz SlurmScheduler's
    # singularity support.  These env commands run inside the container.
    #
    # NOTE: The dawgz version at dawgz_with_python_singularity does NOT have
    # built-in singularity support in SlurmScheduler.  If you need singularity,
    # either:
    #   (a) switch to the dawgz_tatsu fork which has singularity= kwarg, or
    #   (b) wrap the python command in the env commands below.
    #
    # The env commands below assume option (b): the container is entered via
    # a wrapper in `env`.  Adjust the overlay and .sif paths for your system.
    singularity_cmd = slurm_cfg.get("singularity", None)
    project_dir = str(Path(__file__).resolve().parent)
    priors_dir = str(Path(__file__).resolve().parent.parent.parent)

    if singularity_cmd:
        # Inject PYTHONPATH *inside* the singularity container command, right
        # before {python_command}.  We must do it there (not in the outer host
        # shell) because `conda activate` inside the container resets
        # PYTHONPATH, so any value set on the host is lost.
        pythonpath_export = (
            f"export PYTHONPATH={project_dir}:{priors_dir}:${{PYTHONPATH:-}}"
        )
        # Insert the PYTHONPATH export just before {python_command} in the
        # singularity template.  The template typically ends with something
        # like:  ... && conda activate jax_base && {python_command}'
        singularity_cmd = singularity_cmd.replace(
            "{python_command}",
            f"{pythonpath_export} && {{python_command}}",
        )
        env_commands = [
            "export PYTHONUNBUFFERED=1",
            "export WANDB_SILENT=true",
            f"cd {project_dir} || exit 1",
        ]
    else:
        env_commands = [
            "export PYTHONUNBUFFERED=1",
            "export WANDB_SILENT=true",
            f"export PYTHONPATH={project_dir}:{priors_dir}:${{PYTHONPATH:-}}",
            f"cd {project_dir} || exit 1",
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
    num_laps = slurm_cfg.max_jobs
    all_jobs = []       # flat list of all jobs for schedule()
    prev_train_job = None  # the training job from the previous lap

    # Separate walltimes for generation (longer) and training (shorter)
    # Guard: YAML parses unquoted HH:MM:SS as sexagesimal integers (e.g. 14:00:00 → 50400).
    # Convert back to HH:MM:SS string if that happened.
    def _ensure_time_str(val, default="14:00:00"):
        if val is None:
            return default
        if isinstance(val, int):
            h, rem = divmod(val, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}"
        return str(val)

    time_gen = _ensure_time_str(slurm_cfg.get("time_gen", None), "14:00:00")
    time_train = _ensure_time_str(slurm_cfg.get("time_train", None), "12:00:00")

    for lap in range(num_laps):
        # --- gen_train job ---
        gen_train_fn = partial(
            generate_data, cfg=cfg_resolved, runid=runid,
            lap=lap, src=src, split="train",
        )
        j_gen_train = job(
            gen_train_fn,
            name=f"{slurm_cfg.name}_lap{lap:02d}_gen_train",
            cpus=slurm_cfg.cpus,
            gpus=slurm_cfg.gpus,
            ram=slurm_cfg.ram,
            time=time_gen,
            account=slurm_cfg.account,
        )
        j_gen_train.ensure(lambda _l=lap: _generated_data_exists(_l, "train"))

        # --- gen_test job ---
        gen_test_fn = partial(
            generate_data, cfg=cfg_resolved, runid=runid,
            lap=lap, src=src, split="test",
        )
        j_gen_test = job(
            gen_test_fn,
            name=f"{slurm_cfg.name}_lap{lap:02d}_gen_test",
            cpus=slurm_cfg.cpus,
            gpus=slurm_cfg.gpus,
            ram=slurm_cfg.ram,
            time=time_gen,
            account=slurm_cfg.account,
        )
        j_gen_test.ensure(lambda _l=lap: _generated_data_exists(_l, "test"))

        # --- train job ---
        train_fn = partial(
            train_only, cfg=cfg_resolved, runid=runid,
            lap=lap, src=src,
        )
        j_train = job(
            train_fn,
            name=f"{slurm_cfg.name}_lap{lap:02d}_train",
            cpus=slurm_cfg.cpus,
            gpus=slurm_cfg.gpus,
            ram=slurm_cfg.ram,
            time=time_train,
            account=slurm_cfg.account,
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
    if slurm_cfg.dry_run:
        all_jobs = all_jobs[:3]  # first lap only (gen_train, gen_test, train)
    elif slurm_cfg.dry_run_scheduler:
        all_jobs = all_jobs[:slurm_cfg.max_jobs * 3]

    print(f"\nSubmitting {len(all_jobs)} jobs ({len(all_jobs)//3} laps × 3 jobs/lap) via Slurm...")
    print(f"DAG structure per lap: gen_train ∥ gen_test → train\n")
    for i, j in enumerate(all_jobs):
        deps = list(j.dependencies.keys())
        dep_names = [str(d) for d in deps] if deps else ["none"]
        print(f"  [{i:3d}] {j.name} (depends on: {', '.join(dep_names)})")

    # Submit — prune=True skips jobs whose postconditions are already satisfied
    schedule(
        *all_jobs,
        backend="slurm",
        name=f"Training_{slurm_cfg.name}_{runid}",
        interpreter="python -u",
        env=env_commands,
        singularity=singularity_cmd if singularity_cmd else None,
        export='ALL',
        prune=True,
    )

    print(f"\nAll jobs submitted successfully.")


if __name__ == '__main__':
    main()