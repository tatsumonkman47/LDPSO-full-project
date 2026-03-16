#!/usr/bin/env python
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
import numpy as np # type: ignore
import optax                 # type: ignore
#import wandb                 # type: ignore
import jax.numpy as jnp # type: ignore
import pickle
import csv
import uuid

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
from priors.data import prefetch
from priors.image import random_flip, random_hue, random_saturation, to_pil, flatten
from priors.common import dump_module, ppca, fit_moments, load_module
from priors.optim import Adam, EMA

from functools import partial
from tqdm import trange
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from utils import make_model, sample, measure, PATH
import zarr # type: ignore
import time

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
    # Force eval mode during generation
    original_training = getattr(model, 'training', True)
    # Set eval mode with proper RNG context
    model.train(False)
    N = dataset['y'][:23340].shape[0]
    
    def sample_batch(model_fn, y_batch, A_batch, key_batch):
        # Explicitly pass all parameters to sample
        return sample(
            model_fn, y_batch, A_batch, key_batch,
            **kwargs
        )
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
        # Ensure batch size is divisible by 4
        current_batch_size = end - start
        if current_batch_size % 4 != 0:
            # Adjust end to make batch size divisible by 4
            adjusted_batch_size = (current_batch_size // 4) * 4
            if adjusted_batch_size == 0:
                # Skip this batch if it would be empty
                continue
            end = start + adjusted_batch_size
        # Extract data for this batch
        y_batch = dataset['y'][start:end]
        A_batch = dataset['A'][start:end]
        # Use a single key for the batch
        batch_key = rng.split()
        # Put data on devices
        y_batch = jax.device_put(y_batch)
        A_batch = jax.device_put(A_batch)
        # Execute batch sampling
        x_batch = sample_batch(model, y_batch, A_batch, batch_key)
        # Move results back to host memory
        xs.append(np.asarray(x_batch))
        
    # Combine all batches
    xs = np.concatenate(xs, axis=0)
    model.train(original_training)
    return {'x': xs}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def train(cfg: DictConfig, runid: str, lap: int, src: str):
    """Main training loop for a single training lap."""
    start_time = time.time()
    
    """
    # Initialize wandb
    if lap == 0:
        run = wandb.init(
            project=cfg.wandb.project,
            id=runid,
            resume='never',
            dir=PATH,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=None,
            tags=['single_lap_training', f'lap_{lap}'] + cfg.wandb.tags
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
            tags=[f'lap_{lap}']
        )
    """
    class DummyRun:
        def __init__(self, cfg):
            self.name = f"test_run_{runid}"
            self.id = runid
            self.config = cfg
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass
    run = DummyRun(cfg)  # ADD cfg parameter here
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)
    config = run.config
    # Create log file
    log_file = runpath / f'training_log_lap{lap:02d}.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'timestamp'])
    print(f"=" * 80)
    print(f"Starting Training Lap {lap}")
    print(f"Run ID: {runid}")
    print(f"Run Path: {runpath}")
    print(f"=" * 80, flush=True)

    print(f"TRAIN DEBUG: Starting lap {lap}, PID={os.getpid()}", flush=True)
    sys.stdout.flush()
    
    # REMOVE THIS LINE - it forces single device
    # os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    jax.config.update('jax_compilation_cache_dir', '/tmp')
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    
    print(f"TRAIN DEBUG: JAX config set", flush=True)
    print(f"Starting Lap {lap} with runid {runid}")
    print(f"Source directory: {src}")
    print(f"JAX devices available: {jax.devices()}")

    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    base_seed = hash((runpath, lap)) % 2**16
    main_rng = inox.random.PRNG(base_seed)
    init_rng = inox.random.PRNG(main_rng.split())
    dropout_rng = inox.random.PRNG(main_rng.split())
    sampling_rng = inox.random.PRNG(main_rng.split())

    sde = VESDE(**cfg.sde)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load HuggingFace-formatted LLC4320 dataset
    t0 = time.time()
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/test", mode="r")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Validation data (fixed samples)
    y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C
    jax.debug.print(f"[{time.strftime('%X')}] Loaded dataset in {time.time() - t0:.2f} seconds")

    # If lap >0, load previous checkpoint, else fit prior Gaussian model
    t1 = time.time()
    # Usage in your main training function:
    use_tikhonov = cfg.generate.get('use_tikhonov', True)
    if lap > 0:
        checkpoint_path = runpath / f'checkpoint_lap{lap-1:02d}.pkl'
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        # Convert config parameters to native Python types
        converted_model_cfg = convert_omegaconf_to_native(cfg.model)
         # Create model once
        model = make_model(key=init_rng.split(), in_channels=C, out_channels=C, **converted_model_cfg)
        # Load parameters
        model.mu_x = checkpoint_data['mu_x']
        if checkpoint_data.get('cov_x') is not None:
            model.cov_x = checkpoint_data['cov_x']
        static_part, _ = model.partition()
        model = static_part(checkpoint_data['params'])
        model.train(False)
        # REUSE the same model instance for generation AND training
        previous = model
        print(f"[{time.strftime('%X')}] Model loaded successfully from lap {lap-1}")
        if lap > 1 and cfg.generate.get('tikhonov_warmup_only', True):
            use_tikhonov = False  # Disable Tikhonov regularization after first lap
    else:
        y_fit, A_fit = trainset_yA['y'][:16384], trainset_yA['A'][:16384]
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        jax.debug.print(f"[{time.strftime('%X')}] Loaded fitting dataset in {time.time() - t0:.2f} seconds")
        B, H, W, C = y_fit.shape
        D = H * W * C
        t1a = time.time()
        mu_x, cov_x = fit_moments(
            features=D, # The dimensionality of the latent variable x
            rank=320, # This is the low-rank dimension of your approximate posterior or prior covariance matrix
            shard=True,
            A=inox.tree.Partial(measure, A_fit, H=H, W=W, C=C),
            y=flatten(y_fit),
            cov_y=cfg.training.cov_y, # Expected observation noise covariance
            sampler='ddim',
            sde=sde,
            steps=256,
            maxiter=cfg.training.fit_moments_maxiter, # Increased for robustness
            key=main_rng.split(),
            method=cfg.training.fit_moments_method,
        )
        jax.debug.print(f"[{time.strftime('%X')}] fit_moments completed in {time.time() - t1a:.2f} seconds")
        del y_fit, A_fit
        previous = GaussianDenoiser(mu_x, cov_x)
        jax.debug.print(f"[{time.strftime('%X')}] GaussianDenoiser created in {time.time() - t1:.2f} seconds")

    # Prepare the previous model for sampling new training targets
    t2 = time.time()
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)
    print(f"[{time.strftime('%X')}] Model partitioned and moved to device in {time.time() - t2:.2f} seconds")

    # Generate synthetic training and testing data (denoised reconstructions)
    t3 = time.time()
    num_gpus = len(jax.devices())
    trainset = zarr_generate(
        model=previous,
        dataset=trainset_yA,
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
        tikhonov_min_reg = cfg.generate.tikhonov_min_reg,
        tikhonov_base_reg = cfg.generate.tikhonov_base_reg,
        use_tikhonov = use_tikhonov
    )
    print(f"[{time.strftime('%X')}] Generated trainset in {time.time() - t3:.2f} seconds")
    t3b = time.time()
    testset = zarr_generate(
        model=previous,
        dataset=testset_yA,
        rng=main_rng,
        batch_size=cfg.generate.batch_size,
        shape = (H, W, C),
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
        use_tikhonov = use_tikhonov,
    )
    jax.debug.print(f"[{time.strftime('%X')}] Generated testset in {time.time() - t3b:.2f} seconds")

    t4 = time.time()
    x_fit = trainset['x'][:16384]
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=main_rng.split())
    del x_fit
    print(f"[{time.strftime('%X')}] PPCA fit in {time.time() - t4:.2f} seconds")

    t5 = time.time()
    if lap > 0:
        model = previous
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
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

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
    print(f"[{time.strftime('%X')}] Setup complete, entering training loop. Total setup time: {time.time() - start_time:.2f} seconds")

    for epoch in (bar := trange(cfg.training.epochs, ncols=88)):
        epoch_start = time.time()
        N = trainset['x'].shape[0]
        shuffle_seed = base_seed + lap * cfg.training.epochs + epoch
        indices = np.random.RandomState(shuffle_seed).permutation(N)
        losses = []
        
        for x_batch in prefetch(zarr_batch_iterator(trainset['x'], cfg.training.batch_size, indices=indices, drop_last_batch=True)):
            assert x_batch is not None, "x_batch is None!"
            x_batch = jax.device_put(x_batch, distributed)
            x_batch = flatten(x_batch)
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x_batch, key=main_rng.split())
            losses.append(loss)
        loss_train = np.stack(losses).mean()

        val_start = time.time()
        losses = []
        for x_batch in prefetch(zarr_batch_iterator(testset['x'], cfg.training.batch_size, drop_last_batch=True)):
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
            log_dict = {
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'sample_time': time.time() - sample_start,
                'lap': lap,
                'global_epoch': lap * cfg.training.epochs + epoch,
            }
            """
            if isinstance(pil_images, list):
                for i, img in enumerate(pil_images):
                    log_dict[f'samples_channel_{i}'] = wandb.Image(img)
            else:
                log_dict['samples'] = wandb.Image(pil_images)
            """
            if isinstance(pil_images, list):
                for i, img in enumerate(pil_images):
                    img.save(runpath / f'sample_epoch{epoch:04d}_ch{i}.png')
            else:
                pil_images.save(runpath / f'sample_epoch{epoch:04d}.png')
            #run.log(log_dict) # Wandb logging is disabled in this version
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss_train, time.time()])
            print(f"Epoch {epoch}: train/loss = {loss_train:.6f}", flush=True)
            print(f"[{time.strftime('%X')}] Generated example images in {time.time() - t4b:.2f} seconds")
            print(f"[{time.strftime('%X')}] Epoch {epoch+1}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, epoch_time={time.time() - epoch_start:.2f}s, val_time={val_time:.2f}s, sample_time={time.time() - sample_start:.2f}s")
        else:
            """wandb loggin is disabled in this version
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'lap': lap,
                'global_epoch': lap*cfg.training.epochs + epoch,
            })"""
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss_train, time.time()])  # Change from loss.item()
            print(f"[{time.strftime('%X')}] Epoch {epoch+1}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, epoch_time={time.time() - epoch_start:.2f}s, val_time={val_time:.2f}s")
    
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
    with open(runpath / f'checkpoint_lap{lap:02d}.pkl', 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"[{time.strftime('%X')}] Saved checkpoint in {time.time() - t_save:.2f} seconds")

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    #wandb.login() # type: ignore
    
    # Get lap number from environment variable or default to 0
    lap = int(os.environ.get('LAP_NUMBER', 0))
    # Save/load runid to maintain consistency across laps
    runid_file = PATH / 'current_runid.txt'
    if lap == 0:
        #runid = wandb.util.generate_id() # type: ignore
        runid = str(uuid.uuid4())[:8]  # Generate a simple UUID-based runid
        runid_file.write_text(runid)
    else:
        if not runid_file.exists():
            raise FileNotFoundError(f"Cannot resume lap {lap}: runid file not found")
        runid = runid_file.read_text().strip()
    src = cfg.data.src
    print(f"Running single lap: {lap}")
    print(f"Run ID: {runid}")
    # Run training directly
    train(cfg=cfg, runid=runid, lap=lap, src=src)

if __name__ == '__main__':
    main()