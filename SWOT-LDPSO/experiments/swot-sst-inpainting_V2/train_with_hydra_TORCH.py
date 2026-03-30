#!/usr/bin/env python
# Core Libraries
import inox                  # type: ignore # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # type: ignore # type: ignore # Neural network components
from inox import random as inox_random # type: ignore
import jax                   # type: ignore # JAX for high-performance computing
import numpy as np # type: ignore
import optax                 # type: ignore # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking
import jax.numpy as jnp # type: ignore # JAX's numpy for array operations
import sys
import os
import pickle

# Hydra imports
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

# Workflow management
from dawgz import job, schedule # type: ignore
from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
from priors.data import prefetch
from priors.image import random_flip, random_hue, random_saturation, to_pil, flatten
from priors.common import dump_module, ppca, fit_moments, load_module
from priors.optim import Adam, EMA

from functools import partial
from tqdm import trange
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import zarr # type: ignore
import time
from pathlib import Path

def zarr_batch_iterator(array, batch_size, indices=None, drop_last_batch=True):
    N = array.shape[0]
    if indices is None:
        indices = np.arange(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if drop_last_batch and (end - start) < batch_size:
            break
        yield array[indices[start:end]]

def zarr_generate(model, dataset, rng, batch_size, shape, num_gpus, **kwargs):
    """Generate outputs for a dataset (Zarr or dict of arrays) in batches."""
    from utils import sample # type: ignore
    # Force eval mode during generation
    original_training = getattr(model, 'training', True)
    model.train(False)
    N = dataset['y'].shape[0]

    # Call sample directly on the batch (sample already handles the batch dim)
    def sample_batch(model_fn, y_batch, A_batch, key_batch):
        return sample(model_fn, y_batch, A_batch, key_batch, **kwargs)
    # Pre-compile on a small batch
    _ = sample_batch(model, dataset['y'][:4], dataset['A'][:4], 
                     rng.split())

    # Make batch size a multiple of GPU count for better utilization
    adjusted_batch_size = (batch_size // num_gpus) * num_gpus
    if adjusted_batch_size != batch_size:
        print(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} for GPU efficiency")
        batch_size = adjusted_batch_size
    
    xs = []
    for start in (bar := trange(0, N, batch_size, desc="Generating batches", ncols=88)):
        end = min(start + batch_size, N)
        current_batch_size = end - start
        # Extract data for this batch
        y_batch = dataset['y'][start:end]
        A_batch = dataset['A'][start:end]
        
        # Pad the last batch to be divisible by num_gpus
        remainder = current_batch_size % num_gpus
        if remainder != 0:
            pad_size = num_gpus - remainder
            # Repeat the last sample to fill the pad
            y_pad = np.repeat(y_batch[-1:], pad_size, axis=0)
            A_pad = np.repeat(A_batch[-1:], pad_size, axis=0)
            y_batch = np.concatenate([y_batch, y_pad], axis=0)
            A_batch = np.concatenate([A_batch, A_pad], axis=0)
        
        # Generate a single key for the batch (not one per sample)
        batch_key = rng.split()
        # Put data on devices in a sharded manner
        y_batch = jax.device_put(y_batch)
        A_batch = jax.device_put(A_batch)
        # Execute the batch sampling
        x_batch = sample_batch(model, y_batch, A_batch, batch_key)
        # Move results back to host memory and trim padding
        x_batch = np.asarray(x_batch)
        if remainder != 0:
            x_batch = x_batch[:current_batch_size]
        xs.append(x_batch)
    
    # Combine all batches
    xs = np.concatenate(xs, axis=0)
    model.train(original_training)
    return {'x': xs}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Capture the project directory for later use in imports and file paths
PROJECT_DIR = str(Path(__file__).resolve().parent)

def train(cfg: DictConfig, runid: str, lap: int, src: str):
    """
    Main training loop for a single training 'lap' (iteration).
    Each lap can be seen as one cycle of training, optionally starting from a prior checkpoint.
    """
    # Ensure the project directory is in the Python path for imports
    import sys
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Import utilities after setting the path
    from utils import make_model, sample, measure, PATH

    print(f"Starting Lap {lap} with runid {runid}")
    
    # Force early logging before ANY JAX operations
    # Initialize Weights & Biases
    start_time = time.time()
    # REPLACE the wandb.init section with:
    if lap == 0:
        # First lap - create new run
        run = wandb.init(
            project=cfg.wandb.project,
            id=runid,
            resume='never',  # Ensure fresh start for lap 0
            dir=PATH,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=None,  # Let wandb generate the name first
            tags=['multi_lap_training', f'lap_{lap}'] + cfg.wandb.tags
        )
        # Get the auto-generated name and modify it
        auto_name = run.name
        custom_name = f'{auto_name}_{cfg.slurm.name}_{runid}'
        run.name = custom_name
    else:
        # Subsequent laps - resume existing run
        run = wandb.init(
            project=cfg.wandb.project,
            id=runid,  # SAME ID as lap 0
            resume='must',  # Must resume existing run
            dir=PATH,
            # Don't pass config again for resumed runs
            tags=[f'lap_{lap}']  # Add lap-specific tag
        )
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)
    config = run.config

    print(f"TRAIN DEBUG: Starting lap {lap}, PID={os.getpid()}", flush=True)
    sys.stdout.flush()
    # Set JAX compilation flags to be more verbose and lazy
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    jax.config.update('jax_compilation_cache_dir', '/tmp')
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    
    print(f"TRAIN DEBUG: JAX config set", flush=True)
    print(f"Starting Lap {lap} with runid {runid}")
    print(f"Source directory: {src}")

    # Enable partitioning for reproducible RNG across shards
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)  # Use float32 everywhere
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # Initialize PRNG with lap-specific seed
    base_seed = hash((runpath, lap)) % 2**16
    
    # Create multiple RNG streams for different purposes
    main_rng = inox.random.PRNG(base_seed)
    init_rng = inox.random.PRNG(main_rng.split())
    dropout_rng = inox.random.PRNG(main_rng.split())
    sampling_rng = inox.random.PRNG(main_rng.split())

    # Create the SDE object (Variance Exploding SDE)
    sde = VESDE(**cfg.sde)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load HuggingFace-formatted LLC4320 dataset
    t0 = time.time()
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/train", mode="r")
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
    if lap > 0:
        checkpoint_path = runpath / f'checkpoint_lap{lap-1:02d}.pkl'
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        # Create model once
        model = make_model(key=init_rng.split(), in_channels=C, out_channels=C, **cfg.model)
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
    )
    jax.debug.print(f"[{time.strftime('%X')}] Generated testset in {time.time() - t3b:.2f} seconds")

    # Fit low-rank covariance (PPCA) on generated training data
    t4 = time.time()
    x_fit = trainset['x'][:16384]
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=main_rng.split())
    del x_fit
    print(f"[{time.strftime('%X')}] PPCA fit in {time.time() - t4:.2f} seconds")

    # Initialize model
    t5 = time.time()

    if lap > 0:
        model = previous
    else:
        model = make_model(key=main_rng.split(), in_channels=C, out_channels=C, **cfg.model)
    model.train(True)
    print(f"[{time.strftime('%X')}] Model initialized in {time.time() - t5:.2f} seconds")

    # Set model's prior mean
    model.mu_x = mu_x
    # Configure model's covariance heuristic
    if cfg.training.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif cfg.training.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif cfg.training.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif cfg.training.heuristic == 'cov_x':
        model.cov_x = cov_x

    # Partition model parameters
    static, params, others = model.partition(nn.Parameter)
    # Define denoising loss
    objective = DenoiserLoss(sde=sde)
    # Build optimizer
    steps = cfg.training.epochs * len(trainset_yA) // cfg.training.batch_size
    optimizer = Adam(steps=steps, **cfg.optimizer)
    opt_state = optimizer.init(params)
    # Exponential moving average for parameter stabilization
    ema = EMA(decay=cfg.training.ema_decay)
    avrg = params
    # Put everything onto devices
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

    # Data augmentation function (random flips, hue, saturation)
    @jax.jit
    @jax.vmap
    def augment(x, key):
        keys = jax.random.split(key, 3)
        x = random_flip(x, keys[0], axis=-2)
        x = random_hue(x, keys[1], delta=1e-2)
        x = random_saturation(x, keys[2], lower=0.95, upper=1.05)
        return x

    # Loss computation
    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)
        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])
        return objective(static(params, others), x, z, t, key=keys[2])

    # Single SGD update step
    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)
        return loss, avrg, params, opt_state
    
    # Check memory usage
    for i, device in enumerate(jax.devices()):
        memory_info = device.memory_stats()
        print(f"GPU {i}: {memory_info['bytes_in_use'] / 1e9:.1f}GB used")
    print(f"[{time.strftime('%X')}] Setup complete, entering training loop. Total setup time: {time.time() - start_time:.2f} seconds")

    # Training loop over epochs
    for epoch in (bar := trange(cfg.training.epochs, ncols=88)):
        epoch_start = time.time()
        # Shuffle training set per epoch
        N = trainset['x'].shape[0]  # or whatever your dataset size is
        shuffle_seed = base_seed + lap * cfg.training.epochs + epoch
        indices = np.random.RandomState(shuffle_seed).permutation(N)
        losses = []
        #for batch in prefetch(loader):
        for x_batch in prefetch(zarr_batch_iterator(trainset['x'], cfg.training.batch_size, indices=indices, drop_last_batch=True)):
            assert x_batch is not None, "x_batch is None!"
            x_batch = jax.device_put(x_batch, distributed)
            x_batch = flatten(x_batch)
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x_batch, key=main_rng.split())
            losses.append(loss)
        loss_train = np.stack(losses).mean()

        # Validation evaluation
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

        # Every 16 epochs, sample validation images and log to wandb
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
            model.train(True)  # Restore training mode if continuing to train
            num = x.shape[0]
            cols = int(np.sqrt(num))
            rows = num // cols
            x = x.reshape(rows, cols, H, W, C)
            
            # Fix: Handle the list returned by to_pil
            pil_images = to_pil(x, zoom=4)
            log_dict = {
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'sample_time': time.time() - sample_start,
                'lap': lap,  # Add lap info
                'global_epoch': lap * cfg.training.epochs + epoch,
            }
            # Handle single image or multiple channels
            if isinstance(pil_images, list):
                # Log each channel separately
                for i, img in enumerate(pil_images):
                    log_dict[f'samples_channel_{i}'] = wandb.Image(img)
            else:
                # Single image case
                log_dict['samples'] = wandb.Image(pil_images)
            run.log(log_dict)
            jax.debug.print(f"[{time.strftime('%X')}] Generated example images in {time.time() - t4b:.2f} seconds")
            jax.debug.print(f"[{time.strftime('%X')}] Epoch {epoch+1}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, epoch_time={time.time() - epoch_start:.2f}s, val_time={val_time:.2f}s, sample_time={time.time() - sample_start:.2f}s")
        else:
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'lap': lap,
                'global_epoch': lap*cfg.training.epochs + epoch,
            })
            jax.debug.print(f"[{time.strftime('%X')}] Epoch {epoch+1}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, epoch_time={time.time() - epoch_start:.2f}s, val_time={val_time:.2f}s")

    # Save checkpoint
    t_save = time.time()
    model = static(avrg, others)
    model.train(False)
    
    # Save only parameters, not the full model
    static_part, params_part = model.partition()
    checkpoint_data = {
        'params': params_part,
        'others': others,
        'mu_x': model.mu_x,  # Save the prior mean
        'cov_x': getattr(model, 'cov_x', None),  # Save covariance if it exists
        'lap': lap,
        'config': OmegaConf.to_container(cfg, resolve=True),  # Save config for reconstruction
    }
    with open(runpath / f'checkpoint_lap{lap:02d}.pkl', 'wb') as f:
        pickle.dump(checkpoint_data, f)
    """
    dump_module(model, runpath / f'checkpoint_{lap}.pkl')
    """
    print(f"[{time.strftime('%X')}] Saved checkpoint in {time.time() - t_save:.2f} seconds")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb.login() # type: ignore
    runid = wandb.util.generate_id() # type: ignore
    jobs = []
    src = cfg.data.src
    
    # Schedule multiple laps as Slurm jobs
    for lap in range(0, cfg.slurm.max_jobs):
        jobs.append(
            job(
                partial(train, cfg=cfg, runid=runid, lap=lap, src=src),
                name=f'train_{lap}',
                cpus=cfg.slurm.cpus,
                gpus=cfg.slurm.gpus,
                ram=cfg.slurm.ram,
                time=cfg.slurm.time,
                account=cfg.slurm.account,
                #partition=cfg.slurm.partition,
                #delay="00:05:00",
                #wrap='\"hostname && sleep infinity\"'
           )
        )
        if len(jobs) > 1:
            jobs[-1].after(jobs[-2], status="success")
    
    if cfg.slurm.dry_run:
        jobs = jobs[:1]
    if cfg.slurm.dry_run_scheduler:
        jobs = jobs[:cfg.slurm.max_jobs]

    # Add debug prints to see what DAWGZ is actually doing
    print(f"DAWGZ DEBUG: Created {len(jobs)} jobs")
    for i, job_i in enumerate(jobs):
        print(f"DAWGZ DEBUG: Job {i}: {job_i}")
        if hasattr(job_i, 'dependencies'):
            print(f"DAWGZ DEBUG: Job {i} dependencies: {job_i.dependencies}")

    schedule(
        *jobs,
        name=f'Training {runid}',
        backend='slurm',
        #debug=True,
        export='ALL',
        env=['export WANDB_SILENT=true'],
        #dry_run=False,
        singularity=cfg.slurm.singularity
    )

if __name__ == '__main__':
    main()