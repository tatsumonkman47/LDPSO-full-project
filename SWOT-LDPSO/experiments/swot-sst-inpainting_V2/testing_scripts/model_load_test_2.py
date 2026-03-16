#!/usr/bin/env python
"""
Test that replicates the exact training script conditions before checkpoint loading.
"""

import sys
import os
import time
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import zarr
import wandb

# Add your project paths
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting_V2')
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/inox_local')
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/Tatsu')

import inox
import inox.random
from utils import make_model, PATH
from priors.diffusion import VESDE

CONFIG = {
    'hid_channels': (128, 256, 384),
    'hid_blocks': (5, 5, 5),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {1: 4},
    'dropout': 0.1,
    'checkpoint_layers': (),
    'cov_y': 1e-4**2,
    'sampler': 'ddpm',
    'sde': {'a': 1e-4, 'b': 1e2},
    'heuristic': None,
    'discrete': 256,
    'maxiter': 10,
    'generation_batch_size': 128,
    'epochs': 2,
    'batch_size': 304,
    'scheduler': 'constant',
    'lr_init': 2e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.9999,
}

def debug_print(msg):
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] DEBUG: {msg}", flush=True)

def load_checkpoint_with_rng_context(checkpoint_path, C, init_rng, dropout_rng):
    """Exact copy of your function."""
    inox.random.INOX_RNG.clear()
    with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        previous = make_model(
            key=init_rng.split(), 
            in_channels=C, 
            out_channels=C, 
            **CONFIG
        )
        previous.mu_x = checkpoint_data['mu_x']
        if checkpoint_data.get('cov_x') is not None:
            previous.cov_x = checkpoint_data['cov_x']
        static_part, _ = previous.partition()
        previous = static_part(checkpoint_data['params'])
        previous.train(False)
        return previous, checkpoint_data

def test_training_context_loading():
    """Test checkpoint loading with exact training script context."""
    debug_print("=== Testing with exact training script context ===")
    
    # Replicate your exact setup
    runid = "test_context"
    lap = 1  # Simulate lap 1 (loading lap 0)
    src = "/home/tm3076/scratch/priors_precomputed_datasets/precomputed_data_sst/sst_crho_0.4"
    
    debug_print("Step 1: Initialize W&B (like your script)")
    try:
        # Don't actually log, just test initialization
        debug_print("W&B would be initialized here")
        runpath = PATH / f'runs/test_context'
        runpath.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        debug_print(f"W&B setup failed: {e}")
        return False
    
    debug_print("Step 2: Set JAX config (like your script)")
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    jax.config.update('jax_compilation_cache_dir', '/tmp')
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    
    debug_print("Step 3: Create mesh and sharding (like your script)")
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    
    debug_print("Step 4: Initialize RNG (like your script)")
    base_seed = hash((runpath, lap)) % 2**16
    main_rng = inox.random.PRNG(base_seed)
    init_rng = inox.random.PRNG(main_rng.split())
    dropout_rng = inox.random.PRNG(main_rng.split())
    sampling_rng = inox.random.PRNG(main_rng.split())
    
    debug_print("Step 5: Create SDE and load data (like your script)")
    # This is the CRITICAL part - you create the main RNG context BEFORE loading
    with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
        debug_print("Created main RNG context")
        
        # Create SDE
        sde = VESDE(**CONFIG.get('sde'))
        debug_print("SDE created")
        
        # Load dataset (simulate)
        try:
            if os.path.exists(src):
                trainset_yA = zarr.open_group(f"{src}/train", mode="r")
                testset_yA = zarr.open_group(f"{src}/train", mode="r")
                y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
                y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
                B, H, W, C = y_eval.shape
                debug_print(f"Dataset loaded: shape {y_eval.shape}")
            else:
                debug_print("Dataset path not found, using dummy data")
                C = 3  # Assume 3 channels
        except Exception as e:
            debug_print(f"Dataset loading failed: {e}, using C=3")
            C = 3
        
        # Now the critical test - load checkpoint WITHIN the main context
        debug_print("Step 6: Load checkpoint WITHIN main RNG context")
        
        # Find checkpoint
        checkpoint_path = None
        for run_dir in (PATH / "runs").iterdir():
            if run_dir.is_dir():
                for f in run_dir.glob("checkpoint_*.pkl"):
                    checkpoint_path = f
                    break
            if checkpoint_path:
                break
        
        if not checkpoint_path:
            debug_print("No checkpoint found - test incomplete")
            return False
            
        debug_print(f"Found checkpoint: {checkpoint_path}")
        
        # This replicates your exact nested context situation
        debug_print("About to call nested RNG context loading...")
        
        load_init_rng = inox.random.PRNG(init_rng.split())
        load_dropout_rng = inox.random.PRNG(dropout_rng.split())
        
        # This should trigger the hang if it's a context nesting issue
        debug_print("Calling load_checkpoint_with_rng_context...")
        try:
            previous, checkpoint_data = load_checkpoint_with_rng_context(
                checkpoint_path, 
                C,
                load_init_rng, 
                load_dropout_rng
            )
            debug_print("✅ Checkpoint loaded successfully in nested context!")
            return True
        except Exception as e:
            debug_print(f"❌ Checkpoint loading failed in nested context: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    debug_print("Testing checkpoint loading with training script context...")
    
    # Test the exact scenario from your training script
    success = test_training_context_loading()
    
    if success:
        debug_print("✅ Test passed - no context issues detected")
    else:
        debug_print("❌ Test failed - context issue detected")

if __name__ == "__main__":
    main()