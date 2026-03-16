#!/usr/bin/env python
"""
Test zarr_generate function specifically to isolate potential hangs.
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

# Add your project paths
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting_V2')
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/inox_local')
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/Tatsu')

import inox
import inox.random
from utils import make_model, PATH, sample
from priors.diffusion import VESDE
from train import zarr_generate, CONFIG

def debug_print(msg):
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] DEBUG: {msg}", flush=True)

def test_zarr_generate_with_loaded_model():
    """Test zarr_generate with a loaded checkpoint model (replicating your exact scenario)."""
    debug_print("=== Testing zarr_generate with loaded checkpoint model ===")
    
    # Replicate your exact setup
    runid = "test_zarr_gen"
    lap = 1  # Simulate lap 1 (loading lap 0)
    src = "/home/tm3076/scratch/priors_precomputed_datasets/precomputed_data_sst/sst_crho_0.4"
    
    # Set JAX config exactly like training script
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    jax.config.update('jax_compilation_cache_dir', '/tmp')
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    
    # Create mesh and sharding
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    
    # Initialize RNG
    runpath = PATH / f'runs/test_zarr_gen'
    runpath.mkdir(parents=True, exist_ok=True)
    base_seed = hash((runpath, lap)) % 2**16
    main_rng = inox.random.PRNG(base_seed)
    init_rng = inox.random.PRNG(main_rng.split())
    dropout_rng = inox.random.PRNG(main_rng.split())
    sampling_rng = inox.random.PRNG(main_rng.split())
    
    debug_print("Step 1: Load dataset")
    try:
        if os.path.exists(src):
            trainset_yA = zarr.open_group(f"{src}/train", mode="r")
            testset_yA = zarr.open_group(f"{src}/train", mode="r")
            y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
            y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
            B, H, W, C = y_eval.shape
            debug_print(f"Dataset loaded: shape {y_eval.shape}")
        else:
            debug_print("Dataset path not found, creating dummy data")
            B, H, W, C = 16, 64, 64, 3
            y_eval = np.random.randn(B, H, W, C).astype(np.float32)
            A_eval = np.ones((B, H, W, C), dtype=np.float32)
            y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
            
            # Create dummy zarr dataset
            dummy_zarr = {
                'y': np.random.randn(1000, H, W, C).astype(np.float32),
                'A': np.ones((1000, H, W, C), dtype=np.float32)
            }
            trainset_yA = dummy_zarr
            testset_yA = dummy_zarr
    except Exception as e:
        debug_print(f"Dataset setup failed: {e}")
        return False
    
    debug_print("Step 2: Create SDE and load checkpoint model within RNG context")
    with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
        # Create SDE
        sde = VESDE(**CONFIG.get('sde'))
        debug_print("SDE created")
        
        # Find and load checkpoint
        checkpoint_path = None
        for run_dir in (PATH / "runs").iterdir():
            if run_dir.is_dir() and "lucky-lion" in run_dir.name:  # Adjust as needed
                for f in run_dir.glob("checkpoint_*.pkl"):
                    checkpoint_path = f
                    break
            if checkpoint_path:
                break
        
        if not checkpoint_path:
            debug_print("No checkpoint found - creating fresh model for testing")
            # Create a fresh model for testing
            previous = make_model(
                key=init_rng.split(),
                in_channels=C,
                out_channels=C,
                **CONFIG
            )
            # Set dummy prior
            previous.mu_x = jnp.zeros(H * W * C)
            previous.cov_x = None
        else:
            debug_print(f"Loading checkpoint: {checkpoint_path}")
            # Load checkpoint directly (no nested context)
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
            debug_print("Checkpoint loaded successfully")
    
    debug_print("Step 3: Prepare model for zarr_generate (like training script)")
    # This is exactly what your training script does before zarr_generate
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)
    debug_print("Model partitioned and moved to device")
    
    debug_print("Step 4: Test zarr_generate with SMALL batch first")
    num_gpus = len(jax.devices())
    debug_print(f"Number of GPUs: {num_gpus}")
    
    # Create a small subset for testing
    small_dataset = {
        'y': trainset_yA['y'][:32],  # Just 32 samples
        'A': trainset_yA['A'][:32]
    }
    
    try:
        debug_print("About to call zarr_generate with small dataset...")
        debug_print(f"Dataset shapes: y={small_dataset['y'].shape}, A={small_dataset['A'].shape}")
        
        start_time = time.time()
        
        # This is the exact call from your training script
        trainset_generated = zarr_generate(
            model=previous,
            dataset=small_dataset,
            rng=main_rng,
            batch_size=16,  # Small batch size
            shape=(H, W, C),
            num_gpus=num_gpus,
            shard=True,
            sampler=CONFIG['sampler'],
            sde=sde,
            steps=CONFIG['discrete'],
            maxiter=CONFIG['maxiter'],
        )
        
        elapsed = time.time() - start_time
        debug_print(f"✅ zarr_generate completed successfully in {elapsed:.2f} seconds!")
        debug_print(f"Generated shape: {trainset_generated['x'].shape}")
        return True
        
    except Exception as e:
        debug_print(f"❌ zarr_generate failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zarr_generate_components():
    """Test individual components of zarr_generate to isolate the issue."""
    debug_print("=== Testing zarr_generate components ===")
    
    # Set up minimal environment
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    
    # Create minimal test data
    B, H, W, C = 8, 128, 128, 1
    y_test = np.random.randn(B, H, W, C).astype(np.float32)
    A_test = np.ones((B, H, W, C), dtype=np.float32)
    
    # Create simple model
    main_rng = inox.random.PRNG(42)
    init_rng = inox.random.PRNG(main_rng.split())
    dropout_rng = inox.random.PRNG(main_rng.split())
    
    debug_print("Step 1: Test model creation")
    with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
        model = make_model(
            key=init_rng.split(),
            in_channels=C,
            out_channels=C,
            **CONFIG
        )
        model.mu_x = jnp.zeros(H * W * C)
        model.cov_x = None
        model.train(False)
        debug_print("Model created")
    
    debug_print("Step 2: Test single sample call")
    try:
        with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
            sde = VESDE(**CONFIG.get('sde'))
            
            # Test single sample
            x_single = sample(
                model=model,
                y=y_test[:2],
                A=A_test[:2],
                key=main_rng.split(),
                shard=False,
                sampler=CONFIG['sampler'],
                sde=sde,
                steps=CONFIG['discrete'],
                maxiter=CONFIG['maxiter'],
            )
            debug_print(f"✅ Single sample successful: {x_single.shape}")
    except Exception as e:
        debug_print(f"❌ Single sample failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    debug_print("Step 3: Test zarr_generate with minimal data")
    try:
        test_dataset = {'y': y_test, 'A': A_test}
        
        result = zarr_generate(
            model=model,
            dataset=test_dataset,
            rng=main_rng,
            batch_size=4,
            shape=(H, W, C),
            num_gpus=1,
            shard=False,  # Start without sharding
            sampler=CONFIG['sampler'],
            sde=sde,
            steps=16,  # Fewer steps for speed
            maxiter=2,  # Fewer iterations
        )
        debug_print(f"✅ Minimal zarr_generate successful: {result['x'].shape}")
        return True
    except Exception as e:
        debug_print(f"❌ Minimal zarr_generate failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    debug_print("Starting zarr_generate diagnostics...")
    debug_print(f"Python PID: {os.getpid()}")
    
    # Test components first
    debug_print("=== Testing individual components ===")
    if not test_zarr_generate_components():
        debug_print("❌ Component test failed - stopping here")
        return
    
    debug_print("✅ Component test passed")
    
    # Test with loaded model (the actual scenario)
    debug_print("=== Testing with loaded checkpoint model ===")
    success = test_zarr_generate_with_loaded_model()
    
    if success:
        debug_print("✅ All zarr_generate tests passed!")
    else:
        debug_print("❌ zarr_generate test failed")

if __name__ == "__main__":
    main()