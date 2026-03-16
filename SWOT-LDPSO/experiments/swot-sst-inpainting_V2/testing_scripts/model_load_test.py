#!/usr/bin/env python
"""
Standalone checkpoint loading diagnostic script.
This will help isolate whether the hang is in:
1. Pickle loading
2. Model creation 
3. Parameter loading
4. RNG context issues
5. JAX compilation
"""

import sys
import os
import time
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Add your project paths
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting_V2')
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/inox_local')
sys.path.append('/home/tm3076/scratch/project/LDP_OEM/Tatsu')

import inox
import inox.random
from utils import make_model, PATH

# Configuration (copy from your main script)
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
    """Thread-safe debug printing with timestamp."""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] DEBUG: {msg}", flush=True)

def test_basic_jax():
    """Test basic JAX functionality."""
    debug_print("Testing basic JAX operations...")
    try:
        x = jax.random.normal(jax.random.PRNGKey(42), (10, 10))
        y = jnp.sum(x)
        debug_print(f"JAX basic test passed. Sum: {y}")
        return True
    except Exception as e:
        debug_print(f"JAX basic test failed: {e}")
        return False

def test_inox_rng():
    """Test inox RNG functionality."""
    debug_print("Testing inox RNG...")
    try:
        # Test basic PRNG creation
        rng = inox.random.PRNG(42)
        key = rng.split()
        debug_print(f"Basic PRNG created: {key}")
        
        # Test context manager
        with inox.random.set_rng(test=inox.random.PRNG(123)):
            test_rng = inox.random.get_rng('test')
            test_key = test_rng.split()
            debug_print(f"Context RNG test passed: {test_key}")
        
        return True
    except Exception as e:
        debug_print(f"Inox RNG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation without checkpoint loading."""
    debug_print("Testing model creation...")
    try:
        # Clear any existing RNG state
        inox.random.INOX_RNG.clear()
        
        # Create RNG context
        init_rng = inox.random.PRNG(42)
        dropout_rng = inox.random.PRNG(43)
        
        with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
            model = make_model(
                key=init_rng.split(),
                in_channels=3,  # Assuming C=3
                out_channels=3,
                **CONFIG
            )
            debug_print("Model created successfully")
            
            # Test model partitioning
            static_part, params_part = model.partition()
            debug_print("Model partitioning successful")
            
            # Test model reconstruction
            reconstructed = static_part(params_part)
            debug_print("Model reconstruction successful")
            
        return True, model
    except Exception as e:
        debug_print(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_checkpoint_loading(checkpoint_path):
    """Test loading checkpoint step by step."""
    debug_print(f"Testing checkpoint loading from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        debug_print(f"Checkpoint file does not exist: {checkpoint_path}")
        return False
    
    # Step 1: Test pickle loading
    debug_print("Step 1: Loading pickle file...")
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        debug_print(f"Pickle loaded successfully. Keys: {list(checkpoint_data.keys())}")
    except Exception as e:
        debug_print(f"Pickle loading failed: {e}")
        return False
    
    # Step 2: Test model creation with clear RNG state
    debug_print("Step 2: Creating fresh model...")
    try:
        # Clear RNG state
        inox.random.INOX_RNG.clear()
        debug_print("RNG state cleared")
        
        # Create fresh RNG
        init_rng = inox.random.PRNG(100)
        dropout_rng = inox.random.PRNG(101)
        
        with inox.random.set_rng(init=init_rng, dropout=dropout_rng):
            debug_print("Creating model...")
            model = make_model(
                key=init_rng.split(),
                in_channels=3,
                out_channels=3,
                **CONFIG
            )
            debug_print("Fresh model created")
    except Exception as e:
        debug_print(f"Fresh model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test parameter loading
    debug_print("Step 3: Loading saved parameters...")
    try:
        # Load statistics
        if 'mu_x' in checkpoint_data:
            model.mu_x = checkpoint_data['mu_x']
            debug_print("mu_x loaded")
        
        if checkpoint_data.get('cov_x') is not None:
            model.cov_x = checkpoint_data['cov_x']
            debug_print("cov_x loaded")
    except Exception as e:
        debug_print(f"Statistics loading failed: {e}")
        return False
    
    # Step 4: Test parameter reconstruction (this is often where it hangs)
    debug_print("Step 4: Reconstructing model with loaded parameters...")
    try:
        static_part, _ = model.partition()
        debug_print("Model partitioned")
        
        # This is the critical step that often causes hangs
        debug_print("About to reconstruct model with loaded params...")
        loaded_model = static_part(checkpoint_data['params'])
        debug_print("Model reconstructed successfully!")
        
        # Test setting eval mode
        loaded_model.train(False)
        debug_print("Model set to eval mode")
        
        return True
    except Exception as e:
        debug_print(f"Parameter reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function."""
    debug_print("Starting checkpoint loading diagnostics...")
    debug_print(f"Python PID: {os.getpid()}")
    
    # Set up JAX
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    
    # Run tests sequentially
    tests = [
        ("Basic JAX", test_basic_jax),
        ("Inox RNG", test_inox_rng),
        ("Model Creation", lambda: test_model_creation()[0]),
    ]
    
    for test_name, test_func in tests:
        debug_print(f"Running test: {test_name}")
        if not test_func():
            debug_print(f"Test failed: {test_name}")
            return
        debug_print(f"Test passed: {test_name}")
    
    # Find a checkpoint file to test
    debug_print("Looking for checkpoint files...")
    
    # You'll need to modify this path to point to your actual checkpoint
    # Look in your runs directory for a checkpoint file
    possible_paths = [
        PATH / "runs",
        Path("/home/tm3076/scratch/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting_V2/runs")
    ]
    
    checkpoint_path = None
    for base_path in possible_paths:
        if base_path.exists():
            debug_print(f"Searching in: {base_path}")
            for run_dir in base_path.iterdir():
                if run_dir.is_dir() and "lucky-lion" in run_dir.name:  # Adjust as needed
                    for f in run_dir.glob("checkpoint_*.pkl"):
                        checkpoint_path = f
                        debug_print(f"Found checkpoint: {checkpoint_path}")
                        break
                if checkpoint_path:
                    break
    
    if not checkpoint_path:
        debug_print("No checkpoint file found. Creating a dummy test...")
        # Create a simple model and save it for testing
        debug_print("Creating dummy checkpoint for testing...")
        try:
            success, model = test_model_creation()
            if success:
                static_part, params_part = model.partition()
                dummy_checkpoint = {
                    'params': params_part,
                    'others': {},
                    'mu_x': jnp.zeros(10),
                    'cov_x': None,
                    'lap': 0,
                    'config': dict(CONFIG)
                }
                dummy_path = PATH / "dummy_checkpoint.pkl"
                with open(dummy_path, 'wb') as f:
                    pickle.dump(dummy_checkpoint, f)
                checkpoint_path = dummy_path
                debug_print(f"Created dummy checkpoint: {checkpoint_path}")
        except Exception as e:
            debug_print(f"Failed to create dummy checkpoint: {e}")
            return
    
    # Test checkpoint loading
    if checkpoint_path:
        debug_print("Testing checkpoint loading...")
        success = test_checkpoint_loading(checkpoint_path)
        if success:
            debug_print("✅ All tests passed! Checkpoint loading works.")
        else:
            debug_print("❌ Checkpoint loading failed.")
    
    debug_print("Diagnostics complete.")

if __name__ == "__main__":
    main()