r"""CIFAR experiment helpers"""
import sys
from pathlib import Path
import os

# Add local priors package to path (before conda environment version)
_local_priors_path = Path(__file__).resolve().parent.parent.parent
if str(_local_priors_path) not in sys.path:
    sys.path.insert(0, str(_local_priors_path))

import inox # type: ignore
import inox.random as inox_random # type: ignore
import inox.nn as nn # type: ignore
import jax # type: ignore
from jax import Array # type: ignore
from inox.tree import Partial as Partial # type: ignore

from typing import Dict, Sequence, Optional


# isort: split
from priors.diffusion import Denoiser
from priors.image import flatten, unflatten
from priors.common import distribute, sample_any
from priors.nn import UNet

# Set the output path to scratch if available, otherwise use the current directory
if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'LDPSO-full-project/SWOT-LDPSO/experiments/swot-sst-inpainting_V2'
else:
    PATH = Path(__file__).resolve().parent

PATH.mkdir(parents=True, exist_ok=True)

def measure(A: Array, x: Array, H: int, W: int, C: int) -> Array:
    x_unflat = unflatten(x, H, W,)
    return flatten(A * x_unflat) 

def sample(
    model: nn.Module,
    y: Array,
    A: Array,
    key: Array,
    shard: bool = False,
    cov_y: float =1e-3**2,
    **kwargs,
) -> Array:
    if shard:
        y, A = distribute((y, A))

    B, H, W, C = y.shape
    
    # Split the key for different RNG purposes
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Create fresh PRNG instances for this sampling operation
    sample_init_rng = inox.random.PRNG(key1)
    sample_dropout_rng = inox.random.PRNG(key2)
    
    # Always set comprehensive RNG context for sampling operations
    with inox.random.set_rng(
        init=sample_init_rng,
        dropout=sample_dropout_rng,
    ):
        x = sample_any(
            model=model,
            shape=flatten(y).shape,
            shard=shard,
            A=inox.tree.Partial(measure, A, H=H, W=W, C=C),
            y=flatten(y),
            cov_y=cov_y,
            key=key3,
            **kwargs,
        )

    return unflatten(x, H, W)

def make_model(
    key: Array,
    in_channels: int = 3,
    out_channels: int = 3,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: Optional[float] = None,
    checkpoint_layers=(1, 2),
    **absorb,
) -> Denoiser:
    # Split the key for different components
    network_key, denoiser_key = jax.random.split(key, 2)
    return Denoiser(
        network=FlatUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            checkpoint_layers=checkpoint_layers,
            key=network_key,  # ← ADD THIS LINE
        ),
        emb_features=emb_features,
        #key=denoiser_key,  # ← AND THIS LINE IF DENOISER NEEDS IT
    )

class FlatUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=128, height=128)
        x = super().__call__(x, t, key)  # No RNG context management!
        x = flatten(x)
        return x