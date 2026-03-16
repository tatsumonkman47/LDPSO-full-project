r"""Neural networks"""

import inox # type: ignore
import inox.nn as nn # type: ignore # type: ignore
from inox.random import PRNG, get_rng, set_rng # type: ignore
import jax.numpy as jnp # type: ignore

from einops import rearrange # type: ignore
from jax import Array # type: ignore # type: ignore
import jax # type: ignore
from typing import *
from contextlib import nullcontext


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        key: A PRNG key for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hid_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.SiLU,
        normalize: bool = False,
        key: Array = None,
    ):
        if key is None:
            raise ValueError("MLP must be given a key")
        keys = jax.random.split(key, len(hid_features) + 1)

        layers = []

        for i, (before, after) in enumerate(zip(
            (in_features, *hid_features),
            (*hid_features, out_features),
        )):
            layers.extend([
                nn.Linear(before, after, key=keys[i]),
                activation(),
                nn.LayerNorm() if normalize else None,
            ])

        layers = filter(lambda l: l is not None, layers[:-2])

        super().__init__(*layers)


class Modulation(nn.Module):
    r"""Creates an adaptive modulation module."""

    def __init__(self, channels: int, emb_features: int, key: Array):
        k1, k2 = jax.random.split(key)
        self.mlp = nn.Sequential(
            nn.Linear(emb_features, emb_features, key=k1),
            nn.SiLU(),
            nn.Linear(emb_features, 3 * channels, key=k2),
            nn.Rearrange('... C -> ... 1 1 C'),
        )

        layer = self.mlp.layers[-2]
        layer.weight.value = layer.weight.value * 1e-1

    @inox.jit
    def __call__(self, t: Array) -> Tuple[Array, Array, Array]:
        return jnp.array_split(self.mlp(t), 3, axis=-1)


class ResBlock(nn.Module):
    r"""Creates a residual block."""

    def __init__(
        self,
        channels: int,
        emb_features: int,
        dropout: Optional[float] = None,
        checkpoint: bool = False, # Whether to use checkpointing for memory efficiency
        key: Array = None,
        **kwargs,
    ):
        k1, k2, k3 = jax.random.split(key, num=3)
        self.modulation = Modulation(channels, emb_features, key=k1)
        self.block = nn.Sequential(
            nn.LayerNorm(),
            nn.Conv(channels, channels, key=k2, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.TrainingDropout(dropout),
            nn.Conv(channels, channels, key=k3, **kwargs),
        )
        self.checkpoint = checkpoint
    
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        if self.checkpoint:
            return self._checkpointed_forward(x, t, key)
        else:
            return self._forward(x, t, key)

    @inox.checkpoint
    def _checkpointed_forward(self, x: Array, t: Array, key: Array = None) -> Array:
        return self._forward(x, t, key)
    
    def _forward(self, x: Array, t: Array, key: Array = None) -> Array:
        a, b, c = self.modulation(t)
        y = (a + 1) * x + b
        y = self.block(y, key=key)
        y = x + c * y
        return y / jnp.sqrt(1 + c**2)


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block."""

    def __init__(
            self, 
            channels: int, 
            emb_features: int, 
            heads: int = 1, 
            key: Array = None
    ):
        k1, k2 = jax.random.split(key)
        self.modulation = Modulation(channels, emb_features, key=k1)
        self.norm = nn.LayerNorm()
        self.attn = nn.MultiheadAttention(
            heads=heads,
            in_features=channels,
            out_features=channels,
            hid_features=channels // heads,
            key=k2,
        )

    @inox.checkpoint
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        a, b, c = self.modulation(t)
        y = (a + 1) * x + b
        y = self.norm(y)
        y = rearrange(y, '... H W C -> ... (H W) C')
        y = self.attn(y)
        y = y.reshape(x.shape)
        y = x + c * y

        return y / jnp.sqrt(1 + c**2)


class UNet(nn.Module):
    r"""Creates a time (or noise) conditional U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Sequence[int] = (3, 3),
        emb_features: int = 64,
        heads: Dict[int, int] = {},
        dropout: Optional[float] = None,
        checkpoint_layers: Sequence[int] = (1, 2),
        key: Array = None,
    ):
        if key is None:
            get_rng().split()
        key = jax.random.split(key, sum(hid_blocks) * 6 + len(heads) * 4 + 10) # Overallocate keys for safety
        k_iter = iter(key)

        stride = [2 for k in kernel_size]
        kwargs = dict(
            kernel_size=kernel_size,
            padding=[(k // 2, k // 2) for k in kernel_size],
        )

        self.descent, self.ascent = [], []

        for i, blocks in enumerate(hid_blocks):
            do, up = [], []

            # Checkpoint deeper layers (higher memory usage)
            use_checkpoint = i in checkpoint_layers

            for _ in range(blocks):
                do.append(ResBlock(hid_channels[i], 
                                   emb_features, 
                                   dropout=dropout, 
                                   key=next(k_iter), 
                                   checkpoint=use_checkpoint,
                                   **kwargs
                                   ))
                up.append(ResBlock(hid_channels[i], 
                                   emb_features, 
                                   dropout=dropout, 
                                   key=next(k_iter), 
                                   checkpoint=use_checkpoint,
                                   **kwargs
                                   ))
                if i in heads:
                    do.append(AttBlock(hid_channels[i], emb_features, heads[i], key=next(k_iter)))
                    up.append(AttBlock(hid_channels[i], emb_features, heads[i], key=next(k_iter)))

            if i > 0:
                do.insert(
                    0,
                    nn.Sequential(
                        nn.Conv(
                            hid_channels[i - 1],
                            hid_channels[i],
                            stride=stride,  
                            key=next(k_iter),
                            **kwargs,
                        ),
                        nn.LayerNorm(),
                    ),
                )

                up.append(
                    nn.Sequential(
                        nn.LayerNorm(),
                        nn.Resample(factor=stride, method='nearest'),
                    )
                )
            else:
                do.insert(0, nn.Conv(in_channels, hid_channels[i], key=next(k_iter), **kwargs))
                up.append(nn.Linear(hid_channels[i], out_channels, key=next(k_iter),))

            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    nn.Conv(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        key=next(k_iter),
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        r"""
        Arguments:
            x: The noisy tensor, with shape :math:`(*, H, W, C)`.
            t: The time embedding, with shape :math:`(*, T)`.
            key: A PRNG key.
        """
        memory = []
        # Optionally split the key for each block if you want independent randomness per block
        key_iter = iter(jax.random.split(key, len(self.descent) + len(self.ascent))) if key is not None else None

        # Down path
        for i, blocks in enumerate(self.descent):
            block_key = next(key_iter) if key_iter is not None else None
            for block in blocks:
                if isinstance(block, (ResBlock, AttBlock)):
                    x = block(x, t, key=block_key)
                else:
                    x = block(x)
            memory.append(x)
        # Up path
        for i, blocks in enumerate(self.ascent):
            y = memory.pop()
            if x is not y:
                x = jnp.concatenate((x, y), axis=-1)
            block_key = next(key_iter) if key_iter is not None else None
            for block in blocks:
                if isinstance(block, (ResBlock, AttBlock)):
                    x = block(x, t, key=block_key)
                else:
                    x = block(x)

        return x
