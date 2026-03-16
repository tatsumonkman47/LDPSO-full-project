r"""Image helpers"""

import dm_pix as pix # type: ignore
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore

from einops import rearrange # type: ignore
from jax import Array # type: ignore
from pathlib import Path
from PIL import Image # type: ignore
from typing import *


def flatten(x: Array) -> Array:
    return rearrange(x, '... H W C -> ... (H W C)')


def unflatten(x: Array, height: int, width: int) -> Array:
    return rearrange(x, '... (H W C) -> ... H W C', H=height, W=width)


def from_pil(img: Image.Image) -> Array:
    x = np.asarray(img)
    x = x * (4 / 256) - 2

    return x


def to_pil(
    x: np.ndarray,
    pad: int = 0,
    background: int = 255,
    zoom: int = 1,
    file: Optional[Union[str, Path]] = None,
    cmaps: list = ["Spectral_r","seismic"],
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert a batched grid of images into a single PIL Image or a list of images (one per channel).

    x: Array of shape (M, N, H, W, C)
       where:
         M: rows in grid
         N: cols in grid
         H: height
         W: width
         C: channels (1 or more)
    cmap: matplotlib colormap name (used for single-channel images)
    """
    try:
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for this function. Install with: pip install matplotlib")

    x = np.asarray(x)
    
    # Debug info to help diagnose monochrome images
    jax.debug.print(f"Input shape: {x.shape}")
    jax.debug.print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    jax.debug.print(f"Input mean: {x.mean():.4f}")
    jax.debug.print(f"Input std: {x.std():.4f}")
    
    # Subtract mean value from each image
    # Compute mean for each image in the batch (across H, W, C dimensions)
    mean_per_image = x.mean(axis=(2, 3, 4), keepdims=True)
    jax.debug.print(f"Mean per image range: [{mean_per_image.min():.4f}, {mean_per_image.max():.4f}]")
    
    x_centered = x - mean_per_image
    
    jax.debug.print(f"After centering range: [{x_centered.min():.4f}, {x_centered.max():.4f}]")
    jax.debug.print(f"After centering mean: {x_centered.mean():.4f}")
    jax.debug.print(f"After centering std: {x_centered.std():.4f}")
    
    # Check if input is effectively constant
    if np.allclose(x_centered, x_centered.flat[0]):
        jax.debug.print("WARNING: Centered array has constant values - this will produce monochrome images")
    
    # Scale to uint8 using percentile-based scaling for better visualization
    # Use 1st and 99th percentiles to handle outliers
    eps = 1e-8
    vmin = np.percentile(x_centered, 1)
    vmax = np.percentile(x_centered, 99)
    jax.debug.print(f"Percentile range (1%, 99%): [{vmin:.4f}, {vmax:.4f}]")
    
    x_scaled = np.clip((x_centered - vmin) / (vmax - vmin + eps) * 255, 0, 255)
    jax.debug.print(f"After scaling range: [{x_scaled.min():.4f}, {x_scaled.max():.4f}]")
    
    x = np.rint(x_scaled).astype(np.uint8)
    # Pad
    x = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)),
        constant_values=background
    )
    # Rearrange grid to single large image per channel
    _, _, _, _, C = x.shape
    images = []
    for c in range(C):
        x_c = rearrange(x[..., c], 'M N H W -> (M H) (N W)')
        
        # Use modulo to handle cases where c >= len(cmaps)
        cmap_name = cmaps[c % len(cmaps)]
        cmap_fn = cm.get_cmap(cmap_name)
        
        x_norm = x_c.astype(np.float32) / 255.0
        x_rgb_array = cmap_fn(x_norm)
        x_rgb = (x_rgb_array[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(x_rgb, mode='RGB')
        # Resize (zoom)
        if zoom > 1:
            img = img.resize(
                (zoom * img.width, zoom * img.height),
                Image.NEAREST
            )
        images.append(img)

    # Save images if file is specified
    if file is not None:
        if C == 1:
            images[0].save(file)
        else:
            file = Path(file)
            for c, img in enumerate(images):
                img.save(str(file.with_stem(f"{file.stem}_ch{c}")))
    return images[0] if C == 1 else images

def collate(
    images: List[List[Image.Image]],
    pad: int = 0,
    background: int = 255,
    file: Optional[Union[str, Path]] = None,
) -> Image.Image:
    M, N = len(images), max(map(len, images))
    W, H = None, None
    for i in range(M):
        for j in range(N):
            try:
                W, H = images[i][j].size
            except IndexError:
                continue
        if W is not None and H is not None:
            break  # Exit outer loop if size found

    if W is None or H is None:
        raise ValueError("No valid images found to determine canvas size.")
    canvas = Image.new(
        'RGB',
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=background,
    )
    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )
            try:
                canvas.paste(images[i][j], offset)
            except IndexError:
                continue

    if file is not None:
        canvas.save(file)

    return canvas


def random_flip(x: Array, key: Array, axis: int = -2) -> Array:
    return jnp.where(
        jax.random.bernoulli(key),
        x,
        jnp.flip(x, axis=axis),
    )


def random_hue(x: Array, key: Array, delta: float = 1e-2) -> Array:
    x = (x + 2) / 4
    x = pix.random_hue(key, x, delta)
    x = x * 4 - 2

    return x


def random_saturation(x: Array, key: Array, lower: float = 0.95, upper: float = 1.05) -> Array:
    x = (x + 2) / 4
    x = pix.random_saturation(key, x, lower, upper)
    x = x * 4 - 2

    return x


def random_shake(x: Array, key: Array, delta: int = 1, mode: str = 'reflect') -> Array:
    i = jax.random.randint(key, shape=(3,), minval=0, maxval=2 * delta + 1)
    i = i.at[-1].set(0)

    return jax.lax.dynamic_slice(
        jnp.pad(
            x,
            pad_width=((delta, delta), (delta, delta), (0, 0)),
            mode=mode,
        ),
        start_indices=i,
        slice_sizes=x.shape,
    )


def psnr(a: Array, b: Array) -> Array:
    return pix.psnr((a + 2) / 4, (b + 2) / 4)


def ssim(a: Array, b: Array) -> Array:
    return pix.ssim((a + 2) / 4, (b + 2) / 4)
