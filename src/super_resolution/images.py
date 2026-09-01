"""Pillow and tensor image conversion helpers."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a Pillow image to a float CHW tensor in the [0, 1] range."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW or single-item NCHW tensor to an RGB Pillow image."""
    image = tensor.detach().cpu()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("tensor must have shape (3, height, width) or (1, 3, height, width)")
    array = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def bicubic_pair(high_resolution: Image.Image, scale: int) -> tuple[Image.Image, Image.Image]:
    """Create a bicubic-upsampled input paired with its aligned target."""
    if scale < 2:
        raise ValueError("scale must be at least 2")
    rgb = high_resolution.convert("RGB")
    width = rgb.width - rgb.width % scale
    height = rgb.height - rgb.height % scale
    if width < scale or height < scale:
        raise ValueError("image is too small for the selected scale")
    target = rgb.crop((0, 0, width, height))
    low_resolution = target.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
    bicubic = low_resolution.resize((width, height), Image.Resampling.BICUBIC)
    return bicubic, target

