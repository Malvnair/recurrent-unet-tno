"""Pillow-backed image I/O helpers used by dataset loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | Path) -> Image.Image:
    return Image.open(path)


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width_height = (size[1], size[0])
    return image.resize(width_height, BILINEAR)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    width_height = (size[1], size[0])
    return mask.resize(width_height, NEAREST)


def mask_to_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask, dtype=np.uint8)
