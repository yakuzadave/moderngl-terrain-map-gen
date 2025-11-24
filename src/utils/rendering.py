"""CPU-based rendering helpers for generic terrain outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
from PIL import Image

from .artifacts import TerrainMaps


def _height_array(terrain) -> np.ndarray:
    return TerrainMaps.ensure(terrain).height


def shade_heightmap(
    terrain,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 2.0,
    colormap: str = "terrain",
    blend_mode: Literal["soft", "overlay", "hsv"] = "soft",
) -> np.ndarray:
    """Return an RGB uint8 array representing shaded relief."""
    height = _height_array(terrain)
    ls = LightSource(azdeg=float(azimuth), altdeg=float(altitude))
    # Approximate dx/dy so shading stays stable across resolutions.
    dx = 1.0 / max(1, height.shape[1])
    dy = 1.0 / max(1, height.shape[0])
    rgb = ls.shade(
        height,
        cmap=cm.get_cmap(colormap),
        vert_exag=float(vert_exag),
        blend_mode=blend_mode,
        dx=dx,
        dy=dy,
    )
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def save_shaded_relief_png(
    path: str | Path,
    terrain,
    **kwargs,
) -> Path:
    """Save :func:`shade_heightmap` output as a PNG file."""
    target = Path(path)
    img = Image.fromarray(shade_heightmap(terrain, **kwargs), mode="RGB")
    img.save(target)
    return target


def slope_intensity(
    terrain,
    *,
    scale: float = 1.0,
    gamma: float = 0.8,
    normalize: bool = True,
) -> np.ndarray:
    """Return a grayscale slope map derived from height gradients."""
    height = _height_array(terrain)
    gy, gx = np.gradient(height)
    slope = np.sqrt(gx**2 + gy**2) * float(scale)
    if normalize and slope.size:
        max_val = float(np.max(slope))
        if max_val > 0.0:
            slope /= max_val
    slope = np.clip(slope, 0.0, 1.0)
    slope = slope ** float(gamma)
    return (slope * 255.0).astype(np.uint8)


def save_slope_map_png(
    path: str | Path,
    terrain,
    **kwargs,
) -> Path:
    """Save :func:`slope_intensity` output as an 8-bit PNG."""
    target = Path(path)
    img = Image.fromarray(slope_intensity(terrain, **kwargs), mode="L")
    img.save(target)
    return target


__all__ = [
    "shade_heightmap",
    "save_shaded_relief_png",
    "slope_intensity",
    "save_slope_map_png",
]
