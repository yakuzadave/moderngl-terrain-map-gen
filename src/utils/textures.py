"""Additional texture export formats for game engines and DCC tools."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import TerrainMaps

__all__ = [
    "save_splatmap_rgba",
    "save_ao_map",
    "save_curvature_map",
    "save_packed_texture",
    "save_scatter_map",
]


def save_scatter_map(
    path: str | Path,
    terrain,
) -> Path:
    """Save the scatter density map (Trees, Rocks, Grass).

    Args:
        path: Output file path
        terrain: TerrainMaps instance

    Returns:
        Path to saved file
    """
    target = Path(path)
    maps = TerrainMaps.ensure(terrain)

    img = Image.fromarray(maps.scatter_map_u8(), mode="RGB")
    img.save(target)
    return target


def save_splatmap_rgba(
    path: str | Path,
    terrain,
    height_thresholds: tuple[float, float, float] = (0.3, 0.5, 0.7),
    slope_threshold: float = 0.7,
) -> Path:
    """Export a 4-channel splatmap for texture blending in game engines.

    Channels:
        R: Low terrain (beach/grass)
        G: Mid terrain (grass/dirt)
        B: High terrain (rock/snow)
        A: Steep slopes (cliffs)

    Args:
        path: Output file path
        terrain: TerrainMaps instance
        height_thresholds: (low, mid, high) height breakpoints
        slope_threshold: Normal.y threshold for steep detection

    Returns:
        Path to saved file
    """
    target = Path(path)
    maps = TerrainMaps.ensure(terrain)

    height = maps.height
    normals = maps.normals

    low_t, mid_t, high_t = height_thresholds

    # Calculate weights per channel
    r = np.clip((height - low_t) / (mid_t - low_t + 1e-6), 0.0, 1.0)
    r = 1.0 - r  # Invert for low terrain

    g = np.where(
        height < mid_t,
        np.clip((height - low_t) / (mid_t - low_t + 1e-6), 0.0, 1.0),
        np.clip((high_t - height) / (high_t - mid_t + 1e-6), 0.0, 1.0),
    )

    b = np.clip((height - mid_t) / (high_t - mid_t + 1e-6), 0.0, 1.0)

    # Alpha for steep slopes
    a = np.where(normals[:, :, 1] < slope_threshold, 1.0, 0.0)

    # Normalize to sum to 1 (except steep slopes)
    total = r + g + b
    total = np.where(total > 0, total, 1.0)
    r = r / total
    g = g / total
    b = b / total

    # Blend with slope channel
    blend = 1.0 - a
    r = r * blend
    g = g * blend
    b = b * blend

    # Convert to uint8
    rgba = np.stack([r, g, b, a], axis=-1)
    rgba_u8 = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

    img = Image.fromarray(rgba_u8, mode="RGBA")
    img.save(target)
    return target


def save_ao_map(
    path: str | Path,
    terrain,
    radius: int = 3,
    strength: float = 1.0,
) -> Path:
    """Generate and save an ambient occlusion approximation.

    Uses local height variance as a proxy for occlusion.

    Args:
        path: Output file path
        terrain: TerrainMaps instance
        radius: Sample radius in pixels
        strength: AO intensity multiplier

    Returns:
        Path to saved file
    """
    target = Path(path)
    maps = TerrainMaps.ensure(terrain)
    height = maps.height

    # Compute local variance
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(height, size=radius * 2 + 1)
    mean_sq = uniform_filter(height**2, size=radius * 2 + 1)
    variance = mean_sq - mean**2

    # Variance as occlusion proxy
    ao = 1.0 - np.clip(variance * strength * 10.0, 0.0, 1.0)
    ao_u8 = (ao * 255).astype(np.uint8)

    img = Image.fromarray(ao_u8, mode="L")
    img.save(target)
    return target


def save_curvature_map(
    path: str | Path,
    terrain,
    scale: float = 1.0,
) -> Path:
    """Generate and save a curvature map (convex=bright, concave=dark).

    Args:
        path: Output file path
        terrain: TerrainMaps instance
        scale: Curvature intensity scaling

    Returns:
        Path to saved file
    """
    target = Path(path)
    maps = TerrainMaps.ensure(terrain)
    height = maps.height

    # Compute second derivatives (Laplacian)
    from scipy.ndimage import laplace

    curvature = laplace(height.astype(np.float64))

    # Normalize and scale
    curvature = curvature * scale
    curvature = (curvature + 1.0) * 0.5  # Map to [0, 1]
    curvature_u8 = (np.clip(curvature, 0.0, 1.0) * 255).astype(np.uint8)

    img = Image.fromarray(curvature_u8, mode="L")
    img.save(target)
    return target


def save_packed_texture(
    path: str | Path,
    terrain,
    pack_mode: str = "unity_mask",
) -> Path:
    """Save a multi-channel packed texture for efficient storage.

    Pack modes:
        'unity_mask': R=Metallic, G=AO, B=Detail, A=Smoothness
        'ue_orm': R=AO, G=Roughness, B=Metallic
        'height_normal_ao': R=Height, G=NormalX, B=NormalZ, A=AO

    Args:
        path: Output file path
        terrain: TerrainMaps instance
        pack_mode: Packing scheme

    Returns:
        Path to saved file
    """
    target = Path(path)
    maps = TerrainMaps.ensure(terrain)

    if pack_mode == "unity_mask":
        # Generate basic channels
        metallic = np.zeros_like(maps.height)
        ao = _compute_simple_ao(maps.height)
        detail = maps.erosion_channel()
        smoothness = 1.0 - detail

        packed = np.stack([metallic, ao, detail, smoothness], axis=-1)

    elif pack_mode == "ue_orm":
        ao = _compute_simple_ao(maps.height)
        roughness = maps.erosion_channel()
        metallic = np.zeros_like(maps.height)

        packed = np.stack([ao, roughness, metallic], axis=-1)

    elif pack_mode == "height_normal_ao":
        height = maps.height
        nx = (maps.normals[:, :, 0] + 1.0) * 0.5
        nz = (maps.normals[:, :, 2] + 1.0) * 0.5
        ao = _compute_simple_ao(maps.height)

        packed = np.stack([height, nx, nz, ao], axis=-1)

    else:
        raise ValueError(f"Unknown pack_mode: {pack_mode}")

    packed_u8 = (np.clip(packed, 0.0, 1.0) * 255).astype(np.uint8)

    mode = "RGBA" if packed.shape[-1] == 4 else "RGB"
    img = Image.fromarray(packed_u8, mode=mode)
    img.save(target)
    return target


def _compute_simple_ao(height: np.ndarray, radius: int = 2) -> np.ndarray:
    """Fast AO approximation using height variance."""
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(height, size=radius * 2 + 1)
    mean_sq = uniform_filter(height**2, size=radius * 2 + 1)
    variance = mean_sq - mean**2

    ao = 1.0 - np.clip(variance * 10.0, 0.0, 1.0)
    return ao
