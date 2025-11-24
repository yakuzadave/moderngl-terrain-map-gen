"""Matplotlib-based helpers for previewing terrain outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LightSource
import numpy as np

from .artifacts import TerrainMaps

__all__ = [
    "plot_terrain_panels",
    "create_turntable_animation",
    "save_panel_overview",
    "save_turntable_gif",
]


def _ensure_arrays(terrain) -> tuple[np.ndarray, np.ndarray]:
    maps = TerrainMaps.ensure(terrain)
    erosion = maps.erosion_mask if maps.erosion_mask is not None else np.zeros_like(
        maps.height)
    return maps.height, erosion


def plot_terrain_panels(terrain, figsize=(18, 6), dpi=100):
    """Render shaded relief, raw height, and erosion mask panels."""
    height, erosion = _ensure_arrays(terrain)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(
        height,
        cmap=plt.get_cmap("terrain"),
        blend_mode="soft",
        vert_exag=2.0,
        dx=1.0 / height.shape[0],
        dy=1.0 / height.shape[1],
    )
    axes[0].imshow(rgb, origin="lower")
    axes[0].set_title("Shaded Relief", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(height, cmap="terrain", origin="lower")
    axes[1].set_title("Height", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(erosion, cmap="RdYlBu_r", origin="lower")
    axes[2].set_title("Erosion", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_turntable_animation(
    terrain,
    frames: int = 60,
    interval: int = 50,
    figsize=(8, 8),
):
    """Create a simple turntable animation using rotating light direction."""
    height, _ = _ensure_arrays(terrain)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    im = ax.imshow(height, cmap="terrain", origin="lower")

    def _update(frame):
        azimuth = (frame * 360) / frames
        ls = LightSource(azdeg=azimuth, altdeg=45)
        rgb = ls.shade(height, cmap=plt.get_cmap("terrain"),
                       blend_mode="soft", vert_exag=2.0)
        im.set_data(rgb)
        return [im]

    ani = animation.FuncAnimation(
        fig, _update, frames=frames, interval=interval, blit=True)
    return ani


def save_panel_overview(path: str | Path, terrain, **kwargs) -> Path:
    target = Path(path)
    fig = plot_terrain_panels(terrain, **kwargs)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def save_turntable_gif(
    path: str | Path,
    terrain,
    frames: int = 90,
    interval: int = 40,
    dpi: int = 100,
) -> Path:
    target = Path(path)
    ani = create_turntable_animation(terrain, frames=frames, interval=interval)
    fps = max(1, int(1000 / max(1, interval)))
    writer = animation.PillowWriter(fps=fps)
    ani.save(target, writer=writer, dpi=dpi)
    fig = getattr(ani, "_fig", None)
    if fig is not None:
        plt.close(fig)
    return target
