"""Advanced rendering utilities with multiple output methods and animation support."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import numpy as np
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from PIL import Image

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .artifacts import TerrainMaps


__all__ = [
    "render_turntable_frames",
    "save_turntable_video",
    "render_multi_angle",
    "create_comparison_grid",
    "render_lighting_study",
    "save_animation_sequence",
]


def _height_array(terrain) -> np.ndarray:
    """Extract height array from terrain data."""
    return TerrainMaps.ensure(terrain).height


def render_turntable_frames(
    terrain,
    frames: int = 36,
    *,
    altitude: float = 45.0,
    vert_exag: float = 2.0,
    colormap: str = "terrain",
    blend_mode: Literal["soft", "overlay", "hsv"] = "soft",
) -> list[np.ndarray]:
    """
    Render a turntable sequence by rotating the light source around the terrain.

    Args:
        terrain: Terrain data (TerrainMaps or compatible dict)
        frames: Number of frames to generate (one full rotation)
        altitude: Sun altitude angle in degrees
        vert_exag: Vertical exaggeration for shading
        colormap: Matplotlib colormap name
        blend_mode: Shading blend mode

    Returns:
        List of RGB uint8 arrays, one per frame
    """
    height = _height_array(terrain)
    dx = 1.0 / max(1, height.shape[1])
    dy = 1.0 / max(1, height.shape[0])

    result = []
    for i in range(frames):
        azimuth = (i * 360.0) / frames
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        rgb = ls.shade(
            height,
            cmap=plt.get_cmap(colormap),
            vert_exag=vert_exag,
            blend_mode=blend_mode,
            dx=dx,
            dy=dy,
        )
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        result.append(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8))

    return result


def save_turntable_video(
    path: str | Path,
    terrain,
    frames: int = 36,
    fps: int = 12,
    **render_kwargs,
) -> Path:
    """
    Save a turntable animation as an MP4 video file.

    Requires ffmpeg to be installed on the system.
    Falls back to GIF if ffmpeg is not available.

    Args:
        path: Output file path (.mp4 or .gif)
        terrain: Terrain data
        frames: Number of frames in full rotation
        fps: Frames per second
        **render_kwargs: Additional arguments passed to render_turntable_frames

    Returns:
        Path to saved file
    """
    target = Path(path)
    frames_data = render_turntable_frames(
        terrain, frames=frames, **render_kwargs)

    # Try to save as video
    if target.suffix.lower() == ".mp4":
        try:
            import imageio
            with imageio.get_writer(target, fps=fps, codec="libx264") as writer:
                for frame in frames_data:
                    writer.append_data(frame)  # type: ignore[attr-defined]
            return target
        except (ImportError, RuntimeError):
            # Fall back to GIF
            target = target.with_suffix(".gif")

    # Save as GIF
    images = [Image.fromarray(f, mode="RGB") for f in frames_data]
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return target


def render_multi_angle(
    terrain,
    angles: Sequence[tuple[float, float]] | None = None,
    **render_kwargs,
) -> list[np.ndarray]:
    """
    Render terrain from multiple lighting angles.

    Args:
        terrain: Terrain data
        angles: List of (azimuth, altitude) tuples in degrees.
                Defaults to cardinal directions + zenith
        **render_kwargs: Additional rendering parameters

    Returns:
        List of rendered images, one per angle
    """
    if angles is None:
        # Default: four cardinal directions plus overhead
        angles = [
            (315.0, 45.0),  # Northwest (classic)
            (45.0, 45.0),   # Northeast
            (135.0, 45.0),  # Southeast
            (225.0, 45.0),  # Southwest
            (0.0, 90.0),    # Directly overhead
        ]

    height = _height_array(terrain)
    dx = 1.0 / max(1, height.shape[1])
    dy = 1.0 / max(1, height.shape[0])

    result = []
    cmap = plt.get_cmap(render_kwargs.get("colormap", "terrain"))
    vert_exag = render_kwargs.get("vert_exag", 2.0)
    blend_mode = render_kwargs.get("blend_mode", "soft")

    for azimuth, altitude in angles:
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        rgb = ls.shade(
            height,
            cmap=cmap,
            vert_exag=vert_exag,
            blend_mode=blend_mode,
            dx=dx,
            dy=dy,
        )
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        result.append(np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8))

    return result


def create_comparison_grid(
    terrains: Sequence[Any],
    labels: Sequence[str] | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int = 100,
    **render_kwargs,
) -> Figure:  # type: ignore[name-defined]
    """
    Create a grid comparison of multiple terrains.

    Args:
        terrains: Sequence of terrain data objects
        labels: Optional labels for each terrain
        figsize: Figure size (width, height) in inches
        dpi: Figure DPI
        **render_kwargs: Rendering parameters

    Returns:
        Matplotlib figure
    """
    n = len(terrains)
    if labels is None:
        labels = [f"Terrain {i+1}" for i in range(n)]

    # Calculate grid dimensions
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 5, rows * 5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.get_cmap(render_kwargs.get("colormap", "terrain"))
    azimuth = render_kwargs.get("azimuth", 315.0)
    altitude = render_kwargs.get("altitude", 45.0)
    vert_exag = render_kwargs.get("vert_exag", 2.0)
    blend_mode = render_kwargs.get("blend_mode", "soft")

    ls = LightSource(azdeg=azimuth, altdeg=altitude)

    for idx, (terrain, label) in enumerate(zip(terrains, labels)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        height = _height_array(terrain)
        dx = 1.0 / max(1, height.shape[1])
        dy = 1.0 / max(1, height.shape[0])

        rgb = ls.shade(
            height,
            cmap=cmap,
            vert_exag=vert_exag,
            blend_mode=blend_mode,
            dx=dx,
            dy=dy,
        )

        ax.imshow(rgb, origin="lower")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


def render_lighting_study(
    terrain,
    azimuth_steps: int = 8,
    altitude_steps: int = 3,
    figsize: tuple[int, int] | None = None,
    dpi: int = 80,
    **render_kwargs,
) -> Figure:  # type: ignore[name-defined]
    """
    Create a lighting study showing terrain under various light conditions.

    Useful for understanding how lighting affects terrain appearance.

    Args:
        terrain: Terrain data
        azimuth_steps: Number of azimuth angles to test
        altitude_steps: Number of altitude angles to test
        figsize: Figure size override
        dpi: Figure DPI
        **render_kwargs: Additional rendering parameters

    Returns:
        Matplotlib figure with grid of lighting variations
    """
    azimuths = np.linspace(0, 360, azimuth_steps, endpoint=False)
    altitudes = np.linspace(15, 75, altitude_steps)

    if figsize is None:
        figsize = (azimuth_steps * 2, altitude_steps * 2)

    fig, axes = plt.subplots(
        altitude_steps,
        azimuth_steps,
        figsize=figsize,
        dpi=dpi,
    )
    if altitude_steps == 1:
        axes = axes.reshape(1, -1)
    if azimuth_steps == 1:
        axes = axes.reshape(-1, 1)

    height = _height_array(terrain)
    dx = 1.0 / max(1, height.shape[1])
    dy = 1.0 / max(1, height.shape[0])
    cmap = plt.get_cmap(render_kwargs.get("colormap", "terrain"))
    vert_exag = render_kwargs.get("vert_exag", 2.0)
    blend_mode = render_kwargs.get("blend_mode", "soft")

    for i, altitude in enumerate(altitudes):
        for j, azimuth in enumerate(azimuths):
            ax = axes[i, j]
            ls = LightSource(azdeg=azimuth, altdeg=altitude)
            rgb = ls.shade(
                height,
                cmap=cmap,
                vert_exag=vert_exag,
                blend_mode=blend_mode,
                dx=dx,
                dy=dy,
            )
            ax.imshow(rgb, origin="lower")
            if i == 0:
                ax.set_title(f"{azimuth:.0f}°", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{altitude:.0f}°", fontsize=8)
            ax.axis("off")

    plt.suptitle(
        "Lighting Study: Azimuth (columns) vs Altitude (rows)", fontsize=14)
    plt.tight_layout()
    return fig


def save_animation_sequence(
    output_dir: str | Path,
    terrain,
    sequence_func: Callable[[int, int], np.ndarray],
    frame_count: int,
    name_pattern: str = "frame_{:04d}.png",
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """
    Save a custom animation sequence as individual PNG frames.

    Args:
        output_dir: Directory to save frames
        terrain: Terrain data
        sequence_func: Function(frame_index, total_frames) -> RGB array
        frame_count: Total number of frames to generate
        name_pattern: Filename pattern (must include one format placeholder for frame number)
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of paths to saved frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i in range(frame_count):
        frame = sequence_func(i, frame_count)
        frame_path = output_path / name_pattern.format(i)

        img = Image.fromarray(frame, mode="RGB")
        img.save(frame_path)
        saved_paths.append(frame_path)

        if progress_callback is not None:
            progress_callback(i + 1, frame_count)

    return saved_paths
