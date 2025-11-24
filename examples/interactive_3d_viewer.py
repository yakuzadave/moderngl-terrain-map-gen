"""
Interactive 3D Terrain Viewer using PyVista

This example demonstrates how to view generated terrain interactively in 3D
without relying on static image exports.

Requirements:
    pip install pyvista

Usage:
    python examples/interactive_3d_viewer.py
    python examples/interactive_3d_viewer.py --seed 123 --resolution 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Check for pyvista
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not installed. Install with: pip install pyvista")
    print("Falling back to matplotlib 3D plot...")


def terrain_to_structured_grid(
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
):
    """
    Convert TerrainMaps to a PyVista StructuredGrid for 3D visualization.
    
    Args:
        terrain: TerrainMaps instance with height data
        scale: Horizontal scale (XZ extent)
        height_scale: Vertical scale multiplier
        
    Returns:
        pyvista.StructuredGrid with terrain geometry and scalar data
    """
    height = terrain.height
    h, w = height.shape
    
    # Create coordinate arrays
    x = np.linspace(0, scale, w)
    z = np.linspace(0, scale, h)
    xx, zz = np.meshgrid(x, z)
    yy = height * height_scale
    
    # Create structured grid
    grid = pv.StructuredGrid(xx, yy, zz)
    
    # Add scalar data for coloring
    grid["height"] = height.flatten(order="F")
    
    # Add erosion mask if available
    if terrain.erosion_mask is not None:
        grid["erosion"] = terrain.erosion_mask.flatten(order="F")
    
    # Calculate slope from normals if available
    if terrain.normals is not None:
        # Slope = 1 - normal.y (more vertical = steeper)
        slope = 1.0 - terrain.normals[:, :, 1]
        grid["slope"] = slope.flatten(order="F")
    
    return grid


def view_terrain_pyvista(
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
    cmap: str = "terrain",
    show_edges: bool = False,
    lighting: str = "three_lights",
):
    """
    Display terrain in an interactive 3D PyVista window.
    
    Args:
        terrain: TerrainMaps instance
        scale: Horizontal extent
        height_scale: Vertical exaggeration
        cmap: Colormap name (matplotlib colormaps)
        show_edges: Show wireframe edges
        lighting: Lighting preset ("three_lights", "light_kit", "none")
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista required for 3D viewing. pip install pyvista")
    
    grid = terrain_to_structured_grid(terrain, scale, height_scale)
    
    # Create plotter
    plotter = pv.Plotter(title="GPU Terrain Generator - 3D Viewer")
    
    # Configure lighting
    if lighting == "three_lights":
        plotter.enable_3_lights()
    elif lighting == "light_kit":
        plotter.enable_lightkit()
    
    # Add terrain mesh
    plotter.add_mesh(
        grid,
        scalars="height",
        cmap=cmap,
        show_edges=show_edges,
        smooth_shading=True,
        specular=0.2,
        specular_power=15,
    )
    
    # Add axes
    plotter.add_axes()
    
    # Set camera position (isometric-ish view)
    plotter.camera_position = [
        (scale * 1.5, scale * 1.0, scale * 1.5),  # Camera position
        (scale / 2, 0, scale / 2),                 # Focus point
        (0, 1, 0),                                  # Up vector
    ]
    
    # Add instruction text
    plotter.add_text(
        "LMB: Rotate | RMB: Zoom | MMB: Pan | Q: Quit",
        position="lower_left",
        font_size=10,
    )
    
    # Show interactive window
    plotter.show()


def view_terrain_matplotlib(terrain, scale: float = 10.0, height_scale: float = 2.0):
    """Fallback 3D view using matplotlib (less interactive but no extra deps)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    height = terrain.height
    h, w = height.shape
    
    # Downsample for matplotlib performance
    step = max(1, w // 100)
    
    x = np.linspace(0, scale, w)[::step]
    z = np.linspace(0, scale, h)[::step]
    xx, zz = np.meshgrid(x, z)
    yy = height[::step, ::step] * height_scale
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(
        xx, zz, yy,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Height")
    ax.set_title("Terrain (matplotlib fallback - drag to rotate)")
    
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D terrain viewer"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for terrain generation"
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Terrain resolution (256-1024 recommended for interactive)"
    )
    parser.add_argument(
        "--preset", choices=["canyon", "mountains", "plains", "natural"],
        default="canyon",
        help="Terrain preset"
    )
    parser.add_argument(
        "--cmap", default="terrain",
        help="Colormap for height (terrain, viridis, gist_earth, etc)"
    )
    parser.add_argument(
        "--scale", type=float, default=10.0,
        help="Horizontal terrain extent"
    )
    parser.add_argument(
        "--height-scale", type=float, default=2.0,
        help="Vertical exaggeration"
    )
    parser.add_argument(
        "--edges", action="store_true",
        help="Show wireframe edges"
    )
    parser.add_argument(
        "--export-vtk", type=str, default=None,
        help="Export to VTK file instead of displaying"
    )
    
    args = parser.parse_args()
    
    # Import generators
    from src.generators.erosion import ErosionTerrainGenerator, ErosionParams
    
    # Get preset params
    preset_funcs = {
        "canyon": ErosionParams.canyon,
        "mountains": ErosionParams.mountains,
        "plains": ErosionParams.plains,
        "natural": ErosionParams.natural,
    }
    params = preset_funcs[args.preset]()
    
    print(f"Generating terrain (seed={args.seed}, resolution={args.resolution})...")
    
    # Pass preset params as defaults to the generator
    gen = ErosionTerrainGenerator(resolution=args.resolution, defaults=params)
    try:
        terrain = gen.generate_heightmap(seed=args.seed)
        
        print(f"  Height range: {terrain.height.min():.3f} - {terrain.height.max():.3f}")
        
        if args.export_vtk and HAS_PYVISTA:
            grid = terrain_to_structured_grid(terrain, args.scale, args.height_scale)
            grid.save(args.export_vtk)
            print(f"Exported to {args.export_vtk}")
        else:
            # Display interactive viewer
            if HAS_PYVISTA:
                print("Opening PyVista 3D viewer...")
                view_terrain_pyvista(
                    terrain,
                    scale=args.scale,
                    height_scale=args.height_scale,
                    cmap=args.cmap,
                    show_edges=args.edges,
                )
            else:
                print("Opening matplotlib 3D viewer (install pyvista for better experience)...")
                view_terrain_matplotlib(terrain, args.scale, args.height_scale)
    
    finally:
        gen.cleanup()


if __name__ == "__main__":
    main()
