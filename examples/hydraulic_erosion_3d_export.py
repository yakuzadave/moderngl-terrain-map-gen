#!/usr/bin/env python
"""
Hydraulic Erosion 3D Model Export Demo

This script demonstrates the complete pipeline for generating 3D models
from hydraulic erosion simulation results:

1. Generate base terrain using the erosion generator
2. Apply hydraulic erosion simulation with varying iterations
3. Export 3D meshes in OBJ, STL, and glTF formats
4. Generate comparison visualizations

Output Structure:
    hydraulic_3d_output/
    ├── base/
    │   ├── terrain.obj
    │   ├── terrain.stl
    │   ├── terrain.gltf
    │   ├── heightmap.png
    │   ├── normal.png
    │   └── shaded.png
    ├── eroded_50iter/
    │   └── ... (same structure)
    ├── eroded_100iter/
    │   └── ...
    ├── eroded_200iter/
    │   └── ...
    └── comparison/
        ├── all_stages.png
        └── erosion_progression.gif (if ffmpeg available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.erosion import ErosionTerrainGenerator, ErosionParams
from src.generators.hydraulic import HydraulicErosionGenerator, HydraulicParams
from src.utils import (
    TerrainMaps,
    save_heightmap_png,
    save_normal_map_png,
    save_shaded_relief_png,
    save_splatmap_rgba,
    export_obj_mesh,
    export_stl_mesh,
    export_gltf_mesh,
    export_all_formats,
    save_ao_map,
    render_turntable_frames,
    save_turntable_video,
    create_comparison_grid,
)


class ExportResult(NamedTuple):
    """Container for export results."""
    name: str
    terrain: TerrainMaps
    output_dir: Path
    obj_path: Path | None
    stl_path: Path | None
    gltf_path: Path | None


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_info(label: str, value: str) -> None:
    """Print labeled information."""
    print(f"  {label}: {value}")


def generate_base_terrain(
    resolution: int = 512,
    seed: int = 42,
    preset: str = "mountains"
) -> TerrainMaps:
    """
    Generate a base terrain using the erosion generator.
    
    Args:
        resolution: Terrain resolution (power of 2 recommended)
        seed: Random seed for reproducibility
        preset: Terrain preset (canyon, mountains, plains, natural)
    
    Returns:
        TerrainMaps with generated terrain
    """
    print_header("Generating Base Terrain")
    print_info("Resolution", f"{resolution}x{resolution}")
    print_info("Seed", str(seed))
    print_info("Preset", preset)
    
    # Select preset
    preset_map = {
        "canyon": ErosionParams.canyon,
        "mountains": ErosionParams.mountains,
        "plains": ErosionParams.plains,
        "natural": ErosionParams.natural,
    }
    params = preset_map.get(preset, ErosionParams.mountains)()
    
    gen = ErosionTerrainGenerator(resolution=resolution, defaults=params)
    terrain = gen.generate_heightmap(seed=seed)
    gen.cleanup()
    
    # Normalize heightmap to 0-1 range
    h = terrain.height
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h_norm = h_norm.astype(np.float32)
    
    print_info("Height Range", f"{h_norm.min():.3f} - {h_norm.max():.3f}")
    print_info("Mean Height", f"{h_norm.mean():.3f}")
    
    return TerrainMaps(height=h_norm, normals=terrain.normals)


def apply_hydraulic_erosion(
    base_terrain: TerrainMaps,
    iterations: int = 100,
    params: HydraulicParams | None = None,
    resolution: int = 512
) -> TerrainMaps:
    """
    Apply hydraulic erosion to a terrain.
    
    Args:
        base_terrain: Input terrain to erode
        iterations: Number of erosion iterations
        params: Custom hydraulic parameters (uses defaults if None)
        resolution: Simulation resolution
    
    Returns:
        TerrainMaps with eroded terrain
    """
    print(f"\n  Applying {iterations} iterations of hydraulic erosion...")
    
    if params is None:
        params = HydraulicParams(
            iterations=iterations,
            dt=0.02,
            rain_rate=0.015,
            evaporation_rate=0.008,
            sediment_capacity=1.0,
            soil_dissolving=0.5,
            sediment_deposition=0.5,
            thermal_erosion_rate=0.15,
            talus_angle=0.008,
        )
    else:
        # Override iterations
        params = HydraulicParams(
            iterations=iterations,
            dt=params.dt,
            rain_rate=params.rain_rate,
            evaporation_rate=params.evaporation_rate,
            sediment_capacity=params.sediment_capacity,
            soil_dissolving=params.soil_dissolving,
            sediment_deposition=params.sediment_deposition,
            thermal_erosion_rate=params.thermal_erosion_rate,
            talus_angle=params.talus_angle,
        )
    
    gen = HydraulicErosionGenerator(resolution=resolution)
    result = gen.simulate(base_terrain.height.copy(), params)
    gen.cleanup()
    
    # Calculate erosion statistics
    diff = result.height - base_terrain.height
    print_info("Height Change", f"mean={diff.mean():.4f}, std={diff.std():.4f}")
    print_info("Eroded Range", f"{result.height.min():.3f} - {result.height.max():.3f}")
    
    return result


def export_terrain_3d(
    terrain: TerrainMaps,
    output_dir: Path,
    name: str = "terrain",
    scale: float = 10.0,
    height_scale: float = 2.0,
    formats: list[str] | None = None,
) -> ExportResult:
    """
    Export terrain to 3D mesh formats and supporting textures.
    
    Args:
        terrain: Terrain data to export
        output_dir: Output directory
        name: Base filename
        scale: Horizontal mesh scale
        height_scale: Vertical exaggeration
        formats: List of formats to export (obj, stl, gltf)
    
    Returns:
        ExportResult with paths to exported files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ["obj", "stl", "gltf"]
    
    obj_path = None
    stl_path = None
    gltf_path = None
    
    # Export 3D meshes
    if "obj" in formats:
        obj_path = export_obj_mesh(
            output_dir / f"{name}.obj",
            terrain,
            scale=scale,
            height_scale=height_scale
        )
        print_info("OBJ Mesh", str(obj_path))
    
    if "stl" in formats:
        stl_path = export_stl_mesh(
            output_dir / f"{name}.stl",
            terrain,
            scale=scale,
            height_scale=height_scale
        )
        print_info("STL Mesh", str(stl_path))
    
    if "gltf" in formats:
        gltf_path = export_gltf_mesh(
            output_dir / f"{name}.gltf",
            terrain,
            scale=scale,
            height_scale=height_scale,
            embed_textures=True
        )
        print_info("glTF Mesh", str(gltf_path))
    
    # Export supporting textures
    save_heightmap_png(output_dir / f"{name}_height.png", terrain)
    save_normal_map_png(output_dir / f"{name}_normal.png", terrain)
    save_shaded_relief_png(
        output_dir / f"{name}_shaded.png",
        terrain,
        colormap="terrain",
        vert_exag=2.0,
        azimuth=315.0,
        altitude=45.0
    )
    
    # Export game engine textures
    try:
        save_splatmap_rgba(output_dir / f"{name}_splatmap.png", terrain)
        save_ao_map(output_dir / f"{name}_ao.png", terrain, radius=3, strength=1.5)
    except Exception as e:
        print(f"  Warning: Could not generate all textures: {e}")
    
    return ExportResult(
        name=name,
        terrain=terrain,
        output_dir=output_dir,
        obj_path=obj_path,
        stl_path=stl_path,
        gltf_path=gltf_path,
    )


def create_comparison_visualization(
    results: list[ExportResult],
    output_dir: Path,
    colormap: str = "terrain"
) -> None:
    """
    Create comparison visualizations of erosion stages.
    
    Args:
        results: List of export results to compare
        output_dir: Output directory for comparison images
    """
    print_header("Creating Comparison Visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shaded images for comparison
    images = []
    labels = []
    
    for result in results:
        shaded_path = result.output_dir / f"{result.name}_shaded.png"
        if shaded_path.exists():
            img = Image.open(shaded_path)
            images.append(img)
            labels.append(result.name.replace("_", " ").title())
    
    if not images:
        print("  No images found for comparison")
        return
    
    # Create horizontal comparison strip
    strip_height = 300
    strips = []
    for img in images:
        ratio = strip_height / img.height
        resized = img.resize(
            (int(img.width * ratio), strip_height), 
            Image.Resampling.LANCZOS
        )
        strips.append(resized)
    
    total_width = sum(s.width for s in strips)
    comparison = Image.new("RGB", (total_width, strip_height + 30), (255, 255, 255))
    
    # Paste images
    x = 0
    for i, strip in enumerate(strips):
        comparison.paste(strip, (x, 0))
        # Add label (if PIL has ImageDraw support)
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison)
            label = labels[i]
            # Use default font
            draw.text((x + 10, strip_height + 5), label, fill=(0, 0, 0))
        except ImportError:
            pass
        x += strip.width
    
    comparison_path = output_dir / "erosion_comparison_strip.png"
    comparison.save(comparison_path)
    print_info("Comparison Strip", str(comparison_path))
    
    # Create side-by-side grid (2x2 if 4 stages, etc.)
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    thumb_size = 512
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        thumb = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        grid.paste(thumb, (col * thumb_size, row * thumb_size))
    
    grid_path = output_dir / "erosion_grid.png"
    grid.save(grid_path)
    print_info("Grid Comparison", str(grid_path))


def create_turntable_animation(
    terrain: TerrainMaps,
    output_path: Path,
    frames: int = 60,
    fps: int = 15
) -> None:
    """
    Create a turntable animation of the terrain.
    
    Args:
        terrain: Terrain data to animate
        output_path: Output path for animation (gif or mp4)
        frames: Number of frames
        fps: Frames per second
    """
    print(f"\n  Creating turntable animation ({frames} frames)...")
    
    try:
        save_turntable_video(
            output_path,
            terrain,
            frames=frames,
            fps=fps,
            altitude=50.0,
            vert_exag=2.5,
            colormap="terrain"
        )
        print_info("Animation", str(output_path))
    except Exception as e:
        print(f"  Warning: Could not create animation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D models from hydraulic erosion simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default generation (512x512, mountains preset)
  python hydraulic_erosion_3d_export.py
  
  # High resolution with more erosion stages
  python hydraulic_erosion_3d_export.py --resolution 1024 --stages 50,100,200,400
  
  # Canyon preset with custom seed
  python hydraulic_erosion_3d_export.py --preset canyon --seed 12345
  
  # Export only OBJ format with higher mesh scale
  python hydraulic_erosion_3d_export.py --formats obj --scale 20 --height-scale 3
        """
    )
    
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=512,
        help="Terrain resolution (default: 512)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for base terrain (default: 42)"
    )
    parser.add_argument(
        "--preset", "-p",
        choices=["canyon", "mountains", "plains", "natural"],
        default="mountains",
        help="Terrain preset (default: mountains)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="50,100,200",
        help="Erosion iteration stages, comma-separated (default: 50,100,200)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Horizontal mesh scale (default: 10.0)"
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=2.0,
        help="Vertical mesh scale (default: 2.0)"
    )
    parser.add_argument(
        "--formats", "-f",
        type=str,
        default="obj,stl,gltf",
        help="Export formats, comma-separated (default: obj,stl,gltf)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="hydraulic_3d_output",
        help="Output directory (default: hydraulic_3d_output)"
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip turntable animation generation"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    stages = [int(x.strip()) for x in args.stages.split(",")]
    formats = [x.strip().lower() for x in args.formats.split(",")]
    output_base = Path(args.output_dir)
    
    print_header("Hydraulic Erosion 3D Export Pipeline")
    print_info("Resolution", f"{args.resolution}x{args.resolution}")
    print_info("Seed", str(args.seed))
    print_info("Preset", args.preset)
    print_info("Erosion Stages", str(stages))
    print_info("Export Formats", ", ".join(formats))
    print_info("Output Directory", str(output_base.absolute()))
    
    # Generate base terrain
    base_terrain = generate_base_terrain(
        resolution=args.resolution,
        seed=args.seed,
        preset=args.preset
    )
    
    # Export base terrain
    print_header("Exporting Base Terrain (No Erosion)")
    results = [
        export_terrain_3d(
            base_terrain,
            output_base / "base",
            name="terrain",
            scale=args.scale,
            height_scale=args.height_scale,
            formats=formats
        )
    ]
    
    # Apply hydraulic erosion at different stages
    print_header("Applying Hydraulic Erosion")
    
    for iterations in stages:
        print(f"\n  --- {iterations} Iterations ---")
        eroded = apply_hydraulic_erosion(
            base_terrain,
            iterations=iterations,
            resolution=args.resolution
        )
        
        stage_name = f"eroded_{iterations}iter"
        result = export_terrain_3d(
            eroded,
            output_base / stage_name,
            name="terrain",
            scale=args.scale,
            height_scale=args.height_scale,
            formats=formats
        )
        results.append(result)
    
    # Create comparison visualizations
    create_comparison_visualization(
        results,
        output_base / "comparison"
    )
    
    # Create turntable animation of final result
    if not args.no_animation and len(results) > 1:
        print_header("Creating Turntable Animations")
        
        # Animation of base terrain
        create_turntable_animation(
            results[0].terrain,
            output_base / "comparison" / "base_turntable.gif",
            frames=36,
            fps=12
        )
        
        # Animation of most eroded terrain
        create_turntable_animation(
            results[-1].terrain,
            output_base / "comparison" / "eroded_turntable.gif",
            frames=36,
            fps=12
        )
    
    # Summary
    print_header("Export Complete!")
    print(f"\n  Output directory: {output_base.absolute()}")
    print(f"\n  Generated {len(results)} terrain stages:")
    
    for result in results:
        print(f"    - {result.name}")
        if result.obj_path:
            size = result.obj_path.stat().st_size / 1024 / 1024
            print(f"      OBJ: {size:.2f} MB")
        if result.stl_path:
            size = result.stl_path.stat().st_size / 1024 / 1024
            print(f"      STL: {size:.2f} MB")
        if result.gltf_path:
            size = result.gltf_path.stat().st_size / 1024 / 1024
            print(f"      glTF: {size:.2f} MB")
    
    print("\n  Files are ready for import into:")
    print("    - Blender (OBJ, glTF)")
    print("    - Unity (glTF, OBJ)")
    print("    - Unreal Engine (glTF, OBJ)")
    print("    - 3D Printing software (STL)")
    print("    - Three.js / Web (glTF)")


if __name__ == "__main__":
    main()
