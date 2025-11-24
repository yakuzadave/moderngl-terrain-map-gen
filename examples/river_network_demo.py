"""
River Network Generation Demo
=============================

This example demonstrates the River Network Generator which creates realistic
river systems using GPU-accelerated flow accumulation algorithms.

Features:
- D8 flow direction calculation
- Iterative flow accumulation
- River channel carving
- Moisture/wetland maps
- Integration with existing terrain generators

Usage:
    python examples/river_network_demo.py [--resolution 512] [--seed 42] [--preset gentle]
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src import (
    ErosionTerrainGenerator,
    ErosionParams,
    RiverGenerator,
    RiverParams,
    utils,
)

# Add parent to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_visualization(
    original_height: np.ndarray,
    carved_height: np.ndarray,
    river_map: np.ndarray,
    moisture_map: np.ndarray,
) -> np.ndarray:
    """Create a composite visualization of river generation results."""
    h, w = original_height.shape

    # Create 2x2 grid visualization
    vis = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # Top-left: Original terrain
    orig_u8 = (np.clip(original_height, 0, 1) * 255).astype(np.uint8)
    vis[:h, :w, 0] = orig_u8
    vis[:h, :w, 1] = orig_u8
    vis[:h, :w, 2] = orig_u8

    # Top-right: Carved terrain with rivers highlighted
    carved_u8 = (np.clip(carved_height, 0, 1) * 255).astype(np.uint8)
    river_mask = (river_map > 0.3).astype(np.float32)
    vis[h:h*2, :w, 0] = np.where(river_mask > 0,
                                 50, carved_u8).astype(np.uint8)
    vis[h:h*2, :w, 1] = np.where(river_mask > 0,
                                 100, carved_u8).astype(np.uint8)
    vis[h:h*2, :w, 2] = np.where(river_mask > 0,
                                 200, carved_u8).astype(np.uint8)

    # Bottom-left: River network only
    river_u8 = (np.clip(river_map, 0, 1) * 255).astype(np.uint8)
    vis[:h, w:w*2, 0] = river_u8
    vis[:h, w:w*2, 1] = river_u8
    vis[:h, w:w*2, 2] = river_u8

    # Bottom-right: Moisture map
    moisture_u8 = (np.clip(moisture_map, 0, 1) * 255).astype(np.uint8)
    vis[h:h*2, w:w*2, 0] = 0
    vis[h:h*2, w:w*2, 1] = moisture_u8
    vis[h:h*2, w:w*2, 2] = moisture_u8 // 2

    return vis


def create_shaded_river_overlay(
    terrain,
    river_map: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Create shaded terrain with blue river overlay."""
    # Get base shaded terrain
    shaded = utils.shade_heightmap(
        terrain,
        azimuth=azimuth,
        altitude=altitude,
        colormap="terrain",
        vert_exag=2.0,
    )

    # Overlay rivers in blue
    river_mask = (river_map > 0.2)[..., np.newaxis]
    river_color = np.array([50, 100, 200], dtype=np.uint8)

    result = np.where(river_mask, river_color, shaded)
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="River Network Generation Demo")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Terrain resolution (default: 512)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--preset", choices=["gentle", "major", "delta", "canyon"],
                        default="major",
                        help="River preset (default: major)")
    parser.add_argument("--output-dir", type=str, default="river_output",
                        help="Output directory (default: river_output)")
    parser.add_argument("--terrain-preset", choices=["canyon", "plains", "mountains"],
                        default="mountains",
                        help="Base terrain preset (default: mountains)")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🏔️  Generating base terrain ({args.resolution}x{args.resolution})...")

    # Step 1: Generate base terrain
    terrain_presets = {
        "canyon": ErosionParams.canyon,
        "plains": ErosionParams.plains,
        "mountains": ErosionParams.mountains,
    }

    gen = ErosionTerrainGenerator(
        resolution=args.resolution,
        defaults=terrain_presets[args.terrain_preset](),
    )
    try:
        base_terrain = gen.generate_heightmap(seed=args.seed)
    finally:
        gen.cleanup()

    print(
        f"   Height range: {base_terrain.height.min():.3f} - {base_terrain.height.max():.3f}")

    # Step 2: Generate river network
    river_presets = {
        "gentle": RiverParams.gentle_streams,
        "major": RiverParams.major_rivers,
        "delta": RiverParams.delta_wetlands,
        "canyon": RiverParams.canyon_rivers,
    }

    river_params = river_presets[args.preset]()
    river_params.seed = float(args.seed)

    print(f"🌊 Generating river network (preset: {args.preset})...")
    print(f"   Iterations: {river_params.iterations}")
    print(f"   River depth: {river_params.river_depth}")
    print(f"   River width: {river_params.river_width}")

    river_gen = RiverGenerator(resolution=args.resolution)
    try:
        def progress(current, total):
            if current % 16 == 0 or current == total:
                pct = (current / total) * 100
                print(f"   Flow propagation: {pct:.0f}%", end="\r")

        river_result = river_gen.generate(
            base_terrain.height,
            params=river_params,
            progress_callback=progress,
        )
    finally:
        river_gen.cleanup()

    print()  # Newline after progress

    # Extract results
    river_map = river_result["river_map"]
    moisture_map = river_result["moisture_map"]
    carved_height = river_result["carved_height"]
    water_depth = river_result["water_depth"]
    flow_accum = river_result["flow_accumulation"]

    # Calculate river coverage
    river_coverage = (river_map > 0.3).sum() / river_map.size * 100
    print(f"   River coverage: {river_coverage:.2f}%")
    print(f"   Max flow accumulation: {flow_accum.max():.4f}")

    # Step 3: Create updated TerrainMaps
    from src.utils.artifacts import TerrainMaps

    river_terrain = TerrainMaps(
        height=carved_height,
        normals=base_terrain.normals,  # Note: Should recalculate for carved terrain
        erosion_mask=base_terrain.erosion_mask,
        river_map=river_map,
        moisture_map=moisture_map,
        water_depth=water_depth,
        flow_accumulation=flow_accum,
    )

    # Step 4: Save outputs
    print(f"\n💾 Saving outputs to {output_dir}/...")

    # Basic maps
    utils.save_heightmap_png(output_dir / "height_original.png", base_terrain)
    utils.save_heightmap_png(output_dir / "height_carved.png", river_terrain)

    # River maps - save directly via PIL since utils might not have these functions yet
    river_u8 = (np.clip(river_map, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(river_u8, mode="L").save(output_dir / "river_mask.png")

    moisture_u8 = (np.clip(moisture_map, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(moisture_u8, mode="L").save(output_dir / "moisture.png")

    water_depth_u8 = (np.clip(water_depth, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(water_depth_u8, mode="L").save(
        output_dir / "water_depth.png")

    # Flow accumulation (normalized for visibility)
    flow_normalized = flow_accum / (flow_accum.max() + 1e-6)
    flow_u8 = (np.clip(flow_normalized, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(flow_u8, mode="L").save(
        output_dir / "flow_accumulation.png")

    # Visualization composite
    vis = create_visualization(
        base_terrain.height,
        carved_height,
        river_map,
        moisture_map,
    )
    Image.fromarray(vis, mode="RGB").save(
        output_dir / "visualization_grid.png")

    # Shaded relief with rivers
    shaded_rivers = create_shaded_river_overlay(river_terrain, river_map)
    Image.fromarray(shaded_rivers, mode="RGB").save(
        output_dir / "shaded_with_rivers.png")

    # Original shaded for comparison
    shaded_original = utils.shade_heightmap(
        base_terrain, colormap="terrain", vert_exag=2.0)
    Image.fromarray(shaded_original, mode="RGB").save(
        output_dir / "shaded_original.png")

    # Export 3D mesh with rivers
    print("   Exporting OBJ mesh...")
    utils.export_obj_mesh(
        output_dir / "terrain_with_rivers.obj",
        river_terrain,
        scale=10.0,
        height_scale=2.0,
    )

    # NPZ bundle with all data
    np.savez_compressed(
        output_dir / "river_terrain.npz",
        height_original=base_terrain.height,
        height_carved=carved_height,
        normals=base_terrain.normals,
        river_map=river_map,
        moisture_map=moisture_map,
        water_depth=water_depth,
        flow_accumulation=flow_accum,
    )

    print("\n✅ River generation complete!")
    print("\nOutput files:")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name} ({size_kb:.1f} KB)")

    print("\n📊 Summary:")
    print(f"   Base terrain: {args.terrain_preset}")
    print(f"   River preset: {args.preset}")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   River coverage: {river_coverage:.2f}%")
    print(
        f"   Height reduction (max): {(base_terrain.height - carved_height).max():.4f}")


if __name__ == "__main__":
    main()
