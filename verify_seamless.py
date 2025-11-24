#!/usr/bin/env python3
"""
Visual verification of seamless terrain tiling.
Creates a 2x2 grid of the same terrain to check for visible seams.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from src import ErosionTerrainGenerator
from src.utils import save_shaded_relief_png, shade_heightmap


def verify_seamless_tiling():
    """Generate terrain and create 2x2 tiled visualization."""

    print("=" * 60)
    print("Seamless Terrain Tiling Verification")
    print("=" * 60)
    print()

    # Configuration
    RESOLUTION = 512
    SEED = 42
    OUTPUT_DIR = Path("seamless_verification")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"Seed: {SEED}")
    print(f"Output: {OUTPUT_DIR}/")
    print()

    # Initialize generator
    gen = ErosionTerrainGenerator(resolution=RESOLUTION)

    # Generate both seamless and non-seamless for comparison
    print("Generating terrains...")
    print("  [1/2] Non-seamless terrain...", end=" ", flush=True)
    terrain_normal = gen.generate_heightmap(seed=SEED, seamless=False)
    print("✓")

    print("  [2/2] Seamless terrain...", end=" ", flush=True)
    terrain_seamless = gen.generate_heightmap(seed=SEED, seamless=True)
    print("✓")
    print()

    # Cleanup GPU resources
    gen.cleanup()

    # Create 2x2 tiled versions
    print("Creating 2×2 tiled compositions...")

    h_normal = terrain_normal.height
    h_seamless = terrain_seamless.height

    # Tile both heightmaps
    tiled_normal = np.block([[h_normal, h_normal], [h_normal, h_normal]])
    tiled_seamless = np.block(
        [[h_seamless, h_seamless], [h_seamless, h_seamless]])

    print(f"  Single tile: {RESOLUTION}×{RESOLUTION}")
    print(f"  Tiled grid: {tiled_normal.shape[1]}×{tiled_normal.shape[0]}")
    print()

    # Render shaded versions
    print("Rendering shaded reliefs...")

    # Single tiles
    print("  [1/6] Non-seamless single...", end=" ", flush=True)
    shaded_normal_single = save_shaded_relief_png(
        str(OUTPUT_DIR / "normal_single.png"),
        terrain_normal,
        azimuth=315,
        altitude=45
    )
    print("✓")

    print("  [2/6] Seamless single...", end=" ", flush=True)
    shaded_seamless_single = save_shaded_relief_png(
        str(OUTPUT_DIR / "seamless_single.png"),
        terrain_seamless,
        azimuth=315,
        altitude=45
    )
    print("✓")

    # 2x2 grids (create temporary TerrainMaps for shading)
    from src.utils.artifacts import TerrainMaps

    print("  [3/6] Non-seamless 2×2...", end=" ", flush=True)
    tiled_terrain_normal = TerrainMaps(
        height=tiled_normal,
        normals=np.zeros((*tiled_normal.shape, 3), dtype=np.float32),
        erosion_mask=None
    )
    shaded_normal_grid = save_shaded_relief_png(
        str(OUTPUT_DIR / "normal_2x2_grid.png"),
        tiled_terrain_normal,
        azimuth=315,
        altitude=45
    )
    print("✓")

    print("  [4/6] Seamless 2×2...", end=" ", flush=True)
    tiled_terrain_seamless = TerrainMaps(
        height=tiled_seamless,
        normals=np.zeros((*tiled_seamless.shape, 3), dtype=np.float32),
        erosion_mask=None
    )
    shaded_seamless_grid = save_shaded_relief_png(
        str(OUTPUT_DIR / "seamless_2x2_grid.png"),
        tiled_terrain_seamless,
        azimuth=315,
        altitude=45
    )
    print("✓")

    # Create side-by-side comparison
    print("  [5/6] Creating comparison image...", end=" ", flush=True)

    # Load rendered images
    img_normal_grid = Image.open(OUTPUT_DIR / "normal_2x2_grid.png")
    img_seamless_grid = Image.open(OUTPUT_DIR / "seamless_2x2_grid.png")

    # Create side-by-side comparison
    width = img_normal_grid.width
    height = img_normal_grid.height
    comparison = Image.new("RGB", (width * 2 + 40, height + 80), (40, 40, 40))

    # Paste images
    comparison.paste(img_normal_grid, (20, 60))
    comparison.paste(img_seamless_grid, (width + 20, 60))

    # Add labels (using PIL's default font)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)

    # Title
    draw.text((width + 20, 10), "Seamless Tiling Comparison",
              fill=(255, 255, 255))

    # Labels
    draw.text((width // 2, 30), "Non-Seamless (Visible Seams)",
              fill=(255, 100, 100))
    draw.text((width + width // 2, 30),
              "Seamless (No Seams)", fill=(100, 255, 100))

    comparison.save(OUTPUT_DIR / "comparison.png")
    print("✓")

    # Analyze edge continuity
    print("  [6/6] Analyzing edge continuity...", end=" ", flush=True)

    # Check horizontal edges
    left_edge_normal = h_normal[:, 0]
    right_edge_normal = h_normal[:, -1]
    h_diff_normal = np.max(np.abs(left_edge_normal - right_edge_normal))

    left_edge_seamless = h_seamless[:, 0]
    right_edge_seamless = h_seamless[:, -1]
    h_diff_seamless = np.max(np.abs(left_edge_seamless - right_edge_seamless))

    # Check vertical edges
    top_edge_normal = h_normal[0, :]
    bottom_edge_normal = h_normal[-1, :]
    v_diff_normal = np.max(np.abs(top_edge_normal - bottom_edge_normal))

    top_edge_seamless = h_seamless[0, :]
    bottom_edge_seamless = h_seamless[-1, :]
    v_diff_seamless = np.max(np.abs(top_edge_seamless - bottom_edge_seamless))

    print("✓")
    print()

    # Report results
    print("=" * 60)
    print("Edge Continuity Analysis")
    print("=" * 60)
    print()

    print("Non-Seamless Terrain:")
    print(f"  Horizontal edge diff: {h_diff_normal:.6f}")
    print(f"  Vertical edge diff: {v_diff_normal:.6f}")
    print(f"  Maximum diff: {max(h_diff_normal, v_diff_normal):.6f}")
    print(
        f"  Seamless: {'❌ NO' if max(h_diff_normal, v_diff_normal) > 0.01 else '✅ YES'}")
    print()

    print("Seamless Terrain:")
    print(f"  Horizontal edge diff: {h_diff_seamless:.6f}")
    print(f"  Vertical edge diff: {v_diff_seamless:.6f}")
    print(f"  Maximum diff: {max(h_diff_seamless, v_diff_seamless):.6f}")
    print(
        f"  Seamless: {'✅ YES' if max(h_diff_seamless, v_diff_seamless) < 0.01 else '❌ NO'}")
    print()

    print("Improvement:")
    improvement_h = ((h_diff_normal - h_diff_seamless) /
                     h_diff_normal * 100) if h_diff_normal > 0 else 0
    improvement_v = ((v_diff_normal - v_diff_seamless) /
                     v_diff_normal * 100) if v_diff_normal > 0 else 0
    print(f"  Horizontal: {improvement_h:.1f}% reduction")
    print(f"  Vertical: {improvement_v:.1f}% reduction")
    print()

    # Summary
    print("=" * 60)
    print("Output Files")
    print("=" * 60)
    print()
    print(f"  📁 {OUTPUT_DIR}/")
    print(f"     ├── normal_single.png       (non-seamless, 1 tile)")
    print(f"     ├── seamless_single.png     (seamless, 1 tile)")
    print(f"     ├── normal_2x2_grid.png     (non-seamless, 2×2 grid)")
    print(f"     ├── seamless_2x2_grid.png   (seamless, 2×2 grid)")
    print(f"     └── comparison.png          (side-by-side comparison)")
    print()

    print("Visual Inspection:")
    print(f"  1. Open: {OUTPUT_DIR}/normal_2x2_grid.png")
    print(f"     → Look for visible seams at center cross (tile boundaries)")
    print()
    print(f"  2. Open: {OUTPUT_DIR}/seamless_2x2_grid.png")
    print(f"     → Should show NO visible seams at center cross")
    print()
    print(f"  3. Open: {OUTPUT_DIR}/comparison.png")
    print(f"     → Side-by-side comparison highlighting the difference")
    print()

    # Final verdict
    seamless_works = max(h_diff_seamless, v_diff_seamless) < 0.01
    print("=" * 60)
    if seamless_works:
        print("✅ VERIFICATION PASSED")
        print("   Seamless mode successfully eliminates tiling artifacts!")
    else:
        print("❌ VERIFICATION FAILED")
        print("   Seamless mode still shows edge discontinuities.")
        print("   Further debugging needed.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    verify_seamless_tiling()
