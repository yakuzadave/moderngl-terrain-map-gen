"""
Demonstration of all export format capabilities.

This script generates a sample terrain and exports it to multiple formats:
- PNG (16-bit heightmap, normal map, erosion mask)
- RAW (16-bit little-endian)
- R32 (32-bit float)
- OBJ (Wavefront mesh with UVs)
- STL (binary STL for 3D printing)
- glTF (glTF 2.0 with embedded textures)
- NPZ (NumPy compressed bundle)
"""
from __future__ import annotations

from pathlib import Path

from src import ErosionTerrainGenerator, ErosionParams, utils


def main() -> None:
    output_dir = Path("export_demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Export Formats Demonstration")
    print("=" * 60)

    # Generate a canyon preset terrain
    print("\n[1/8] Generating terrain...")
    gen = ErosionTerrainGenerator(resolution=512)
    try:
        terrain = gen.generate_heightmap(
            seed=12345,
            seamless=True,
            preset=ErosionParams.canyon()
        )
        print(
            f"    ✓ Generated {terrain.resolution}x{terrain.resolution} terrain")
    finally:
        gen.cleanup()

    # Individual format exports
    print("\n[2/8] Exporting PNG heightmap (16-bit)...")
    utils.save_heightmap_png(output_dir / "terrain_height.png", terrain)
    print("    ✓ terrain_height.png")

    print("\n[3/8] Exporting normal map...")
    utils.save_normal_map_png(output_dir / "terrain_normal.png", terrain)
    print("    ✓ terrain_normal.png")

    print("\n[4/8] Exporting RAW heightmap (16-bit LE)...")
    utils.save_heightmap_raw(output_dir / "terrain.raw", terrain)
    print("    ✓ terrain.raw")

    print("\n[5/8] Exporting R32 heightmap (32-bit float LE)...")
    utils.save_heightmap_r32(output_dir / "terrain.r32", terrain)
    print("    ✓ terrain.r32")

    print("\n[6/8] Exporting OBJ mesh...")
    utils.export_obj_mesh(
        output_dir / "terrain.obj",
        terrain,
        scale=10.0,
        height_scale=2.0
    )
    print("    ✓ terrain.obj")

    print("\n[7/8] Exporting STL mesh...")
    utils.export_stl_mesh(
        output_dir / "terrain.stl",
        terrain,
        scale=10.0,
        height_scale=2.0
    )
    print("    ✓ terrain.stl")

    print("\n[8/8] Exporting glTF 2.0 mesh with textures...")
    utils.export_gltf_mesh(
        output_dir / "terrain.gltf",
        terrain,
        scale=10.0,
        height_scale=2.0,
        embed_textures=True
    )
    print("    ✓ terrain.gltf (with embedded textures)")

    # Batch export all formats
    print("\n" + "=" * 60)
    print("Batch Export (All Formats)")
    print("=" * 60)

    batch_dir = output_dir / "batch"
    results = utils.export_all_formats(
        batch_dir,
        terrain,
        base_name="canyon",
        formats=['png', 'raw', 'r32', 'obj', 'stl', 'gltf', 'npz'],
        scale=10.0,
        height_scale=2.0
    )

    print(f"\nExported {len(results)} files to {batch_dir}:")
    for fmt, path in sorted(results.items()):
        file_size = path.stat().st_size / 1024
        print(f"  • {fmt:20s} → {path.name:25s} ({file_size:>8.1f} KB)")

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nFile formats:")
    print("  PNG     - 16-bit grayscale heightmap (lossless)")
    print("  RAW     - 16-bit little-endian binary (Unity/Unreal compatible)")
    print("  R32     - 32-bit float little-endian (full precision)")
    print("  OBJ     - Wavefront mesh with UV coordinates")
    print("  STL     - Binary STL mesh (3D printing ready)")
    print("  glTF    - glTF 2.0 with embedded PBR materials and textures")
    print("  NPZ     - NumPy compressed bundle (all maps)")


if __name__ == "__main__":
    main()
