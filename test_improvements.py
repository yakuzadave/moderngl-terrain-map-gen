"""
Quick test to verify core functionality works after improvements.
Run this to ensure the codebase is working correctly.
"""
import sys
from pathlib import Path


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from src import (
            ErosionTerrainGenerator,  # noqa: F401
            MorphologicalTerrainGPU,
            TerrainMaps,
            utils,
        )
        from src.generators.erosion import ErosionParams
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_generation():
    """Test basic terrain generation."""
    print("\nTesting basic terrain generation...")
    try:
        from src import ErosionTerrainGenerator

        gen = ErosionTerrainGenerator(resolution=128)
        terrain = gen.generate_heightmap(seed=42)
        gen.cleanup()

        assert terrain.height.shape == (128, 128)
        assert terrain.normals.shape == (128, 128, 3)

        print("✓ Basic generation works")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_presets():
    """Test terrain presets."""
    print("\nTesting terrain presets...")
    try:
        from src import ErosionTerrainGenerator
        from src.generators.erosion import ErosionParams

        for name in ["canyon", "plains", "mountains"]:
            preset = getattr(ErosionParams, name)()
            gen = ErosionTerrainGenerator(resolution=128, defaults=preset)
            terrain = gen.generate_heightmap(seed=42)
            gen.cleanup()

            assert terrain.height.shape == (128, 128)

        print("✓ All presets work")
        return True
    except Exception as e:
        print(f"✗ Presets failed: {e}")
        return False


def test_rendering():
    """Test basic rendering functions."""
    print("\nTesting rendering functions...")
    try:
        from src import ErosionTerrainGenerator, utils

        gen = ErosionTerrainGenerator(resolution=128)
        terrain = gen.generate_heightmap(seed=42)
        gen.cleanup()

        # Test shade_heightmap
        img = utils.shade_heightmap(terrain)
        assert img.shape == (128, 128, 3)
        assert img.dtype == "uint8"

        # Test slope_intensity
        slope = utils.slope_intensity(terrain)
        assert slope.shape == (128, 128)
        assert slope.dtype == "uint8"

        print("✓ Basic rendering works")
        return True
    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_rendering():
    """Test advanced rendering functions."""
    print("\nTesting advanced rendering functions...")
    try:
        from src import ErosionTerrainGenerator, utils

        gen = ErosionTerrainGenerator(resolution=128)
        terrain = gen.generate_heightmap(seed=42)
        gen.cleanup()

        # Test turntable frames
        frames = utils.render_turntable_frames(terrain, frames=4)
        assert len(frames) == 4
        assert frames[0].shape == (128, 128, 3)

        # Test multi-angle
        renders = utils.render_multi_angle(terrain)
        assert len(renders) == 5

        print("✓ Advanced rendering works")
        return True
    except Exception as e:
        print(f"✗ Advanced rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export():
    """Test export functions."""
    print("\nTesting export functions...")
    try:
        from src import ErosionTerrainGenerator, utils
        import tempfile

        gen = ErosionTerrainGenerator(resolution=128)
        terrain = gen.generate_heightmap(seed=42)
        gen.cleanup()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test heightmap export
            utils.save_heightmap_png(tmppath / "height.png", terrain)
            assert (tmppath / "height.png").exists()

            # Test normal map export
            utils.save_normal_map_png(tmppath / "normal.png", terrain)
            assert (tmppath / "normal.png").exists()

            # Test OBJ export
            utils.export_obj_mesh(tmppath / "mesh.obj", terrain)
            assert (tmppath / "mesh.obj").exists()

        print("✓ Export functions work")
        return True
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Running terrain generation tests...")
    print("="*70)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Basic Generation", test_basic_generation()))
    results.append(("Presets", test_presets()))
    results.append(("Rendering", test_rendering()))
    results.append(("Advanced Rendering", test_advanced_rendering()))
    results.append(("Export", test_export()))

    print("\n" + "="*70)
    print("Test Summary:")
    print("="*70)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("="*70)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*70)

    if failed > 0:
        print("\n⚠ Some tests failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed! The codebase is working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
