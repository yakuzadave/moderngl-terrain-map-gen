"""Tests for terrain export functionality."""
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.artifacts import TerrainMaps
from src.utils.export import (
    export_gltf_mesh,
    export_obj_mesh,
    export_stl_mesh,
    save_heightmap_png,
    save_heightmap_r32,
    save_heightmap_raw,
    save_normal_map_png,
)


@pytest.fixture
def sample_terrain():
    """Create a sample terrain for testing."""
    resolution = 64
    # Create a simple pyramid heightmap
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    height = 1.0 - np.maximum(np.abs(X), np.abs(Y))
    height = height.astype(np.float32)

    # Create simple normals (pointing up)
    normals = np.zeros((resolution, resolution, 3), dtype=np.float32)
    normals[:, :, 1] = 1.0  # Up direction

    return TerrainMaps(height=height, normals=normals)


class TestHeightmapExports:
    def test_heightmap_png(self, sample_terrain):
        """Test 16-bit PNG heightmap export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "height.png"
            result = save_heightmap_png(path, sample_terrain)

            assert result.exists()
            assert result.suffix == ".png"
            assert result.stat().st_size > 0

    def test_heightmap_raw(self, sample_terrain):
        """Test raw 16-bit heightmap export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "height.raw"
            result = save_heightmap_raw(path, sample_terrain)

            assert result.exists()
            # Should be exactly resolution^2 * 2 bytes (16-bit)
            expected_size = 64 * 64 * 2
            assert result.stat().st_size == expected_size

    def test_heightmap_r32(self, sample_terrain):
        """Test raw 32-bit float heightmap export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "height.r32"
            result = save_heightmap_r32(path, sample_terrain)

            assert result.exists()
            # Should be exactly resolution^2 * 4 bytes (32-bit float)
            expected_size = 64 * 64 * 4
            assert result.stat().st_size == expected_size


class TestNormalMapExport:
    def test_normal_map_png(self, sample_terrain):
        """Test normal map PNG export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normal.png"
            result = save_normal_map_png(path, sample_terrain)

            assert result.exists()
            assert result.suffix == ".png"


class TestMeshExports:
    def test_obj_mesh_export(self, sample_terrain):
        """Test OBJ mesh export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terrain.obj"
            result = export_obj_mesh(
                path, sample_terrain, scale=10.0, height_scale=2.0)

            assert result.exists()
            assert result.suffix == ".obj"

            # Verify OBJ structure
            content = result.read_text()
            lines = content.split("\n")

            vertex_count = sum(1 for line in lines if line.startswith("v "))
            face_count = sum(1 for line in lines if line.startswith("f "))
            uv_count = sum(1 for line in lines if line.startswith("vt "))

            # 64x64 = 4096 vertices
            assert vertex_count == 64 * 64
            # Each quad = 2 triangles, (63*63) quads
            assert face_count == 63 * 63 * 2
            # UVs should match vertices
            assert uv_count == vertex_count

    def test_stl_mesh_export(self, sample_terrain):
        """Test STL mesh export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terrain.stl"
            result = export_stl_mesh(
                path, sample_terrain, scale=10.0, height_scale=2.0)

            assert result.exists()
            assert result.suffix == ".stl"

            # Verify binary STL structure
            with open(result, "rb") as f:
                # Skip 80-byte header
                f.read(80)
                # Read triangle count
                triangle_count = struct.unpack("<I", f.read(4))[0]

            # Each quad = 2 triangles, (63*63) quads
            expected_triangles = 63 * 63 * 2
            assert triangle_count == expected_triangles

    def test_gltf_mesh_export(self, sample_terrain):
        """Test glTF mesh export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terrain.gltf"
            result = export_gltf_mesh(
                path,
                sample_terrain,
                scale=10.0,
                height_scale=2.0,
                embed_textures=True,
            )

            assert result.exists()
            assert result.suffix == ".gltf"

            # Verify glTF structure
            with open(result, "r") as f:
                gltf = json.load(f)

            assert gltf["asset"]["version"] == "2.0"
            assert len(gltf["meshes"]) == 1
            assert len(gltf["materials"]) == 1
            assert len(gltf["accessors"]) == 4  # position, normal, uv, indices

            # Check vertex count
            pos_accessor = gltf["accessors"][0]
            assert pos_accessor["count"] == 64 * 64

            # Check index count (6 indices per quad for 2 triangles)
            idx_accessor = gltf["accessors"][3]
            assert idx_accessor["count"] == 63 * 63 * 6

    def test_gltf_mesh_without_textures(self, sample_terrain):
        """Test glTF mesh export without embedded textures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terrain.gltf"
            result = export_gltf_mesh(
                path,
                sample_terrain,
                scale=10.0,
                height_scale=2.0,
                embed_textures=False,
            )

            assert result.exists()

            with open(result, "r") as f:
                gltf = json.load(f)

            # Should not have embedded textures
            assert "textures" not in gltf or len(gltf.get("textures", [])) == 0


class TestMeshScaling:
    def test_mesh_scale_parameters(self, sample_terrain):
        """Test that scale parameters affect mesh output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export with different scales
            path1 = Path(tmpdir) / "terrain1.obj"
            path2 = Path(tmpdir) / "terrain2.obj"

            export_obj_mesh(path1, sample_terrain,
                            scale=10.0, height_scale=2.0)
            export_obj_mesh(path2, sample_terrain,
                            scale=20.0, height_scale=4.0)

            content1 = path1.read_text()
            content2 = path2.read_text()

            # First vertex line (should have different values)
            v1_line = [l for l in content1.split(
                "\n") if l.startswith("v ")][0]
            v2_line = [l for l in content2.split(
                "\n") if l.startswith("v ")][0]

            # Parse vertex coordinates
            v1_coords = [float(x) for x in v1_line.split()[1:]]
            v2_coords = [float(x) for x in v2_line.split()[1:]]

            # X and Z should be roughly 2x for scale=20 vs scale=10
            assert abs(v2_coords[0] / v1_coords[0] -
                       2.0) < 0.01 if v1_coords[0] != 0 else True
            assert abs(v2_coords[2] / v1_coords[2] -
                       2.0) < 0.01 if v1_coords[2] != 0 else True
