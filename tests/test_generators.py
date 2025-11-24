import numpy as np
from src.generators.erosion import ErosionTerrainGenerator
from src.generators.hydraulic import HydraulicErosionGenerator, HydraulicParams
from src.generators.morphological import MorphologicalTerrainGPU
from src.utils.artifacts import TerrainMaps


class TestErosionGenerator:
    def test_initialization(self, ctx):
        """Test that the generator initializes correctly."""
        with ErosionTerrainGenerator(resolution=256, ctx=ctx) as gen:
            assert gen.resolution == 256
            assert gen.ctx == ctx

    def test_generation_output_shape(self, ctx):
        """Test that generation produces arrays of correct shape."""
        resolution = 128
        with ErosionTerrainGenerator(resolution=resolution, ctx=ctx) as gen:
            result = gen.generate_heightmap(seed=42)

            assert isinstance(result, TerrainMaps)
            assert result.height.shape == (resolution, resolution)
            assert result.normals.shape == (resolution, resolution, 3)
            assert result.erosion_mask.shape == (resolution, resolution)

            # Check value ranges
            assert result.height.min() >= 0.0
            assert result.height.max() <= 1.0

    def test_seamless_generation(self, ctx):
        """Test seamless generation flag."""
        with ErosionTerrainGenerator(resolution=128, ctx=ctx) as gen:
            result = gen.generate_heightmap(seed=42, seamless=True)

            # Check continuity at edges (basic check)
            # Left edge should match Right edge
            # Top edge should match Bottom edge
            # Note: Due to float precision and interpolation, they might not be exactly equal,
            # but should be very close.

            h = result.height
            diff_x = np.abs(h[:, 0] - h[:, -1]).mean()
            diff_y = np.abs(h[0, :] - h[-1, :]).mean()

            assert diff_x < 0.01, f"Horizontal seam detected: {diff_x}"
            assert diff_y < 0.01, f"Vertical seam detected: {diff_y}"


class TestHydraulicGenerator:
    def test_simulation(self, ctx):
        """Test hydraulic erosion simulation."""
        resolution = 128
        # Create a simple slope
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        initial_height = X.astype(np.float32)  # Slope

        with HydraulicErosionGenerator(resolution=resolution, ctx=ctx) as gen:
            params = HydraulicParams(iterations=10)
            result = gen.simulate(initial_height, params)

            assert isinstance(result, TerrainMaps)
            assert result.height.shape == (resolution, resolution)
            # Height should have changed
            assert not np.array_equal(result.height, initial_height)


class TestMorphologicalGenerator:
    def test_generation(self, ctx):
        """Test morphological generation."""
        resolution = 128
        with MorphologicalTerrainGPU(ctx=ctx) as gen:
            result = gen.generate(resolution=resolution, seed=42)

            assert isinstance(result, TerrainMaps)
            assert result.height.shape == (resolution, resolution)
            assert result.height.dtype == np.float32
