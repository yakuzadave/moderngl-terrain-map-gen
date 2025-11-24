import pytest
import numpy as np
from src.generators.erosion import ErosionTerrainGenerator
from src.generators.hydraulic import HydraulicErosionGenerator, HydraulicParams


def test_hydraulic_erosion_integration(ctx, tmp_path):
    """
    Integration test for hydraulic erosion.
    Verifies that the simulation runs and produces different output than the input.
    """
    resolution = 128

    # 1. Generate base terrain
    with ErosionTerrainGenerator(resolution=resolution, use_erosion=False, ctx=ctx) as base_gen:
        base_terrain = base_gen.generate_heightmap(seed=42)

    # 2. Apply hydraulic erosion
    with HydraulicErosionGenerator(resolution=resolution, ctx=ctx) as hydro_gen:
        params = HydraulicParams(
            iterations=10,  # Low iterations for speed
            dt=0.002,
            sediment_capacity=1.0,
            soil_dissolving=0.5,
            sediment_deposition=0.5,
            evaporation_rate=0.01
        )
        eroded_terrain = hydro_gen.simulate(base_terrain.height, params)

        # Check that erosion actually happened
        diff = base_terrain.height - eroded_terrain.height
        assert np.abs(diff).max(
        ) > 0, "Hydraulic erosion should modify the terrain"

        # Check output shapes
        assert eroded_terrain.height.shape == (resolution, resolution)
        if eroded_terrain.erosion_mask is not None:
            assert eroded_terrain.erosion_mask.shape == (
                resolution, resolution)
