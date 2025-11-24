import pytest
import numpy as np
from src.utils import PRESET_CONFIGS, shade_heightmap
from src.utils.artifacts import TerrainMaps


def test_render_configs(ctx):
    """
    Test that all preset render configurations can be applied without error.
    """
    resolution = 64
    # Create dummy terrain data
    height = np.random.rand(resolution, resolution).astype(np.float32)
    normals = np.random.rand(resolution, resolution, 3).astype(np.float32)
    erosion = np.random.rand(resolution, resolution).astype(np.float32)

    terrain = TerrainMaps(
        height=height,
        normals=normals,
        erosion_mask=erosion
    )

    for config_name, config_func in PRESET_CONFIGS.items():
        config = config_func()

        # Try to render with this config
        try:
            img_array = shade_heightmap(
                terrain,
                azimuth=config.azimuth,
                altitude=config.altitude,
                vert_exag=config.vert_exag,
                colormap=config.colormap,
                blend_mode=config.blend_mode,
            )

            assert img_array is not None
            assert img_array.shape[0] == resolution
            assert img_array.shape[1] == resolution

        except Exception as e:
            pytest.fail(f"Config '{config_name}' failed to render: {e}")
