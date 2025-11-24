import numpy as np
from src.generators.erosion import ErosionTerrainGenerator, ErosionParams
from src.generators.hydraulic import HydraulicErosionGenerator, HydraulicParams


def test_raymarch_rendering(ctx):
    """
    Test the raymarch rendering functionality.
    """
    resolution = 128  # Lower resolution for testing

    # Setup generators
    defaults = ErosionParams.mountains()
    with ErosionTerrainGenerator(resolution=resolution, use_erosion=False, defaults=defaults, ctx=ctx) as gen:
        with HydraulicErosionGenerator(resolution=resolution, ctx=ctx) as hydro_gen:

            # Generate base
            base_terrain = gen.generate_heightmap(seed=42)

            # Erode
            params = HydraulicParams(iterations=10)
            eroded_terrain = hydro_gen.simulate(base_terrain.height, params)

            # Pack data (simulating what happens in the script)
            height = eroded_terrain.height.astype('f4')
            normals = eroded_terrain.normals.astype('f4')
            erosion = eroded_terrain.erosion_mask.astype(
                'f4') if eroded_terrain.erosion_mask is not None else np.zeros_like(height)

            data = np.zeros((resolution, resolution, 4), dtype='f4')
            data[:, :, 0] = height
            data[:, :, 1] = normals[:, :, 0]
            data[:, :, 2] = normals[:, :, 2]
            data[:, :, 3] = erosion

            gen.heightmap_texture.write(data.tobytes())

            # Render
            img_array = gen.render_raymarch(
                camera_pos=(0.0, 1.5, -1.5),
                look_at=(0.0, 0.0, 0.0),
                water_height=0.45,
                sun_dir=(-0.5, 0.2, 0.5),
                exposure=1.2,
                fog_density=0.05,
                fog_height=0.4,
                sun_intensity=3.0
            )

            assert img_array is not None
            assert img_array.shape == (resolution, resolution, 3)
            assert img_array.dtype == np.uint8
