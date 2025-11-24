import numpy as np
from PIL import Image
from src.generators.erosion import ErosionTerrainGenerator, ErosionParams
from src.generators.hydraulic import HydraulicErosionGenerator, HydraulicParams
from src.utils import gl_context
import moderngl


def test_raymarch():
    print("Initializing context...")
    try:
        ctx = gl_context.create_context()
    except ValueError:
        ctx = moderngl.create_standalone_context()

    resolution = 1024

    print("Initializing generators...")
    # Use mountains preset for base terrain to get more variation
    defaults = ErosionParams.mountains()
    # Increase amplitude and gain for more dramatic terrain
    defaults.height_amp = 1.2  # Much higher amplitude
    defaults.height_gain = 0.5
    defaults.height_octaves = 8

    gen = ErosionTerrainGenerator(
        resolution=resolution, use_erosion=False, defaults=defaults, ctx=ctx)
    hydro_gen = HydraulicErosionGenerator(resolution=resolution, ctx=ctx)

    print("Generating base terrain...")
    gen.generate_heightmap(seed=42)

    # Read back the base heightmap to pass to hydraulic sim
    # We can read it from the texture or just use the returned object
    # generate_heightmap returns a TerrainMaps object
    base_terrain = gen.generate_heightmap(seed=42)
    print(
        f"Base height stats: min={base_terrain.height.min():.4f}, max={base_terrain.height.max():.4f}")

    print("Applying hydraulic erosion...")
    params = HydraulicParams(
        iterations=100,
        dt=0.002,
        sediment_capacity=1.0,
        soil_dissolving=0.5,
        sediment_deposition=0.5,
        evaporation_rate=0.01
    )
    eroded_terrain = hydro_gen.simulate(base_terrain.height, params)

    print("Packing data for renderer...")
    # Pack into (Height, NormalX, NormalZ, Erosion)
    height = eroded_terrain.height.astype('f4')
    normals = eroded_terrain.normals.astype('f4')
    erosion = eroded_terrain.erosion_mask.astype(
        'f4') if eroded_terrain.erosion_mask is not None else np.zeros_like(height)

    data = np.zeros((resolution, resolution, 4), dtype='f4')
    data[:, :, 0] = height
    data[:, :, 1] = normals[:, :, 0]  # nx
    data[:, :, 2] = normals[:, :, 2]  # nz
    data[:, :, 3] = erosion

    # Upload to texture
    gen.heightmap_texture.write(data.tobytes())

    print(
        f"Height stats: min={height.min():.4f}, max={height.max():.4f}, mean={height.mean():.4f}")
    print(f"Erosion stats: min={erosion.min():.4f}, max={erosion.max():.4f}")

    print("Rendering raymarch view...")
    # Render with new atmospheric settings
    # Moved camera further back and up to see more context
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

    print("Saving output...")
    Image.fromarray(img_array).save("test_raymarch_improved.png")

    gen.cleanup()
    hydro_gen.cleanup()
    print("Done! Saved to test_raymarch_improved.png")


if __name__ == "__main__":
    test_raymarch()
