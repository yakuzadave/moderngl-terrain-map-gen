import moderngl
import numpy as np
from PIL import Image
from src.generators.erosion import ErosionTerrainGenerator
from src.generators.hydraulic import HydraulicErosionGenerator, HydraulicParams
from src.utils import gl_context


def main():
    # Create context if not exists
    try:
        ctx = gl_context.create_context()
    except ValueError:
        ctx = moderngl.create_standalone_context()

    resolution = 512

    # 1. Generate base terrain (noise only)
    print("Generating base terrain...")
    base_gen = ErosionTerrainGenerator(
        resolution=resolution, use_erosion=False, ctx=ctx)
    base_terrain = base_gen.generate_heightmap(seed=42)

    # Save base
    Image.fromarray((np.clip(base_terrain.height, 0, 1) *
                    255).astype('uint8')).save("test_hydraulic_base.png")

    # 2. Apply hydraulic erosion
    print("Applying hydraulic erosion...")
    hydro_gen = HydraulicErosionGenerator(resolution=resolution, ctx=ctx)

    params = HydraulicParams(
        iterations=100,
        dt=0.002,  # Smaller dt for stability
        sediment_capacity=1.0,
        soil_dissolving=0.5,
        sediment_deposition=0.5,
        evaporation_rate=0.01
    )

    eroded_terrain = hydro_gen.simulate(base_terrain.height, params)

    # Save result
    Image.fromarray((np.clip(eroded_terrain.height, 0, 1) *
                    255).astype('uint8')).save("test_hydraulic_eroded.png")

    # Save difference
    diff = base_terrain.height - eroded_terrain.height
    print(f"Diff range: {diff.min()} to {diff.max()}")

    # Normalize diff for viz
    if diff.max() > diff.min():
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min())
        Image.fromarray((diff_norm * 255).astype('uint8')
                        ).save("test_hydraulic_diff.png")

    # Save sediment mask
    if eroded_terrain.erosion_mask is not None:
        sed = eroded_terrain.erosion_mask
        print(f"Sediment range: {sed.min()} to {sed.max()}")
        sed_norm = (sed - sed.min()) / (sed.max() - sed.min() + 1e-6)
        Image.fromarray((sed_norm * 255).astype('uint8')
                        ).save("test_hydraulic_sediment.png")

    print("Done!")


if __name__ == "__main__":
    main()
