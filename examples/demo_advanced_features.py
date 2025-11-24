"""Demo script showcasing advanced rendering and post-processing."""
from __future__ import annotations
from src.utils.postprocessing import (
    apply_tonemapping,
    apply_color_grading,
    apply_bloom_effect,
    apply_sharpening,
    apply_atmospheric_perspective,
)
from src.utils import shade_heightmap, RenderConfig
from src import ErosionTerrainGenerator, ErosionParams
from PIL import Image
import numpy as np

import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))


def demo_postprocessing_effects():
    """Demonstrate various post-processing effects."""
    print("="*70)
    print("ADVANCED POST-PROCESSING DEMO")
    print("="*70)

    output_dir = Path("postprocessing_demo_output")
    output_dir.mkdir(exist_ok=True)

    # Generate terrain
    print("\n1. Generating terrain...")
    gen = ErosionTerrainGenerator(
        resolution=512, defaults=ErosionParams.mountains())
    try:
        terrain = gen.generate_heightmap(seed=12345)
    finally:
        gen.cleanup()

    # Base render
    print("\n2. Creating base render...")
    config = RenderConfig.dramatic()
    base_img = shade_heightmap(
        terrain,
        azimuth=config.azimuth,
        altitude=config.altitude,
        vert_exag=config.vert_exag,
        colormap=config.colormap,
        blend_mode=config.blend_mode,
    ).astype(float) / 255.0

    # Save base
    Image.fromarray((base_img * 255).astype(np.uint8)).save(
        output_dir / "01_base.png"
    )
    print("   ✓ Saved: 01_base.png")

    # Apply tonemapping variations
    print("\n3. Applying tonemapping variations...")
    for method in ["reinhard", "filmic", "aces", "uncharted2"]:
        tonemap_img = apply_tonemapping(base_img, method=method, exposure=1.2)
        Image.fromarray((tonemap_img * 255).astype(np.uint8)).save(
            output_dir / f"02_tonemap_{method}.png"
        )
        print(f"   ✓ Saved: 02_tonemap_{method}.png")

    # Apply color grading
    print("\n4. Applying color grading...")

    # Warm sunset
    warm_img = apply_color_grading(
        base_img,
        temperature=0.3,
        tint=-0.1,
        saturation=1.2,
        contrast=1.1,
    )
    Image.fromarray((warm_img * 255).astype(np.uint8)).save(
        output_dir / "03_grading_warm_sunset.png"
    )
    print("   ✓ Saved: 03_grading_warm_sunset.png")

    # Cool morning
    cool_img = apply_color_grading(
        base_img,
        temperature=-0.2,
        tint=0.05,
        saturation=0.9,
        contrast=1.05,
        brightness=0.05,
    )
    Image.fromarray((cool_img * 255).astype(np.uint8)).save(
        output_dir / "03_grading_cool_morning.png"
    )
    print("   ✓ Saved: 03_grading_cool_morning.png")

    # High contrast
    contrast_img = apply_color_grading(
        base_img,
        saturation=0.0,
        contrast=1.5,
        gamma=0.9,
    )
    Image.fromarray((contrast_img * 255).astype(np.uint8)).save(
        output_dir / "03_grading_high_contrast.png"
    )
    print("   ✓ Saved: 03_grading_high_contrast.png")

    # Apply bloom
    print("\n5. Applying bloom effect...")
    bloom_img = apply_bloom_effect(
        base_img,
        threshold=0.7,
        intensity=0.5,
        blur_radius=15.0,
    )
    Image.fromarray((bloom_img * 255).astype(np.uint8)).save(
        output_dir / "04_bloom.png"
    )
    print("   ✓ Saved: 04_bloom.png")

    # Apply sharpening
    print("\n6. Applying sharpening...")
    sharp_img = apply_sharpening(base_img, amount=0.8, method="unsharp")
    Image.fromarray((sharp_img * 255).astype(np.uint8)).save(
        output_dir / "05_sharpened.png"
    )
    print("   ✓ Saved: 05_sharpened.png")

    # Apply atmospheric perspective
    print("\n7. Applying atmospheric perspective...")
    height_data = terrain.height

    atmos_img = apply_atmospheric_perspective(
        base_img,
        height_data,
        fog_color=(0.75, 0.82, 0.9),
        fog_density=0.4,
        fog_height_falloff=3.0,
    )
    Image.fromarray((atmos_img * 255).astype(np.uint8)).save(
        output_dir / "06_atmospheric.png"
    )
    print("   ✓ Saved: 06_atmospheric.png")

    # Combined effects
    print("\n8. Creating combined effect renders...")

    # "Cinematic" look
    cinematic = base_img.copy()
    cinematic = apply_tonemapping(cinematic, method="filmic", exposure=1.1)
    cinematic = apply_color_grading(
        cinematic,
        temperature=0.15,
        saturation=1.1,
        contrast=1.2,
        gamma=0.95,
    )
    cinematic = apply_bloom_effect(cinematic, threshold=0.75, intensity=0.3)
    cinematic = apply_sharpening(cinematic, amount=0.5)
    Image.fromarray((cinematic * 255).astype(np.uint8)).save(
        output_dir / "07_combined_cinematic.png"
    )
    print("   ✓ Saved: 07_combined_cinematic.png")

    # "Ethereal" look
    ethereal = base_img.copy()
    ethereal = apply_atmospheric_perspective(
        ethereal, height_data,
        fog_color=(0.8, 0.85, 0.95),
        fog_density=0.5,
        fog_height_falloff=2.5,
    )
    ethereal = apply_color_grading(
        ethereal,
        temperature=-0.1,
        tint=0.1,
        saturation=0.8,
        brightness=0.05,
    )
    ethereal = apply_bloom_effect(ethereal, threshold=0.8, intensity=0.6)
    Image.fromarray((ethereal * 255).astype(np.uint8)).save(
        output_dir / "07_combined_ethereal.png"
    )
    print("   ✓ Saved: 07_combined_ethereal.png")

    # "Dramatic noir" look
    noir = base_img.copy()
    noir = apply_color_grading(
        noir,
        saturation=0.0,
        contrast=1.6,
        brightness=-0.05,
        gamma=0.85,
    )
    noir = apply_sharpening(noir, amount=1.0)
    Image.fromarray((noir * 255).astype(np.uint8)).save(
        output_dir / "07_combined_noir.png"
    )
    print("   ✓ Saved: 07_combined_noir.png")

    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    print("="*70)


def demo_multi_lighting_comparison():
    """Create comparison of different lighting setups."""
    print("\n" + "="*70)
    print("MULTI-LIGHTING COMPARISON DEMO")
    print("="*70)

    output_dir = Path("lighting_comparison_output")
    output_dir.mkdir(exist_ok=True)

    print("\n1. Generating terrain...")
    gen = ErosionTerrainGenerator(
        resolution=512, defaults=ErosionParams.canyon())
    try:
        terrain = gen.generate_heightmap(seed=99999)
    finally:
        gen.cleanup()

    print("\n2. Rendering with various lighting conditions...")

    lighting_scenarios = [
        ("dawn", 90.0, 5.0, "Dawn - Low angle from east"),
        ("morning", 120.0, 25.0, "Morning - East-southeast"),
        ("noon", 180.0, 85.0, "Noon - Overhead"),
        ("afternoon", 240.0, 35.0, "Afternoon - Southwest"),
        ("sunset", 270.0, 8.0, "Sunset - Low angle from west"),
        ("dusk", 300.0, -2.0, "Dusk - Below horizon (rim lighting)"),
    ]

    for name, azimuth, altitude, desc in lighting_scenarios:
        img = shade_heightmap(
            terrain,
            azimuth=azimuth,
            altitude=max(1.0, altitude),  # Clamp to positive
            vert_exag=2.5,
            colormap="terrain",
            blend_mode="overlay",
        ).astype(float) / 255.0

        # Apply cinematic grading
        img = apply_tonemapping(img, method="filmic", exposure=1.1)
        img = apply_color_grading(img, contrast=1.15, saturation=1.1)

        Image.fromarray((img * 255).astype(np.uint8)).save(
            output_dir / f"lighting_{name}.png"
        )
        print(f"   ✓ Saved: lighting_{name}.png - {desc}")

    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    start_time = time.perf_counter()

    demo_postprocessing_effects()
    demo_multi_lighting_comparison()

    elapsed = time.perf_counter() - start_time
    print(f"\n✓ Total demo time: {elapsed:.2f}s")
