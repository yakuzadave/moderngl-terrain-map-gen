"""Test script for enhanced render configurations and quality assessment."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src import ErosionTerrainGenerator, ErosionParams
from src.utils import (
    RenderConfig,
    PRESET_CONFIGS,
    shade_heightmap,
    save_shaded_relief_png,
)


def calculate_image_metrics(img_array: np.ndarray) -> dict[str, float]:
    """Calculate quality metrics for rendered image."""
    # Normalize to 0-1 range
    img = img_array.astype(float) / 255.0

    # Calculate contrast (standard deviation)
    contrast = float(np.std(img))

    # Calculate brightness (mean)
    brightness = float(np.mean(img))

    # Calculate dynamic range
    dynamic_range = float(np.max(img) - np.min(img))

    # Calculate edge strength (approximation using gradient)
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img

    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_strength = float(np.mean(grad_x) + np.mean(grad_y))

    # Calculate color saturation (if RGB)
    saturation = 0.0
    if len(img.shape) == 3 and img.shape[2] >= 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        chroma = max_rgb - min_rgb
        saturation = float(np.mean(chroma))

    return {
        "contrast": contrast,
        "brightness": brightness,
        "dynamic_range": dynamic_range,
        "edge_strength": edge_strength,
        "saturation": saturation,
    }


def test_single_config(
    terrain_data,
    config_name: str,
    config: RenderConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Test a single render configuration and return metrics."""
    print(f"  Testing '{config_name}' configuration...")

    start_time = time.perf_counter()

    # Render using the configuration
    img_array = shade_heightmap(
        terrain_data,
        azimuth=config.azimuth,
        altitude=config.altitude,
        vert_exag=config.vert_exag,
        colormap=config.colormap,
        blend_mode=config.blend_mode,
    )

    render_time = time.perf_counter() - start_time

    # Save the output
    output_path = output_dir / f"render_{config_name}.png"
    save_shaded_relief_png(str(output_path), terrain_data,
                           azimuth=config.azimuth,
                           altitude=config.altitude,
                           vert_exag=config.vert_exag,
                           colormap=config.colormap,
                           blend_mode=config.blend_mode)

    # Calculate metrics
    metrics: dict[str, Any] = dict(calculate_image_metrics(img_array))
    metrics["render_time"] = render_time
    metrics["config_name"] = config_name
    metrics["output_file"] = str(output_path.name)

    # Add configuration details
    metrics["azimuth"] = config.azimuth
    metrics["altitude"] = config.altitude
    metrics["vert_exag"] = config.vert_exag
    metrics["colormap"] = config.colormap
    metrics["blend_mode"] = config.blend_mode

    print(f"    ✓ Rendered in {render_time:.3f}s | "
          f"Contrast: {metrics['contrast']:.3f} | "
          f"Brightness: {metrics['brightness']:.3f}")

    return metrics


def create_comparison_sheet(
    output_dir: Path,
    metrics_list: list[dict[str, Any]],
    grid_cols: int = 4,
) -> None:
    """Create a comparison sheet showing all renders."""
    print("\nCreating comparison sheet...")

    images = []
    labels = []

    for metrics in metrics_list:
        img_path = output_dir / metrics["output_file"]
        if img_path.exists():
            img = Image.open(img_path)
            images.append(img)
            labels.append(metrics["config_name"])

    if not images:
        print("  No images to compare!")
        return

    # Calculate grid dimensions
    n_images = len(images)
    grid_rows = (n_images + grid_cols - 1) // grid_cols

    # Get image dimensions
    img_width, img_height = images[0].size

    # Create composite image
    composite_width = img_width * grid_cols
    composite_height = img_height * grid_rows
    composite = Image.new(
        "RGB", (composite_width, composite_height), (50, 50, 50))

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * img_width
        y = row * img_height
        composite.paste(img, (x, y))

    comparison_path = output_dir / "comparison_grid.png"
    composite.save(comparison_path)
    print(f"  ✓ Saved comparison grid to {comparison_path.name}")


def run_comprehensive_test(
    resolution: int = 512,
    seed: int = 42,
    preset: str = "canyon",
) -> list[dict[str, Any]]:
    """Run comprehensive render configuration tests."""
    print("="*70)
    print("RENDER CONFIGURATION COMPREHENSIVE TEST")
    print("="*70)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "test_render_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Generate terrain
    print(
        f"\nGenerating terrain (resolution={resolution}, seed={seed}, preset={preset})...")

    if preset == "canyon":
        params = ErosionParams.canyon()
    elif preset == "plains":
        params = ErosionParams.plains()
    elif preset == "mountains":
        params = ErosionParams.mountains()
    else:
        params = ErosionParams()

    gen = ErosionTerrainGenerator(resolution=resolution, defaults=params)

    try:
        start = time.perf_counter()
        terrain = gen.generate_heightmap(seed=seed)
        gen_time = time.perf_counter() - start
        print(f"  ✓ Generated in {gen_time:.3f}s")
    finally:
        gen.cleanup()

    # Test all preset configurations
    print(f"\nTesting {len(PRESET_CONFIGS)} render configurations...")
    print("-"*70)

    all_metrics = []

    for config_name, config_func in PRESET_CONFIGS.items():
        config = config_func()
        try:
            metrics = test_single_config(
                terrain, config_name, config, output_dir)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    ✗ Error testing '{config_name}': {e}")

    # Create comparison sheet
    create_comparison_sheet(output_dir, all_metrics)

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if all_metrics:
        avg_render_time = np.mean([m["render_time"] for m in all_metrics])
        print(f"\nAverage render time: {avg_render_time:.3f}s")

        print("\nTop 5 by Contrast:")
        sorted_by_contrast = sorted(
            all_metrics, key=lambda m: m["contrast"], reverse=True)
        for i, m in enumerate(sorted_by_contrast[:5], 1):
            print(f"  {i}. {m['config_name']:20s} - {m['contrast']:.4f}")

        print("\nTop 5 by Dynamic Range:")
        sorted_by_range = sorted(
            all_metrics, key=lambda m: m["dynamic_range"], reverse=True)
        for i, m in enumerate(sorted_by_range[:5], 1):
            print(f"  {i}. {m['config_name']:20s} - {m['dynamic_range']:.4f}")

        print("\nTop 5 by Edge Strength:")
        sorted_by_edges = sorted(
            all_metrics, key=lambda m: m["edge_strength"], reverse=True)
        for i, m in enumerate(sorted_by_edges[:5], 1):
            print(f"  {i}. {m['config_name']:20s} - {m['edge_strength']:.4f}")

        print("\nMost Saturated:")
        sorted_by_sat = sorted(
            all_metrics, key=lambda m: m["saturation"], reverse=True)
        for i, m in enumerate(sorted_by_sat[:5], 1):
            print(f"  {i}. {m['config_name']:20s} - {m['saturation']:.4f}")

    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    print("="*70)

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test render configurations")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Terrain resolution (default: 512)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--preset", type=str, default="canyon",
                        choices=["canyon", "plains", "mountains", "default"],
                        help="Terrain preset (default: canyon)")

    args = parser.parse_args()

    run_comprehensive_test(
        resolution=args.resolution,
        seed=args.seed,
        preset=args.preset,
    )
