"""Visual inspection and quality analysis of rendered terrains."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def detect_render_issues(img_path: Path) -> dict[str, Any]:
    """Detect potential rendering issues in an image."""
    img = Image.open(img_path)
    img_array = np.array(img).astype(float) / 255.0

    issues = []

    # Check for clipping (overexposure/underexposure)
    white_pixels = np.sum(np.all(img_array >= 0.98, axis=2))
    black_pixels = np.sum(np.all(img_array <= 0.02, axis=2))
    total_pixels = img_array.shape[0] * img_array.shape[1]

    white_percent = (white_pixels / total_pixels) * 100
    black_percent = (black_pixels / total_pixels) * 100

    if white_percent > 5.0:
        issues.append(f"Overexposure: {white_percent:.1f}% white pixels")
    if black_percent > 5.0:
        issues.append(f"Underexposure: {black_percent:.1f}% black pixels")

    # Check for low contrast
    contrast = np.std(img_array)
    if contrast < 0.15:
        issues.append(f"Low contrast: {contrast:.3f}")

    # Check for color banding (limited dynamic range usage)
    if len(img_array.shape) == 3:
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        max_possible = 256 ** 3
        usage_percent = (unique_colors / min(total_pixels, max_possible)) * 100

        if usage_percent < 1.0:
            issues.append(
                f"Color banding detected: only {unique_colors} unique colors")

    # Check brightness distribution
    brightness = np.mean(img_array)
    if brightness < 0.3:
        issues.append(f"Very dark overall: {brightness:.2f}")
    elif brightness > 0.8:
        issues.append(f"Very bright overall: {brightness:.2f}")

    # Check for lack of detail (uniform regions)
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_strength = np.mean(grad_x) + np.mean(grad_y)

    if edge_strength < 0.02:
        issues.append(f"Very low detail: edge strength {edge_strength:.4f}")

    return {
        "white_percent": float(white_percent),
        "black_percent": float(black_percent),
        "contrast": float(contrast),
        "brightness": float(brightness),
        "edge_strength": float(edge_strength),
        "issues": issues,
        "quality_score": calculate_quality_score(
            float(contrast),
            float(brightness),
            float(white_percent),
            float(black_percent),
            float(edge_strength),
        ),
    }


def calculate_quality_score(
    contrast: float,
    brightness: float,
    white_percent: float,
    black_percent: float,
    edge_strength: float,
) -> float:
    """Calculate an overall quality score (0-100)."""
    score = 100.0

    # Penalize clipping
    score -= white_percent * 2.0
    score -= black_percent * 2.0

    # Reward good contrast (optimal around 0.3-0.4)
    if contrast < 0.2:
        score -= (0.2 - contrast) * 100
    elif contrast > 0.5:
        score -= (contrast - 0.5) * 50

    # Penalize extreme brightness
    if brightness < 0.3:
        score -= (0.3 - brightness) * 50
    elif brightness > 0.8:
        score -= (brightness - 0.8) * 50

    # Reward detail
    score += edge_strength * 100

    return max(0.0, min(100.0, score))


def analyze_all_renders(output_dir: Path) -> dict[str, Any]:
    """Analyze all renders and generate report."""
    results = {}

    for img_path in sorted(output_dir.glob("render_*.png")):
        config_name = img_path.stem.replace("render_", "")
        print(f"Analyzing {config_name}...")

        analysis = detect_render_issues(img_path)
        results[config_name] = analysis

        if analysis["issues"]:
            print("  ⚠ Issues found:")
            for issue in analysis["issues"]:
                print(f"    - {issue}")
        else:
            print("  ✓ No major issues detected")

        print(f"  Quality Score: {analysis['quality_score']:.1f}/100")

    return results


def generate_recommendations(analysis_results: dict[str, Any]) -> list[str]:
    """Generate rendering improvement recommendations."""
    recommendations = []

    # Find common issues
    overexposed_count = sum(1 for r in analysis_results.values()
                            if r["white_percent"] > 5.0)
    underexposed_count = sum(1 for r in analysis_results.values()
                             if r["black_percent"] > 5.0)
    low_contrast_count = sum(1 for r in analysis_results.values()
                             if r["contrast"] < 0.15)

    if overexposed_count > 3:
        recommendations.append(
            "Multiple configs show overexposure. Consider reducing sun_intensity "
            "or adding exposure compensation in post-processing."
        )

    if underexposed_count > 3:
        recommendations.append(
            "Multiple configs show underexposure. Consider increasing ambient_strength "
            "or adjusting altitude angle for better lighting."
        )

    if low_contrast_count > 3:
        recommendations.append(
            "Multiple configs show low contrast. Consider increasing vert_exag "
            "or using overlay blend mode for more dramatic relief."
        )

    # Check if any config performs exceptionally well
    best_config = max(analysis_results.items(),
                      key=lambda x: x[1]["quality_score"])

    if best_config[1]["quality_score"] > 80:
        recommendations.append(
            f"Config '{best_config[0]}' achieved quality score {best_config[1]['quality_score']:.1f}. "
            f"Consider using its parameters as a baseline for similar renders."
        )

    return recommendations


def main():
    """Run visual inspection analysis."""
    output_dir = Path(__file__).parent.parent / "output" / "test_render_output"

    if not output_dir.exists():
        print(f"Error: {output_dir} directory not found!")
        print("Run test_render_configs.py first to generate test renders.")
        return

    print("="*70)
    print("VISUAL QUALITY ANALYSIS")
    print("="*70)
    print()

    analysis_results = analyze_all_renders(output_dir)

    print("\n" + "="*70)
    print("QUALITY RANKINGS")
    print("="*70)

    sorted_results = sorted(
        analysis_results.items(),
        key=lambda x: x[1]["quality_score"],
        reverse=True
    )

    print("\nTop 10 by Quality Score:")
    for i, (config_name, result) in enumerate(sorted_results[:10], 1):
        print(
            f"  {i:2d}. {config_name:20s} - {result['quality_score']:5.1f}/100")

    print("\nBottom 5 by Quality Score:")
    for i, (config_name, result) in enumerate(sorted_results[-5:], 1):
        rank = len(sorted_results) - 5 + i
        print(
            f"  {rank:2d}. {config_name:20s} - {result['quality_score']:5.1f}/100")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    recommendations = generate_recommendations(analysis_results)

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n✓ All renders look good! No major improvements needed.")

    # Save detailed analysis
    analysis_path = output_dir / "quality_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✓ Detailed analysis saved to {analysis_path}")
    print("="*70)


if __name__ == "__main__":
    main()
