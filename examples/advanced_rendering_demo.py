"""
Demonstration of advanced rendering capabilities.

This script shows how to use the new rendering features including:
- Turntable animations
- Multi-angle renders
- Lighting studies
- Comparison grids
"""
from pathlib import Path

from src import ErosionTerrainGenerator, ErosionParams
from src import utils


def demo_turntable():
    """Generate a turntable animation of canyon terrain."""
    print("=" * 70)
    print("Demo 1: Turntable Animation")
    print("=" * 70)

    # Generate canyon terrain
    params = ErosionParams.canyon()
    gen = ErosionTerrainGenerator(resolution=512, defaults=params)
    terrain = gen.generate_heightmap(seed=42)
    gen.cleanup()

    # Save as video (requires ffmpeg) or GIF
    output_path = Path("output/turntable_canyon.gif")
    output_path.parent.mkdir(exist_ok=True)

    saved = utils.save_turntable_video(
        output_path,
        terrain,
        frames=60,
        fps=15,
        altitude=45.0,
        vert_exag=2.5,
        colormap="terrain",
    )

    print(f"✓ Saved turntable animation to: {saved}")
    print()


def demo_multi_angle():
    """Render the same terrain from multiple lighting angles."""
    print("=" * 70)
    print("Demo 2: Multi-Angle Rendering")
    print("=" * 70)

    # Generate mountain terrain
    params = ErosionParams.mountains()
    gen = ErosionTerrainGenerator(resolution=512, defaults=params)
    terrain = gen.generate_heightmap(seed=123)
    gen.cleanup()

    # Define custom angles (azimuth, altitude)
    angles = [
        (315.0, 45.0),  # Northwest (classic shading)
        (135.0, 45.0),  # Southeast (opposite lighting)
        (0.0, 90.0),    # Directly overhead
        (45.0, 15.0),   # Low angle from northeast
    ]

    # Render all angles
    renders = utils.render_multi_angle(
        terrain,
        angles=angles,
        colormap="gist_earth",
        vert_exag=3.0,
    )

    # Save each render
    output_dir = Path("output/multi_angle")
    output_dir.mkdir(parents=True, exist_ok=True)

    angle_names = ["northwest", "southeast", "overhead", "low_angle"]
    for name, render in zip(angle_names, renders):
        from PIL import Image
        img = Image.fromarray(render, mode="RGB")
        img.save(output_dir / f"mountains_{name}.png")
        print(f"✓ Saved: mountains_{name}.png")

    print()


def demo_lighting_study():
    """Create a comprehensive lighting study."""
    print("=" * 70)
    print("Demo 3: Lighting Study")
    print("=" * 70)

    # Generate plains terrain
    params = ErosionParams.plains()
    gen = ErosionTerrainGenerator(resolution=512, defaults=params)
    terrain = gen.generate_heightmap(seed=789)
    gen.cleanup()

    # Create lighting study figure
    fig = utils.render_lighting_study(
        terrain,
        azimuth_steps=8,
        altitude_steps=3,
        colormap="terrain",
        vert_exag=2.0,
    )

    output_path = Path("output/lighting_study.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"✓ Saved lighting study to: {output_path}")
    print()


def demo_comparison_grid():
    """Compare different terrain presets side by side."""
    print("=" * 70)
    print("Demo 4: Terrain Comparison Grid")
    print("=" * 70)

    # Generate multiple terrains with different presets
    presets = [
        ("Default", ErosionParams()),
        ("Canyon", ErosionParams.canyon()),
        ("Plains", ErosionParams.plains()),
        ("Mountains", ErosionParams.mountains()),
    ]

    terrains = []
    labels = []
    seed = 42

    for name, params in presets:
        gen = ErosionTerrainGenerator(resolution=512, defaults=params)
        terrain = gen.generate_heightmap(seed=seed)
        gen.cleanup()

        terrains.append(terrain)
        labels.append(name)
        print(f"  Generated: {name}")

    # Create comparison grid
    fig = utils.create_comparison_grid(
        terrains,
        labels=labels,
        figsize=(12, 10),
        dpi=120,
        azimuth=315.0,
        altitude=45.0,
        colormap="terrain",
    )

    output_path = Path("output/preset_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")

    print(f"✓ Saved comparison grid to: {output_path}")
    print()


def demo_custom_animation():
    """Create a custom animation sequence with changing parameters."""
    print("=" * 70)
    print("Demo 5: Custom Animation Sequence")
    print("=" * 70)

    # Generate base terrain
    gen = ErosionTerrainGenerator(resolution=512)
    terrain = gen.generate_heightmap(seed=999)
    gen.cleanup()

    # Define custom sequence function
    def zoom_and_rotate(frame: int, total: int):
        """Combine rotation and vertical exaggeration change."""
        from matplotlib.colors import LightSource
        from matplotlib import cm
        import numpy as np

        # Rotate light
        azimuth = (frame * 360.0) / total

        # Vary vertical exaggeration
        vert_exag = 1.0 + 2.0 * np.sin(frame * 2 * np.pi / total)

        height = terrain.height
        dx = 1.0 / height.shape[1]
        dy = 1.0 / height.shape[0]

        ls = LightSource(azdeg=azimuth, altdeg=45.0)
        rgb = ls.shade(
            height,
            cmap=cm.get_cmap("terrain"),
            vert_exag=vert_exag,
            blend_mode="soft",
            dx=dx,
            dy=dy,
        )

        return (np.clip(rgb * 255.0, 0.0, 255.0)).astype("uint8")

    output_dir = Path("output/custom_sequence")

    saved_frames = utils.save_animation_sequence(
        output_dir,
        terrain,
        sequence_func=zoom_and_rotate,
        frame_count=30,
        name_pattern="frame_{:04d}.png",
        progress_callback=lambda cur, total: print(
            f"  Frame {cur}/{total}", end="\r"
        ),
    )

    print(f"\n✓ Saved {len(saved_frames)} frames to: {output_dir}")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Advanced Rendering Demos" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        demo_turntable()
        demo_multi_angle()
        demo_lighting_study()
        demo_comparison_grid()
        demo_custom_animation()

        print("=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nCheck the 'output/' directory for generated files.")

    except Exception as e:
        print(f"\n❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
