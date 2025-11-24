"""Command-line entry point for GPU terrain helpers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src import (
    ErosionTerrainGenerator,
    ErosionParams,
    MorphologicalTerrainGPU,
    HydraulicErosionGenerator,
    HydraulicParams,
    TerrainMaps,
    utils,
)


def _vec3(arg: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated floats, e.g. 0.6,0.7,0.8")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate procedural terrains with ModernGL")
    parser.add_argument(
        "--generator", choices=["erosion", "morph", "hydraulic"], default="erosion")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seamless", action="store_true",
                        help="Force seamless tiling (rounds frequency to integer)")
    parser.add_argument("--preset", choices=["default", "canyon", "plains", "mountains", "natural"], default="default",
                        help="Terrain preset (only for erosion generator)")
    parser.add_argument("--disable-erosion", action="store_true",
                        help="Turn off erosion layers in the erosion shader")
    parser.add_argument(
        "--render", choices=["none", "viz", "ray", "shade"], default="viz")
    parser.add_argument("--mode", type=int, default=0,
                        help="Visualization mode for --render=viz (0-4)")
    parser.add_argument("--water-height", type=float, default=0.45)
    parser.add_argument("--heightmap-out", type=Path, default=None)
    parser.add_argument("--raw-out", type=Path, default=None,
                        help="Save raw 16-bit little-endian heightmap")
    parser.add_argument("--r32-out", type=Path, default=None,
                        help="Save raw 32-bit float little-endian heightmap")
    parser.add_argument("--normals-out", type=Path, default=None)
    parser.add_argument("--erosion-out", type=Path, default=None)
    parser.add_argument("--bundle-out", type=Path, default=None)
    parser.add_argument("--panel-out", type=Path, default=None,
                        help="Save Matplotlib overview figure")
    parser.add_argument("--turntable-out", type=Path,
                        default=None, help="Save animated GIF turntable")
    parser.add_argument("--turntable-frames", type=int, default=90)
    parser.add_argument("--turntable-interval", type=int, default=40)
    parser.add_argument("--render-out", type=Path, default=None)
    parser.add_argument("--obj-out", type=Path, default=None)
    parser.add_argument("--stl-out", type=Path, default=None,
                        help="Save binary STL mesh")
    parser.add_argument("--shaded-out", type=Path, default=None,
                        help="Save CPU shaded-relief preview")
    parser.add_argument("--shade-azimuth", type=float, default=315.0)
    parser.add_argument("--shade-altitude", type=float, default=45.0)
    parser.add_argument("--shade-vert-exag", type=float, default=2.0)
    parser.add_argument("--shade-colormap", type=str, default="terrain")
    parser.add_argument("--slope-out", type=Path, default=None,
                        help="Save grayscale slope intensity map")
    parser.add_argument("--slope-scale", type=float, default=1.0)
    parser.add_argument("--slope-gamma", type=float, default=0.8)
    parser.add_argument("--warp-strength", type=float, default=None,
                        help="Domain warping strength (overrides preset default)")
    parser.add_argument("--ridge-noise", action="store_true",
                        help="Enable ridge noise for sharp peaks")

    # Thermal Erosion options
    parser.add_argument("--thermal-iterations", type=int, default=0,
                        help="Number of thermal erosion iterations")
    parser.add_argument("--thermal-threshold", type=float, default=0.001,
                        help="Height difference threshold for thermal erosion")
    parser.add_argument("--thermal-strength", type=float, default=0.5,
                        help="Strength of thermal erosion material transfer")

    # Hydraulic Erosion options (for --generator=hydraulic)
    parser.add_argument("--hydro-iterations", type=int, default=100,
                        help="Number of hydraulic erosion iterations")
    parser.add_argument("--hydro-dt", type=float, default=0.002,
                        help="Time step for hydraulic erosion")

    # Chained generation (context-aware)
    parser.add_argument("--chain-count", type=int, default=0,
                        help="Generate N terrains in sequence, adapting params from previous run")
    parser.add_argument("--chain-out", type=Path, default=Path("chain_output"),
                        help="Output directory for chained generation artifacts")
    parser.add_argument("--chain-prefix", type=str, default="chain",
                        help="Filename prefix for chained outputs")
    parser.add_argument("--chain-adapt", action="store_true",
                        help="Adapt erosion/height params based on previous terrain roughness")
    parser.add_argument("--preset-file", type=Path, default=None,
                        help="Load generator parameters from JSON file")
    parser.add_argument("--save-preset", type=Path, default=None,
                        help="Save effective generator parameters to JSON")
    parser.add_argument("--hdr-out", type=Path, default=None,
                        help="Save HDR (EXR) copy of render (shade/ray modes only)")
    parser.add_argument("--compare-seeds", type=str, default="",
                        help="Comma-separated seeds for comparison grid render (GPU viz/ray)")
    parser.add_argument("--compare-out", type=Path, default=None,
                        help="Output path for comparison grid PNG")
    parser.add_argument("--ray-exposure", type=float, default=1.0,
                        help="Exposure multiplier for raymarch render")
    parser.add_argument("--ray-fog-density", type=float, default=0.1,
                        help="Fog density for raymarch render")
    parser.add_argument("--ray-fog-height", type=float, default=0.35,
                        help="Fog height falloff (world units) for raymarch render")
    parser.add_argument("--ray-fog-color", type=_vec3, default=(0.6, 0.7, 0.8),
                        help="Fog color as comma-separated floats (r,g,b)")
    parser.add_argument("--ray-sun-intensity", type=float, default=2.0,
                        help="Direct sun intensity multiplier for raymarch render")
    parser.add_argument("--ray-max-steps", type=int, default=128,
                        help="Max raymarch steps (higher = better quality, slower)")
    parser.add_argument("--ray-min-step", type=float, default=0.002,
                        help="Minimum raymarch step size")
    parser.add_argument("--ray-shadow-softness", type=float, default=16.0,
                        help="Softness factor for raytraced shadows (higher = sharper)")
    parser.add_argument("--ray-ao-strength", type=float, default=1.0,
                        help="Strength of ambient occlusion (0.0 to 1.0+)")

    # Advanced rendering options
    parser.add_argument("--multi-angle-out", type=Path, default=None,
                        help="Save multi-angle renders to directory")
    parser.add_argument("--lighting-study-out", type=Path, default=None,
                        help="Save comprehensive lighting study figure")
    parser.add_argument("--turntable-video-out", type=Path, default=None,
                        help="Save turntable as MP4/GIF (requires ffmpeg for MP4)")
    parser.add_argument("--turntable-fps", type=int, default=12,
                        help="Frames per second for turntable video")

    # Texture exports
    parser.add_argument("--splatmap-out", type=Path, default=None,
                        help="Save RGBA splatmap for texture blending")
    parser.add_argument("--ao-out", type=Path, default=None,
                        help="Save ambient occlusion approximation")
    parser.add_argument("--curvature-out", type=Path, default=None,
                        help="Save curvature map (convex/concave)")
    parser.add_argument("--packed-out", type=Path, default=None,
                        help="Save packed texture (use --pack-mode)")
    parser.add_argument("--pack-mode", choices=["unity_mask", "ue_orm", "height_normal_ao"],
                        default="unity_mask", help="Packed texture channel layout")
    parser.add_argument("--scatter-out", type=Path, default=None,
                        help="Save scatter density map (R=Trees, G=Rocks, B=Grass)")

    # Batch generation
    parser.add_argument("--batch-count", type=int, default=0,
                        help="Generate multiple terrains with sequential seeds")
    parser.add_argument("--batch-dir", type=Path, default=Path("batch_output"),
                        help="Output directory for batch generation")
    parser.add_argument("--batch-formats", type=str, default="png,shaded",
                        help="Comma-separated export formats for batch (png,obj,stl,npz,shaded)")
    return parser.parse_args()


def _ensure_image(path: Path, data: np.ndarray) -> None:
    if data.dtype == np.uint16:
        img = Image.fromarray(data, mode="I;16")
    else:
        img = Image.fromarray(data)
    img.save(path)


def _terrain_stats(terrain: TerrainMaps) -> dict[str, float]:
    """Compute simple stats to drive adaptive chaining."""
    maps = TerrainMaps.ensure(terrain)
    height = maps.height
    gy, gx = np.gradient(height)
    slope = np.sqrt(gx**2 + gy**2)
    return {
        "height_mean": float(height.mean()),
        "height_std": float(height.std()),
        "slope_mean": float(slope.mean()),
        "slope_std": float(slope.std()),
    }


def _adapt_overrides(prev_overrides: dict[str, float], stats: dict[str, float]) -> dict[str, float]:
    """Adjust a small set of parameters based on previous terrain roughness."""
    overrides = dict(prev_overrides)
    target_slope = 0.12
    slope = stats["slope_mean"]

    # Nudge height amplitude and erosion strength toward target slope.
    if slope < target_slope * 0.85:
        overrides["height_amp"] = overrides.get("height_amp", 0.25) * 1.05
        overrides["erosion_strength"] = overrides.get(
            "erosion_strength", 0.04) * 1.08
    elif slope > target_slope * 1.15:
        overrides["height_amp"] = overrides.get("height_amp", 0.25) * 0.95
        overrides["erosion_strength"] = overrides.get(
            "erosion_strength", 0.04) * 0.92

    # Keep values in reasonable ranges.
    overrides["height_amp"] = float(
        np.clip(overrides["height_amp"], 0.05, 0.6))
    overrides["erosion_strength"] = float(
        np.clip(overrides["erosion_strength"], 0.01, 0.12))
    return overrides


def _load_preset(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            "Preset JSON must be an object of parameter keys/values.")
    result: dict[str, float] = {}
    for k, v in data.items():
        try:
            result[str(k)] = float(v)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid value for '{k}': {v}") from exc
    return result


def _save_preset(path: Path, params: ErosionParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.uniforms(), indent=2))


def _save_hdr_image(path: Path, data: np.ndarray) -> None:
    """Save float HDR output; prefer EXR via imageio, fallback to NPY."""
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        iio = None  # type: ignore

    arr = data.astype(np.float32)
    if arr.max() > 1.5:  # assume 0-255 input, de-gamma
        arr = np.clip(arr / 255.0, 0.0, 1.0)
        arr = np.power(arr, 2.2)
    path.parent.mkdir(parents=True, exist_ok=True)

    if iio:
        for kwargs in (
            {"format_hint": ".exr"},
            {"extension": ".exr"},
            {},
        ):
            try:
                iio.imwrite(path, arr, **kwargs)
                print(f"✓ Saved HDR to {path}")
                return
            except Exception:
                continue

    # Fallback: save as NPY to preserve float data
    npy_path = path.with_suffix(".npy")
    np.save(npy_path, arr)
    print(f"⚠ HDR EXR export unavailable; saved float data to {npy_path}")


def main() -> None:
    """
    Main entry point for the GPU terrain generation CLI.
    Handles argument parsing, generator initialization, execution, and artifact export.
    """
    args = _parse_args()
    preset_overrides: dict[str, float] = {}
    if args.preset_file:
        preset_overrides = _load_preset(args.preset_file)

    if args.warp_strength is not None:
        preset_overrides["warp_strength"] = args.warp_strength
    
    if args.ridge_noise:
        preset_overrides["ridge_noise"] = 1

    if args.thermal_iterations > 0:
        preset_overrides["thermal_iterations"] = args.thermal_iterations
        preset_overrides["thermal_threshold"] = args.thermal_threshold
        preset_overrides["thermal_strength"] = args.thermal_strength

    # Chained generation mode (adaptive multi-step)
    if args.chain_count > 0:
        if args.generator != "erosion":
            raise SystemExit(
                "Chained generation currently supports only the erosion generator.")

        args.chain_out.mkdir(parents=True, exist_ok=True)
        print(
            f"Chained generation: {args.chain_count} terrains → {args.chain_out}")

        gen_kwargs: dict[str, float | int] = {}
        if args.preset == "canyon":
            defaults = ErosionParams.canyon()
            gen_kwargs = defaults.uniforms()
        elif args.preset == "plains":
            defaults = ErosionParams.plains()
            gen_kwargs = defaults.uniforms()
        elif args.preset == "mountains":
            defaults = ErosionParams.mountains()
            gen_kwargs = defaults.uniforms()
        elif args.preset == "natural":
            defaults = ErosionParams.natural()
            gen_kwargs = defaults.uniforms()
        else:
            defaults = ErosionParams()

        if preset_overrides:
            defaults = defaults.override(preset_overrides)
            gen_kwargs = defaults.uniforms()
        if args.save_preset:
            _save_preset(args.save_preset, defaults)

        overrides: dict[str, float] = {}
        seed = args.seed
        erosion_gen = ErosionTerrainGenerator(
            resolution=args.resolution,
            use_erosion=not args.disable_erosion,
            defaults=defaults,
        )
        try:
            for idx in range(args.chain_count):
                terrain = erosion_gen.generate_heightmap(
                    seed=seed,
                    seamless=args.seamless,
                    **overrides,
                )
                stats = _terrain_stats(terrain)

                # Save artifacts per step
                prefix = f"{args.chain_prefix}_{idx:03d}"
                _ensure_image(args.chain_out /
                              f"{prefix}_height.png", terrain.height_u16())
                _ensure_image(args.chain_out / f"{prefix}_viz.png", erosion_gen.render_visualization(
                    water_height=args.water_height, mode=args.mode))
                _ensure_image(args.chain_out / f"{prefix}_ray.png", erosion_gen.render_raymarch(
                    water_height=args.water_height,
                    exposure=args.ray_exposure,
                    fog_color=args.ray_fog_color,
                    fog_density=args.ray_fog_density,
                    fog_height=args.ray_fog_height,
                    sun_intensity=args.ray_sun_intensity,
                    ray_max_steps=args.ray_max_steps,
                    ray_min_step=args.ray_min_step,
                    shadow_softness=args.ray_shadow_softness,
                    ao_strength=args.ray_ao_strength,
                ))

                print(
                    f"[{idx+1}/{args.chain_count}] seed={seed} slope_mean={stats['slope_mean']:.4f}")

                if args.chain_adapt:
                    overrides = _adapt_overrides(overrides, stats)
                seed += 1
        finally:
            erosion_gen.cleanup()
        return

    # Batch generation mode
    if args.batch_count > 0:
        formats = [f.strip() for f in args.batch_formats.split(",")]
        print(f"Batch generating {args.batch_count} terrains...")

        gen_kwargs = {}
        if args.generator == "erosion":
            if args.preset == "canyon":
                defaults = ErosionParams.canyon()
                gen_kwargs = defaults.uniforms()
            elif args.preset == "plains":
                defaults = ErosionParams.plains()
                gen_kwargs = defaults.uniforms()
            elif args.preset == "mountains":
                defaults = ErosionParams.mountains()
                gen_kwargs = defaults.uniforms()
            elif args.preset == "natural":
                defaults = ErosionParams.natural()
                gen_kwargs = defaults.uniforms()
            else:
                defaults = ErosionParams()
                gen_kwargs = defaults.uniforms()
            if preset_overrides:
                defaults = defaults.override(preset_overrides)
                gen_kwargs = defaults.uniforms()

        utils.generate_terrain_set(
            count=args.batch_count,
            base_seed=args.seed,
            generator=args.generator,
            resolution=args.resolution,
            output_dir=args.batch_dir,
            formats=formats,
            **gen_kwargs,
        )
        return

    erosion_gen: ErosionTerrainGenerator | None = None
    morph_gen: MorphologicalTerrainGPU | None = None
    hydro_gen: HydraulicErosionGenerator | None = None

    try:
        if args.generator == "erosion":
            # Select preset
            if args.preset == "canyon":
                defaults = ErosionParams.canyon()
            elif args.preset == "plains":
                defaults = ErosionParams.plains()
            elif args.preset == "mountains":
                defaults = ErosionParams.mountains()
            else:
                defaults = ErosionParams()

            if preset_overrides:
                defaults = defaults.override(preset_overrides)
            if args.save_preset:
                _save_preset(args.save_preset, defaults)

            erosion_gen = ErosionTerrainGenerator(
                resolution=args.resolution,
                use_erosion=not args.disable_erosion,
                defaults=defaults,
            )
            terrain = erosion_gen.generate_heightmap(
                seed=args.seed,
                seamless=args.seamless,
            )
        elif args.generator == "hydraulic":
            # Generate base terrain first (using erosion generator without erosion)
            # Keep it alive for rendering

            # Select preset for base terrain
            if args.preset == "canyon":
                defaults = ErosionParams.canyon()
            elif args.preset == "plains":
                defaults = ErosionParams.plains()
            elif args.preset == "mountains":
                defaults = ErosionParams.mountains()
            elif args.preset == "natural":
                defaults = ErosionParams.natural()
            else:
                defaults = ErosionParams()

            if preset_overrides:
                defaults = defaults.override(preset_overrides)

            erosion_gen = ErosionTerrainGenerator(
                resolution=args.resolution,
                use_erosion=False,
                defaults=defaults,
            )
            base_terrain = erosion_gen.generate_heightmap(
                seed=args.seed,
                seamless=args.seamless,
            )

            # Share context
            hydro_gen = HydraulicErosionGenerator(
                resolution=args.resolution, ctx=erosion_gen.ctx)
            params = HydraulicParams(
                iterations=100,
                dt=0.002,
            )
            # Apply overrides if any (TODO: Map preset overrides to hydraulic params)

            terrain = hydro_gen.simulate(base_terrain.height, params)

            # Upload hydraulic result to erosion_gen for rendering
            height = terrain.height.astype('f4')
            normals = terrain.normals.astype('f4')
            erosion = terrain.erosion_mask.astype(
                'f4') if terrain.erosion_mask is not None else np.zeros_like(height)

            data = np.zeros((args.resolution, args.resolution, 4), dtype='f4')
            data[:, :, 0] = height
            data[:, :, 1] = normals[:, :, 0]
            data[:, :, 2] = normals[:, :, 2]
            data[:, :, 3] = erosion

            erosion_gen.heightmap_texture.write(data.tobytes())
        else:
            morph_gen = MorphologicalTerrainGPU()
            terrain = morph_gen.generate(
                resolution=args.resolution, seed=args.seed)

        terrain = TerrainMaps.ensure(terrain)

        # Comparison grid render (GPU path) using viz/ray
        if args.compare_out and args.compare_seeds:
            seeds = [int(s)
                     for s in args.compare_seeds.split(",") if s.strip()]
            if erosion_gen is None:
                raise SystemExit(
                    "Comparison grid currently supports only erosion generator.")
            imgs: list[np.ndarray] = []
            labels: list[str] = []
            for s in seeds:
                _ = erosion_gen.generate_heightmap(
                    seed=s, seamless=args.seamless)
                if args.render == "viz":
                    img = erosion_gen.render_visualization(
                        water_height=args.water_height, mode=args.mode)
                else:
                    img = erosion_gen.render_raymarch(
                        water_height=args.water_height,
                        exposure=args.ray_exposure,
                        fog_color=args.ray_fog_color,
                        fog_density=args.ray_fog_density,
                        fog_height=args.ray_fog_height,
                        sun_intensity=args.ray_sun_intensity,
                        ray_max_steps=args.ray_max_steps,
                        ray_min_step=args.ray_min_step,
                        shadow_softness=args.ray_shadow_softness,
                        ao_strength=args.ray_ao_strength,
                    )
                imgs.append(img)
                labels.append(f"Seed {s}")

            try:
                import matplotlib.pyplot as plt
            except Exception as exc:
                raise SystemExit(f"Comparison grid requires matplotlib: {exc}")

            n = len(imgs)
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(4 * cols, 4 * rows), dpi=120)
            if rows == 1 and cols == 1:
                axes = [[axes]]  # type: ignore
            elif rows == 1:
                axes = [axes]  # type: ignore
            elif cols == 1:
                axes = [[ax] for ax in axes]  # type: ignore

            for idx, (img, label) in enumerate(zip(imgs, labels)):
                r = idx // cols
                c = idx % cols
                axes[r][c].imshow(img)
                axes[r][c].set_title(label)
                axes[r][c].axis("off")

            for ax in np.array(axes).ravel()[n:]:
                ax.axis("off")

            args.compare_out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(args.compare_out)
            plt.close(fig)
            print(f"✓ Saved comparison grid to {args.compare_out}")

        shade_kwargs = dict(
            azimuth=args.shade_azimuth,
            altitude=args.shade_altitude,
            vert_exag=args.shade_vert_exag,
            colormap=args.shade_colormap,
        )

        if args.heightmap_out:
            utils.save_heightmap_png(args.heightmap_out, terrain)
            print(f"✓ Saved heightmap to {args.heightmap_out}")

        if args.raw_out:
            utils.save_heightmap_raw(args.raw_out, terrain)
            print(f"✓ Saved raw heightmap to {args.raw_out}")

        if args.r32_out:
            utils.save_heightmap_r32(args.r32_out, terrain)
            print(f"✓ Saved raw float heightmap to {args.r32_out}")

        if args.normals_out:
            utils.save_normal_map_png(args.normals_out, terrain)
            print(f"✓ Saved normal map to {args.normals_out}")

        if args.erosion_out:
            utils.save_erosion_mask_png(args.erosion_out, terrain)
            print(f"✓ Saved erosion mask to {args.erosion_out}")

        if args.bundle_out:
            utils.save_npz_bundle(args.bundle_out, terrain)
            print(f"✓ Saved NPZ bundle to {args.bundle_out}")

        if args.obj_out:
            utils.export_obj_mesh(args.obj_out, terrain)
            print(f"✓ Saved OBJ mesh to {args.obj_out}")

        if args.stl_out:
            utils.export_stl_mesh(args.stl_out, terrain)
            print(f"✓ Saved STL mesh to {args.stl_out}")

        if args.panel_out:
            utils.save_panel_overview(args.panel_out, terrain)
            print(f"✓ Saved overview panel to {args.panel_out}")

        if args.turntable_out:
            utils.save_turntable_gif(
                args.turntable_out,
                terrain,
                frames=args.turntable_frames,
                interval=args.turntable_interval,
            )
            print(f"✓ Saved turntable animation to {args.turntable_out}")

        if args.shaded_out:
            utils.save_shaded_relief_png(
                args.shaded_out, terrain, **shade_kwargs)
            print(f"✓ Saved shaded relief preview to {args.shaded_out}")

        if args.slope_out:
            utils.save_slope_map_png(
                args.slope_out,
                terrain,
                scale=args.slope_scale,
                gamma=args.slope_gamma,
            )
            print(f"✓ Saved slope map to {args.slope_out}")

        # Texture exports
        if args.splatmap_out:
            utils.save_splatmap_rgba(args.splatmap_out, terrain)
            print(f"✓ Saved RGBA splatmap to {args.splatmap_out}")

        if args.ao_out:
            utils.save_ao_map(args.ao_out, terrain)
            print(f"✓ Saved AO map to {args.ao_out}")

        if args.curvature_out:
            utils.save_curvature_map(args.curvature_out, terrain)
            print(f"✓ Saved curvature map to {args.curvature_out}")

        if args.packed_out:
            utils.save_packed_texture(
                args.packed_out, terrain, pack_mode=args.pack_mode)
            print(
                f"✓ Saved packed texture ({args.pack_mode}) to {args.packed_out}")

        if args.scatter_out:
            utils.save_scatter_map(args.scatter_out, terrain)
            print(f"✓ Saved scatter map to {args.scatter_out}")

        # Advanced rendering exports
        if args.multi_angle_out:
            output_dir = args.multi_angle_out
            output_dir.mkdir(parents=True, exist_ok=True)

            renders = utils.render_multi_angle(
                terrain,
                azimuth=args.shade_azimuth,
                altitude=args.shade_altitude,
                vert_exag=args.shade_vert_exag,
                colormap=args.shade_colormap,
            )

            angles = ["northwest", "northeast",
                      "southeast", "southwest", "overhead"]
            for angle_name, render in zip(angles, renders):
                _ensure_image(output_dir / f"angle_{angle_name}.png", render)

            print(
                f"✓ Saved {len(renders)} multi-angle renders to {output_dir}")

        if args.lighting_study_out:
            fig = utils.render_lighting_study(
                terrain,
                azimuth_steps=8,
                altitude_steps=3,
                colormap=args.shade_colormap,
                vert_exag=args.shade_vert_exag,
            )
            fig.savefig(args.lighting_study_out, dpi=150, bbox_inches="tight")
            print(f"✓ Saved lighting study to {args.lighting_study_out}")

        if args.turntable_video_out:
            saved_path = utils.save_turntable_video(
                args.turntable_video_out,
                terrain,
                frames=args.turntable_frames,
                fps=args.turntable_fps,
                altitude=args.shade_altitude,
                vert_exag=args.shade_vert_exag,
                colormap=args.shade_colormap,
            )
            print(f"✓ Saved turntable video to {saved_path}")

        if args.render == "shade":
            img = utils.shade_heightmap(terrain, **shade_kwargs)
            if args.render_out:
                _ensure_image(args.render_out, img)
                print(f"✓ Saved shaded render to {args.render_out}")
        elif args.render != "none" and erosion_gen is not None:
            if args.render == "viz":
                img = erosion_gen.render_visualization(
                    water_height=args.water_height, mode=args.mode)
            else:
                img = erosion_gen.render_raymarch(
                    water_height=args.water_height,
                    exposure=args.ray_exposure,
                    fog_color=args.ray_fog_color,
                    fog_density=args.ray_fog_density,
                    fog_height=args.ray_fog_height,
                    sun_intensity=args.ray_sun_intensity,
                    ray_max_steps=args.ray_max_steps,
                    ray_min_step=args.ray_min_step,
                    shadow_softness=args.ray_shadow_softness,
                    ao_strength=args.ray_ao_strength,
                )
            if args.render_out:
                _ensure_image(args.render_out, img)
                print(f"✓ Saved render to {args.render_out}")
                if args.hdr_out and args.render in ("ray", "shade"):
                    _save_hdr_image(args.hdr_out, img)
        elif args.render != "none" and args.generator == "morph":
            print("⚠ GPU renders are only available for the erosion generator; try --render shade for CPU previews.")
    finally:
        if erosion_gen is not None:
            erosion_gen.cleanup()
        if morph_gen is not None:
            morph_gen.cleanup()
        if hydro_gen is not None:
            hydro_gen.cleanup()


if __name__ == "__main__":
    main()
