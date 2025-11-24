"""Batch terrain generation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import time

import numpy as np

from .artifacts import TerrainMaps
from . import export, rendering

if TYPE_CHECKING:
    from ..generators import ErosionTerrainGenerator, MorphologicalTerrainGPU

__all__ = ["BatchGenerator", "generate_terrain_set"]


class BatchGenerator:
    """Manages batch generation of terrain maps with varied parameters."""

    def __init__(
        self,
        generator_type: str = "erosion",
        resolution: int = 512,
        output_dir: str | Path = "batch_output",
    ) -> None:
        self.generator_type = generator_type
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if generator_type == "erosion":
            from ..generators import ErosionTerrainGenerator
            self.gen: ErosionTerrainGenerator | MorphologicalTerrainGPU = (
                ErosionTerrainGenerator(resolution=resolution)
            )
        else:
            from ..generators import MorphologicalTerrainGPU
            self.gen = MorphologicalTerrainGPU()

    def generate_set(
        self,
        seeds: list[int],
        prefix: str = "terrain",
        export_formats: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        **generation_kwargs: Any,
    ) -> list[TerrainMaps]:
        """Generate multiple terrains with different seeds.

        Args:
            seeds: List of random seeds for variation
            prefix: Filename prefix for exports
            export_formats: List of formats to export ('png', 'obj', 'stl', 'npz')
            progress_callback: Optional callback(current, total, status_msg)
            **generation_kwargs: Parameters passed to generator

        Returns:
            List of generated TerrainMaps
        """
        export_formats = export_formats or ["png"]
        results = []
        total = len(seeds)

        start_time = time.time()

        for idx, seed in enumerate(seeds, 1):
            if progress_callback:
                progress_callback(idx, total, f"Generating seed {seed}...")

            # Generate terrain
            if self.generator_type == "erosion":
                terrain = self.gen.generate_heightmap(
                    seed=seed, **generation_kwargs)  # type: ignore
            else:
                terrain = self.gen.generate(  # type: ignore
                    resolution=self.resolution, seed=seed, **generation_kwargs
                )

            terrain = TerrainMaps.ensure(terrain)
            results.append(terrain)

            # Export in requested formats
            base_name = f"{prefix}_{seed:04d}"
            self._export_terrain(terrain, base_name, export_formats)

            if progress_callback:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = avg_time * (total - idx)
                progress_callback(
                    idx,
                    total,
                    f"Completed {idx}/{total} (ETA: {remaining:.1f}s)",
                )

        return results

    def _export_terrain(
        self, terrain: TerrainMaps, base_name: str, formats: list[str]
    ) -> None:
        """Export terrain in multiple formats."""
        for fmt in formats:
            if fmt == "png":
                export.save_heightmap_png(
                    self.output_dir / f"{base_name}_height.png", terrain
                )
                export.save_normal_map_png(
                    self.output_dir / f"{base_name}_normal.png", terrain
                )
                if terrain.erosion_mask is not None:
                    export.save_erosion_mask_png(
                        self.output_dir / f"{base_name}_erosion.png", terrain
                    )
            elif fmt == "shaded":
                rendering.save_shaded_relief_png(
                    self.output_dir / f"{base_name}_shaded.png", terrain
                )
            elif fmt == "obj":
                export.export_obj_mesh(
                    self.output_dir / f"{base_name}.obj", terrain
                )
            elif fmt == "stl":
                export.export_stl_mesh(
                    self.output_dir / f"{base_name}.stl", terrain
                )
            elif fmt == "npz":
                export.save_npz_bundle(
                    self.output_dir / f"{base_name}.npz", terrain
                )

    def cleanup(self) -> None:
        """Release GPU resources."""
        if hasattr(self.gen, "cleanup"):
            self.gen.cleanup()


def generate_terrain_set(
    count: int,
    base_seed: int = 0,
    generator: str = "erosion",
    resolution: int = 512,
    output_dir: str | Path = "terrain_set",
    formats: list[str] | None = None,
    **kwargs: Any,
) -> list[TerrainMaps]:
    """Convenience function to generate a set of terrains.

    Args:
        count: Number of terrains to generate
        base_seed: Starting seed (will increment)
        generator: 'erosion' or 'morph'
        resolution: Texture resolution
        output_dir: Output directory path
        formats: Export formats ('png', 'obj', 'stl', 'npz', 'shaded')
        **kwargs: Additional generation parameters

    Returns:
        List of generated TerrainMaps
    """
    seeds = list(range(base_seed, base_seed + count))
    batch_gen = BatchGenerator(generator, resolution, output_dir)

    def _progress(current: int, total: int, msg: str) -> None:
        print(f"[{current}/{total}] {msg}")

    try:
        results = batch_gen.generate_set(
            seeds, prefix="terrain", export_formats=formats, progress_callback=_progress, **kwargs
        )
        print(f"\n✓ Generated {count} terrains in {output_dir}/")
        return results
    finally:
        batch_gen.cleanup()
