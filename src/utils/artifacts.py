"""Shared terrain data containers and helpers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["TerrainMaps"]


@dataclass(slots=True)
class TerrainMaps:
    """Container for GPU terrain outputs with handy conversions."""

    height: np.ndarray
    normals: np.ndarray
    erosion_mask: np.ndarray | None = None
    scatter_map: np.ndarray | None = None
    moisture_map: np.ndarray | None = None
    temperature_map: np.ndarray | None = None
    biome_map: np.ndarray | None = None

    @classmethod
    def ensure(cls, data: "TerrainMaps | Mapping[str, np.ndarray]") -> "TerrainMaps":
        if isinstance(data, TerrainMaps):
            return data
        if isinstance(data, Mapping):
            return cls(
                height=np.asarray(data["height"], dtype=np.float32),
                normals=np.asarray(data["normals"], dtype=np.float32),
                erosion_mask=np.asarray(
                    data.get("erosion_mask"), dtype=np.float32)
                if data.get("erosion_mask") is not None
                else None,
                scatter_map=np.asarray(
                    data.get("scatter_map"), dtype=np.float32)
                if data.get("scatter_map") is not None
                else None,
                moisture_map=np.asarray(
                    data.get("moisture_map"), dtype=np.float32)
                if data.get("moisture_map") is not None
                else None,
                temperature_map=np.asarray(
                    data.get("temperature_map"), dtype=np.float32)
                if data.get("temperature_map") is not None
                else None,
                biome_map=np.asarray(
                    data.get("biome_map"), dtype=np.float32)
                if data.get("biome_map") is not None
                else None,
            )
        raise TypeError(
            "Terrain data must be TerrainMaps or mapping with 'height'/'normals'.")

    @property
    def resolution(self) -> tuple[int, int]:
        return tuple(map(int, self.height.shape))  # type: ignore[arg-type]

    def as_dict(self, include_optional: bool = True) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "height": self.height,
            "normals": self.normals,
        }
        if include_optional:
            if self.erosion_mask is not None:
                payload["erosion_mask"] = self.erosion_mask
            if self.scatter_map is not None:
                payload["scatter_map"] = self.scatter_map
            if self.moisture_map is not None:
                payload["moisture_map"] = self.moisture_map
            if self.temperature_map is not None:
                payload["temperature_map"] = self.temperature_map
            if self.biome_map is not None:
                payload["biome_map"] = self.biome_map
        return payload

    # ------------------------------------------------------------------
    # Derived buffers
    # ------------------------------------------------------------------
    def _erosion_or_zero(self) -> np.ndarray:
        if self.erosion_mask is None:
            return np.zeros_like(self.height, dtype=np.float32)
        return self.erosion_mask

    def height_u16(self) -> np.ndarray:
        return (np.clip(self.height, 0.0, 1.0) * 65535).astype(np.uint16)

    def normal_map_u8(self) -> np.ndarray:
        normals = np.clip((self.normals * 0.5) + 0.5, 0.0, 1.0)
        return (normals * 255).astype(np.uint8)

    def erosion_mask_u8(self) -> np.ndarray:
        mask = np.clip(self._erosion_or_zero(), 0.0, 1.0)
        return (mask * 255).astype(np.uint8)

    def erosion_channel(self) -> np.ndarray:
        return self._erosion_or_zero()

    def scatter_map_u8(self) -> np.ndarray:
        if self.scatter_map is None:
            return np.zeros((*self.resolution, 3), dtype=np.uint8)
        # Scatter map is RGBA, but we usually only care about RGB for density
        # If it's 4 channels, take first 3
        scatter = self.scatter_map
        if scatter.shape[-1] == 4:
            scatter = scatter[..., :3]

        scatter = np.clip(scatter, 0.0, 1.0)
        return (scatter * 255).astype(np.uint8)
