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
        if include_optional and self.erosion_mask is not None:
            payload["erosion_mask"] = self.erosion_mask
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
