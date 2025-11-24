"""Unified configuration model for terrain generation and rendering."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Any
import json

import yaml

from .generators.erosion import ErosionParams
from .generators.hydraulic import HydraulicParams
from .utils.render_configs import RenderConfig

__all__ = [
    "TerrainConfig",
    "load_config",
    "save_config",
]


@dataclass
class TerrainConfig:
    """Complete configuration for terrain generation and rendering.

    This combines terrain generation parameters, render settings, and
    output configuration into a single serializable structure.
    """

    # Version for schema migration
    version: str = "1.0"

    # Output settings
    resolution: int = 512
    seed: int = 42
    output_dir: str = "output"

    # Terrain generation parameters
    generator_type: str = "erosion"  # erosion, hydraulic, morph
    terrain_preset: str = "canyon"  # canyon, plains, mountains, custom
    seamless: bool = False  # Enable seamless/tileable generation
    height_tiles: float = 3.0
    height_octaves: int = 3
    height_amp: float = 0.25
    height_gain: float = 0.1
    height_lacunarity: float = 2.0
    water_height: float = 0.45
    erosion_tiles: float = 3.0
    erosion_octaves: int = 5
    erosion_gain: float = 0.5
    erosion_lacunarity: float = 2.0
    erosion_slope_strength: float = 3.0
    erosion_branch_strength: float = 3.0
    erosion_strength: float = 0.04
    use_erosion: bool = True

    # Hydraulic erosion parameters
    hydraulic_iterations: int = 100
    hydraulic_dt: float = 0.02
    hydraulic_pipe_length: float = 1.0
    hydraulic_sediment_capacity: float = 1.0
    hydraulic_soil_dissolving: float = 0.5
    hydraulic_sediment_deposition: float = 0.5
    hydraulic_evaporation_rate: float = 0.01
    hydraulic_rain_rate: float = 0.01

    # Render configuration
    render_preset: str = "classic"  # classic, dramatic, etc or custom
    azimuth: float = 315.0
    altitude: float = 45.0
    vert_exag: float = 2.0
    blend_mode: Literal["soft", "overlay", "hsv"] = "soft"
    colormap: str = "terrain"
    ambient_strength: float = 0.1
    sun_intensity: float = 2.0
    tonemap_enabled: bool = True
    gamma: float = 2.2
    exposure: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    viz_mode: int = 0

    # Output formats
    export_heightmap: bool = True
    export_normals: bool = True
    export_shaded: bool = True
    export_obj: bool = False

    def get_erosion_params(self) -> ErosionParams:
        """Extract terrain generation parameters."""
        return ErosionParams(
            height_tiles=self.height_tiles,
            height_octaves=self.height_octaves,
            height_amp=self.height_amp,
            height_gain=self.height_gain,
            height_lacunarity=self.height_lacunarity,
            water_height=self.water_height,
            erosion_tiles=self.erosion_tiles,
            erosion_octaves=self.erosion_octaves,
            erosion_gain=self.erosion_gain,
            erosion_lacunarity=self.erosion_lacunarity,
            erosion_slope_strength=self.erosion_slope_strength,
            erosion_branch_strength=self.erosion_branch_strength,
            erosion_strength=self.erosion_strength,
        )

    def get_hydraulic_params(self) -> HydraulicParams:
        """Extract hydraulic erosion parameters."""
        return HydraulicParams(
            iterations=self.hydraulic_iterations,
            dt=self.hydraulic_dt,
            pipe_length=self.hydraulic_pipe_length,
            sediment_capacity=self.hydraulic_sediment_capacity,
            soil_dissolving=self.hydraulic_soil_dissolving,
            sediment_deposition=self.hydraulic_sediment_deposition,
            evaporation_rate=self.hydraulic_evaporation_rate,
            rain_rate=self.hydraulic_rain_rate,
        )

    def get_render_config(self) -> RenderConfig:
        """Extract rendering parameters."""
        return RenderConfig(
            azimuth=self.azimuth,
            altitude=self.altitude,
            vert_exag=self.vert_exag,
            blend_mode=self.blend_mode,
            colormap=self.colormap,
            water_height=self.water_height,
            ambient_strength=self.ambient_strength,
            sun_intensity=self.sun_intensity,
            tonemap_enabled=self.tonemap_enabled,
            gamma=self.gamma,
            exposure=self.exposure,
            contrast=self.contrast,
            saturation=self.saturation,
            viz_mode=self.viz_mode,
        )

    def apply_terrain_preset(self, preset: str) -> None:
        """Apply a terrain generation preset."""
        self.terrain_preset = preset

        if preset == "canyon":
            params = ErosionParams.canyon()
        elif preset == "plains":
            params = ErosionParams.plains()
        elif preset == "mountains":
            params = ErosionParams.mountains()
        else:
            return  # Keep current values for "custom"

        # Update parameters
        self.height_tiles = params.height_tiles
        self.height_octaves = params.height_octaves
        self.height_amp = params.height_amp
        self.height_gain = params.height_gain
        self.height_lacunarity = params.height_lacunarity
        self.water_height = params.water_height
        self.erosion_tiles = params.erosion_tiles
        self.erosion_octaves = params.erosion_octaves
        self.erosion_gain = params.erosion_gain
        self.erosion_lacunarity = params.erosion_lacunarity
        self.erosion_slope_strength = params.erosion_slope_strength
        self.erosion_branch_strength = params.erosion_branch_strength
        self.erosion_strength = params.erosion_strength

    def apply_render_preset(self, preset: str) -> None:
        """Apply a render configuration preset."""
        from .utils.render_configs import PRESET_CONFIGS

        self.render_preset = preset

        if preset == "custom":
            return  # Keep current values

        if preset not in PRESET_CONFIGS:
            return

        config = PRESET_CONFIGS[preset]()

        # Update parameters
        self.azimuth = config.azimuth
        self.altitude = config.altitude
        self.vert_exag = config.vert_exag
        self.blend_mode = config.blend_mode
        self.colormap = config.colormap
        self.ambient_strength = config.ambient_strength
        self.sun_intensity = config.sun_intensity
        self.tonemap_enabled = config.tonemap_enabled
        self.gamma = config.gamma
        self.exposure = config.exposure
        self.contrast = config.contrast
        self.saturation = config.saturation
        self.viz_mode = config.viz_mode

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainConfig":
        """Create from dictionary."""
        # Filter out unknown keys for forward compatibility
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def save_config(config: TerrainConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f,
                  default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> TerrainConfig:
    """Load configuration from YAML or JSON file."""
    path = Path(path)

    with open(path, "r") as f:
        if path.suffix == ".json":
            data = json.load(f)
        else:  # .yaml or .yml
            data = yaml.safe_load(f)

    return TerrainConfig.from_dict(data)
