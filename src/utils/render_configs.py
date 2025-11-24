"""Render configuration presets for various artistic and technical styles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "RenderConfig",
    "PRESET_CONFIGS",
]


@dataclass
class RenderConfig:
    """Comprehensive rendering configuration for terrain visualization."""

    # Lighting parameters
    azimuth: float = 315.0  # Sun azimuth angle (0-360 degrees)
    altitude: float = 45.0  # Sun altitude angle (0-90 degrees)

    # Shading parameters
    vert_exag: float = 2.0  # Vertical exaggeration for relief
    blend_mode: Literal["soft", "overlay", "hsv"] = "soft"

    # Colormap settings
    colormap: str = "terrain"

    # Water rendering
    water_height: float = 0.45
    water_color: tuple[float, float, float] = (0.0, 0.05, 0.1)
    water_shore_color: tuple[float, float, float] = (0.0, 0.25, 0.25)
    water_smoothness: float = 0.9

    # Material properties
    ambient_strength: float = 0.1
    sun_intensity: float = 2.0

    # Post-processing
    tonemap_enabled: bool = True
    gamma: float = 2.2
    exposure: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0

    # Debug/visualization mode
    viz_mode: int = 0  # 0=full, 1=height, 2=normals, 3=erosion, 4=slope, 5=curvature

    @classmethod
    def classic(cls) -> "RenderConfig":
        """Classic northwest lighting - most common cartographic style."""
        return cls(
            azimuth=315.0,
            altitude=45.0,
            vert_exag=2.0,
            blend_mode="soft",
            colormap="terrain",
        )

    @classmethod
    def dramatic(cls) -> "RenderConfig":
        """Low-angle dramatic lighting with high contrast."""
        return cls(
            azimuth=90.0,
            altitude=15.0,
            vert_exag=3.0,
            blend_mode="overlay",
            colormap="gist_earth",
            sun_intensity=2.5,
            contrast=1.2,
        )

    @classmethod
    def noon(cls) -> "RenderConfig":
        """High overhead sun - minimal shadows."""
        return cls(
            azimuth=0.0,
            altitude=85.0,
            vert_exag=1.5,
            blend_mode="soft",
            colormap="terrain",
            ambient_strength=0.15,
        )

    @classmethod
    def sunrise(cls) -> "RenderConfig":
        """Warm sunrise lighting from the east."""
        return cls(
            azimuth=90.0,
            altitude=10.0,
            vert_exag=2.5,
            blend_mode="overlay",
            colormap="copper",
            sun_intensity=1.8,
            exposure=1.1,
            saturation=1.2,
        )

    @classmethod
    def sunset(cls) -> "RenderConfig":
        """Warm sunset lighting from the west."""
        return cls(
            azimuth=270.0,
            altitude=12.0,
            vert_exag=2.5,
            blend_mode="overlay",
            colormap="autumn",
            sun_intensity=1.8,
            exposure=1.1,
            saturation=1.3,
        )

    @classmethod
    def flat(cls) -> "RenderConfig":
        """Flat shading - no relief, pure elevation colors."""
        return cls(
            azimuth=0.0,
            altitude=90.0,
            vert_exag=0.1,
            blend_mode="soft",
            colormap="terrain",
            ambient_strength=1.0,
            sun_intensity=0.0,
        )

    @classmethod
    def contour(cls) -> "RenderConfig":
        """High contrast for contour-like appearance."""
        return cls(
            azimuth=315.0,
            altitude=45.0,
            vert_exag=4.0,
            blend_mode="overlay",
            colormap="gray",
            contrast=1.5,
            saturation=0.0,
        )

    @classmethod
    def technical(cls) -> "RenderConfig":
        """Clean technical visualization."""
        return cls(
            azimuth=315.0,
            altitude=45.0,
            vert_exag=2.0,
            blend_mode="soft",
            colormap="viridis",
            ambient_strength=0.2,
            tonemap_enabled=False,
        )

    @classmethod
    def artistic_vibrant(cls) -> "RenderConfig":
        """Vibrant artistic style with saturated colors."""
        return cls(
            azimuth=135.0,
            altitude=35.0,
            vert_exag=2.5,
            blend_mode="hsv",
            colormap="rainbow",
            sun_intensity=2.2,
            saturation=1.4,
            contrast=1.1,
        )

    @classmethod
    def monochrome(cls) -> "RenderConfig":
        """Black and white relief shading."""
        return cls(
            azimuth=315.0,
            altitude=45.0,
            vert_exag=2.5,
            blend_mode="soft",
            colormap="gray",
            saturation=0.0,
        )

    @classmethod
    def underwater(cls) -> "RenderConfig":
        """Underwater bathymetry style."""
        return cls(
            azimuth=315.0,
            altitude=60.0,
            vert_exag=1.5,
            blend_mode="soft",
            colormap="ocean",
            water_height=1.0,  # No water surface
            ambient_strength=0.3,
        )

    @classmethod
    def alpine(cls) -> "RenderConfig":
        """Alpine terrain with snow caps."""
        return cls(
            azimuth=315.0,
            altitude=40.0,
            vert_exag=3.0,
            blend_mode="overlay",
            colormap="terrain",
            sun_intensity=2.5,
            exposure=1.15,
        )

    @classmethod
    def desert(cls) -> "RenderConfig":
        """Hot desert appearance."""
        return cls(
            azimuth=270.0,
            altitude=25.0,
            vert_exag=2.0,
            blend_mode="soft",
            colormap="YlOrBr",
            sun_intensity=2.8,
            exposure=1.2,
            saturation=1.2,
        )

    @classmethod
    def debug_normals(cls) -> "RenderConfig":
        """Show normal map visualization."""
        return cls(
            viz_mode=2,
            colormap="terrain",
        )

    @classmethod
    def debug_erosion(cls) -> "RenderConfig":
        """Show erosion mask visualization."""
        return cls(
            viz_mode=3,
            colormap="hot",
        )

    @classmethod
    def debug_slope(cls) -> "RenderConfig":
        """Show slope angle visualization."""
        return cls(
            viz_mode=4,
            colormap="plasma",
        )

    @classmethod
    def debug_curvature(cls) -> "RenderConfig":
        """Show surface curvature visualization."""
        return cls(
            viz_mode=5,
            colormap="coolwarm",
        )


# Dictionary of all preset configurations
PRESET_CONFIGS = {
    "classic": RenderConfig.classic,
    "dramatic": RenderConfig.dramatic,
    "noon": RenderConfig.noon,
    "sunrise": RenderConfig.sunrise,
    "sunset": RenderConfig.sunset,
    "flat": RenderConfig.flat,
    "contour": RenderConfig.contour,
    "technical": RenderConfig.technical,
    "vibrant": RenderConfig.artistic_vibrant,
    "monochrome": RenderConfig.monochrome,
    "underwater": RenderConfig.underwater,
    "alpine": RenderConfig.alpine,
    "desert": RenderConfig.desert,
    "debug_normals": RenderConfig.debug_normals,
    "debug_erosion": RenderConfig.debug_erosion,
    "debug_slope": RenderConfig.debug_slope,
    "debug_curvature": RenderConfig.debug_curvature,
}
