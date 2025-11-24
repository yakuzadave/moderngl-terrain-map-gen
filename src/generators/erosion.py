"""Hydraulic erosion terrain generator backed by ModernGL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import moderngl
import numpy as np

from ..utils import (
    TerrainMaps,
    create_context,
    create_detail_texture,
    load_shader,
)

__all__ = ["ErosionTerrainGenerator"]


@dataclass
class ErosionParams:
    """
    Represents configurable parameters for terrain height and erosion
    generation, including presets for canyons, plains, mountains, and
    natural terrain. Provides methods to create modified copies and export
    parameters as dictionaries.
    """
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
    warp_strength: float = 0.0  # Domain warping strength
    ridge_noise: int = 0        # 0 = Standard FBM, 1 = Ridge Noise
    thermal_iterations: int = 0
    thermal_threshold: float = 0.001
    thermal_strength: float = 0.5

    @classmethod
    def canyon(cls) -> "ErosionParams":
        """Preset: Deep erosion with branching valleys."""
        return cls(
            height_amp=0.3,
            erosion_octaves=6,
            erosion_strength=0.06,
            erosion_slope_strength=4.0,
            erosion_branch_strength=4.5,
            warp_strength=0.5,
            ridge_noise=1,
            thermal_iterations=10,
            thermal_threshold=0.002,
            thermal_strength=0.5,
        )

    @classmethod
    def plains(cls) -> "ErosionParams":
        """Preset: Gentle rolling hills with minimal erosion."""
        return cls(
            height_amp=0.15,
            erosion_octaves=3,
            erosion_strength=0.02,
            erosion_slope_strength=1.5,
            erosion_branch_strength=1.0,
            warp_strength=0.2,
        )

    @classmethod
    def mountains(cls) -> "ErosionParams":
        """Preset: Sharp peaks with moderate erosion."""
        return cls(
            height_amp=0.35,
            height_gain=0.15,
            erosion_octaves=5,
            erosion_strength=0.04,
            erosion_slope_strength=3.5,
            erosion_branch_strength=2.5,
            warp_strength=0.8,
        )

    @classmethod
    def natural(cls) -> "ErosionParams":
        """Preset: Organic terrain combining ridge noise and domain warping."""
        return cls(
            height_amp=0.3,
            height_gain=0.1,
            erosion_octaves=6,
            erosion_strength=0.05,
            erosion_slope_strength=3.5,
            erosion_branch_strength=3.0,
            warp_strength=0.5,
            ridge_noise=1,
        )

    def override(self, overrides: Dict[str, Any]) -> "ErosionParams":
        """
        Updates the current ErosionParams instance by applying the given
        overrides dictionary and returns a new ErosionParams object with the
        updated values.
        """
        data = self.__dict__.copy()
        data.update(overrides)
        return ErosionParams(**data)

    def uniforms(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ErosionTerrainGenerator:
    """GPU-accelerated generator with optional visualization pipelines."""

    def __init__(
        self,
        resolution: int = 512,
        use_erosion: bool = True,
        defaults: ErosionParams | None = None,
        ctx: moderngl.Context | None = None,
    ) -> None:
        self.resolution = int(resolution)
        self.use_erosion = bool(use_erosion)
        self.defaults = defaults or ErosionParams()

        self.ctx = ctx or create_context()
        self._own_ctx = ctx is None

        self.height_program = self._compile_program(
            "quad.vert", "erosion_heightmap.frag")
        self.thermal_program = self._compile_program(
            "quad.vert", "thermal_erosion.frag")
        self.normal_program = self._compile_program(
            "quad.vert", "recompute_normals.frag")
        self.viz_program = self._compile_program(
            "quad.vert", "erosion_viz.frag")
        self.ray_program = self._compile_program(
            "quad.vert", "erosion_raymarch.frag")
        self.scatter_program = self._compile_program(
            "quad.vert", "scatter_density.frag")

        self._create_framebuffers()
        self._create_geometry()
        self.detail_texture = create_detail_texture(self.ctx)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _compile_program(self, vert: str, frag: str) -> moderngl.Program:
        """
        Compiles and returns a moderngl.Program by loading and combining the
        specified vertex (vert) and fragment (frag) shader source files.
        """
        return self.ctx.program(
            vertex_shader=load_shader(vert),
            fragment_shader=load_shader(frag),
        )

    def _create_framebuffers(self) -> None:
        """
        Initializes heightmap and visualization framebuffers and textures using
        the current OpenGL context, configuring their size, format, and
        filtering for terrain generation and visualization.
        """
        size = (self.resolution, self.resolution)
        self.heightmap_texture = self.ctx.texture(size, 4, dtype="f4")
        self.heightmap_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.heightmap_fbo = self.ctx.framebuffer([self.heightmap_texture])

        self.pingpong_texture = self.ctx.texture(size, 4, dtype="f4")
        self.pingpong_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.pingpong_fbo = self.ctx.framebuffer([self.pingpong_texture])

        self.viz_texture = self.ctx.texture(size, 3, dtype="f1")
        self.viz_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.viz_fbo = self.ctx.framebuffer([self.viz_texture])

        self.scatter_texture = self.ctx.texture(size, 4, dtype="f1")
        self.scatter_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scatter_fbo = self.ctx.framebuffer([self.scatter_texture])

    def _create_geometry(self) -> None:
        """
        Initializes a quad geometry by creating vertex and index buffers and
        sets up a vertex array object for rendering with the height shader
        program using moderngl.
        """
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ], dtype="f4")
        indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.quad = self.ctx.vertex_array(
            self.height_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )
        self.thermal_quad = self.ctx.vertex_array(
            self.thermal_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )
        self.normal_quad = self.ctx.vertex_array(
            self.normal_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )
        self.viz_quad = self.ctx.vertex_array(
            self.viz_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )
        self.ray_quad = self.ctx.vertex_array(
            self.ray_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )
        self.scatter_quad = self.ctx.vertex_array(
            self.scatter_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )

    # ------------------------------------------------------------------
    # Generation API
    # ------------------------------------------------------------------
    def generate_heightmap(self, seed: int = 0, seamless: bool = False, **overrides) -> TerrainMaps:
        """
        Generates a terrain heightmap using the specified seed and optional
        seamless tiling, applying parameter overrides. Renders the heightmap
        with erosion effects via OpenGL, extracts height, normal, and erosion
        mask data, and returns them as a TerrainMaps object.
        """
        params = self.defaults.override(overrides)
        if seamless:
            params.height_tiles = max(1.0, round(params.height_tiles))
            params.erosion_tiles = max(1.0, round(params.erosion_tiles))

        uniforms = params.uniforms()
        self.heightmap_fbo.use()
        self.heightmap_fbo.clear(0.0, 0.0, 0.0, 1.0)

        program = self.height_program
        program["useErosion"].value = 1 if self.use_erosion else 0  # type: ignore
        program["u_seamless"].value = 1 if seamless else 0  # type: ignore
        program["u_seed"].value = float(seed)  # type: ignore
        texel = 1.0 / float(self.resolution)
        program["u_texelSize"].value = (texel, texel)  # type: ignore

        for key, uniform in uniforms.items():
            gl_name = self._uniform_name(key)
            if gl_name not in program:
                continue
            if "octaves" in key or "ridge_noise" in key:
                program[gl_name].value = int(uniform)  # type: ignore
            else:
                program[gl_name].value = float(uniform)  # type: ignore

        self.quad.render(moderngl.TRIANGLES)

        # Apply thermal erosion if requested
        if params.thermal_iterations > 0:
            self.apply_thermal_erosion(
                params.thermal_iterations,
                params.thermal_threshold,
                params.thermal_strength
            )

        raw = self.heightmap_fbo.read(components=4, dtype="f4")
        data = np.frombuffer(raw, dtype="f4").reshape(
            (self.resolution, self.resolution, 4))
        data = np.flip(data, axis=0)

        height = data[:, :, 0]
        nx = data[:, :, 1]
        nz = data[:, :, 2]
        erosion = data[:, :, 3]
        ny_sq = np.clip(1.0 - nx**2 - nz**2, 0.0, 1.0)
        ny = np.sqrt(ny_sq)
        normals = np.stack([nx, ny, nz], axis=-1)

        # Generate scatter map
        scatter_map = self.generate_scatter_map(params.water_height)

        return TerrainMaps(height=height, normals=normals, erosion_mask=erosion, scatter_map=scatter_map)

    def generate_scatter_map(self, water_height: float) -> np.ndarray:
        """
        Generates a scatter density map (Trees, Rocks, Grass) based on the
        current heightmap state.
        """
        self.scatter_fbo.use()
        self.scatter_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.heightmap_texture.use(location=0)

        program = self.scatter_program
        program["u_heightmap"].value = 0  # type: ignore
        program["u_waterHeight"].value = float(water_height)  # type: ignore

        self.scatter_quad.render(moderngl.TRIANGLES)

        raw = self.scatter_fbo.read(components=4, dtype="f1")
        data = np.frombuffer(raw, dtype="u1").reshape(
            (self.resolution, self.resolution, 4))
        return np.flip(data, axis=0)

    def apply_thermal_erosion(self, iterations: int, threshold: float, strength: float) -> None:
        """
        Applies thermal erosion (talus deposition) to the heightmap.
        This simulates material falling down steep slopes.
        """
        if iterations <= 0:
            return

        texel = 1.0 / float(self.resolution)

        # Ping-pong loop for thermal erosion
        for _ in range(iterations):
            # Render from heightmap -> pingpong
            self.pingpong_fbo.use()
            self.heightmap_texture.use(location=0)

            self.thermal_program["u_heightmap"].value = 0  # type: ignore
            self.thermal_program["u_texelSize"].value = (
                texel, texel)  # type: ignore
            self.thermal_program["u_talusThreshold"].value = float(
                threshold)  # type: ignore
            self.thermal_program["u_thermalStrength"].value = float(
                strength)  # type: ignore

            self.thermal_quad.render(moderngl.TRIANGLES)

            # Swap textures and FBOs so heightmap_texture holds the result
            self.heightmap_texture, self.pingpong_texture = self.pingpong_texture, self.heightmap_texture
            self.heightmap_fbo, self.pingpong_fbo = self.pingpong_fbo, self.heightmap_fbo

        # Recompute normals after erosion
        # Render from current heightmap -> pingpong (to get new normals)
        self.pingpong_fbo.use()
        self.heightmap_texture.use(location=0)

        self.normal_program["u_heightmap"].value = 0  # type: ignore
        self.normal_program["u_texelSize"].value = (
            texel, texel)  # type: ignore

        self.normal_quad.render(moderngl.TRIANGLES)

        # Swap back so heightmap_texture has the final result with normals
        self.heightmap_texture, self.pingpong_texture = self.pingpong_texture, self.heightmap_texture
        self.heightmap_fbo, self.pingpong_fbo = self.pingpong_fbo, self.heightmap_fbo

    # ------------------------------------------------------------------
    # Rendering options
    # ------------------------------------------------------------------
    def render_visualization(
        self,
        water_height: float = 0.45,
        sun_dir: Tuple[float, float, float] = (-1.0, 0.1, 0.25),
        mode: int = 0,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Renders a visualization of the terrain using the erosion_viz shader.

        Args:
            water_height: Height of the water plane (0.0 to 1.0).
            sun_dir: Direction vector of the sun light.
            mode: Visualization mode (0=Standard, 1=Height, 2=Normals,
                  3=Erosion, 4=Slope, 5=Curvature).
            time: Time in seconds for animation (water, clouds).

        Returns:
            A numpy array of shape (resolution, resolution, 3) containing the
            rendered image data (RGB).
        """
        self.viz_fbo.use()
        self.viz_fbo.clear(0.0, 0.0, 0.0, 1.0)

        self.heightmap_texture.use(location=0)
        self.detail_texture.use(location=1)

        program = self.viz_program
        program["u_heightmap"].value = 0  # type: ignore
        program["u_detail"].value = 1  # type: ignore
        program["u_waterHeight"].value = float(water_height)  # type: ignore
        program["u_sunDir"].value = tuple(map(float, sun_dir))  # type: ignore
        program["u_mode"].value = int(mode)  # type: ignore
        if "u_time" in program:
            program["u_time"].value = float(time)  # type: ignore

        # Enable triplanar if mode is 0 (Standard) or maybe add a new mode?
        # For now, let's enable it by default for standard mode if the uniform exists
        if "u_useTriplanar" in program:
            program["u_useTriplanar"].value = 1  # type: ignore

        self.viz_quad.render(moderngl.TRIANGLES)

        raw = self.viz_fbo.read(components=3, dtype="f1")
        data = np.frombuffer(raw, dtype="u1").reshape(
            (self.resolution, self.resolution, 3))
        return np.flip(data, axis=0)

    def render_raymarch(
        self,
        camera_pos: Tuple[float, float, float] = (0.0, 0.8, -1.2),
        look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        water_height: float = 0.45,
        sun_dir: Tuple[float, float, float] = (-1.0, 0.1, 0.25),
        *,
        time: float = 0.0,
        exposure: float = 1.0,
        fog_color: Tuple[float, float, float] = (0.6, 0.7, 0.8),
        fog_density: float = 0.1,
        fog_height: float = 0.35,
        sun_intensity: float = 2.0,
        ray_max_steps: int = 128,
        ray_min_step: float = 0.002,
        shadow_softness: float = 16.0,
        ao_strength: float = 1.0,
    ) -> np.ndarray:
        """
        Renders a high-quality raymarched view of the terrain with advanced
        lighting effects.

        Args:
            camera_pos: Position of the camera in world space.
            look_at: Target point the camera is looking at.
            water_height: Height of the water plane.
            sun_dir: Direction vector of the sun light.
            time: Time in seconds for animation.
            exposure: Exposure multiplier for tone mapping.
            fog_color: Color of the atmospheric fog.
            fog_density: Density of the distance fog.
            fog_height: Height falloff for the height fog.
            sun_intensity: Intensity multiplier for direct sunlight.
            ray_max_steps: Maximum number of raymarching steps.
            ray_min_step: Minimum step size for raymarching.
            shadow_softness: Softness factor for shadows (higher = sharper).
            ao_strength: Strength of ambient occlusion (0.0 to 1.0+).

        Returns:
            A numpy array of shape (resolution, resolution, 3) containing the
            rendered image data (RGB).
        """
        self.viz_fbo.use()
        self.viz_fbo.clear(0.0, 0.0, 0.0, 1.0)

        self.heightmap_texture.use(location=0)
        self.detail_texture.use(location=1)

        program = self.ray_program
        program["u_heightmap"].value = 0  # type: ignore
        program["u_detail"].value = 1  # type: ignore
        program["u_res"].value = (  # type: ignore
            float(self.resolution), float(self.resolution))
        if "u_time" in program:
            program["u_time"].value = float(time)  # type: ignore
        program["u_camPos"].value = tuple(map(float, camera_pos))  # type: ignore
        program["u_lookAt"].value = tuple(map(float, look_at))  # type: ignore
        program["u_waterHeight"].value = float(water_height)  # type: ignore
        program["u_sunDir"].value = tuple(map(float, sun_dir))  # type: ignore
        if "u_exposure" in program:
            program["u_exposure"].value = float(exposure)  # type: ignore
        if "u_fogColor" in program:
            program["u_fogColor"].value = tuple(map(float, fog_color))  # type: ignore
        if "u_fogDensity" in program:
            program["u_fogDensity"].value = float(fog_density)  # type: ignore
        if "u_fogHeight" in program:
            program["u_fogHeight"].value = float(fog_height)  # type: ignore
        if "u_sunIntensity" in program:
            program["u_sunIntensity"].value = float(sun_intensity)  # type: ignore
        if "u_rayMaxSteps" in program:
            program["u_rayMaxSteps"].value = int(ray_max_steps)  # type: ignore
        if "u_rayMinStep" in program:
            program["u_rayMinStep"].value = float(ray_min_step)  # type: ignore
        if "u_shadowSoftness" in program:
            program["u_shadowSoftness"].value = float(shadow_softness)  # type: ignore
        if "u_aoStrength" in program:
            program["u_aoStrength"].value = float(ao_strength)  # type: ignore

        self.ray_quad.render(moderngl.TRIANGLES)

        raw = self.viz_fbo.read(components=3, dtype="f1")
        data = np.frombuffer(raw, dtype="u1").reshape(
            (self.resolution, self.resolution, 3))
        return np.flip(data, axis=0)

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        self.heightmap_fbo.release()
        self.pingpong_fbo.release()
        self.viz_fbo.release()
        self.scatter_fbo.release()
        self.heightmap_texture.release()
        self.pingpong_texture.release()
        self.viz_texture.release()
        self.scatter_texture.release()
        self.vbo.release()
        self.ibo.release()
        self.quad.release()
        self.thermal_quad.release()
        self.normal_quad.release()
        self.viz_quad.release()
        self.ray_quad.release()
        self.scatter_quad.release()
        self.detail_texture.release()
        if self._own_ctx:
            self.ctx.release()

    @staticmethod
    def _uniform_name(field_name: str) -> str:
        """
        Converts a snake_case field name to a GLSL uniform name (camelCase with
        u_ prefix).
        """
        parts = field_name.split("_")
        camel = parts[0] + "".join(word.capitalize() for word in parts[1:])
        return f"u_{camel}" if not camel.startswith("u_") else camel
