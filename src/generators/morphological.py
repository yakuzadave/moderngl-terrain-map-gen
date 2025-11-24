"""ModernGL morphological terrain generator."""
from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from ..utils import TerrainMaps, create_context, load_shader

__all__ = ["MorphologicalTerrainGPU"]


@dataclass
class MorphologicalParams:
    scale: float = 5.0
    octaves: int = 8
    persistence: float = 0.5
    lacunarity: float = 2.0
    radius: int = 2
    strength: float = 0.5


class MorphologicalTerrainGPU:
    def __init__(self, ctx: moderngl.Context | None = None) -> None:
        self.ctx = ctx or create_context()
        self._own_ctx = ctx is None

        shader = load_shader("quad.vert")
        self.noise_prog = self.ctx.program(
            vertex_shader=shader, fragment_shader=load_shader("morph_noise.frag"))
        self.erosion_prog = self.ctx.program(
            vertex_shader=shader, fragment_shader=load_shader("morph_erosion.frag"))

        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ], dtype="f4")
        indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        layout = [(self.vbo, "2f", "in_position")]
        self.quad_noise = self.ctx.vertex_array(
            self.noise_prog, layout, self.ibo)
        self.quad_erosion = self.ctx.vertex_array(
            self.erosion_prog, layout, self.ibo)

    def generate(
        self,
        resolution: int = 512,
        seed: int = 42,
        params: MorphologicalParams | None = None,
    ) -> TerrainMaps:
        resolution = int(resolution)
        params = params or MorphologicalParams()

        tex_noise = self.ctx.texture((resolution, resolution), 1, dtype="f4")
        tex_final = self.ctx.texture((resolution, resolution), 1, dtype="f4")
        fbo_noise = self.ctx.framebuffer([tex_noise])
        fbo_final = self.ctx.framebuffer([tex_final])

        fbo_noise.use()
        # pyright: ignore[reportAttributeAccessIssue]
        self.noise_prog["u_seed"].value = float(seed)
        # pyright: ignore[reportAttributeAccessIssue]
        self.noise_prog["u_scale"].value = float(params.scale)
        # pyright: ignore[reportAttributeAccessIssue]
        self.noise_prog["u_octaves"].value = int(params.octaves)
        # pyright: ignore[reportAttributeAccessIssue]
        self.noise_prog["u_persistence"].value = float(params.persistence)
        # pyright: ignore[reportAttributeAccessIssue]
        self.noise_prog["u_lacunarity"].value = float(params.lacunarity)
        self.quad_noise.render(moderngl.TRIANGLES)

        fbo_final.use()
        tex_noise.use(location=0)
        self.erosion_prog["u_texture"].value = 0
        self.erosion_prog["u_resolution"].value = (
            float(resolution), float(resolution))
        self.erosion_prog["u_radius"].value = float(params.radius)
        self.erosion_prog["u_strength"].value = float(params.strength)
        self.quad_erosion.render(moderngl.TRIANGLES)

        base = np.flipud(np.frombuffer(fbo_noise.read(
            components=1, dtype="f4"), dtype="f4").reshape((resolution, resolution)))
        eroded = np.flipud(np.frombuffer(fbo_final.read(
            components=1, dtype="f4"), dtype="f4").reshape((resolution, resolution)))

        dy, dx = np.gradient(eroded)
        normal_strength = 50.0 / resolution
        nx = -dx * normal_strength
        nz = -dy * normal_strength
        ny = np.ones_like(nx)
        length = np.sqrt(nx**2 + ny**2 + nz**2)
        normals = np.dstack(
            (nx / length, ny / length, nz / length)).astype("f4")

        erosion_mask = np.clip((base - eroded) * 10.0, 0.0, 1.0).astype("f4")

        tex_noise.release()
        tex_final.release()
        fbo_noise.release()
        fbo_final.release()

        return TerrainMaps(height=eroded.astype("f4"), normals=normals, erosion_mask=erosion_mask)

    def cleanup(self) -> None:
        self.vbo.release()
        self.ibo.release()
        self.quad_noise.release()
        self.quad_erosion.release()
        self.noise_prog.release()
        self.erosion_prog.release()
        if self._own_ctx:
            self.ctx.release()

    def __enter__(self) -> "MorphologicalTerrainGPU":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
