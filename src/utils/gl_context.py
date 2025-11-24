"""Helper utilities for ModernGL context and GPU resources."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import moderngl
import numpy as np

__all__ = ["create_context", "create_detail_texture"]


def create_context(require: int = 330, backend: Optional[str] = "egl") -> moderngl.Context:
    """Create a standalone ModernGL context.

    Falls back to the default backend if EGL is unavailable.
    """
    if backend is None:
        return moderngl.create_standalone_context(require=require)

    try:
        return moderngl.create_standalone_context(require=require, backend=backend)
    except Exception:
        return moderngl.create_standalone_context(require=require)


def create_detail_texture(ctx: moderngl.Context, size: int = 256) -> moderngl.Texture:
    """Create a small tiling noise texture used by the raymarch renderer."""
    from .shader_loader import load_shader

    program = ctx.program(
        vertex_shader=load_shader("quad.vert"),
        fragment_shader=load_shader("detail_noise.frag"),
    )

    texture = ctx.texture((size, size), 3, dtype="f1")
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    fbo = ctx.framebuffer([texture])

    vertices = np.array([
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    ], dtype="f4")
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")

    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())

    vao = ctx.vertex_array(program, [(vbo, "2f", "in_position")], ibo)
    fbo.use()
    vao.render(moderngl.TRIANGLES)

    texture.build_mipmaps()

    # Cleanup
    vbo.release()
    ibo.release()
    vao.release()
    fbo.release()
    program.release()

    return texture
