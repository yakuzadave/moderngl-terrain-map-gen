"""Utility helpers for loading GLSL shader sources from disk."""
from __future__ import annotations

from pathlib import Path

__all__ = ["load_shader"]

_SHADER_ROOT = Path(__file__).resolve().parents[1] / "shaders"


def load_shader(filename: str) -> str:
    """Return the contents of a shader file located under ``src/shaders``."""
    shader_path = _SHADER_ROOT / filename
    if not shader_path.exists():
        raise FileNotFoundError(
            f"Shader '{filename}' not found under {_SHADER_ROOT}")
    return shader_path.read_text(encoding="utf-8")
