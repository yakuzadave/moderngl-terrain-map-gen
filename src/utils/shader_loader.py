"""Utility helpers for loading GLSL shader sources from disk."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["load_shader"]


def load_shader(filename: str) -> str:
    """
    Return the contents of a shader file located under ``src/shaders``.

    Uses importlib.resources to locate the file within the 'src' package.
    """
    try:
        # Locate the 'src' package resources
        src_pkg = files("src")
        # Navigate to 'shaders' subdirectory
        shader_path = src_pkg.joinpath("shaders").joinpath(filename)

        if not shader_path.is_file():
            raise FileNotFoundError(
                f"Shader '{filename}' not found in src/shaders")

        return shader_path.read_text(encoding="utf-8")

    except (ImportError, ModuleNotFoundError):
        # Fallback for when 'src' is not installed as a package
        # (e.g. running a script inside src/ without package context)
        root = Path(__file__).resolve().parents[1] / "shaders"
        shader_path = root / filename
        if not shader_path.exists():
            raise FileNotFoundError(
                f"Shader '{filename}' not found at {root}")
        return shader_path.read_text(encoding="utf-8")
