# How to Preview Shaders

This guide explains how to use the `viz_shader.py` tool to preview GLSL shaders using Desktop OpenGL (3.3+). This tool is useful for verifying shaders that are not compatible with WebGL-based extensions (like `vscode-glsl-canvas`).

## Prerequisites

Ensure you have the project dependencies installed:

```bash
pip install -r requirements.txt
```

This tool relies on `moderngl-window`.

## Usage

Run the script from the command line, providing the path to your fragment shader:

```bash
python viz_shader.py path/to/shader.frag
```

### Optional Arguments

- `--texture`: Path to an input texture (e.g., a heightmap or noise texture).

```bash
python viz_shader.py src/shaders/erosion_viz.frag --texture my_heightmap.png
```

## Controls

- **R**: Reload the shader (useful for iterating on code without restarting the viewer).
- **Mouse**: Updates the `u_mouse` uniform.

## Standard Uniforms

The viewer automatically provides the following uniforms if they are declared in your shader:

- `uniform float u_time;` - Time in seconds since start.
- `uniform vec2 u_resolution;` - Window resolution in pixels.
- `uniform vec2 u_mouse;` - Mouse position (pixel coordinates).
- `uniform sampler2D u_texture_0;` - The texture provided via `--texture`.

## Example: Verifying Seamless Tiling

To check if a generated texture is seamless, use the provided `viz_seamless_desktop.glsl` shader:

1. Generate a seamless heightmap:
   ```bash
   python gpu_terrain.py --resolution 512 --seamless --heightmap-out test_seamless.png
   ```

2. Run the viewer with the seamless check shader:
   ```bash
   python viz_shader.py viz_seamless_desktop.glsl --texture test_seamless.png
   ```

The shader tiles the texture 2x2. If you see hard edges in the middle of the window, the texture is not seamless.
