# Interactive Visualization Guide

This guide covers the various ways to view generated terrain beyond static image files.

## Current Visualization Options

### 1. Streamlit Web UI (`app/ui_streamlit.py`)

**Best for:** Quick parameter exploration with immediate visual feedback.

```bash
streamlit run app/ui_streamlit.py
```

**Features:**
- Real-time preview of terrain generation
- Adjustable lighting parameters (azimuth, altitude, intensity)
- Multiple colormap options
- Export to PNG, OBJ, Normal Maps
- Preset management (save/load configurations)
- Debug visualization modes (height, normals, erosion, slope)

**Limitations:**
- 2D rendering only (shaded relief)
- Single static image at a time
- Requires Streamlit server running

---

### 2. Marimo Reactive Notebook (`terrain_gen_marimo.py`)

**Best for:** Interactive exploration with 2D/3D toggle, data export.

```bash
marimo run terrain_gen_marimo.py
# Or for editing:
marimo edit terrain_gen_marimo.py
```

**Features:**
- 2D Shaded Relief view with LightSource
- **3D Raymarch view** with camera controls (yaw, pitch, distance, FOV)
- Generator selection (Hydraulic vs Morphological)
- Parameter sliders with live updates
- Direct download buttons (PNG, OBJ, NPY)

**Limitations:**
- Regenerates on every slider change (no explicit "Generate" gating)
- 3D view is static (no interactive rotation)

---

### 3. GPU Shader Preview (`scripts/viz_shader.py`)

**Best for:** Live shader development and debugging.

```bash
python scripts/viz_shader.py src/shaders/erosion_viz.frag --texture output/terrain.png
```

**Features:**
- Real-time GLSL shader preview using `moderngl-window`
- Hot-reload shaders with 'R' key
- Mouse position uniform for interaction
- Time-based animations

**Limitations:**
- Requires writing/modifying shaders
- More developer-focused than end-user tool

---

### 4. VS Code 3D Viewer Extension (Already Installed)

**Best for:** Viewing exported meshes without leaving the editor.

**Supported formats:** `.obj`, `.stl`, `.gltf`, `.glb`, `.fbx`, `.dae`, `.3ds`

**Usage:**
1. Generate terrain with `--obj-out terrain.obj` or `--gltf-out terrain.gltf`
2. Right-click the file in VS Code → "3D Viewer: Open Model"
3. Rotate, zoom, and inspect the mesh interactively

**Workflow:**
```bash
python gpu_terrain.py --preset canyon --obj-out output/canyon.obj
# Then open output/canyon.obj in VS Code
```

---

### 5. Turntable Animation Export

**Best for:** Creating preview videos/GIFs for sharing.

```python
from src.utils.advanced_rendering import save_turntable_video

save_turntable_video(
    "turntable.mp4",
    terrain,
    frames=60,
    fps=30,
    altitude=50.0,
    colormap="terrain"
)
```

**CLI:**
```bash
python gpu_terrain.py --preset canyon --turntable-out turntable.mp4 --turntable-frames 60
```

---

## Recommended Visualization Paths

### Quick Preview Workflow
1. Use **Streamlit UI** for rapid iteration
2. Adjust parameters and see 2D shaded relief instantly
3. Export OBJ when satisfied
4. View 3D mesh in **VS Code 3D Viewer**

### 3D Exploration Workflow
1. Use **Marimo notebook** with "3D Raymarch" mode
2. Adjust camera yaw/pitch/distance
3. Export GLTF for external 3D software

### Professional Output Workflow
1. Generate terrain via CLI or Python API
2. Export **GLTF** for game engines (Unity, Unreal, Godot)
3. Export **PNG + Normal maps** for texturing
4. Create **turntable video** for presentations

---

## Future Enhancement Options

### Option A: Integrated 3D Viewer (pyvista)

**Pros:**
- Mature, stable VTK-based visualization
- Excellent Jupyter/notebook integration
- Interactive rotation, zoom, lighting
- Surface mesh rendering with colormaps
- Export to various formats

**Cons:**
- Large dependency (VTK)
- Desktop-focused (less web-friendly)

**Example Integration:**
```python
import pyvista as pv
import numpy as np

# Create mesh from heightmap
def terrain_to_pyvista(terrain, scale=10.0, height_scale=2.0):
    h, w = terrain.height.shape
    x = np.linspace(0, scale, w)
    z = np.linspace(0, scale, h)
    xx, zz = np.meshgrid(x, z)
    yy = terrain.height * height_scale
    
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["height"] = terrain.height.flatten()
    return grid

# Interactive display
plotter = pv.Plotter()
mesh = terrain_to_pyvista(terrain)
plotter.add_mesh(mesh, scalars="height", cmap="terrain")
plotter.show()
```

### Option B: WebGPU Viewer (pygfx)

**Pros:**
- Modern WebGPU backend (future-proof)
- Jupyter integration
- Scene-graph API (cameras, lights, meshes)
- Cross-platform (desktop + web)

**Cons:**
- Still maturing (v0.15, targeting v1.0 mid-2026)
- Requires Python 3.10+
- Smaller community than VTK

### Option C: Web-Based Viewer (three.js via Panel/Streamlit)

**Pros:**
- Full 3D interactivity in browser
- No additional Python dependencies
- Works with existing GLTF exports

**Cons:**
- Requires JavaScript/WebGL knowledge
- More complex integration

### Option D: Enhanced Marimo with Anywidget

**Pros:**
- Stays within current tech stack
- Reactive notebooks already working
- Can embed three.js via anywidget

**Cons:**
- Requires custom widget development

---

## Summary Table

| Method | Interactive | 3D View | Setup Complexity | Best For |
|--------|-------------|---------|------------------|----------|
| Streamlit UI | ✅ | ❌ | Low | Quick 2D preview |
| Marimo Notebook | ✅ | ✅ (raymarch) | Low | 2D/3D exploration |
| VS Code 3D Viewer | ✅ | ✅ | None | Mesh inspection |
| Shader Preview | ✅ | Limited | Medium | Shader dev |
| Turntable Export | ❌ | ✅ | Low | Sharing |
| pyvista (future) | ✅ | ✅ | Medium | Scientific viz |
| pygfx (future) | ✅ | ✅ | Medium | Modern WebGPU |

---

## See Also

- [ADVANCED_RENDERING.md](../ADVANCED_RENDERING.md) - Turntable and lighting studies
- [EXPORT_FORMATS.md](../EXPORT_FORMATS.md) - Supported export formats
- [api-reference.md](../api-reference.md) - Python API for rendering
