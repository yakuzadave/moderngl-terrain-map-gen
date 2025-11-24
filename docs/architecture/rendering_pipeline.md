# Rendering Pipeline Architecture

The `map_gen` project employs a dual-pipeline architecture for rendering terrain, catering to different use cases: **Offline High-Quality Rendering** (CPU/Matplotlib) and **Real-time Preview Rendering** (GPU/GLSL).

## 1. Pipeline Overview

```mermaid
graph TD
    subgraph "Generation (GPU)"
        Gen[Terrain Generator] -->|Compute Shader| Heightmap[Heightmap Texture]
        Gen -->|Compute Shader| Normal[Normal Map Texture]
        Gen -->|Compute Shader| Erosion[Erosion Mask Texture]
        Gen -->|Compute Shader| Scatter[Scatter Map Texture]
    end

    subgraph "Readback (GPU -> CPU)"
        Heightmap -->|fbo.read()| NumPy[NumPy Arrays]
    end

    subgraph "Offline Rendering (CPU)"
        NumPy -->|Matplotlib| LightSource[LightSource Shading]
        LightSource -->|Post-Process| Effects[Bloom/Tone Mapping]
        Effects -->|Save| ImageFile[PNG/JPG]
    end

    subgraph "Real-time Rendering (GPU)"
        Heightmap -->|Texture Bind| VizShader[Visualization Shader]
        Normal -->|Texture Bind| VizShader
        Erosion -->|Texture Bind| VizShader
        VizShader -->|Raymarching| Screen[Screen/FBO]
    end
```

## 2. Offline Rendering (CPU)

Used for high-quality static exports, scientific visualization, and batch processing.

- **Engine**: `matplotlib.colors.LightSource`
- **Input**: NumPy arrays (float32)
- **Features**:
  - Hillshading with custom azimuth/altitude
  - Colormap application (e.g., 'terrain', 'gist_earth')
  - Vertical exaggeration
  - **Post-Processing**:
    - Tonemapping (ACES, Filmic, Reinhard)
    - Bloom
    - Color Grading (LUT-like adjustments)
    - Atmospheric Perspective (CPU-based fog)

**Key Modules**:
- `src/utils/rendering.py`: Core shading logic.
- `src/utils/postprocessing.py`: Image effects.
- `src/utils/advanced_rendering.py`: Turntables and lighting studies.

## 3. Real-time Rendering (GPU)

Used for interactive previews, fast iteration, and 3D visualization.

- **Engine**: ModernGL + GLSL Fragment Shaders
- **Input**: OpenGL Textures (Height, Normal, Erosion)
- **Shaders**:
  - `erosion_viz.frag`: Standard PBR visualization.
  - `erosion_raymarch.frag`: Advanced raymarching with shadows.

### 3.1 Visualization Shader (`erosion_viz.frag`)

Implements a standard forward rendering pass on a fullscreen quad.

- **PBR Material System**:
  - Uses **Tri-planar Mapping** to texture steep slopes without stretching.
  - Blends 4 materials based on height and slope:
    - **Water**: Low height, high smoothness.
    - **Sand**: Low height, moderate roughness.
    - **Grass**: Medium height, flat slopes.
    - **Rock**: Steep slopes.
    - **Snow**: High altitude.
- **Lighting**:
  - Blinn-Phong specular model.
  - Directional sun light.

### 3.2 Raymarching Shader (`erosion_raymarch.frag`)

Implements screen-space raymarching for advanced effects.

- **Technique**: Raymarches through the heightmap texture to find intersections.
- **Soft Shadows**: Raymarches towards the light source from surface points to calculate occlusion.
- **Atmospheric Fog**:
  - **Height Fog**: Density decays exponentially with height ($e^{-h}$).
  - **Mie Scattering**: Simulates sun glare/halo effects.
- **Ambient Occlusion**: Horizon-Based Ambient Occlusion (HBAO) approximation.

## 4. Data Flow

1.  **Generation**: `ErosionTerrainGenerator` runs compute shaders to populate `fbo_height`, `fbo_erosion`, and `fbo_scatter`.
2.  **Normalization**: Height data is typically in `[0, 1]` range.
3.  **Export**:
    - For CPU rendering, `fbo.read()` transfers data to RAM.
    - For GPU rendering, textures are kept on VRAM and bound to the visualization program.

## 5. Design Decisions (ADRs)

- **Fragment Shaders for Compute**: We use fragment shaders rendering to FBOs instead of Compute Shaders for broader compatibility (OpenGL 3.3 vs 4.3) and simplicity in 2D grid operations.
- **Dual Pipeline**: We maintain both CPU and GPU pipelines because Matplotlib offers superior scientific colormaps and publication-quality plots, while GLSL offers real-time performance and PBR fidelity.
