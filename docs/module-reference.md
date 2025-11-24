# Module Reference

Detailed reference for all Python modules in the GPU Terrain Generator.

## Package Structure

```
src/
├── __init__.py           # Main package exports
├── config.py             # Configuration models
├── generators/           # Terrain generation algorithms
│   ├── __init__.py
│   ├── erosion.py        # Erosion-based generator
│   ├── hydraulic.py      # Hydraulic erosion simulator
│   └── morphological.py  # Morphological generator
├── shaders/              # GLSL shader files
│   ├── quad.vert
│   ├── erosion_heightmap.frag
│   ├── erosion_viz.frag
│   ├── hydraulic/*.frag
│   └── ...
└── utils/                # Utility modules
    ├── __init__.py
    ├── advanced_rendering.py
    ├── artifacts.py
    ├── batch.py
    ├── export.py
    ├── gl_context.py
    ├── postprocessing.py
    ├── render_configs.py
    ├── rendering.py
    ├── shader_loader.py
    ├── textures.py
    └── visualization.py
```

---

## Core Package: `src`

**Module:** `src/__init__.py`

**Purpose:** Main package entry point, exports public API.

**Public Exports:**
```python
from src import (
    # Generators
    ErosionTerrainGenerator,
    ErosionParams,
    MorphologicalTerrainGPU,
    MorphologicalParams,
    HydraulicErosionGenerator,
    HydraulicParams,
    
    # Data structures
    TerrainMaps,
    TerrainConfig,
    RenderConfig,
    
    # Utilities namespace
    utils,
)
```

---

## Configuration: `src.config`

**Module:** `src/config.py`

**Purpose:** Unified configuration system for terrain generation and rendering.

### Classes

#### `TerrainConfig`
Comprehensive configuration dataclass combining generation, rendering, and export settings.

**Key Methods:**
- `get_erosion_params() -> ErosionParams` - Extract erosion parameters
- `get_hydraulic_params() -> HydraulicParams` - Extract hydraulic parameters
- `get_render_config() -> RenderConfig` - Extract render configuration
- `to_dict() -> dict` - Serialize to dictionary
- `from_dict(data: dict) -> TerrainConfig` - Deserialize from dictionary

### Functions

#### `load_config(path: str | Path) -> TerrainConfig`
Load configuration from YAML or JSON file.

**Supported formats:** `.yaml`, `.yml`, `.json`

#### `save_config(config: TerrainConfig, path: str | Path) -> Path`
Save configuration to YAML or JSON file.

**Auto-detects format from extension.**

---

## Generators Package: `src.generators`

### Module: `src.generators.erosion`

**Purpose:** GPU-accelerated erosion-based terrain generation using fractal noise.

#### Classes

**`ErosionParams`** - Configuration dataclass for erosion parameters
- Presets: `canyon()`, `plains()`, `mountains()`, `natural()`
- Custom parameter modification support

**`ErosionTerrainGenerator`** - Main terrain generator
- GPU shader-based generation
- Real-time heightmap creation
- Support for seamless/tileable terrain

**Key Shaders Used:**
- `quad.vert` - Fullscreen quad vertex shader
- `erosion_heightmap.frag` - Height + erosion computation
- `erosion_viz.frag` - PBR visualization (optional)

**Uniform Naming Convention:**
Python `snake_case` → GLSL `u_camelCase`
- `height_tiles` → `u_heightTiles`
- `erosion_slope_strength` → `u_erosionSlopeStrength`

---

### Module: `src.generators.hydraulic`

**Purpose:** Physically-based hydraulic erosion simulation using pipe model.

#### Classes

**`HydraulicParams`** - Simulation parameters
- Physical constants (Kc, Ks, Kd, Ke)
- Iteration and timestep control
- Thermal erosion parameters

**`HydraulicErosionGenerator`** - Multi-stage erosion simulator

**Simulation Pipeline:**
1. Flux computation (water movement between cells)
2. Water velocity & depth update
3. Erosion & deposition (sediment transport)
4. Sediment advection (movement with flow)
5. Evaporation
6. Thermal erosion (slope stabilization)

**Key Shaders:**
- `hydraulic/flux.frag` - Water flux calculation
- `hydraulic/water_velocity.frag` - Velocity update
- `hydraulic/erosion_deposition.frag` - Sediment transport
- `hydraulic/sediment_advection.frag` - Sediment movement
- `hydraulic/evaporation.frag` - Water evaporation
- `hydraulic/thermal_erosion.frag` - Thermal weathering

**Texture Resources:**
- Ping-pong buffers for: height, water, sediment, flux
- Float32 RGBA textures for multi-channel data

---

### Module: `src.generators.morphological`

**Purpose:** Alternative terrain generation using Voronoi patterns and morphological operations.

#### Classes

**`MorphologicalParams`** - Generation parameters
- Noise scale and octave control
- Morphological operation radius and strength

**`MorphologicalTerrainGPU`** - Voronoi-based generator

**Key Shaders:**
- `morph_noise.frag` - Voronoi noise generation
- `morph_erosion.frag` - Distance field operations

**Characteristics:**
- Faster than erosion generator
- Different aesthetic (more cellular/organic)
- Simpler parameter space

---

## Utilities Package: `src.utils`

### Module: `src.utils.artifacts`

**Purpose:** Core data structures for terrain representation.

#### Classes

**`TerrainMaps`** - Primary terrain data container

**Attributes:**
- `height: np.ndarray` - Float32 heightmap (HxW)
- `normals: np.ndarray` - Float32 normal vectors (HxWx3)
- `erosion_mask: np.ndarray | None` - Erosion intensity
- `scatter_map: np.ndarray | None` - Vegetation scatter data

**Methods:**
- `ensure(data)` - Convert dict/other formats to TerrainMaps
- `resolution` - Property returning (height, width)
- `as_dict()` - Convert to dictionary
- `height_u16()` - Height as uint16
- `normal_map_u8()` - Normals as RGB uint8
- `erosion_mask_u8()` - Erosion as uint8
- `scatter_map_u8()` - Scatter as RGB uint8

---

### Module: `src.utils.export`

**Purpose:** File I/O for terrain data in various formats.

**Functions:**

**Heightmap Exports:**
- `save_heightmap_png()` - 16-bit PNG
- `save_heightmap_raw()` - 16-bit binary
- `save_heightmap_r32()` - 32-bit float binary

**Map Exports:**
- `save_normal_map_png()` - RGB normal map
- `save_erosion_mask_png()` - Grayscale erosion mask

**Bundle Export:**
- `save_npz_bundle()` - Compressed NumPy archive

**Mesh Exports:**
- `export_obj_mesh()` - Wavefront OBJ with UVs
- `export_stl_mesh()` - Binary STL (3D printing)
- `export_gltf_mesh()` - glTF 2.0 (requires pygltflib)

**Batch Export:**
- `export_all_formats()` - Export multiple formats at once

---

### Module: `src.utils.rendering`

**Purpose:** CPU-based rendering using matplotlib's LightSource.

**Functions:**

**Shaded Relief:**
- `shade_heightmap()` - Generate shaded relief RGB array
- `save_shaded_relief_png()` - Render and save to PNG

**Slope Analysis:**
- `slope_intensity()` - Calculate slope magnitude
- `save_slope_map_png()` - Export slope map

**Key Features:**
- Resolution-independent shading
- Multiple colormap support
- Configurable lighting (azimuth, altitude)
- Vertical exaggeration
- Blend modes: soft, overlay, hsv

---

### Module: `src.utils.textures`

**Purpose:** Game engine texture generation and packing.

**Functions:**

**Splatmaps:**
- `save_splatmap_rgba()` - 4-channel texture blending weights

**Procedural Textures:**
- `save_ao_map()` - Ambient occlusion
- `save_curvature_map()` - Surface curvature
- `save_scatter_map()` - Vegetation scatter density

**Packed Textures:**
- `save_packed_texture()` - Multi-channel packed formats
  - `unity_mask` mode: (Metallic, AO, Detail, Smoothness)
  - `ue_orm` mode: (AO, Roughness, Metallic)
  - `custom` mode: User-defined channel mapping

**Target Engines:**
- Unity HDRP/URP
- Unreal Engine
- Godot
- Custom engines

---

### Module: `src.utils.advanced_rendering`

**Purpose:** Advanced visualization and animation generation.

**Functions:**

**Animations:**
- `render_turntable_frames()` - Generate rotation sequence
- `save_turntable_video()` - Save as MP4/GIF
- `save_animation_sequence()` - Export frame sequence

**Multi-View:**
- `render_multi_angle()` - Multiple lighting directions
- `render_lighting_study()` - Grid of lighting variations
- `create_comparison_grid()` - Side-by-side terrain comparison

**Output Formats:**
- MP4 (requires ffmpeg)
- Animated GIF
- PNG sequence

---

### Module: `src.utils.batch`

**Purpose:** Batch terrain generation and processing.

#### Classes

**`BatchGenerator`** - Manages batch generation with parameter variation

**Methods:**
- `generate_set()` - Generate multiple terrains with different seeds
- `cleanup()` - Release GPU resources

**Features:**
- Progress callbacks
- Flexible export format control
- Memory-efficient (reuses GPU context)
- Parallel-safe (one generator per thread)

#### Functions

**`generate_terrain_set()`** - Convenience wrapper for common batch operations

---

### Module: `src.utils.gl_context`

**Purpose:** ModernGL context management and helpers.

**Functions:**

**`create_context() -> moderngl.Context`**
- Creates standalone ModernGL context
- Handles platform-specific initialization
- Validates OpenGL version

**`create_detail_texture(ctx, data, ...) -> moderngl.Texture`**
- Helper for texture creation with common defaults
- Handles filtering and wrapping modes

---

### Module: `src.utils.shader_loader`

**Purpose:** GLSL shader file loading.

**Functions:**

**`load_shader(name: str) -> str`**
- Loads shader source from `src/shaders/` directory
- Handles path resolution
- Provides clear error messages

**Shader Organization:**
- `quad.vert` - Standard fullscreen quad
- `erosion_*.frag` - Erosion generator shaders
- `hydraulic/*.frag` - Hydraulic simulation shaders
- `morph_*.frag` - Morphological generator shaders

---

### Module: `src.utils.render_configs`

**Purpose:** Rendering configuration presets.

#### Classes

**`RenderConfig`** - Rendering parameters dataclass
- Lighting configuration
- Color and tone mapping
- Post-processing settings

**Constants:**

**`PRESET_CONFIGS: dict[str, RenderConfig]`**
- `"classic"` - Standard visualization
- `"dramatic"` - High contrast, low sun
- `"soft"` - Diffuse lighting
- `"topographic"` - Map-style rendering

---

### Module: `src.utils.visualization`

**Purpose:** Legacy matplotlib-based visualization utilities.

**Functions:**

**`plot_terrain_panels()`** - Multi-panel terrain analysis plot
**`create_turntable_animation()`** - Legacy turntable generation
**`save_panel_overview()`** - Save analysis overview
**`save_turntable_gif()`** - Save turntable as GIF

**Note:** Most functionality superseded by `advanced_rendering` module.

---

### Module: `src.utils.postprocessing`

**Purpose:** Image post-processing effects.

**Features:**
- Tone mapping
- Gamma correction
- Contrast and saturation adjustment
- Color grading

**Integration:**
Used internally by rendering pipeline for final image output.

---

## Shader Files: `src/shaders`

### Vertex Shaders

**`quad.vert`**
Standard fullscreen quad vertex shader. Maps 0-1 vertex positions to NDC -1 to 1.

```glsl
in vec2 in_position;
out vec2 v_uv;

void main() {
    v_uv = in_position;
    gl_Position = vec4(in_position * 2.0 - 1.0, 0.0, 1.0);
}
```

---

### Erosion Shaders

**`erosion_heightmap.frag`**
- Main terrain generation shader
- Fractal noise (FBM/ridge)
- Domain warping
- Procedural erosion simulation

**Outputs:** `vec4(height, normal.x, normal.z, erosion)`

**`erosion_viz.frag`**
- PBR visualization
- Material-based coloring
- Debug modes (height, normals, erosion, slope)

---

### Hydraulic Shaders

**Pipeline stages:**

1. **`flux.frag`** - Compute water flux between cells using pipe model
2. **`water_velocity.frag`** - Update water depth and velocity from flux
3. **`erosion_deposition.frag`** - Sediment transport based on velocity
4. **`sediment_advection.frag`** - Move sediment with water flow
5. **`evaporation.frag`** - Evaporate water over time
6. **`thermal_erosion.frag`** - Thermal weathering (talus deposition)

**Physics Implementation:**
Based on academic papers on hydraulic erosion simulation.

---

### Morphological Shaders

**`morph_noise.frag`**
- Voronoi cellular noise
- Multi-octave fractal
- Configurable scale and persistence

**`morph_erosion.frag`**
- Distance field operations
- Morphological dilation/erosion
- Radius-based smoothing

---

## Testing Modules: `tests/`

**`tests/conftest.py`** - Pytest fixtures
**`tests/test_generators.py`** - Generator tests
**`tests/test_configs.py`** - Configuration tests
**`tests/test_rendering.py`** - Rendering tests
**`tests/test_hydraulic_integration.py`** - Hydraulic erosion tests

---

## Examples: `examples/`

**`advanced_rendering_demo.py`** - Showcase advanced rendering features
**`export_formats_demo.py`** - Demonstrate all export formats
**`demo_advanced_features.py`** - General feature demonstration

---

## Scripts: `scripts/`

**`generate_sample_renders.py`** - Batch render generation
**`verify_seamless.py`** - Verify seamless terrain tiling
**`inspect_render_quality.py`** - Quality analysis tools
**`check_gl_version.py`** - OpenGL version check utility
**`viz_shader.py`** - Shader visualization tool

---

## Application: `app/`

**`ui_streamlit.py`** - Streamlit web UI
- Interactive parameter controls
- Real-time preview
- Export management
- Preset management

---

## Tools: `tools/`

**`build_knowledge_graph.py`** - Generate codebase knowledge graph for documentation

---

## Module Dependencies

### Core Dependencies
- **moderngl** - OpenGL rendering
- **numpy** - Array operations
- **PIL** (Pillow) - Image I/O
- **matplotlib** - Visualization and colormaps

### Optional Dependencies
- **pygltflib** - glTF export (install separately)
- **ffmpeg** - MP4 video encoding (system install)
- **imageio** - Animation support
- **streamlit** - Web UI (for `app/ui_streamlit.py`)

---

## Import Patterns

### Recommended Import Style

```python
# Main generators
from src import (
    ErosionTerrainGenerator,
    ErosionParams,
    HydraulicErosionGenerator,
    HydraulicParams,
)

# Utilities namespace
from src import utils

# Specific utility functions
from src.utils import (
    save_heightmap_png,
    export_obj_mesh,
    shade_heightmap,
)

# Configuration
from src import TerrainConfig, load_config, save_config
```

### Avoid Direct Imports

```python
# DON'T do this (internal modules may change)
from src.generators.erosion import _uniform_name
from src.utils.gl_context import _create_standalone_context

# DO use public API instead
from src import ErosionTerrainGenerator
```

---

## Extending the Codebase

### Adding a New Generator

1. Create module in `src/generators/`
2. Implement generator class with `generate()` method
3. Return `TerrainMaps` instance
4. Implement `cleanup()` method
5. Export from `src/generators/__init__.py`
6. Export from `src/__init__.py`
7. Add to `gpu_terrain.py` CLI options

### Adding a New Export Format

1. Add function to `src/utils/export.py`
2. Follow naming convention: `export_<format>_<type>()`
3. Accept `TerrainMaps` or compatible input
4. Return `Path` to saved file
5. Export from `src/utils/__init__.py`
6. Document in `docs/EXPORT_FORMATS.md`

### Adding a New Shader

1. Create `.frag` or `.vert` file in `src/shaders/`
2. Use standard naming: `<category>_<purpose>.frag`
3. Document uniforms and outputs
4. Load via `load_shader("filename.frag")`
5. Follow uniform naming: `u_camelCase`

---

## See Also

- [API Reference](api-reference.md) - Complete API documentation
- [Quick Reference](quick-reference.md) - Common code patterns
- [Architecture Overview](architecture/index.md) - System design
- [Contributing Guidelines](../README.md#contributing)
