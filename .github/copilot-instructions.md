# GPU Terrain Generator - AI Agent Guide

## Architecture Overview

This is a **ModernGL-based GPU terrain generator** using GLSL compute shaders for real-time heightmap generation. The architecture has two primary paths:

1. **GPU Generation Pipeline**: GLSL shaders (`src/shaders/*.frag`) → ModernGL context → NumPy arrays → Export utilities
2. **CPU Rendering Pipeline**: NumPy heightmaps → Matplotlib LightSource → PIL images

**Key Components**:

- `src/generators/erosion.py`: Primary generator using fractal noise + hydraulic erosion simulation (GPU)
- `src/generators/morphological.py`: Alternative generator using Voronoi + distance fields (GPU)
- `src/utils/`: Export (PNG/OBJ/STL), rendering (shaded relief), textures (splatmaps/AO), batch generation
- `gpu_terrain.py`: CLI entry point with argparse
- `terrain_gen.ipynb`: Experimental notebook (references src/ modules, not production code)

## Critical Workflows

### Development Environment Setup

```bash
# Always use the virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Testing Changes

```bash
# Quick validation (256x256, ~0.05s)
python gpu_terrain.py --resolution 256 --preset canyon --shaded-out test.png

# Test with all generators
python gpu_terrain.py --generator erosion --shaded-out erosion.png
python gpu_terrain.py --generator morph --shaded-out morph.png

# Validate exports work
python gpu_terrain.py --heightmap-out h.png --obj-out m.obj --normals-out n.png
```

### Running Demo Scripts

```bash
# Test all advanced rendering features
python examples/advanced_rendering_demo.py

# Run automated test suite
python test_improvements.py
```

## Project-Specific Conventions

### Type Hints with ModernGL

ModernGL has incomplete type stubs. When assigning uniforms, **always use `# type: ignore`**:

```python
# CORRECT - Pylance will complain but this is the right way
program["u_seed"].value = float(seed)  # type: ignore
program["u_texelSize"].value = (texel, texel)  # type: ignore

# Type stubs are incomplete for: program[name].value, texture.use(), imageio.Writer
```

This pattern appears in `erosion.py` - preserve these `# type: ignore` comments when editing.

### Shader Loading Pattern

Shaders are loaded from `src/shaders/` using a centralized loader:

```python
from ..utils import load_shader
program = ctx.program(
    vertex_shader=load_shader("quad.vert"),
    fragment_shader=load_shader("erosion_heightmap.frag")
)
```

Never hardcode shader paths - always use `load_shader()`.

### GLSL Shader Structure & Uniform Naming

**Shader Pipeline**: All shaders use a simple fullscreen quad pattern:

1. `quad.vert` - Generic vertex shader (maps 0-1 positions to NDC -1 to 1)
2. Fragment shader - Does all the work using UV coordinates

**Available Shaders**:

- `erosion_heightmap.frag` - Main terrain generation (outputs RGBA: height, normalXZ, erosion)
- `erosion_viz.frag` - PBR visualization with materials
- `erosion_raymarch.frag` - 3D raymarch rendering
- `morph_noise.frag` - Fractal noise for morphological generator
- `morph_erosion.frag` - Voronoi-based morphological terrain

**Uniform Naming Convention**: Python snake*case → GLSL camelCase with `u*` prefix

```python
# Python (ErosionParams dataclass)
height_tiles: float = 3.0
height_octaves: int = 3
erosion_slope_strength: float = 3.0

# GLSL (erosion_heightmap.frag)
uniform float u_heightTiles;
uniform int u_heightOctaves;
uniform float u_erosionSlopeStrength;
```

The conversion is automatic via `_uniform_name()` in `erosion.py`:

```python
# "height_tiles" → "u_HeightTiles"
# Parts split by underscore, capitalize each, add u_ prefix
```

**Standard Uniforms** (erosion_heightmap.frag):

- `u_seed` (float) - Random seed offset
- `u_texelSize` (vec2) - 1/resolution for finite differences
- `useErosion` (int) - Boolean flag (0 or 1)
- Height params: `u_heightTiles`, `u_heightOctaves`, `u_heightAmp`, `u_heightGain`, `u_heightLacunarity`
- Erosion params: `u_erosionTiles`, `u_erosionOctaves`, `u_erosionGain`, `u_erosionLacunarity`, `u_erosionSlopeStrength`, `u_erosionBranchStrength`, `u_erosionStrength`
- Water: `u_waterHeight`

**Visualization Uniforms** (erosion_viz.frag):

- `u_heightmap` (sampler2D) - Input heightmap texture
- `u_waterHeight` (float) - Water level threshold
- `u_sunDir` (vec3) - Sun direction for lighting
- `u_mode` (int) - Debug visualization mode (0=full, 1=height, 2=normals, 3=erosion, 4=slope)

**Output Channels** (erosion_heightmap.frag):

```glsl
fragColor = vec4(height, normal.x, normal.z, erosion);
// R: Height value (0-1 range)
// G: Normal X component
// B: Normal Z component
// A: Erosion mask intensity
// Note: Normal Y is reconstructed from X/Z in downstream code
```

**When Adding New Uniforms**:

1. Add to Python dataclass (snake_case)
2. Add to shader (camelCase with u\_ prefix)
3. Test that `_uniform_name()` conversion matches
4. Set value in generator with `# type: ignore` comment

Example:

```python
# 1. Add to ErosionParams
@dataclass
class ErosionParams:
    my_new_param: float = 1.5

# 2. Add to erosion_heightmap.frag
uniform float u_myNewParam;

# 3. Set in generator
program["u_myNewParam"].value = float(params.my_new_param)  # type: ignore
```

### TerrainMaps Dataclass

All terrain data flows through `TerrainMaps` (defined in `src/utils/artifacts.py`):

```python
@dataclass
class TerrainMaps:
    height: np.ndarray          # Main heightmap (float32)
    normals: np.ndarray | None  # RGB normals (Nx×Ny×3)
    erosion_mask: np.ndarray | None  # Erosion intensity
    resolution: int

    @classmethod
    def ensure(cls, terrain) -> TerrainMaps:
        # Converts dicts/other formats to TerrainMaps
```

Always use `TerrainMaps.ensure(terrain)` when accepting flexible terrain inputs.

### Matplotlib Colormap Access

**NEVER** use `plt.cm.terrain` directly (deprecated). Always use:

```python
from matplotlib import cm
cmap = cm.get_cmap("terrain")  # CORRECT
```

This was a major bug fix - maintain this pattern throughout.

### Resource Cleanup Pattern

ModernGL objects must be explicitly released:

```python
try:
    gen = ErosionTerrainGenerator(resolution=512)
    terrain = gen.generate_heightmap(seed=42)
    # ... use terrain ...
finally:
    gen.cleanup()  # Always cleanup in finally block
```

All generator classes have `.cleanup()` methods that release GPU resources.

## Generator Presets

The erosion generator has three built-in presets in `ErosionParams`:

```python
# Deep erosion with branching valleys
ErosionParams.canyon()

# Gentle rolling hills
ErosionParams.plains()

# Sharp peaks with moderate erosion
ErosionParams.mountains()
```

When adding new presets, follow this pattern in `src/generators/erosion.py`:

1. Add `@classmethod` to `ErosionParams`
2. Update CLI in `gpu_terrain.py` (add to `--preset` choices)
3. Add conditional in `main()` to instantiate the preset

## Export Formats & Game Engine Integration

### Export Format Patterns

Export functions follow consistent naming:

- `save_heightmap_png()` - 16-bit grayscale PNG (full precision)
- `save_normal_map_png()` - RGB tangent-space normals
- `export_obj_mesh()` - Wavefront OBJ with UV coordinates
- `export_stl_mesh()` - Binary STL (3D printing)
- `save_splatmap_rgba()` - RGBA texture blending weights
- `save_packed_texture()` - Multi-channel packed textures

### Game Engine Packed Textures

Packed textures combine multiple maps into RGB(A) channels:

```python
# Unity HDRP Mask Map: (Metallic, AO, Detail, Smoothness)
save_packed_texture("mask.png", terrain, pack_mode="unity_mask")

# Unreal Engine ORM: (AO, Roughness, Metallic)
save_packed_texture("orm.png", terrain, pack_mode="ue_orm")
```

When adding new pack modes, update `src/utils/textures.py:_pack_channels()`.

## Advanced Rendering System

Recent addition (see `ADVANCED_RENDERING.md` for full guide). Key functions in `src/utils/advanced_rendering.py`:

- `render_turntable_frames()` - Rotating animation frames
- `save_turntable_video()` - MP4/GIF output (auto-falls back to GIF if no ffmpeg)
- `render_multi_angle()` - Multiple lighting directions
- `render_lighting_study()` - Grid of lighting conditions
- `create_comparison_grid()` - Side-by-side terrain comparison

These are integrated into CLI via `--multi-angle-out`, `--lighting-study-out`, `--turntable-video-out`.

## Batch Generation

Batch generation uses `src/utils/batch.py`:

```python
from src.utils import generate_terrain_set

# Generate 50 terrains with sequential seeds
generate_terrain_set(
    count=50,
    base_seed=100,
    generator="erosion",
    resolution=512,
    output_dir="output/",
    formats=["png", "obj", "shaded"]
)
```

Batch exports create organized directory structure: `batch_output/terrain_0001/`, etc.

## Common Debugging Patterns

### ModernGL Context Issues

If generation fails silently:

```python
import moderngl
ctx = moderngl.create_standalone_context()
print(ctx.info)  # Check OpenGL version (need 3.3+)
```

### Shader Compilation Errors

ModernGL will raise with line numbers:

```python
try:
    program = ctx.program(vertex_shader=vert, fragment_shader=frag)
except moderngl.Error as e:
    print(f"Shader error: {e}")  # Shows exact line/issue
```

### Verifying Exports

Quick visual checks:

```bash
# Generate test with known seed
python gpu_terrain.py --seed 12345 --shaded-out ref.png

# Verify reproducibility
python gpu_terrain.py --seed 12345 --shaded-out test.png
# ref.png and test.png should be identical
```

## Integration Points

### Python API Usage

```python
from src import ErosionTerrainGenerator, utils

# Generate
gen = ErosionTerrainGenerator(resolution=512)
terrain = gen.generate_heightmap(seed=42, seamless=True)
gen.cleanup()

# Export
utils.save_heightmap_png("height.png", terrain)
utils.export_obj_mesh("mesh.obj", terrain)
```

### Jupyter Notebook Integration

The notebook (`terrain_gen.ipynb`) is for experimentation only. Production code lives in `src/`. When adding features:

1. Implement in `src/` modules first
2. Import and demonstrate in notebook
3. Add CLI integration in `gpu_terrain.py`
4. Update README.md examples

## Performance Expectations

Reference benchmarks (RTX 3060, resolutions 256-2048):

- Erosion generation: 0.05s - 1.2s
- Morph generation: 0.02s - 0.5s
- PNG export: 0.005s - 0.1s
- OBJ export: 0.01s - 0.6s

Use `--resolution 256` for rapid iteration, `1024` for production, `2048` for final assets.

## Documentation Locations

- **ADVANCED_RENDERING.md**: Turntable animations, multi-angle renders, lighting studies
- **TEXTURE_EXPORTS.md**: Splatmaps, AO, curvature, packed textures for game engines
- **BATCH_GENERATION.md**: Batch workflows and automation
- **IMPROVEMENTS.md**: Recent code changes and migration guide
- **README.md**: User-facing quick start and API reference

When adding features, update the relevant markdown file(s) with examples.
