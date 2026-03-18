# GPU Terrain Generator - AI Agent Guide

## Architecture Overview

ModernGL-based GPU terrain generator using GLSL fragment shaders for heightmap generation.

> 📖 **Deep dive**: [Architecture Overview](../docs/architecture/system_overview.md) | [Rendering Pipeline](../docs/architecture/rendering_pipeline.md)

**Data Flow**: GLSL shaders (`src/shaders/*.frag`) → ModernGL context → NumPy arrays (`TerrainMaps`) → Export utilities

**Three Generator Types** (all in `src/generators/`):
- `ErosionTerrainGenerator` - Fractal noise + erosion simulation (primary, most used)
- `HydraulicErosionGenerator` - Multi-pass physical erosion with water/sediment simulation
- `MorphologicalTerrainGPU` - Voronoi + distance fields (alternative style)

**Entry Points**:
- `gpu_terrain.py` - CLI with argparse
- `src/__init__.py` - Python API (`from src import ErosionTerrainGenerator, utils`)
- `app/ui_streamlit.py` - Interactive web UI

## Critical Workflows

```bash
# Quick validation (256x256, ~0.05s)
python gpu_terrain.py --resolution 256 --preset canyon --shaded-out test.png

# Run pytest suite
pytest tests/ -v

# Test specific generators
python gpu_terrain.py --generator erosion --shaded-out erosion.png
python gpu_terrain.py --generator morph --shaded-out morph.png
```

## Project-Specific Conventions

> 📖 **Reference**: [Quick Reference](../docs/quick-reference.md) | [Module Reference](../docs/module-reference.md)

### ModernGL Type Ignore Pattern (CRITICAL)

ModernGL has incomplete stubs. **Always use `# type: ignore`** for uniforms:

```python
program["u_seed"].value = float(seed)  # type: ignore
program["u_texelSize"].value = (texel, texel)  # type: ignore
```

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

### Uniform Naming: Python → GLSL

Automatic conversion via `_uniform_name()` in `erosion.py`:
- Python: `height_tiles` → GLSL: `u_heightTiles`
- Python: `erosion_slope_strength` → GLSL: `u_erosionSlopeStrength`

**Shader Output** (`erosion_heightmap.frag`):
```glsl
fragColor = vec4(height, normal.x, normal.z, erosion);
// R: Height (0-1), G: Normal X, B: Normal Z, A: Erosion mask
```

### TerrainMaps Dataclass

Central data structure (`src/utils/artifacts.py`). Use `TerrainMaps.ensure(terrain)` when accepting flexible inputs.

### Colormap Access

```python
from matplotlib import cm
cmap = cm.get_cmap("terrain")  # CORRECT
# NEVER: plt.cm.terrain (deprecated)
```

### Resource Cleanup

Use context managers or call `.cleanup()` explicitly:

```python
with ErosionTerrainGenerator(resolution=512) as gen:
    terrain = gen.generate_heightmap(seed=42)
# Auto-cleanup on exit
```

## Adding New Features

### New Preset
1. Add `@classmethod` to `ErosionParams` in `src/generators/erosion.py`
2. Add to `--preset` choices in `gpu_terrain.py`
3. (Optional) Add YAML config in `configs/presets/`

### New Uniform
1. Add to `ErosionParams` (snake_case)
2. Add to shader (camelCase with `u_` prefix)
3. Set in generator: `program["u_myParam"].value = val  # type: ignore`

### New Export Format

> 📖 **See**: [Export Formats](../docs/EXPORT_FORMATS.md) | [Export CLI Reference](../docs/EXPORT_CLI_REFERENCE.md)
- Add to `src/utils/export.py` (follow `save_*`/`export_*` naming)
- Export in `src/utils/__init__.py`
- Add CLI arg in `gpu_terrain.py`

## Key Files Reference

| Purpose | File |
|---------|------|
| CLI entry | `gpu_terrain.py` |
| Main generator | `src/generators/erosion.py` |
| Hydraulic sim | `src/generators/hydraulic.py` |
| Data container | `src/utils/artifacts.py` |
| Export utils | `src/utils/export.py` |
| Shader loader | `src/utils/shader_loader.py` |
| Render presets | `src/utils/render_configs.py` |
| Config system | `src/config.py` (`TerrainConfig`) |
| Test fixtures | `tests/conftest.py` (shared `ctx` fixture) |

## Debugging

```bash
# Generate test with known seed
python gpu_terrain.py --seed 12345 --shaded-out ref.png

# Verify reproducibility
python gpu_terrain.py --seed 12345 --shaded-out test.png
# ref.png and test.png should be identical
```

## Integration Points

> 📖 **Full API**: [API Reference](../docs/api-reference.md)

### Python API Usage

```python
from src import ErosionTerrainGenerator, utils

with ErosionTerrainGenerator(resolution=512) as gen:
    terrain = gen.generate_heightmap(seed=42, seamless=True)
    utils.save_heightmap_png("height.png", terrain)
    utils.export_obj_mesh("mesh.obj", terrain)
```

## Performance Expectations

Use `--resolution 256` for rapid iteration, `1024` for production, `2048` for final assets.

## Documentation Locations

> 📖 **Start here**: [Documentation Index](../docs/index.md) | [Documentation Map](../docs/DOCUMENTATION_MAP.md)

- **[ADVANCED_RENDERING.md](../docs/ADVANCED_RENDERING.md)**: Turntable animations, multi-angle renders, lighting studies
- **[TEXTURE_EXPORTS.md](../docs/TEXTURE_EXPORTS.md)**: Splatmaps, AO, curvature, packed textures for game engines
- **[BATCH_GENERATION.md](../docs/BATCH_GENERATION.md)**: Batch workflows and automation
- **[HYDRAULIC_EROSION.md](../HYDRAULIC_EROSION.md)**: Physical erosion simulation details
- **[docs/architecture/](../docs/architecture/)**: System design, patterns, dependency graphs

When adding features, update the relevant markdown file(s) with examples.
