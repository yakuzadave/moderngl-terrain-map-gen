# Batch Generation Guide

This guide covers bulk terrain generation for asset libraries, game development pipelines, and production workflows.

## Overview

Batch generation creates multiple terrain variations with sequential seeds, perfect for:

- Building terrain asset libraries
- A/B testing terrain parameters
- Generating variety for procedural games
- Creating training datasets for ML

## Quick Start

### Basic Batch Generation

```bash
# Generate 10 terrains with default settings
python gpu_terrain.py --batch-count 10

# Generate with specific preset
python gpu_terrain.py --batch-count 5 --preset canyon

# Custom output directory
python gpu_terrain.py --batch-count 20 --batch-dir assets/terrains
```

### Multi-Format Export

```bash
# Export heightmaps and shaded previews
python gpu_terrain.py --batch-count 10 --batch-formats png,shaded

# Full material set: heightmaps, meshes, and NPZ data
python gpu_terrain.py --batch-count 5 --batch-formats png,obj,stl,npz,shaded

# Minimal: heightmaps only
python gpu_terrain.py --batch-count 100 --batch-formats png
```

## Output Structure

Batch generation creates organized directories:

```
batch_output/
  terrain_0042/
    heightmap.png          # 16-bit PNG heightmap
    normal_map.png         # RGB normal map
    erosion_mask.png       # Grayscale erosion intensity
    shaded.png             # CPU-rendered shaded relief
    mesh.obj               # Wavefront OBJ mesh (if requested)
    mesh.stl               # Binary STL mesh (if requested)
    bundle.npz             # NumPy bundle (if requested)
  terrain_0043/
    ...
  terrain_0044/
    ...
```

## CLI Options

### Core Options

```bash
--batch-count N
  Generate N terrains with sequential seeds
  Default: 0 (disabled)
  Example: --batch-count 50

--batch-dir PATH
  Output directory for batch generation
  Default: batch_output/
  Example: --batch-dir assets/canyon_pack

--batch-formats FORMATS
  Comma-separated list of export formats
  Options: png, obj, stl, npz, shaded
  Default: png,shaded
  Example: --batch-formats png,obj,shaded
```

### Generation Options

All standard terrain generation options apply to batch mode:

```bash
--preset PRESET         # Terrain preset: default, canyon, plains, mountains
--resolution SIZE       # Texture resolution (power of 2)
--seed SEED            # Starting seed (increments for each terrain)
--generator TYPE       # Generator: erosion or morph
--seamless             # Force seamless tiling
```

## Examples

### Asset Library for Game

Generate 50 canyon variants for level design:

```bash
python gpu_terrain.py \
  --preset canyon \
  --resolution 512 \
  --batch-count 50 \
  --batch-dir game_assets/canyon_library \
  --batch-formats png,shaded,obj \
  --seed 1000
```

Result: 50 unique canyon terrains with previews and meshes.

### High-Res Terrain Pack

Create 10 high-quality terrains with all data:

```bash
python gpu_terrain.py \
  --preset mountains \
  --resolution 2048 \
  --batch-count 10 \
  --batch-dir highres_pack \
  --batch-formats png,obj,stl,npz,shaded
```

### Testing Multiple Presets

Generate samples from each preset:

```bash
# Canyon variants
python gpu_terrain.py --preset canyon --batch-count 5 --batch-dir test/canyon

# Plains variants
python gpu_terrain.py --preset plains --batch-count 5 --batch-dir test/plains

# Mountain variants
python gpu_terrain.py --preset mountains --batch-count 5 --batch-dir test/mountains
```

### Seamless Tileable Set

Create tileable terrain tiles for open-world games:

```bash
python gpu_terrain.py \
  --seamless \
  --resolution 1024 \
  --batch-count 25 \
  --batch-dir tileable_chunks \
  --batch-formats png,obj
```

### Quick Previews

Generate many low-res previews for selection:

```bash
python gpu_terrain.py \
  --resolution 256 \
  --batch-count 100 \
  --batch-dir previews \
  --batch-formats shaded
```

## Python API

### Using BatchGenerator Class

```python
from src.utils import BatchGenerator

# Create batch generator
batch = BatchGenerator(
    generator_type="erosion",
    resolution=512,
    output_dir="terrain_batch",
)

# Generate with sequential seeds
terrains = batch.generate_set(
    seeds=[42, 43, 44, 45, 46],
    prefix="terrain",
    export_formats=["png", "shaded", "obj"],
)

# Access generated terrain data
for terrain in terrains:
    print(f"Height range: {terrain.height.min():.3f} - {terrain.height.max():.3f}")
```

### Progress Callbacks

Monitor batch generation progress:

```python
def progress_callback(current, total, message):
    progress = (current / total) * 100
    print(f"[{progress:.1f}%] {message}")

batch = BatchGenerator(output_dir="monitored_batch")
terrains = batch.generate_set(
    seeds=list(range(100, 110)),
    progress_callback=progress_callback,
)
```

### Custom Export Pipeline

```python
from src.utils import BatchGenerator, save_splatmap_rgba, save_packed_texture

batch = BatchGenerator(resolution=1024, output_dir="custom_export")

terrains = batch.generate_set(
    seeds=[1, 2, 3, 4, 5],
    export_formats=[],  # Disable auto-export
)

# Custom post-processing
for i, terrain in enumerate(terrains):
    base_dir = batch.output_dir / f"terrain_{i:04d}"
    base_dir.mkdir(exist_ok=True)
    
    # Export custom textures
    save_splatmap_rgba(base_dir / "splat.png", terrain)
    save_packed_texture(base_dir / "mask.png", terrain, pack_mode="unity_mask")
```

### Parallel Batch Generation

For multi-core systems, generate batches in parallel:

```python
from multiprocessing import Pool
from src.utils import generate_terrain_set

def generate_batch(args):
    batch_id, seed_range = args
    generate_terrain_set(
        count=len(seed_range),
        base_seed=seed_range[0],
        output_dir=f"parallel_batch_{batch_id}",
        formats=["png", "shaded"],
    )

# Split work across cores
seed_ranges = [
    range(0, 25),
    range(25, 50),
    range(50, 75),
    range(75, 100),
]

with Pool(4) as pool:
    pool.map(generate_batch, enumerate(seed_ranges))
```

## Format Details

### PNG Exports

Includes three textures per terrain:

- **heightmap.png**: 16-bit grayscale (0-65535 range)
- **normal_map.png**: RGB tangent-space normals
- **erosion_mask.png**: Grayscale erosion intensity

### OBJ Exports

Wavefront OBJ mesh with:

- Vertex positions scaled to world units
- UV coordinates (0-1 range)
- Vertex normals
- Single quad-based mesh

### STL Exports

Binary STL mesh for 3D printing and CAD:

- Triangle mesh (quads split into 2 triangles)
- Vertex normals
- Compact binary format

### NPZ Exports

NumPy compressed archive containing:

- `height`: Float32 height field
- `normals`: Float32 normal map (Nx3)
- `erosion_mask`: Float32 erosion mask (if available)
- `metadata`: Resolution, seed, generator type

### Shaded Exports

CPU-rendered shaded relief preview:

- Hillshade lighting (315° azimuth, 45° altitude)
- Terrain colormap applied
- 8-bit RGB PNG
- Quick visual reference

## Performance

### Benchmarks

Average generation time per terrain (512x512):

- **Erosion generator**: ~0.1-0.2s
- **Morphological generator**: ~0.05-0.1s
- **PNG export**: ~0.01s
- **OBJ export**: ~0.05s
- **STL export**: ~0.1s
- **Shaded export**: ~0.2s

For 100 terrains at 512x512 with `png,shaded` formats:

- **Total time**: ~30-40 seconds
- **Disk usage**: ~50-100 MB

### Optimization Tips

**Reduce resolution for previews:**
```bash
python gpu_terrain.py --resolution 256 --batch-count 100
```

**Minimize exports:**
```bash
python gpu_terrain.py --batch-formats png --batch-count 500
```

**Use SSD for output:**
```bash
python gpu_terrain.py --batch-dir /fast_ssd/terrains
```

## Automation & CI/CD

### Shell Script Example

```bash
#!/bin/bash
# generate_terrain_library.sh

PRESETS=("canyon" "plains" "mountains")
COUNT=20
BASE_DIR="terrain_library"

for preset in "${PRESETS[@]}"; do
    echo "Generating $preset terrains..."
    python gpu_terrain.py \
        --preset $preset \
        --batch-count $COUNT \
        --batch-dir "$BASE_DIR/$preset" \
        --batch-formats png,shaded \
        --resolution 512
done

echo "Library generation complete!"
```

### GitHub Actions Workflow

```yaml
name: Generate Terrain Assets

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Generate terrain library
        run: |
          python gpu_terrain.py \
            --batch-count 50 \
            --batch-dir artifacts/terrains \
            --batch-formats png,shaded
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: terrain-assets
          path: artifacts/terrains/
```

## Advanced: Custom Generator Parameters

Pass custom parameters to the terrain generator:

```python
from src.utils import generate_terrain_set

# Custom erosion parameters for canyons
generate_terrain_set(
    count=10,
    generator="erosion",
    resolution=1024,
    output_dir="custom_canyons",
    formats=["png", "obj"],
    # Custom uniforms
    u_octaves=8,
    u_lacunarity=2.5,
    u_persistence=0.6,
    u_erosion_strength=0.8,
)
```

## Troubleshooting

### Out of Memory

Reduce resolution or generate in smaller batches:
```bash
python gpu_terrain.py --resolution 512 --batch-count 10
# Instead of --resolution 2048 --batch-count 100
```

### Slow Generation

Check GPU availability:
```python
import moderngl
ctx = moderngl.create_standalone_context()
print(f"Renderer: {ctx.info['GL_RENDERER']}")
```

### Disk Space Issues

Monitor disk usage and use compression:
```bash
python gpu_terrain.py --batch-formats npz  # Most compact
```

## See Also

- [TEXTURE_EXPORTS.md](./TEXTURE_EXPORTS.md) - Texture export guide
- [README.md](../README.md) - Main documentation
- [gpu_terrain.py](../gpu_terrain.py) - CLI reference
