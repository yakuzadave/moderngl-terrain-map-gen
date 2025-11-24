# API Quick Reference

Quick reference for common operations in the GPU Terrain Generator.

## Installation & Setup

```python
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Terrain Generation

```python
from src import ErosionTerrainGenerator, ErosionParams

# Initialize generator
gen = ErosionTerrainGenerator(resolution=512)

try:
    # Generate terrain
    terrain = gen.generate_heightmap(seed=42)
    
    # Access data
    print(terrain.height.shape)    # (512, 512)
    print(terrain.normals.shape)   # (512, 512, 3)
    
finally:
    gen.cleanup()
```

### Using Presets

```python
from src import ErosionTerrainGenerator, ErosionParams

params = ErosionParams.canyon()    # Deep valleys
params = ErosionParams.plains()    # Gentle hills
params = ErosionParams.mountains() # Sharp peaks
params = ErosionParams.natural()   # Organic terrain

gen = ErosionTerrainGenerator(resolution=1024)
terrain = gen.generate_heightmap(seed=123, params=params)
gen.cleanup()
```

## Export Operations

### Single File Exports

```python
from src import utils

# Heightmap (16-bit PNG)
utils.save_heightmap_png("height.png", terrain)

# Normal map (RGB PNG)
utils.save_normal_map_png("normals.png", terrain)

# Erosion mask (grayscale PNG)
utils.save_erosion_mask_png("erosion.png", terrain)

# OBJ mesh
utils.export_obj_mesh("terrain.obj", terrain, scale=10.0)

# glTF mesh
utils.export_gltf_mesh("terrain.gltf", terrain, scale=10.0)

# STL mesh (3D printing)
utils.export_stl_mesh("terrain.stl", terrain, scale=10.0)
```

### Batch Export

```python
from src import utils

# Export all formats at once
paths = utils.export_all_formats(
    "output/my_terrain",
    terrain,
    formats=["png", "normals", "obj", "npz"],
    scale=15.0
)

print(paths["obj"])  # output/my_terrain.obj
```

## Rendering

### Shaded Relief

```python
from src import utils

# Render and save
utils.save_shaded_relief_png(
    "render.png",
    terrain,
    azimuth=315.0,      # Light from NW
    altitude=45.0,      # 45° elevation
    colormap="terrain", # Color scheme
    vert_exag=2.0      # Vertical exaggeration
)
```

### Custom Rendering

```python
from src import utils
import numpy as np
from PIL import Image

# Get RGB array
rgb = utils.shade_heightmap(
    terrain,
    azimuth=45.0,
    altitude=60.0,
    colormap="gist_earth",
    blend_mode="soft"
)

# Save with PIL
img = Image.fromarray(rgb, mode="RGB")
img.save("custom_render.png")
```

## Textures for Game Engines

### Splatmap (Texture Blending)

```python
from src import utils

# 4-channel splatmap: R=low, G=mid, B=high, A=steep
utils.save_splatmap_rgba(
    "splatmap.png",
    terrain,
    height_thresholds=(0.3, 0.5, 0.7),
    slope_threshold=0.7
)
```

### Ambient Occlusion

```python
from src import utils

utils.save_ao_map(
    "ao.png",
    terrain,
    samples=64,      # Higher = better quality
    radius=0.05,
    intensity=1.5
)
```

### Packed Textures

```python
from src import utils

# Unity HDRP Mask Map (Metallic, AO, Detail, Smoothness)
utils.save_packed_texture(
    "unity_mask.png",
    terrain,
    pack_mode="unity_mask"
)

# Unreal Engine ORM (AO, Roughness, Metallic)
utils.save_packed_texture(
    "ue_orm.png",
    terrain,
    pack_mode="ue_orm"
)
```

### Scatter Map (Vegetation)

```python
from src import utils

# RGB channels: R=trees, G=rocks, B=grass
utils.save_scatter_map("scatter.png", terrain)
```

## Advanced Rendering

### Turntable Animation

```python
from src import utils

# MP4 video (requires ffmpeg)
utils.save_turntable_video(
    "turntable.mp4",
    terrain,
    frames=60,
    fps=30,
    altitude=50.0,
    colormap="terrain"
)

# Or just get frames
frames = utils.render_turntable_frames(terrain, frames=36)
```

### Multi-Angle Renders

```python
from src import utils

# Render from 4 directions
renders = utils.render_multi_angle(
    terrain,
    angles=[(0, 45), (90, 45), (180, 45), (270, 45)]
)

# Save each
from PIL import Image
for i, rgb in enumerate(renders):
    Image.fromarray(rgb).save(f"angle_{i}.png")
```

### Lighting Study

```python
from src import utils
from PIL import Image

# Grid of different lighting conditions
study = utils.render_lighting_study(
    terrain,
    azimuth_steps=6,
    altitude_steps=4
)

Image.fromarray(study).save("lighting_study.png")
```

## Batch Generation

### Generate Multiple Terrains

```python
from src import utils, ErosionParams

# Generate 50 terrains with sequential seeds
terrains = utils.generate_terrain_set(
    count=50,
    base_seed=1000,
    generator="erosion",
    resolution=512,
    output_dir="batch_output",
    formats=["png", "obj", "shaded"],
    params=ErosionParams.mountains()
)
```

### Custom Batch Processing

```python
from src.utils import BatchGenerator
from src import ErosionParams

batch = BatchGenerator(
    generator_type="erosion",
    resolution=1024,
    output_dir="custom_batch"
)

# Generate with custom seeds
seeds = [42, 123, 456, 789, 1011]
results = batch.generate_set(
    seeds=seeds,
    prefix="terrain",
    export_formats=["png", "normals", "obj"],
    params=ErosionParams.canyon(),
    seamless=True
)

batch.cleanup()
```

## Hydraulic Erosion

### Apply to Existing Terrain

```python
from src import (
    ErosionTerrainGenerator,
    HydraulicErosionGenerator,
    HydraulicParams
)

# Generate base terrain
gen = ErosionTerrainGenerator(resolution=512)
base = gen.generate_heightmap(seed=42)
gen.cleanup()

# Apply hydraulic erosion
hydraulic = HydraulicErosionGenerator(resolution=512)
params = HydraulicParams(
    iterations=200,
    rain_rate=0.02,
    sediment_capacity=1.5,
    soil_dissolving=0.7
)

eroded = hydraulic.simulate(
    base.height,
    params=params,
    progress_callback=lambda i, total: print(f"{i}/{total}")
)

hydraulic.cleanup()
```

## Configuration Files

### Create and Save Config

```python
from src import TerrainConfig, save_config, load_config

# Create config
config = TerrainConfig(
    resolution=1024,
    seed=12345,
    generator_type="erosion",
    terrain_preset="canyon",
    seamless=True,
    export_heightmap=True,
    export_normals=True,
    export_obj=True,
    azimuth=315.0,
    altitude=45.0,
    colormap="terrain"
)

# Save as YAML
save_config(config, "configs/my_terrain.yaml")

# Load later
loaded = load_config("configs/my_terrain.yaml")
```

### Use Config for Generation

```python
from src import (
    TerrainConfig,
    load_config,
    ErosionTerrainGenerator,
    utils
)

# Load config
config = load_config("configs/canyon.yaml")

# Generate
gen = ErosionTerrainGenerator(resolution=config.resolution)
terrain = gen.generate_heightmap(
    seed=config.seed,
    params=config.get_erosion_params(),
    seamless=config.seamless
)
gen.cleanup()

# Export based on config
if config.export_heightmap:
    utils.save_heightmap_png("height.png", terrain)
if config.export_normals:
    utils.save_normal_map_png("normals.png", terrain)
if config.export_obj:
    utils.export_obj_mesh("mesh.obj", terrain)
```

## Working with TerrainMaps

### Accessing Data

```python
# TerrainMaps attributes
height = terrain.height          # (H, W) float32, range 0-1
normals = terrain.normals        # (H, W, 3) float32, range -1 to 1
erosion = terrain.erosion_mask   # (H, W) float32 or None
scatter = terrain.scatter_map    # (H, W, 3) float32 or None

# Get resolution
h, w = terrain.resolution

# Convert to dict
data = terrain.as_dict()
```

### Converting to Other Formats

```python
# 16-bit height for external tools
height_u16 = terrain.height_u16()  # uint16, 0-65535

# RGB normal map
normals_u8 = terrain.normal_map_u8()  # uint8 RGB

# Erosion mask
erosion_u8 = terrain.erosion_mask_u8()  # uint8 grayscale
```

### Creating from Arrays

```python
import numpy as np
from src import TerrainMaps

# Create from scratch
height = np.random.rand(512, 512).astype(np.float32)
normals = np.random.rand(512, 512, 3).astype(np.float32)
terrain = TerrainMaps(height=height, normals=normals)

# From dict
data = {
    "height": height,
    "normals": normals,
    "erosion_mask": np.zeros_like(height)
}
terrain = TerrainMaps.ensure(data)
```

## Command Line Interface

### Basic Usage

```bash
# Generate with preset
python gpu_terrain.py --preset canyon --seed 42 --resolution 1024

# Custom parameters
python gpu_terrain.py \
    --resolution 1024 \
    --seed 12345 \
    --height-amp 0.3 \
    --erosion-strength 0.06 \
    --seamless
```

### Export Options

```bash
# Specify outputs
python gpu_terrain.py \
    --preset mountains \
    --heightmap-out height.png \
    --normals-out normals.png \
    --shaded-out render.png \
    --obj-out mesh.obj
```

### Using Config Files

```bash
# Generate from config
python gpu_terrain.py --config configs/my_terrain.yaml

# Override config values
python gpu_terrain.py \
    --config configs/base.yaml \
    --seed 999 \
    --resolution 2048
```

## Error Handling

### Resource Cleanup

```python
from src import ErosionTerrainGenerator

gen = ErosionTerrainGenerator(resolution=512)

try:
    terrain = gen.generate_heightmap(seed=42)
    # ... process terrain ...
finally:
    gen.cleanup()  # Always cleanup
```

### Context Manager Pattern

```python
from contextlib import contextmanager

@contextmanager
def terrain_generator(resolution=512):
    gen = ErosionTerrainGenerator(resolution=resolution)
    try:
        yield gen
    finally:
        gen.cleanup()

# Use it
with terrain_generator(1024) as gen:
    terrain = gen.generate_heightmap(seed=42)
    # Auto cleanup when done
```

## Performance Tips

### Resolution Guidelines

```python
# Fast preview (0.05s)
gen = ErosionTerrainGenerator(resolution=256)

# Production quality (0.3s)
gen = ErosionTerrainGenerator(resolution=1024)

# High detail (1.2s)
gen = ErosionTerrainGenerator(resolution=2048)

# Maximum quality (5s+)
gen = ErosionTerrainGenerator(resolution=4096)
```

### Batch Optimization

```python
from src.utils import BatchGenerator

# Reuse generator for multiple terrains
batch = BatchGenerator(resolution=512)

seeds = range(100, 200)
for seed in seeds:
    terrain = batch.gen.generate_heightmap(seed=seed)
    # Process...

batch.cleanup()  # Single cleanup at end
```

## Common Patterns

### Seamless Terrain

```python
# Enable seamless mode for tiling
terrain = gen.generate_heightmap(
    seed=42,
    seamless=True  # Edges wrap seamlessly
)
```

### Custom Parameter Tweaking

```python
from src import ErosionParams

# Start with preset
params = ErosionParams.canyon()

# Tweak specific values
params.erosion_strength = 0.08
params.warp_strength = 0.7
params.thermal_iterations = 20

terrain = gen.generate_heightmap(seed=42, params=params)
```

### Comparing Generators

```python
from src import (
    ErosionTerrainGenerator,
    MorphologicalTerrainGPU,
    utils
)

# Generate with both
erosion_gen = ErosionTerrainGenerator(resolution=512)
morph_gen = MorphologicalTerrainGPU()

terrain_a = erosion_gen.generate_heightmap(seed=42)
terrain_b = morph_gen.generate(resolution=512, seed=42)

erosion_gen.cleanup()
morph_gen.cleanup()

# Compare side-by-side
comparison = utils.create_comparison_grid(
    [terrain_a, terrain_b],
    labels=["Erosion", "Morphological"]
)

from PIL import Image
Image.fromarray(comparison).save("comparison.png")
```

## Troubleshooting

### Check GPU Support

```python
import moderngl

# Test context creation
try:
    ctx = moderngl.create_context(standalone=True)
    print(f"OpenGL version: {ctx.version_code}")
    print(f"Vendor: {ctx.info['GL_VENDOR']}")
    ctx.release()
except Exception as e:
    print(f"GPU initialization failed: {e}")
```

### Verify Outputs

```python
import numpy as np

# Check terrain data validity
assert terrain.height.shape == (512, 512)
assert terrain.height.dtype == np.float32
assert 0.0 <= terrain.height.min() <= terrain.height.max() <= 1.0

print(f"Height range: {terrain.height.min():.3f} - {terrain.height.max():.3f}")
print(f"Normal range: {terrain.normals.min():.3f} - {terrain.normals.max():.3f}")
```

---

## See Also

- [Full API Reference](api-reference.md) - Complete API documentation
- [Export Formats Guide](EXPORT_FORMATS.md) - Detailed export format info
- [Configuration Reference](config_ui.md) - Config file structure
- [Architecture Overview](architecture/index.md) - System design
