# GPU Terrain Generator

High-performance procedural terrain generation using ModernGL with multi-format export for game engines and 3D applications.

## 📚 Documentation

- **[Quick Reference](docs/quick-reference.md)** - Common operations and code snippets
- **[API Reference](docs/api-reference.md)** - Complete Python API documentation
- **[Module Reference](docs/module-reference.md)** - Detailed module structure
- **[Documentation Index](docs/index.md)** - All documentation

## Features

### Core Generation
- **GPU-Accelerated**: Real-time terrain generation using OpenGL compute shaders
- **Multiple Algorithms**:
  - Erosion-based terrain (fractal noise + hydraulic erosion simulation)
  - Morphological terrain (Voronoi + distance fields)
- **Terrain Presets**: Canyon, Plains, Mountains with tuned parameters
- **Seamless Tiling**: Generate tileable terrains for open-world games

### Advanced Rendering
- **Atmospheric Fog**: Height-based exponential fog with sun glare (Mie scattering)
- **PBR Materials**: Tri-planar mapping with roughness/metallic properties (Sand, Grass, Rock, Snow)
- **Raymarched Shadows**: Soft shadows using distance field raymarching
- **Ambient Occlusion**: Horizon-Based Ambient Occlusion (HBAO) approximation
- **Thermal Erosion**: Simulation of talus deposition and material collapse

### Export Formats
- **Heightmaps**: 16-bit PNG, RAW, R32 (32-bit float) for game engines
- **Normal Maps**: Tangent-space RGB normal maps
- **Meshes**: Wavefront OBJ, binary STL, glTF 2.0 with embedded textures
- **Textures**: Splatmaps, AO, curvature, packed textures, scatter maps
- **Data**: Compressed NumPy archives (.npz)
- **Previews**: Shaded relief visualizations

### Production Features
- **Batch Generation**: Generate 10s-100s of terrain variations
- **Multi-Format Export**: Export all formats simultaneously
- **Progress Tracking**: Real-time generation progress
- **Organized Output**: Structured directory organization
- **Advanced Rendering**: Turntable animations, multi-angle renders, lighting studies
- **Comparison Tools**: Side-by-side terrain comparisons

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd map_gen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24.0
moderngl>=5.8.0
pillow>=10.0.0
matplotlib>=3.7.0
scipy>=1.11.0
```

### Basic Usage

```bash
# Generate canyon terrain with preview
python gpu_terrain.py --preset canyon

# Export heightmap and mesh
python gpu_terrain.py --heightmap-out terrain.png --gltf-out terrain.gltf

# High-resolution export
python gpu_terrain.py --resolution 2048 --heightmap-out terrain_hires.png

# Export all formats at once
python gpu_terrain.py \
  --preset mountains \
  --export-all exports/ \
  --export-formats png,raw,obj,gltf,npz
```

## Usage Guide

### Terrain Presets

```bash
# Canyon: Deep valleys with steep cliffs
python gpu_terrain.py --preset canyon

# Plains: Rolling hills and gentle terrain
python gpu_terrain.py --preset plains

# Mountains: High peaks with sharp ridges
python gpu_terrain.py --preset mountains

# Default: Balanced multi-scale features
python gpu_terrain.py --preset default
```

### Export Formats

#### Heightmaps
```bash
# 16-bit PNG (0-65535 range)
python gpu_terrain.py --heightmap-out terrain.png

# With normal map
python gpu_terrain.py --heightmap-out terrain.png --normals-out normal.png
```

#### 3D Meshes
```bash
# Wavefront OBJ with UVs
python gpu_terrain.py --obj-out terrain.obj

# Binary STL for 3D printing
python gpu_terrain.py --stl-out terrain.stl

# glTF 2.0 with embedded textures (web/game engines)
python gpu_terrain.py --gltf-out terrain.gltf

# Control mesh scale
python gpu_terrain.py --gltf-out terrain.gltf --mesh-scale 50 --mesh-height-scale 3
```

#### Batch Export (All Formats)
```bash
# Export all formats to a directory
python gpu_terrain.py --export-all exports/ --export-formats png,raw,r32,obj,stl,gltf,npz

# Custom format selection
python gpu_terrain.py --export-all exports/ --export-formats png,gltf,npz

# With custom mesh scaling
python gpu_terrain.py \
  --export-all unity_assets/ \
  --export-formats raw,gltf \
  --mesh-scale 100 \
  --mesh-height-scale 2.5
```

#### Raw Binary Formats
```bash
# 16-bit RAW (Unity/Unreal compatible)
python gpu_terrain.py --raw-out terrain.raw

# 32-bit float R32 (maximum precision)
python gpu_terrain.py --r32-out terrain.r32

# NumPy compressed bundle (all data)
python gpu_terrain.py --bundle-out terrain.npz
```

#### Game Engine Textures
```bash
# RGBA splatmap for texture blending
python gpu_terrain.py --splatmap-out splat.png

# Ambient occlusion map
python gpu_terrain.py --ao-out ao.png

# Curvature map (convex/concave detection)
python gpu_terrain.py --curvature-out curve.png

# Unity HDRP mask map (Metallic/AO/Detail/Smoothness)
python gpu_terrain.py --packed-out mask.png --pack-mode unity_mask

# Unreal Engine ORM texture
python gpu_terrain.py --packed-out orm.png --pack-mode ue_orm

# Scatter density map (R=Trees, G=Rocks, B=Grass)
python gpu_terrain.py --scatter-out biomes.png
```

### Batch Generation

Generate multiple terrain variations:

```bash
# Generate 10 terrains with sequential seeds
python gpu_terrain.py --batch-count 10

# Canyon library for game
python gpu_terrain.py \
  --preset canyon \
  --batch-count 50 \
  --batch-dir game_assets/canyons \
  --batch-formats png,obj,shaded

# High-res production set
python gpu_terrain.py \
  --resolution 2048 \
  --batch-count 20 \
  --batch-formats png,obj,stl,npz,shaded
```

See [BATCH_GENERATION.md](docs/BATCH_GENERATION.md) for complete guide.

### Advanced Options

```bash
# Seamless tileable terrain
python gpu_terrain.py --seamless --resolution 1024

# Custom seed for reproducibility
python gpu_terrain.py --seed 12345

# Disable erosion layers
python gpu_terrain.py --disable-erosion

# Custom shading parameters
python gpu_terrain.py \
  --shaded-out preview.png \
  --shade-azimuth 270 \
  --shade-altitude 60 \
  --shade-colormap gist_earth

# Advanced rendering: turntable animation
python gpu_terrain.py \
  --preset canyon \
  --turntable-video-out turntable.gif \
  --turntable-frames 60 \
  --turntable-fps 15

# Multi-angle renders
python gpu_terrain.py --multi-angle-out ./angles/

# Lighting study
python gpu_terrain.py --lighting-study-out lighting.png

# Chained/adaptive generation (each step nudges params based on previous slope)
python gpu_terrain.py --chain-count 5 --chain-adapt --chain-out chain_output

# Seed comparison grid (GPU viz or ray)
python gpu_terrain.py --render ray --compare-seeds 1,2,3 --compare-out compare.png

# Load/save presets
python gpu_terrain.py --preset-file my_preset.json --save-preset baked_preset.json

# Raymarch tuning & HDR
python gpu_terrain.py --render ray --ray-exposure 1.2 --ray-fog-density 0.08 --ray-max-steps 160 --hdr-out preview.exr
```

## Python API

### Basic Generation

```python
from src import ErosionTerrainGenerator

# Create generator
gen = ErosionTerrainGenerator(resolution=512)

# Generate terrain
terrain = gen.generate_heightmap(seed=42)

# Access data
print(f"Height range: {terrain.height.min():.3f} - {terrain.height.max():.3f}")
print(f"Resolution: {terrain.resolution}")
```

### Using Presets

```python
from src.generators.erosion import ErosionParams, ErosionTerrainGenerator

# Load canyon preset
params = ErosionParams.canyon()
gen = ErosionTerrainGenerator(defaults=params)
terrain = gen.generate_heightmap(seed=42)
```

### Export All Formats

```python
from src import ErosionTerrainGenerator
from src.utils import (
    save_heightmap_png,
    save_normal_map_png,
    export_obj_mesh,
    save_splatmap_rgba,
    save_packed_texture,
    save_scatter_map,
)

gen = ErosionTerrainGenerator(resolution=1024)
terrain = gen.generate_heightmap(seed=42)

# Export everything
save_heightmap_png("height.png", terrain)
save_normal_map_png("normal.png", terrain)
export_obj_mesh("mesh.obj", terrain)
save_splatmap_rgba("splat.png", terrain)
save_packed_texture("mask.png", terrain, pack_mode="unity_mask")
save_scatter_map("biomes.png", terrain)
```

### Batch Generation

```python
from src.utils import BatchGenerator

# Create batch generator
batch = BatchGenerator(
    generator_type="erosion",
    resolution=512,
    output_dir="terrain_library",
)

# Generate multiple terrains
terrains = batch.generate_set(
    seeds=list(range(100, 120)),  # 20 terrains
    export_formats=["png", "obj", "shaded"],
    progress_callback=lambda c, t, m: print(f"{c}/{t}: {m}"),
)
```

### Custom Parameters

```python
from src import ErosionTerrainGenerator

gen = ErosionTerrainGenerator(resolution=1024)

# Custom generation parameters
terrain = gen.generate_heightmap(
    seed=42,
    seamless=True,
    # Override shader uniforms
    u_octaves=8,
    u_lacunarity=2.5,
    u_persistence=0.6,
    u_erosion_strength=0.9,
)
```

## Documentation

- **[docs/EXPORT_FORMATS.md](docs/EXPORT_FORMATS.md)**: Complete guide to all export formats (PNG, RAW, R32, OBJ, STL, glTF, NPZ)
- **[docs/EXPORT_CLI_REFERENCE.md](docs/EXPORT_CLI_REFERENCE.md)**: Quick CLI reference for export commands
- **[ADVANCED_RENDERING.md](docs/ADVANCED_RENDERING.md)**: Complete guide to turntable animations, multi-angle renders, and lighting studies
- **[TEXTURE_EXPORTS.md](docs/TEXTURE_EXPORTS.md)**: Complete texture export guide (splatmaps, AO, curvature, packed textures)
- **[BATCH_GENERATION.md](docs/BATCH_GENERATION.md)**: Batch generation workflows and automation
- **[HYDRAULIC_EROSION.md](HYDRAULIC_EROSION.md)**: Guide to the physical hydraulic erosion simulation
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)**: Summary of recent code improvements and new features
- **[docs/howto/preview-shaders.md](docs/howto/preview-shaders.md)**: Guide to using the standalone GLSL shader viewer
- **[terrain_gen.ipynb](terrain_gen.ipynb)**: Interactive Jupyter notebook with examples

## Project Structure

```
map_gen/
├── gpu_terrain.py              # CLI entry point
├── requirements.txt            # Python dependencies
├── terrain_gen.ipynb          # Interactive notebook
├── src/
│   ├── __init__.py
│   ├── generators/
│   │   ├── erosion.py         # Erosion-based terrain
│   │   └── morphological.py   # Voronoi-based terrain
│   ├── shaders/
│   │   ├── erosion_heightmap.frag
│   │   ├── morph_noise.frag
│   │   └── *.vert/*.frag
│   └── utils/
│       ├── artifacts.py       # TerrainMaps dataclass
│       ├── export.py          # PNG/OBJ/STL export
│       ├── rendering.py       # Shaded relief
│       ├── textures.py        # Game engine textures
│       ├── batch.py           # Batch generation
│       └── visualization.py   # Matplotlib plots
└── notebook_outputs/          # Notebook output directory
```

## CLI Reference

### Generation Options

```
--generator {erosion,morph}    Generator type (default: erosion)
--resolution SIZE              Texture resolution, power of 2 (default: 512)
--seed SEED                    Random seed for reproducibility (default: 42)
--preset {default,canyon,plains,mountains}  Terrain preset (erosion only)
--seamless                     Force seamless tiling
--disable-erosion              Turn off erosion layers
```

### Export Options

```
--heightmap-out FILE           16-bit PNG heightmap
--normals-out FILE             RGB normal map
--erosion-out FILE             Erosion mask (grayscale)
--obj-out FILE                 Wavefront OBJ mesh
--stl-out FILE                 Binary STL mesh
--bundle-out FILE              NumPy .npz bundle
--shaded-out FILE              Shaded relief preview
--splatmap-out FILE            RGBA texture blending map
--ao-out FILE                  Ambient occlusion map
--curvature-out FILE           Curvature map
--packed-out FILE              Packed texture (see --pack-mode)
--pack-mode {unity_mask,ue_orm,height_normal_ao}
--scatter-out FILE             Scatter density map (Trees/Rocks/Grass)
```

### Batch Options

```
--batch-count N                Generate N terrains with sequential seeds
--batch-dir DIR                Output directory (default: batch_output/)
--batch-formats FORMATS        Comma-separated: png,obj,stl,npz,shaded
```

### Visualization Options

```
--render {none,viz,ray,shade}  Rendering mode (default: viz)
--render-out FILE              Save rendered visualization
--panel-out FILE               Matplotlib overview figure
--turntable-out FILE           Animated GIF turntable
```

## Performance

### Benchmarks (NVIDIA RTX 3060)

| Resolution | Erosion Gen | Morph Gen | PNG Export | OBJ Export |
| ---------- | ----------- | --------- | ---------- | ---------- |
| 256x256    | 0.05s       | 0.02s     | 0.005s     | 0.01s      |
| 512x512    | 0.12s       | 0.05s     | 0.01s      | 0.05s      |
| 1024x1024  | 0.35s       | 0.15s     | 0.03s      | 0.15s      |
| 2048x2048  | 1.20s       | 0.50s     | 0.10s      | 0.60s      |

### Optimization Tips

- Use power-of-2 resolutions (256, 512, 1024, 2048)
- Generate lower-res previews first (--resolution 256)
- Use batch mode for multiple terrains
- Enable GPU monitoring to ensure GPU acceleration

## Game Engine Integration

### Unity

```bash
# Generate Unity terrain assets
python gpu_terrain.py \
  --resolution 1024 \
  --heightmap-out Assets/Terrain/height.png \
  --splatmap-out Assets/Terrain/splat.png \
  --packed-out Assets/Terrain/mask.png \
  --pack-mode unity_mask
```

1. Import `height.png` as Raw 16-bit heightmap
2. Apply `splat.png` for texture blending
3. Use `mask.png` in HDRP/URP terrain shader

### Unreal Engine

```bash
# Generate UE landscape
python gpu_terrain.py \
  --resolution 2017 \
  --heightmap-out Landscape_Height.png \
  --packed-out Landscape_ORM.png \
  --pack-mode ue_orm
```

1. Import heightmap to landscape
2. Apply ORM texture in material

### Godot

```bash
# Generate Godot terrain
python gpu_terrain.py \
  --resolution 1024 \
  --heightmap-out terrain_height.png \
  --normals-out terrain_normal.png \
  --obj-out terrain_mesh.obj
```

Import OBJ mesh with heightmap/normal textures.

## Troubleshooting

### ModernGL Error

Ensure OpenGL 3.3+ support:
```python
import moderngl
ctx = moderngl.create_standalone_context()
print(ctx.info)
```

### Out of Memory

Reduce resolution or disable preview:
```bash
python gpu_terrain.py --resolution 512 --render none
```

### Import Errors

Verify all dependencies installed:
```bash
pip install -r requirements.txt
```

## Examples

### Asset Library

```bash
# Generate 100 varied terrains
for preset in canyon plains mountains; do
  python gpu_terrain.py \
    --preset $preset \
    --batch-count 33 \
    --batch-dir library/$preset \
    --batch-formats png,shaded
done
```

### High-Quality Export

```bash
python gpu_terrain.py \
  --preset mountains \
  --resolution 4096 \
  --heightmap-out final_height.png \
  --normals-out final_normal.png \
  --obj-out final_mesh.obj \
  --splatmap-out final_splat.png \
  --shaded-out final_preview.png
```

### Seamless Tile Set

```bash
python gpu_terrain.py \
  --seamless \
  --batch-count 16 \
  --resolution 1024 \
  --batch-formats png,obj
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional terrain presets
- More packed texture formats
- Compute shader optimizations
- Additional export formats (TIFF, EXR)
- Real-time preview window

## License

MIT License - see LICENSE file for details.

## Credits

- ModernGL for GPU acceleration
- NumPy for array operations
- Matplotlib for visualization
- PIL for image export

## See Also

- [ModernGL Documentation](https://moderngl.readthedocs.io/)
- [Terrain Generation Theory](https://www.redblobgames.com/maps/terrain-from-noise/)
- [Unity Terrain Tools](https://docs.unity3d.com/Manual/terrain-Tools.html)
- [Unreal Landscape](https://docs.unrealengine.com/en-US/BuildingWorlds/Landscape/)
