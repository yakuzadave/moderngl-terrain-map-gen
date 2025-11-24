# Quick Start: Exporting Terrain

Get started with terrain exports in under 5 minutes.

## Basic Exports

### Export a single heightmap PNG
```bash
python gpu_terrain.py --heightmap-out terrain.png
```

### Export with normal map
```bash
python gpu_terrain.py \
    --heightmap-out terrain.png \
    --normals-out normals.png
```

### Export as 3D mesh (glTF for modern workflows)
```bash
python gpu_terrain.py --gltf-out terrain.gltf
```

## One-Command Complete Export

Export everything at once:
```bash
python gpu_terrain.py --export-all exports/
```

This creates:
- `terrain_height.png` - 16-bit heightmap
- `terrain_normal.png` - Normal map
- `terrain.obj` - OBJ mesh
- `terrain.gltf` - glTF mesh with textures
- `terrain.npz` - NumPy bundle

## Common Workflows

### For Unity
```bash
python gpu_terrain.py \
    --raw-out terrain.raw \
    --resolution 1024
```
Import in Unity: `Terrain Tools → Import Raw → 16-bit, 1024x1024`

### For Unreal Engine
```bash
python gpu_terrain.py \
    --heightmap-out landscape.png \
    --resolution 2017
```
Import in Unreal: `Landscape → Import from File`

### For Blender
```bash
python gpu_terrain.py --gltf-out terrain.gltf
```
Import in Blender: `File → Import → glTF 2.0`

### For Three.js / Web
```bash
python gpu_terrain.py --gltf-out web_terrain.gltf
```
Load with GLTFLoader in Three.js

### For 3D Printing
```bash
python gpu_terrain.py \
    --stl-out print.stl \
    --mesh-scale 100 \
    --mesh-height-scale 0.8
```
Import into slicer software (Cura, PrusaSlicer)

## Terrain Presets

Try different terrain styles:

```bash
# Deep canyons and valleys
python gpu_terrain.py --preset canyon --export-all canyon/

# Rolling hills
python gpu_terrain.py --preset plains --export-all plains/

# Sharp mountain peaks
python gpu_terrain.py --preset mountains --export-all mountains/

# Balanced natural terrain
python gpu_terrain.py --preset natural --export-all natural/
```

## Custom Scale

Control mesh size:

```bash
python gpu_terrain.py \
    --gltf-out terrain.gltf \
    --mesh-scale 50 \           # 50x50 units horizontal
    --mesh-height-scale 3       # 3x vertical exaggeration
```

## High Resolution

Generate detailed terrain:

```bash
python gpu_terrain.py \
    --resolution 2048 \
    --export-all high_res/ \
    --export-formats png,gltf
```

## Seamless Tiling

For repeating terrain:

```bash
python gpu_terrain.py \
    --seamless \
    --export-all tileable/
```

## Format Selection

Choose specific formats:

```bash
# Only PNG and glTF
python gpu_terrain.py \
    --export-all exports/ \
    --export-formats png,gltf

# Only meshes
python gpu_terrain.py \
    --export-all exports/ \
    --export-formats obj,stl,gltf

# Everything
python gpu_terrain.py \
    --export-all exports/ \
    --export-formats png,raw,r32,obj,stl,gltf,npz
```

## Python API

Use in scripts:

```python
from src import ErosionTerrainGenerator, ErosionParams, utils

# Generate terrain
gen = ErosionTerrainGenerator(resolution=512)
try:
    terrain = gen.generate_heightmap(seed=42, preset=ErosionParams.canyon())
finally:
    gen.cleanup()

# Export all formats
results = utils.export_all_formats(
    "exports/",
    terrain,
    base_name="canyon",
    formats=['png', 'gltf', 'npz'],
    scale=10.0,
    height_scale=2.0
)

print(f"Exported {len(results)} files")
```

## What Format Should I Use?

| Need                | Format | Command                       |
| ------------------- | ------ | ----------------------------- |
| Unity terrain       | RAW    | `--raw-out terrain.raw`       |
| Unreal landscape    | PNG    | `--heightmap-out terrain.png` |
| Blender/3D software | glTF   | `--gltf-out terrain.gltf`     |
| 3D printing         | STL    | `--stl-out terrain.stl`       |
| Web/Three.js        | glTF   | `--gltf-out terrain.gltf`     |
| Max precision       | R32    | `--r32-out terrain.r32`       |
| Python analysis     | NPZ    | `--bundle-out terrain.npz`    |

## Demo Script

Run the full demonstration:

```bash
python examples/export_formats_demo.py
```

This generates a sample terrain and exports it to all formats with detailed output.

## Next Steps

- **Full documentation**: [docs/EXPORT_FORMATS.md](EXPORT_FORMATS.md)
- **CLI reference**: [docs/EXPORT_CLI_REFERENCE.md](EXPORT_CLI_REFERENCE.md)
- **Texture exports**: [TEXTURE_EXPORTS.md](./TEXTURE_EXPORTS.md)
- **Advanced rendering**: [ADVANCED_RENDERING.md](./ADVANCED_RENDERING.md)

## Troubleshooting

### File not found errors
Ensure output directories exist or use `--export-all` which creates them automatically.

### Large file sizes
Use lower resolutions (256-512) for testing, higher (1024-2048) for production.

### Unity RAW import issues
Verify: 16-bit depth, Little-Endian byte order, resolution matches exactly.

### glTF missing textures
Textures are embedded as base64. Ensure terrain includes normal map data.

## Support

- Issues: Check [docs/EXPORT_FORMATS.md](EXPORT_FORMATS.md) troubleshooting section
- Examples: See [examples/export_formats_demo.py](../examples/export_formats_demo.py)
- API: See inline docstrings in `src/utils/export.py`
