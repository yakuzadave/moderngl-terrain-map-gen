# Export Formats Guide

Complete guide to exporting terrain data in multiple formats for use in game engines, 3D software, and GIS applications.

## Quick Reference

| Format | Extension | Use Case                      | Precision         | File Size    |
| ------ | --------- | ----------------------------- | ----------------- | ------------ |
| PNG    | `.png`    | Visual preview, Unity, Unreal | 16-bit            | Small        |
| RAW    | `.raw`    | Unity RAW import, GIS         | 16-bit LE         | Small        |
| R32    | `.r32`    | Full precision heightmaps     | 32-bit float LE   | Medium       |
| OBJ    | `.obj`    | Blender, Maya, general 3D     | Float ASCII       | Large        |
| STL    | `.stl`    | 3D printing, CAD software     | 32-bit binary     | Medium       |
| glTF   | `.gltf`   | Three.js, Babylon.js, Blender | Float JSON+binary | Medium-Large |
| NPZ    | `.npz`    | Python/NumPy analysis         | Float compressed  | Small        |

---

## Format Details

### PNG Heightmap (16-bit Grayscale)

**Use for**: Unity, Unreal Engine, general heightmap import

```python
from src import utils

utils.save_heightmap_png("terrain_height.png", terrain)
```

**CLI**:
```bash
python gpu_terrain.py --heightmap-out terrain_height.png
```

**Properties**:
- Format: 16-bit grayscale PNG (mode "I;16")
- Range: 0-65535 (maps from terrain 0.0-1.0)
- Lossless compression
- Compatible with most game engines and GIS software

**Unity Import**:
1. Import PNG to Unity project
2. Set Texture Type → "Single Channel" or use as heightmap in Terrain Tools
3. Apply to terrain via Terrain Settings

**Unreal Engine Import**:
1. Content Browser → Import
2. Select heightmap PNG
3. Use in Landscape tool (Height Map tab)

---

### RAW Heightmap (16-bit Little-Endian)

**Use for**: Unity RAW import, World Machine, GIS applications

```python
utils.save_heightmap_raw("terrain.raw", terrain)
```

**CLI**:
```bash
python gpu_terrain.py --raw-out terrain.raw
```

**Properties**:
- Format: Raw 16-bit unsigned little-endian binary
- No header, pure data
- Cross-platform compatible (Windows/Mac/Linux)
- Direct memory mapping possible

**Unity RAW Import**:
```
Terrain Tools → Import Raw
- Depth: 16-bit
- Resolution: Match your generation resolution (e.g., 512x512)
- Byte Order: Little-Endian (Windows)
```

---

### R32 Heightmap (32-bit Float)

**Use for**: Maximum precision, scientific visualization, custom pipelines

```python
utils.save_heightmap_r32("terrain.r32", terrain)
```

**CLI**:
```bash
python gpu_terrain.py --r32-out terrain.r32
```

**Properties**:
- Format: Raw 32-bit float little-endian binary
- Full floating-point precision (no quantization)
- Larger file size than 16-bit formats
- Ideal for erosion simulation or scientific applications

**Reading in Python**:
```python
import numpy as np

height = np.fromfile("terrain.r32", dtype="<f4").reshape(512, 512)
```

---

### OBJ Mesh

**Use for**: Blender, Maya, 3ds Max, most 3D software

```python
utils.export_obj_mesh(
    "terrain.obj",
    terrain,
    scale=10.0,          # Horizontal scale
    height_scale=2.0     # Vertical exaggeration
)
```

**CLI**:
```bash
python gpu_terrain.py --obj-out terrain.obj --mesh-scale 10 --mesh-height-scale 2
```

**Properties**:
- ASCII text format (human-readable)
- Includes vertex positions, UVs, and face definitions
- Grid mesh (2 triangles per quad)
- Large file size for high resolutions

**Blender Import**:
```
File → Import → Wavefront (.obj)
```

**Optimization Tips**:
- Use lower resolutions (256-512) for editing, higher (1024-2048) for final renders
- Apply Decimate modifier in Blender to reduce poly count
- OBJ does not include materials; apply in your 3D software

---

### STL Mesh

**Use for**: 3D printing, CAD software, physical models

```python
utils.export_stl_mesh(
    "terrain.stl",
    terrain,
    scale=10.0,
    height_scale=2.0
)
```

**CLI**:
```bash
python gpu_terrain.py --stl-out terrain.stl
```

**Properties**:
- Binary STL format (industry standard)
- Includes computed face normals
- Optimized for 3D printing and CAM software
- Smaller file size than OBJ

**3D Printing Workflow**:
1. Generate STL with appropriate scale
2. Import to slicer (Cura, PrusaSlicer)
3. Add base/supports as needed
4. Print!

**Recommended Settings**:
- `scale=100.0` for small desktop prints (10cm x 10cm base)
- `height_scale=0.5-1.0` to prevent overhangs
- Use lower resolution (128-256) to reduce print time

---

### glTF 2.0 Mesh

**Use for**: Web (Three.js, Babylon.js), Blender, game engines

```python
utils.export_gltf_mesh(
    "terrain.gltf",
    terrain,
    scale=10.0,
    height_scale=2.0,
    embed_textures=True    # Embed PNG textures as base64
)
```

**CLI**:
```bash
python gpu_terrain.py --gltf-out terrain.gltf
```

**Properties**:
- JSON-based scene format (glTF 2.0 spec)
- Embedded binary buffers (base64)
- Includes PBR material definition
- Embedded heightmap and normal map textures
- Optimized for web delivery and modern engines

**Included Data**:
- Vertex positions, normals, UVs
- Indices (optimized triangle list)
- PBR material (rough non-metallic terrain)
- Normal map texture (if available)
- Heightmap as baseColor texture

**Three.js Loading**:
```javascript
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('terrain.gltf', (gltf) => {
    scene.add(gltf.scene);
});
```

**Babylon.js Loading**:
```javascript
BABYLON.SceneLoader.ImportMesh("", "./", "terrain.gltf", scene);
```

---

### NPZ Bundle

**Use for**: Python analysis, pipeline storage, reproducibility

```python
utils.save_npz_bundle("terrain.npz", terrain)
```

**CLI**:
```bash
python gpu_terrain.py --bundle-out terrain.npz
```

**Properties**:
- NumPy compressed archive format
- Contains all terrain maps: height, normals, erosion mask
- Lossless float32 precision
- Fast loading in Python

**Loading**:
```python
import numpy as np

data = np.load("terrain.npz")
height = data['height']          # float32 heightmap
normals = data['normals']        # RGB normal map
erosion = data['erosion_mask']   # Erosion intensity
```

---

## Batch Export

Export all formats at once to a directory:

### Python API

```python
from src import utils

results = utils.export_all_formats(
    output_dir="exports",
    terrain=terrain,
    base_name="terrain",
    formats=['png', 'raw', 'r32', 'obj', 'stl', 'gltf', 'npz'],
    scale=10.0,
    height_scale=2.0,
    embed_textures=True
)

# Results is a dict: {'heightmap_png': Path(...), 'mesh_obj': Path(...), ...}
```

### CLI

```bash
python gpu_terrain.py \
    --export-all exports/ \
    --export-formats png,obj,gltf,npz \
    --mesh-scale 10 \
    --mesh-height-scale 2
```

**Output Structure**:
```
exports/
├── terrain_height.png     # Heightmap
├── terrain_normal.png     # Normal map
├── terrain_erosion.png    # Erosion mask (if available)
├── terrain.raw            # RAW heightmap
├── terrain.r32            # R32 heightmap
├── terrain.obj            # OBJ mesh
├── terrain.stl            # STL mesh
├── terrain.gltf           # glTF mesh with textures
└── terrain.npz            # NumPy bundle
```

---

## Common Workflows

### Unity Terrain Pipeline

```bash
# Generate 1024x1024 terrain
python gpu_terrain.py \
    --resolution 1024 \
    --preset mountains \
    --raw-out unity_height.raw \
    --normals-out unity_normal.png \
    --splatmap-out unity_splat.png

# Unity Import:
# 1. Terrain Tools → Import Raw (unity_height.raw, 16-bit, 1024x1024)
# 2. Apply normal map to terrain material
# 3. Use splatmap for texture blending
```

### Unreal Engine Landscape

```bash
python gpu_terrain.py \
    --resolution 2017 \
    --preset canyon \
    --heightmap-out unreal_height.png

# Unreal Import:
# Landscape → Import from File → unreal_height.png
# Scale: Match to desired world size
```

### Blender Render

```bash
python gpu_terrain.py \
    --resolution 512 \
    --preset natural \
    --gltf-out blender_terrain.gltf

# Blender Import:
# File → Import → glTF 2.0 (.gltf)
# Mesh includes PBR materials and normal maps
```

### 3D Printing

```bash
python gpu_terrain.py \
    --resolution 256 \
    --preset plains \
    --stl-out print_terrain.stl \
    --mesh-scale 100 \
    --mesh-height-scale 0.8

# Slicer Import:
# Load print_terrain.stl
# Add base/supports
# Slice and print
```

### Web Visualization

```bash
python gpu_terrain.py \
    --resolution 512 \
    --preset canyon \
    --gltf-out web_terrain.gltf

# Three.js:
# Use GLTFLoader to load web_terrain.gltf
# Includes textures and PBR materials ready to render
```

---

## Advanced Options

### Mesh Scaling

Control the size of exported meshes:

```bash
python gpu_terrain.py \
    --mesh-scale 50.0 \           # 50x50 units horizontal
    --mesh-height-scale 5.0 \     # 5x vertical exaggeration
    --obj-out terrain.obj
```

**Scale Guidelines**:
- `mesh-scale`: Horizontal extent in world units
- `mesh-height-scale`: Vertical multiplier (terrain 0-1 → 0-N units)
- Unity/Unreal: Use scales that match your game's unit system
- Blender: Scale 1.0 = 1 Blender unit (typically 1 meter)

### Normal Maps

All PNG exports include normal maps when available:

```bash
python gpu_terrain.py \
    --normals-out terrain_normal.png \
    --gltf-out terrain.gltf  # Auto-includes normals in glTF
```

Normal maps are RGB tangent-space normals, compatible with:
- Unity (import as Normal Map texture type)
- Unreal (import with "Normal Map" compression)
- Blender (connect to Normal input of Principled BSDF)

---

## File Size Reference

For a 512x512 terrain:

| Format             | Typical Size  |
| ------------------ | ------------- |
| PNG (16-bit)       | 200-400 KB    |
| RAW (16-bit)       | 524 KB        |
| R32 (32-bit float) | 1.0 MB        |
| OBJ                | 15-30 MB      |
| STL                | 5-10 MB       |
| glTF (embedded)    | 5-15 MB       |
| NPZ                | 500 KB - 1 MB |

**Size Scaling**: File sizes scale roughly with resolution²:
- 256x256: 1/4 the size
- 1024x1024: 4× the size
- 2048x2048: 16× the size

---

## Example Script

See `examples/export_formats_demo.py` for a complete demonstration:

```bash
python examples/export_formats_demo.py
```

This generates a sample terrain and exports it to all supported formats with detailed output.

---

## Troubleshooting

### Large File Sizes

**Problem**: OBJ/glTF files are too large

**Solution**:
- Reduce resolution: Use 256-512 for editing, 1024-2048 for final export
- Use STL instead of OBJ for 3D printing (smaller binary format)
- For web: Use Draco compression (external tool) on glTF files

### Missing Textures in glTF

**Problem**: glTF loads without textures in viewer

**Solution**:
- Ensure `embed_textures=True` in export call
- Check that terrain includes normal map data
- Use a glTF validator: https://github.khronos.org/glTF-Validator/

### Unity RAW Import Issues

**Problem**: Terrain appears wrong when importing RAW

**Solution**:
- Verify byte order: Little-Endian (Windows)
- Verify depth: 16-bit
- Verify resolution matches exactly (e.g., 512x512)
- Mac/Linux users: Raw format is little-endian, should work across platforms

### Blender OBJ Import Slow

**Problem**: Blender hangs when importing large OBJ

**Solution**:
- Use glTF instead: Much faster to load
- Reduce resolution before export
- Split terrain into tiles for massive terrains (>2048²)

---

## API Reference

### `save_heightmap_png(path, terrain) -> Path`
Save 16-bit PNG heightmap.

### `save_heightmap_raw(path, terrain) -> Path`
Save 16-bit little-endian RAW binary.

### `save_heightmap_r32(path, terrain) -> Path`
Save 32-bit float little-endian binary.

### `export_obj_mesh(path, terrain, scale=10.0, height_scale=2.0) -> Path`
Export Wavefront OBJ mesh.

### `export_stl_mesh(path, terrain, scale=10.0, height_scale=2.0) -> Path`
Export binary STL mesh.

### `export_gltf_mesh(path, terrain, scale=10.0, height_scale=2.0, embed_textures=True) -> Path`
Export glTF 2.0 mesh with PBR materials.

### `export_all_formats(output_dir, terrain, base_name="terrain", formats=None, **kwargs) -> dict[str, Path]`
Batch export to multiple formats.

**Parameters**:
- `output_dir`: Output directory path
- `terrain`: TerrainMaps or compatible terrain data
- `base_name`: Base filename for all exports
- `formats`: List of format names or None for all
- `**kwargs`: Additional arguments (scale, height_scale, embed_textures)

**Returns**: Dictionary mapping format names to output paths

---

## See Also

- [README.md](../README.md) - Project overview and quickstart
- [TEXTURE_EXPORTS.md](TEXTURE_EXPORTS.md) - Advanced texture exports (splatmaps, AO, etc.)
- [examples/export_formats_demo.py](../examples/export_formats_demo.py) - Complete demo script
