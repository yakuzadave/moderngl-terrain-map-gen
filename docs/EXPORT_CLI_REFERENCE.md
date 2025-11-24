# Export Formats - CLI Quick Reference

Quick command examples for exporting terrain in different formats.

## Single Format Exports

### PNG Heightmap (16-bit)
```bash
python gpu_terrain.py --preset canyon --heightmap-out terrain.png
```

### Normal Map
```bash
python gpu_terrain.py --preset mountains --normals-out normals.png
```

### RAW Heightmap (Unity/Unreal)
```bash
python gpu_terrain.py --resolution 1024 --raw-out terrain.raw
```

### R32 Heightmap (32-bit float)
```bash
python gpu_terrain.py --r32-out terrain.r32
```

### OBJ Mesh
```bash
python gpu_terrain.py --obj-out terrain.obj --mesh-scale 10 --mesh-height-scale 2
```

### STL Mesh (3D Printing)
```bash
python gpu_terrain.py --stl-out terrain.stl --mesh-scale 100 --mesh-height-scale 0.8
```

### glTF Mesh (Web/Game Engines)
```bash
python gpu_terrain.py --gltf-out terrain.gltf --mesh-scale 10 --mesh-height-scale 2
```

### NPZ Bundle (All Data)
```bash
python gpu_terrain.py --bundle-out terrain.npz
```

---

## Batch Export (All Formats)

Export everything to a directory:

```bash
python gpu_terrain.py \
    --preset canyon \
    --resolution 512 \
    --export-all exports/ \
    --export-formats png,raw,obj,stl,gltf,npz \
    --mesh-scale 10 \
    --mesh-height-scale 2
```

Custom format selection:

```bash
# Only export PNG and glTF
python gpu_terrain.py \
    --export-all exports/ \
    --export-formats png,gltf

# Only export meshes
python gpu_terrain.py \
    --export-all exports/ \
    --export-formats obj,stl,gltf
```

---

## Complete Workflows

### Unity Terrain
```bash
python gpu_terrain.py \
    --preset mountains \
    --resolution 1024 \
    --seamless \
    --raw-out unity_height.raw \
    --normals-out unity_normal.png \
    --splatmap-out unity_splat.png
```

### Unreal Engine Landscape
```bash
python gpu_terrain.py \
    --preset canyon \
    --resolution 2017 \
    --heightmap-out unreal_landscape.png
```

### Blender/3D Software
```bash
python gpu_terrain.py \
    --preset natural \
    --resolution 512 \
    --gltf-out blender_terrain.gltf
```

### 3D Printing
```bash
python gpu_terrain.py \
    --preset plains \
    --resolution 256 \
    --stl-out print_model.stl \
    --mesh-scale 100 \
    --mesh-height-scale 0.8
```

### Web Visualization (Three.js)
```bash
python gpu_terrain.py \
    --preset canyon \
    --resolution 512 \
    --gltf-out web_terrain.gltf
```

---

## Options Reference

### Export Paths
- `--heightmap-out PATH` - 16-bit PNG heightmap
- `--normals-out PATH` - RGB normal map
- `--erosion-out PATH` - Erosion mask
- `--raw-out PATH` - 16-bit RAW binary
- `--r32-out PATH` - 32-bit float binary
- `--obj-out PATH` - OBJ mesh
- `--stl-out PATH` - STL mesh
- `--gltf-out PATH` - glTF 2.0 mesh
- `--bundle-out PATH` - NPZ bundle
- `--export-all PATH` - Export all to directory

### Mesh Scaling
- `--mesh-scale FLOAT` - Horizontal scale (default: 10.0)
- `--mesh-height-scale FLOAT` - Vertical scale (default: 2.0)

### Batch Export
- `--export-formats LIST` - Comma-separated formats (default: png,obj,gltf,npz)

### Generation Options
- `--preset NAME` - Terrain preset: canyon, mountains, plains, natural
- `--resolution INT` - Grid resolution (default: 512)
- `--seed INT` - Random seed (default: 42)
- `--seamless` - Enable seamless tiling
- `--generator NAME` - Generator type: erosion, morph, hydraulic

---

## Format Selection Guide

| Need            | Format      | Command                       |
| --------------- | ----------- | ----------------------------- |
| Unity import    | RAW         | `--raw-out terrain.raw`       |
| Unreal import   | PNG         | `--heightmap-out terrain.png` |
| Blender/Maya    | glTF or OBJ | `--gltf-out terrain.gltf`     |
| 3D printing     | STL         | `--stl-out terrain.stl`       |
| Web (Three.js)  | glTF        | `--gltf-out terrain.gltf`     |
| Python analysis | NPZ         | `--bundle-out terrain.npz`    |
| Max precision   | R32         | `--r32-out terrain.r32`       |

---

## Advanced Examples

### High-Resolution Export for Unreal
```bash
python gpu_terrain.py \
    --preset mountains \
    --resolution 4033 \
    --seed 12345 \
    --heightmap-out unreal_8k_landscape.png \
    --normals-out unreal_8k_normals.png
```

### Seamless Tileable Terrain for Unity
```bash
python gpu_terrain.py \
    --preset natural \
    --resolution 512 \
    --seamless \
    --raw-out unity_tile.raw \
    --splatmap-out unity_splat.png
```

### Complete Asset Pack
```bash
python gpu_terrain.py \
    --preset canyon \
    --resolution 1024 \
    --seed 99999 \
    --export-all asset_pack/ \
    --export-formats png,raw,obj,gltf,npz \
    --mesh-scale 50 \
    --mesh-height-scale 3 \
    --splatmap-out asset_pack/splatmap.png \
    --ao-out asset_pack/ao.png \
    --curvature-out asset_pack/curvature.png
```

### Multiple Seeds for Asset Variation
```bash
# Generate 5 variations
for seed in 100 101 102 103 104; do
    python gpu_terrain.py \
        --preset natural \
        --seed $seed \
        --export-all "exports/terrain_$seed/" \
        --export-formats png,gltf
done
```

---

## See Also

- [EXPORT_FORMATS.md](EXPORT_FORMATS.md) - Complete format documentation
- [README.md](../README.md) - Project overview
- [TEXTURE_EXPORTS.md](TEXTURE_EXPORTS.md) - Texture generation (splatmaps, AO, etc.)
