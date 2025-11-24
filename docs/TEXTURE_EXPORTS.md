# Texture Export Guide

This document describes the texture and material export features for game engine integration.

## Overview

The terrain generator now supports multiple texture export formats optimized for different game engines and workflows:

- **Splatmaps**: 4-channel RGBA blend maps for terrain texture blending
- **Scatter Maps**: RGB density maps for procedural vegetation (Trees, Rocks, Grass)
- **Ambient Occlusion**: Grayscale AO approximation using height variance
- **Curvature Maps**: Convex/concave surface detection
- **Packed Textures**: Multi-channel textures for efficient storage

## Quick Start

### Single Texture Export

```bash
# Generate a splatmap for texture blending
python gpu_terrain.py --preset canyon --splatmap-out terrain_splat.png

# Generate scatter map for vegetation
python gpu_terrain.py --preset canyon --scatter-out terrain_biomes.png

# Generate AO map
python gpu_terrain.py --preset mountains --ao-out terrain_ao.png

# Generate curvature map
python gpu_terrain.py --curvature-out terrain_curve.png

# Generate packed texture for Unity
python gpu_terrain.py --packed-out terrain_mask.png --pack-mode unity_mask
```

### Complete Material Set

```bash
python gpu_terrain.py \
  --preset canyon \
  --resolution 512 \
  --heightmap-out terrain_height.png \
  --normals-out terrain_normal.png \
  --splatmap-out terrain_splat.png \
  --ao-out terrain_ao.png \
  --curvature-out terrain_curve.png \
  --packed-out terrain_mask.png \
  --pack-mode unity_mask
```

## Splatmaps (RGBA)

Splatmaps define how to blend multiple terrain textures based on height and slope.

### Channel Layout

- **R**: Low terrain (beach/grass) - dominant at low heights
- **G**: Mid terrain (grass/dirt) - dominant at medium heights
- **B**: High terrain (rock/snow) - dominant at high heights
- **A**: Steep slopes (cliffs) - dominant on steep faces

### Usage Example

```python
from src.utils import save_splatmap_rgba

# Custom height thresholds
save_splatmap_rgba(
    "custom_splat.png",
    terrain,
    height_thresholds=(0.2, 0.5, 0.8),  # (low, mid, high)
    slope_threshold=0.6,  # Normal.y < 0.6 = steep
)
```

### Integration (Unity)

1. Import splatmap texture
2. Set texture type to "Default" with Alpha channel
3. In terrain shader, sample each channel:
   ```glsl
   float4 splat = tex2D(_SplatMap, uv);
   float3 finalColor = 
       _GrassTex * splat.r +
       _DirtTex * splat.g +
       _RockTex * splat.b +
       _CliffTex * splat.a;
   ```

## Scatter Maps (RGB)

Scatter maps provide density information for procedural asset placement (vegetation, rocks).

### Channel Layout

- **R**: Trees (Moderate slope, specific height range, high moisture)
- **G**: Rocks (Steep slopes)
- **B**: Grass (Flat areas, inverse of trees/rocks)

### Usage Example

```python
from src.utils import save_scatter_map

save_scatter_map(
    "biomes.png",
    terrain,
)
```

### Integration

Use the channels to drive density for instanced meshes:
- **Red Channel** -> Tree instances
- **Green Channel** -> Rock instances
- **Blue Channel** -> Grass/Detail mesh instances

## Ambient Occlusion Maps

AO maps provide ambient lighting approximation for added depth.

### Method

- Uses local height variance as occlusion proxy
- Higher variance = more occlusion (darker)
- Configurable radius and strength

### Usage Example

```python
from src.utils import save_ao_map

save_ao_map(
    "terrain_ao.png",
    terrain,
    radius=3,      # Sample radius in pixels
    strength=1.5,  # AO intensity multiplier
)
```

### Integration

Multiply AO into your ambient lighting term:
```glsl
float ao = tex2D(_AO, uv).r;
float3 ambient = _AmbientColor * ao;
```

## Curvature Maps

Curvature maps highlight convex (bright) and concave (dark) features.

### Method

- Uses Laplacian operator to detect curvature
- Useful for detail masking and weathering effects

### Usage Example

```python
from src.utils import save_curvature_map

save_curvature_map(
    "terrain_curve.png",
    terrain,
    scale=2.0,  # Curvature intensity
)
```

### Integration

Use for detail enhancement or weathering masks:
```glsl
float curve = tex2D(_Curvature, uv).r;
float detail = lerp(0.5, _DetailTex, curve);
```

## Packed Textures

Packed textures combine multiple grayscale channels into RGB/RGBA for efficiency.

### Pack Modes

#### Unity Mask Map (`unity_mask`)

Standard Unity HDRP/URP mask map format:

- **R**: Metallic (0.0 for terrain)
- **G**: Ambient Occlusion
- **B**: Detail mask (erosion intensity)
- **A**: Smoothness (inverse of detail)

```bash
python gpu_terrain.py --packed-out mask.png --pack-mode unity_mask
```

#### Unreal Engine ORM (`ue_orm`)

Unreal Engine 4/5 ORM texture format:

- **R**: Ambient Occlusion
- **G**: Roughness (erosion intensity)
- **B**: Metallic (0.0 for terrain)

```bash
python gpu_terrain.py --packed-out orm.png --pack-mode ue_orm
```

#### Height + Normal + AO (`height_normal_ao`)

Custom packed format for shader reconstruction:

- **R**: Height (0-1)
- **G**: Normal.x remapped to 0-1
- **B**: Normal.z remapped to 0-1
- **A**: Ambient Occlusion

```bash
python gpu_terrain.py --packed-out packed.png --pack-mode height_normal_ao
```

### Usage Example

```python
from src.utils import save_packed_texture

save_packed_texture(
    "unity_mask.png",
    terrain,
    pack_mode="unity_mask",
)
```

## Batch Generation

Generate multiple complete material sets for asset libraries:

```bash
# Generate 10 canyon variants with all textures
python gpu_terrain.py \
  --preset canyon \
  --batch-count 10 \
  --batch-dir assets/canyon_pack \
  --batch-formats png,shaded,obj

# This creates:
# assets/canyon_pack/
#   terrain_0042/
#     heightmap.png
#     shaded.png
#     mesh.obj
#   terrain_0043/
#     ...
```

## Python API

### Direct Export

```python
from src import ErosionTerrainGenerator
from src.utils import (
    save_splatmap_rgba,
    save_scatter_map,
    save_ao_map,
    save_curvature_map,
    save_packed_texture,
)

gen = ErosionTerrainGenerator(resolution=512)
terrain = gen.generate_heightmap(seed=42)

# Export all texture types
save_splatmap_rgba("splat.png", terrain)
save_scatter_map("biomes.png", terrain)
save_ao_map("ao.png", terrain)
save_curvature_map("curve.png", terrain)
save_packed_texture("mask.png", terrain, pack_mode="unity_mask")
```

### Batch Generation

```python
from src.utils import BatchGenerator

batch = BatchGenerator(
    generator_type="erosion",
    resolution=512,
    output_dir="terrain_library",
)

terrains = batch.generate_set(
    seeds=[42, 43, 44, 45, 46],
    prefix="canyon",
    export_formats=["png", "splatmap", "ao", "packed"],
)
```

## Performance Notes

- **Splatmaps**: Fast (simple threshold operations)
- **AO Maps**: Moderate (requires convolution with scipy)
- **Curvature Maps**: Moderate (requires Laplacian calculation)
- **Packed Textures**: Fast (channel reorganization only)

## Dependencies

The texture export features require:

```
numpy
pillow
scipy  # For AO and curvature maps
```

Install via:
```bash
pip install numpy pillow scipy
```

## Examples

### Unity Terrain Material Setup

1. Generate complete material set:
```bash
python gpu_terrain.py \
  --preset mountains \
  --resolution 1024 \
  --heightmap-out height.png \
  --normals-out normal.png \
  --splatmap-out splat.png \
  --packed-out mask.png \
  --pack-mode unity_mask
```

2. In Unity:
   - Import `height.png` as Raw 16-bit heightmap (set to 65535 depth)
   - Import `normal.png` as normal map texture
   - Import `splat.png` as default texture with alpha
   - Import `mask.png` as default texture (RGBA)
   - Create terrain shader with custom splatmap blending

### Unreal Engine Landscape

1. Generate UE-compatible set:
```bash
python gpu_terrain.py \
  --preset canyon \
  --resolution 2017 \  # UE landscape size
  --heightmap-out UE_Height.png \
  --normals-out UE_Normal.png \
  --packed-out UE_ORM.png \
  --pack-mode ue_orm
```

2. In Unreal:
   - Import `UE_Height.png` as landscape heightmap
   - Create landscape material with `UE_ORM.png` as ORM texture
   - Use normal map `UE_Normal.png` for surface detail

## Troubleshooting

### Issue: AO map is too dark/bright

Adjust the `strength` parameter:
```python
save_ao_map("ao.png", terrain, strength=0.5)  # Subtle
save_ao_map("ao.png", terrain, strength=2.0)  # Strong
```

### Issue: Splatmap channels don't blend well

Customize height thresholds:
```python
save_splatmap_rgba(
    "splat.png",
    terrain,
    height_thresholds=(0.25, 0.55, 0.75),  # Wider mid range
)
```

### Issue: Curvature map lacks contrast

Increase scale parameter:
```python
save_curvature_map("curve.png", terrain, scale=3.0)
```

## Advanced: Custom Packed Textures

You can create custom packing schemes:

```python
import numpy as np
from PIL import Image
from src.utils import TerrainMaps

def save_custom_packed(path, terrain):
    maps = TerrainMaps.ensure(terrain)
    
    # Custom channel assignment
    r = maps.height
    g = maps.erosion_channel()
    b = (maps.normals[:, :, 1] + 1.0) * 0.5  # Normal.y
    a = np.ones_like(r)  # Unused
    
    packed = np.stack([r, g, b, a], axis=-1)
    packed_u8 = (np.clip(packed, 0, 1) * 255).astype(np.uint8)
    
    img = Image.fromarray(packed_u8, mode="RGBA")
    img.save(path)
```

## See Also

- [README.md](../README.md) - Main documentation
- [BATCH_GENERATION.md](BATCH_GENERATION.md) - Batch workflow guide
- [gpu_terrain.py](../gpu_terrain.py) - CLI reference
