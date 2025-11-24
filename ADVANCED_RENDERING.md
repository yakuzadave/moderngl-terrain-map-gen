# Advanced Rendering Features

This document describes the advanced rendering capabilities added to the terrain generation system.

## Overview

The advanced rendering module provides multiple output methods for terrain visualization, including:

- **Turntable animations**: Rotate lighting around static terrain
- **Multi-angle renders**: Generate views from multiple lighting directions
- **Lighting studies**: Comprehensive matrix of lighting conditions
- **Comparison grids**: Side-by-side terrain comparisons
- **Custom animations**: Build your own animation sequences

## Command-Line Usage

### Turntable Video

Generate a rotating animation of your terrain:

```bash
# GIF output (no dependencies)
python gpu_terrain.py --preset canyon --turntable-video-out turntable.gif --turntable-frames 60 --turntable-fps 15

# MP4 output (requires ffmpeg)
python gpu_terrain.py --preset mountains --turntable-video-out turntable.mp4 --turntable-frames 90 --turntable-fps 24
```

**Options:**
- `--turntable-video-out PATH`: Output file path (.gif or .mp4)
- `--turntable-frames N`: Number of frames in full rotation (default: 90)
- `--turntable-fps N`: Frames per second (default: 12)
- `--shade-altitude ANGLE`: Sun elevation angle (default: 45.0)
- `--shade-vert-exag SCALE`: Vertical exaggeration (default: 2.0)
- `--shade-colormap NAME`: Matplotlib colormap (default: terrain)

### Multi-Angle Renders

Generate the same terrain lit from multiple directions:

```bash
python gpu_terrain.py --preset plains --multi-angle-out ./angles/
```

This creates 5 renders:
- `angle_northwest.png` - Classic 315° lighting
- `angle_northeast.png` - 45° lighting
- `angle_southeast.png` - 135° lighting
- `angle_southwest.png` - 225° lighting
- `angle_overhead.png` - Directly overhead

**Use Cases:**
- Compare how lighting affects terrain appearance
- Choose best angle for presentation
- Create reference sheets

### Lighting Study

Generate a comprehensive matrix showing terrain under various lighting conditions:

```bash
python gpu_terrain.py --preset canyon --lighting-study-out study.png
```

Creates a grid with:
- **Columns**: Different azimuth angles (0-360°)
- **Rows**: Different altitude angles (15-75°)

**Use Cases:**
- Understand terrain structure
- Identify optimal lighting for screenshots
- Educational material showing lighting effects

### Seed Comparison Grid (GPU Viz/Ray)

Compare multiple seeds in one grid using the GPU renderer:

```bash
python gpu_terrain.py --render ray --compare-seeds 1,2,3,4 --compare-out compare.png --resolution 256
```

Notes:
- Works with `--render viz` or `--render ray`.
- Uses the erosion generator; honors other render flags (fog, exposure, etc.).

## Python API

### Turntable Animation

```python
from src import ErosionTerrainGenerator, utils

# Generate terrain
gen = ErosionTerrainGenerator(resolution=512)
terrain = gen.generate_heightmap(seed=42)
gen.cleanup()

# Create turntable video
utils.save_turntable_video(
    "output.gif",
    terrain,
    frames=60,
    fps=15,
    altitude=45.0,
    vert_exag=2.5,
    colormap="gist_earth",
)
```

**Parameters:**
- `frames`: Number of frames in 360° rotation
- `fps`: Playback speed
- `altitude`: Sun elevation angle (0-90°)
- `vert_exag`: Height exaggeration multiplier
- `colormap`: Any matplotlib colormap name
- `blend_mode`: "soft", "overlay", or "hsv"

### Multi-Angle Rendering

```python
# Render from multiple angles
renders = utils.render_multi_angle(
    terrain,
    angles=[
        (315.0, 45.0),  # (azimuth, altitude)
        (135.0, 45.0),
        (0.0, 90.0),
    ],
    colormap="terrain",
    vert_exag=2.0,
)

# Save renders
from PIL import Image
for i, render in enumerate(renders):
    img = Image.fromarray(render, mode="RGB")
    img.save(f"render_{i}.png")
```

### Lighting Study

```python
# Create comprehensive lighting matrix
fig = utils.render_lighting_study(
    terrain,
    azimuth_steps=8,    # Test 8 rotation angles
    altitude_steps=3,   # Test 3 elevation angles
    colormap="terrain",
    vert_exag=2.0,
)

fig.savefig("lighting_study.png", dpi=150, bbox_inches="tight")
```

### Comparison Grid

```python
# Compare multiple terrains
from src import ErosionParams

terrains = []
labels = []

for name, preset_func in [
    ("Canyon", ErosionParams.canyon),
    ("Plains", ErosionParams.plains),
    ("Mountains", ErosionParams.mountains),
]:
    gen = ErosionTerrainGenerator(defaults=preset_func())
    terrain = gen.generate_heightmap(seed=42)
    gen.cleanup()
    
    terrains.append(terrain)
    labels.append(name)

# Create comparison figure
fig = utils.create_comparison_grid(
    terrains,
    labels=labels,
    figsize=(15, 5),
    dpi=120,
)

fig.savefig("comparison.png", dpi=120, bbox_inches="tight")
```

### Custom Animation Sequences

```python
# Define custom animation function
def my_animation(frame_index: int, total_frames: int):
    """Custom rendering logic per frame."""
    from matplotlib.colors import LightSource
    from matplotlib import cm
    import numpy as np
    
    # Example: Rotate and pulse vertical exaggeration
    azimuth = (frame_index * 360.0) / total_frames
    vert_exag = 1.5 + 1.0 * np.sin(frame_index * 2 * np.pi / total_frames)
    
    ls = LightSource(azdeg=azimuth, altdeg=45.0)
    height = terrain.height
    
    rgb = ls.shade(
        height,
        cmap=cm.get_cmap("terrain"),
        vert_exag=vert_exag,
        blend_mode="soft",
        dx=1.0/height.shape[1],
        dy=1.0/height.shape[0],
    )
    
    return (np.clip(rgb * 255.0, 0.0, 255.0)).astype("uint8")

# Save sequence
utils.save_animation_sequence(
    output_dir="frames/",
    terrain=terrain,
    sequence_func=my_animation,
    frame_count=120,
    name_pattern="frame_{:04d}.png",
)
```

## Complete Example

```python
"""Complete workflow with all rendering features."""
from pathlib import Path
from src import ErosionTerrainGenerator, ErosionParams, utils

# Setup
output = Path("output")
output.mkdir(exist_ok=True)

# Generate canyon terrain
params = ErosionParams.canyon()
gen = ErosionTerrainGenerator(resolution=1024, defaults=params)
terrain = gen.generate_heightmap(seed=12345)
gen.cleanup()

# 1. Save basic exports
utils.save_heightmap_png(output / "height.png", terrain)
utils.export_obj_mesh(output / "mesh.obj", terrain)

# 2. Create turntable animation
utils.save_turntable_video(
    output / "turntable.gif",
    terrain,
    frames=60,
    fps=15,
)

# 3. Generate multi-angle renders
renders = utils.render_multi_angle(terrain)
for i, render in enumerate(renders):
    from PIL import Image
    Image.fromarray(render).save(output / f"angle_{i}.png")

# 4. Create lighting study
fig = utils.render_lighting_study(terrain)
fig.savefig(output / "lighting_study.png", dpi=150, bbox_inches="tight")

# 5. Save shaded relief
utils.save_shaded_relief_png(
    output / "shaded.png",
    terrain,
    azimuth=315.0,
    altitude=45.0,
    vert_exag=2.5,
)

print(f"✓ All outputs saved to {output}/")
```

## Use Cases

### Game Development

**Asset Previews:**
```bash
# Quick preview with multiple angles
python gpu_terrain.py --seed 42 --multi-angle-out ./previews/
```

**Promotional Materials:**
```bash
# High-quality turntable for trailer
python gpu_terrain.py \
  --resolution 2048 \
  --preset mountains \
  --turntable-video-out promo.mp4 \
  --turntable-frames 120 \
  --turntable-fps 30
```

### Art Direction

**Lighting Tests:**
```bash
# Test different lighting on terrain
python gpu_terrain.py \
  --preset canyon \
  --lighting-study-out lighting_options.png
```

**Preset Comparison:**
```python
# Compare all presets with same seed
presets = ["default", "canyon", "plains", "mountains"]
terrains = []

for preset in presets:
    gen = ErosionTerrainGenerator(
        defaults=ErosionParams[preset]() if preset != "default" else None
    )
    terrains.append(gen.generate_heightmap(seed=42))
    gen.cleanup()

utils.create_comparison_grid(terrains, labels=presets)
```

### Education

**Erosion Demonstration:**
```python
# Show effect of erosion strength
strengths = [0.0, 0.02, 0.04, 0.08]
terrains = []

for strength in strengths:
    params = ErosionParams(erosion_strength=strength)
    gen = ErosionTerrainGenerator(defaults=params)
    terrain = gen.generate_heightmap(seed=42)
    gen.cleanup()
    terrains.append(terrain)

fig = utils.create_comparison_grid(
    terrains,
    labels=[f"Erosion: {s}" for s in strengths],
)
```

## Performance Tips

1. **Resolution**: Lower resolution for previews (512x512), high for finals (2048x2048)
2. **Frame Count**: 30-60 frames for web, 90-120 for smooth playback
3. **Batch Processing**: Use `--batch-count` for multiple variations
4. **Caching**: Save `.npz` bundles to avoid regenerating terrain

```bash
# Generate once, render many times
python gpu_terrain.py --seed 42 --bundle-out terrain.npz

# Then use cached data
python -c "
import numpy as np
from src import utils

data = np.load('terrain.npz')
terrain = {'height': data['height'], 'normals': data['normals']}

utils.save_turntable_video('out.gif', terrain, frames=60)
utils.render_lighting_study(terrain)
"
```

## Troubleshooting

**"No module named 'imageio'"**
```bash
pip install imageio imageio-ffmpeg  # For MP4 support
```

**"ffmpeg not found"**
- Install ffmpeg from https://ffmpeg.org/
- Or use GIF output instead (.gif extension)

**"Out of memory"**
- Reduce resolution: `--resolution 512`
- Reduce frame count: `--turntable-frames 30`
- Process in batches

**"Turntable looks wrong"**
- Check colormap name: `--shade-colormap terrain`
- Adjust vertical exaggeration: `--shade-vert-exag 2.0`
- Try different altitude: `--shade-altitude 60`

## API Reference

See inline documentation for complete parameter lists:

```python
help(utils.render_turntable_frames)
help(utils.save_turntable_video)
help(utils.render_multi_angle)
help(utils.create_comparison_grid)
help(utils.render_lighting_study)
help(utils.save_animation_sequence)
```
