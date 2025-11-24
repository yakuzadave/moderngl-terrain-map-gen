# Configuration UI Guide

This guide covers the Streamlit-based configuration interface for the GPU terrain generator.

## Overview

The configuration UI provides a user-friendly interface for:
- Adjusting terrain generation parameters without editing code
- Applying and customizing render presets
- Real-time preview of terrain generation
- Saving and loading configuration presets
- Exporting heightmaps, normal maps, and 3D meshes

The UI is designed for artists, game developers, and technical users who want to quickly iterate on terrain designs without writing Python code.

---

## Starting the UI

### Prerequisites

1. **Install dependencies** (first time only):
   ```bash
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   This installs:
   - `streamlit` (web UI framework)
   - `pyyaml` (configuration file format)
   - `scipy` (post-processing filters)
   - `imageio` (additional export formats)

2. **Verify your environment**:
   - Python 3.11+ required
   - OpenGL 3.3+ capable GPU
   - Tested on Windows with NVIDIA/AMD GPUs

### Launch the UI

```bash
streamlit run app/ui_streamlit.py
```

This will:
1. Start a local web server (default: http://localhost:8501)
2. Automatically open the UI in your default browser
3. Display the terrain configuration interface

---

## Interface Overview

The UI is organized into three main areas:

### 1. Sidebar (Left Panel)

**Quick Controls** - frequently used settings:
- **Resolution**: Output image size (256-2048 pixels)
  - 256: Fast previews (~0.05s generation)
  - 512: Development work (~0.2s)
  - 1024: Production quality (~1s)
  - 2048: High-quality exports (~4s)
- **Seed**: Random seed for deterministic generation
  - Same seed = identical terrain
  - Useful for version control and reproducibility
- **Seamless**: Enable tileable terrain (wraps at edges)

**Terrain Presets** - quick terrain styles:
- **Canyon**: Deep erosion with branching valleys
- **Plains**: Gentle rolling hills
- **Mountains**: Sharp peaks with moderate erosion
- **Custom**: Manual parameter control

**Render Styles** - 17 lighting/visualization presets:
- **Classic**: Standard daylight terrain rendering
- **Dramatic**: High-contrast side lighting
- **Sunrise/Sunset**: Warm atmospheric lighting
- **Noon**: Overhead flat lighting
- **Flat/Contour**: Cartographic visualizations
- **Technical**: Height-based color coding
- **Debug modes**: Normals, erosion, slope, curvature
- *(See RENDERING_ENHANCEMENT_REPORT.md for full preset reference)*

**Actions**:
- **Generate Terrain**: Run GPU generation with current settings
- **Export All**: Save heightmap + normals + shaded render + OBJ mesh

### 2. Preview Area (Center)

Displays the currently generated terrain render with:
- Live image preview (updates after each generation)
- Generation time (GPU terrain creation)
- Render time (CPU shading and post-processing)
- Resolution and seed info

### 3. Advanced Settings (Right Panel - Tabs)

Four tabbed panels for detailed parameter control:

#### Tab 1: Terrain Parameters
**Height Generation** - base fractal noise:
- **Tiles**: Frequency of height features (2-5)
- **Octaves**: Detail layers (2-6)
- **Amplitude**: Overall height scale (0.5-2.0)
- **Gain**: Octave amplitude falloff (0.3-0.7)
- **Lacunarity**: Frequency multiplier per octave (1.8-2.5)

**Erosion Simulation** - hydraulic erosion:
- **Tiles**: Frequency of erosion patterns (2-5)
- **Octaves**: Detail complexity (2-6)
- **Gain**: Detail falloff (0.3-0.7)
- **Lacunarity**: Frequency multiplier (1.8-2.5)
- **Slope Strength**: How much slopes drive erosion (1-5)
- **Branch Strength**: Valley branching intensity (0-1)
- **Overall Strength**: Global erosion intensity (0-1)

**Water Level**:
- Height threshold for water surface (0.0-0.5)

#### Tab 2: Lighting & Camera
**Sun Direction**:
- **Azimuth**: Horizontal angle (0-360°, 0=North, 90=East, 180=South, 270=West)
- **Altitude**: Vertical angle (0-90°, 0=horizon, 90=overhead)

**Camera**:
- **Vertical Exaggeration**: Height scale multiplier (0.5-5.0)

#### Tab 3: Rendering Options
**Color Mapping**:
- **Colormap**: Color scheme (terrain, gist_earth, viridis, etc.)
- **Normalization**: Histogram adjustment (linear, equalize, log)

**Blending**:
- **Mode**: How color blends with shading (soft, overlay, multiply, screen)

#### Tab 4: Post-Processing Effects
**Tone Mapping**:
- **Method**: HDR-to-LDR algorithm (reinhard, aces, filmic, uncharted2)

**Color Grading** (6 adjustments):
- **Contrast**: 0.5-2.0 (1.0=neutral)
- **Brightness**: 0.5-2.0 (1.0=neutral)
- **Saturation**: 0.0-2.0 (1.0=neutral)
- **Highlights**: -1.0 to 1.0 (0=neutral)
- **Shadows**: -1.0 to 1.0 (0=neutral)
- **Gamma**: 0.5-2.2 (1.0=neutral)

**Bloom Effect**:
- **Enable/Disable** toggle
- **Radius**: Glow spread (2-20 pixels)
- **Strength**: Glow intensity (0.1-1.0)

**Sharpening**:
- **Enable/Disable** toggle
- **Method**: unsharp mask or laplacian
- **Amount**: 0.1-2.0

**SSAO (Ambient Occlusion)**:
- **Enable/Disable** toggle
- **Radius**: Sampling distance (5-30 pixels)
- **Strength**: Darkening intensity (0.1-1.0)

**Atmospheric Perspective**:
- **Enable/Disable** toggle
- **Fog Color**: RGB color picker
- **Density**: 0.001-0.05

---

## Preset Management

### Saving Presets

1. Configure all parameters to your desired settings
2. Click **"💾 Save Current Config"** in the sidebar
3. Enter a filename (e.g., `my_canyon_preset`)
4. Config saved to `configs/presets/my_canyon_preset.yaml`

**Preset Format** (YAML):
```yaml
version: 1.0
resolution: 1024
seed: 12345
seamless: false

# Terrain parameters
height_tiles: 3.0
height_octaves: 3
# ... (40+ fields total)

# Render parameters
azimuth: 90
altitude: 15
# ...

# Post-processing
tone_map_method: aces
color_grade_contrast: 1.2
# ...
```

### Loading Presets

1. Click **"📂 Load Preset"** in the sidebar
2. Select a preset file from `configs/presets/`
3. All parameters update to match the preset
4. Click **"Generate Terrain"** to see the result

**Built-in Presets**:
- `canyon_sunrise.yaml` - Deep valleys with warm lighting
- `mountains_dramatic.yaml` - High peaks with side lighting
- `plains_noon.yaml` - Gentle hills with flat lighting

---

## Workflow Examples

### Workflow 1: Creating a New Terrain Style

1. **Start with a preset**:
   - Select "Mountains" terrain preset
   - Select "Classic" render style
   - Click "Generate Terrain" to see baseline

2. **Adjust terrain shape**:
   - Go to "Terrain" tab
   - Increase "Height Amplitude" to 1.5 for taller peaks
   - Decrease "Erosion Branch Strength" to 0.2 for less valley complexity
   - Click "Generate Terrain" to preview

3. **Tune lighting**:
   - Go to "Lighting" tab
   - Set Azimuth to 270 (west-facing light)
   - Set Altitude to 20 (low sun angle)
   - Click "Generate Terrain" to update

4. **Apply post-processing**:
   - Go to "Post-FX" tab
   - Enable "Atmospheric Perspective"
   - Set fog color to cool blue [0.5, 0.6, 0.8]
   - Set density to 0.01
   - Enable "Sharpening" with amount 0.6
   - Click "Generate Terrain" to apply

5. **Save the preset**:
   - Click "💾 Save Current Config"
   - Name it `alpine_evening`
   - Now you can reload this exact configuration anytime

### Workflow 2: Rapid Seed Exploration

**Goal**: Find interesting terrain variations quickly

1. Set resolution to 256 for fast generation (~0.05s)
2. Choose your base preset (e.g., "Canyon")
3. Click in the "Seed" field and use arrow keys or type random numbers
4. Press Enter after each seed change to regenerate
5. When you find an interesting seed:
   - Increase resolution to 1024
   - Generate final high-quality version
   - Export all outputs

### Workflow 3: Lighting Study

**Goal**: Test different lighting conditions on the same terrain

1. Generate terrain with a fixed seed (e.g., 12345)
2. Go to "Lighting" tab
3. Animate through time of day:
   - **Sunrise**: Azimuth=90, Altitude=15
   - **Morning**: Azimuth=120, Altitude=45
   - **Noon**: Azimuth=180, Altitude=90
   - **Afternoon**: Azimuth=240, Altitude=45
   - **Sunset**: Azimuth=270, Altitude=15
4. Keep seed constant - only lighting changes
5. Export each variation for comparison

### Workflow 4: Exporting for Game Engine

1. Generate terrain at 1024 or 2048 resolution
2. Enable all output formats:
   - ☑ Heightmap (16-bit PNG for displacement)
   - ☑ Normals (RGB normal map for materials)
   - ☑ Shaded (preview image for documentation)
   - ☑ OBJ mesh (geometry for direct import)
3. Click "Export All"
4. Files saved to `output/terrain_SEED_*`:
   - `heightmap.png` - Import as displacement map
   - `normals.png` - Import as normal map
   - `shaded.png` - Reference image
   - `mesh.obj` - Geometry file

---

## Performance Tips

### Resolution vs. Generation Time

Benchmarks on RTX 3060:

| Resolution | Gen Time | Render Time | Use Case |
|------------|----------|-------------|----------|
| 256x256    | 0.05s    | 0.1s        | Fast previews, seed exploration |
| 512x512    | 0.15s    | 0.3s        | Development work, parameter tuning |
| 1024x1024  | 0.8s     | 1.5s        | Production quality, final assets |
| 2048x2048  | 3.5s     | 6s          | High-quality exports, print resolution |

**Recommendations**:
- **Iterating on parameters**: Use 256 or 512 resolution
- **Final exports**: Use 1024 (sufficient for most games)
- **Hero assets or print**: Use 2048 (large memory footprint)

### Post-Processing Cost

Effects ordered by performance impact (fast → slow):

1. **Tone mapping** - Negligible (~0.01s)
2. **Color grading** - Fast (~0.02s)
3. **Sharpening** - Fast (~0.03s)
4. **Atmospheric perspective** - Moderate (~0.05s)
5. **Bloom** - Moderate (~0.08s)
6. **SSAO** - Expensive (~0.2s at 1024 resolution)

**Recommendations**:
- Disable SSAO for fast iteration
- Enable bloom and SSAO only for final renders
- Keep all effects on when generating exports (quality matters)

### Memory Usage

| Resolution | GPU Memory | System RAM | Notes |
|------------|------------|------------|-------|
| 256        | ~10 MB     | ~50 MB     | Safe on integrated GPUs |
| 512        | ~20 MB     | ~100 MB    | Safe on all GPUs |
| 1024       | ~60 MB     | ~300 MB    | Requires discrete GPU |
| 2048       | ~250 MB    | ~1.2 GB    | Requires 2GB+ VRAM |

---

## Troubleshooting

### UI won't start

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### Terrain generation is slow

**Issue**: Generation takes >5 seconds at 512 resolution

**Possible causes**:
1. Integrated GPU - upgrade to discrete GPU
2. Old GPU drivers - update to latest drivers
3. Too many post-processing effects enabled

**Solution**:
- Disable SSAO, bloom, and atmospheric perspective for faster iteration
- Use 256 resolution for parameter exploration
- Check GPU utilization in Task Manager (should be >80% during generation)

---

### Preview image looks wrong

**Issue**: Terrain is all black, all white, or very low contrast

**Possible causes**:
1. Extreme color grading settings
2. Vertical exaggeration too high/low
3. Water height covering entire terrain

**Solution**:
- Click "Reset to Default" on the Post-FX tab
- Set vertical exaggeration to 1.5 (default)
- Set water height to 0.0
- Try "Classic" render style for neutral visualization

---

### Saved preset won't load

**Issue**: `FileNotFoundError` or `yaml.scanner.ScannerError`

**Possible causes**:
1. Preset file was manually edited with invalid YAML syntax
2. File was moved or renamed
3. Version mismatch (old preset format)

**Solution**:
- Check that file exists in `configs/presets/`
- Validate YAML syntax (use online YAML validator)
- Compare with working presets (canyon_sunrise.yaml)
- If all else fails, recreate the preset using "Save Current Config"

---

### Export fails

**Issue**: Click "Export All" but no files appear

**Possible causes**:
1. No terrain generated yet
2. Write permissions in output directory
3. Disk full

**Solution**:
- Click "Generate Terrain" first
- Check `output/` directory exists and is writable
- Verify available disk space
- Check Streamlit terminal for error messages

---

## Configuration File Reference

The YAML configuration format contains 40+ fields organized into groups:

### Version & Output Settings
```yaml
version: 1.0                # Config schema version
resolution: 1024            # Output size in pixels
seed: 12345                 # Random seed
seamless: false             # Tileable terrain
output_heightmap: true      # Export 16-bit heightmap PNG
output_normals: true        # Export RGB normal map PNG
output_shaded: true         # Export shaded relief PNG
output_obj: false           # Export OBJ mesh
output_stl: false           # Export STL mesh
```

### Terrain Parameters (12 fields)
```yaml
height_tiles: 3.0           # Base noise frequency
height_octaves: 3           # Detail layers
height_amp: 1.0             # Height scale
height_gain: 0.55           # Octave falloff
height_lacunarity: 2.0      # Frequency multiplier
erosion_tiles: 3.0          # Erosion frequency
erosion_octaves: 5          # Erosion detail
erosion_gain: 0.5           # Erosion falloff
erosion_lacunarity: 2.0     # Erosion frequency mult
erosion_slope_strength: 3.0 # Slope-driven erosion
erosion_branch_strength: 0.5 # Valley branching
erosion_strength: 0.8       # Overall erosion intensity
water_height: 0.0           # Water level threshold
```

### Render Parameters (6 fields)
```yaml
azimuth: 90                 # Sun horizontal angle (0-360)
altitude: 15                # Sun vertical angle (0-90)
vert_exag: 1.5              # Height exaggeration
blend_mode: soft            # Color blending mode
colormap: terrain           # Matplotlib colormap name
cmap_norm: equalize         # Histogram normalization
```

### Post-Processing Parameters (16 fields)
```yaml
tone_map_method: aces       # HDR tone mapping algorithm
color_grade_contrast: 1.2   # Contrast adjustment
color_grade_brightness: 1.0 # Brightness adjustment
color_grade_saturation: 1.1 # Saturation adjustment
color_grade_highlights: 0.0 # Highlight adjustment
color_grade_shadows: 0.0    # Shadow adjustment
color_grade_gamma: 1.0      # Gamma correction
apply_bloom: false          # Enable bloom effect
bloom_radius: 5             # Bloom spread
bloom_strength: 0.3         # Bloom intensity
apply_sharpening: true      # Enable sharpening
sharpen_method: unsharp     # Sharpening algorithm
sharpen_amount: 0.5         # Sharpening intensity
apply_ssao: false           # Enable ambient occlusion
ssao_radius: 10             # AO sampling radius
ssao_strength: 0.3          # AO darkening
apply_atmospheric: false    # Enable fog
atmos_color: [0.6, 0.7, 0.9] # Fog RGB color
atmos_density: 0.01         # Fog thickness
```

---

## Advanced Usage

### Custom Render Styles

To create a new render style preset:

1. Load an existing style as a base (e.g., "Classic")
2. Adjust lighting, colormap, blend mode, and post-processing
3. Save as a new preset with descriptive name
4. To make it a default style, edit `src/utils/render_configs.py`

### Batch Generation from Presets

The UI is designed for interactive exploration. For batch processing:

```bash
# Use the CLI instead
python gpu_terrain.py --resolution 1024 --preset canyon --shaded-out canyon.png

# Or use the batch generation utilities
python -c "
from src.utils import generate_terrain_set
generate_terrain_set(count=50, base_seed=100, generator='erosion', 
                     resolution=512, output_dir='batch_output/')
"
```

See `BATCH_GENERATION.md` for full batch workflows.

### Integrating Custom Shaders

The UI uses shaders from `src/shaders/`. To add custom visualization:

1. Create new `.frag` shader in `src/shaders/`
2. Add corresponding `RenderConfig` entry in `src/utils/render_configs.py`
3. Add preset to UI sidebar in `app/ui_streamlit.py`
4. Restart Streamlit to see new style

See `.github/copilot-instructions.md` for GLSL conventions.

---

## Additional Resources

- **RENDERING_ENHANCEMENT_REPORT.md** - Full reference for all 17 render styles
- **TEXTURE_EXPORTS.md** - Splatmaps, AO, curvature, and game engine formats
- **ADVANCED_RENDERING.md** - Turntable animations and lighting studies
- **BATCH_GENERATION.md** - Automated terrain set generation
- **IMPROVEMENTS.md** - Recent code changes and migration guide

---

## Feedback and Contributions

This UI is actively developed. If you encounter issues or have feature requests:

1. Check the troubleshooting section above
2. Review the Streamlit terminal output for error messages
3. Open an issue with:
   - Your configuration (save and attach the YAML)
   - Error messages from terminal
   - Screenshot of the issue
   - System info (GPU model, Python version, OS)

**Common feature requests**:
- [ ] Undo/redo for parameter changes
- [ ] Side-by-side preset comparison view
- [ ] Animation timeline for parameter interpolation
- [ ] Custom colormap editor
- [ ] Terrain sculpting tools (manual height editing)
