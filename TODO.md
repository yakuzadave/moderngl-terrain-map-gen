## Terrain Gen / Rendering TODO

### Completed ✅

- ✅ **Rendering Improvements** - Enhanced raymarch shader (Nov 23, 2025)
  - Atmospheric scattering (height fog + sun glare)
  - Improved water rendering (Beer's law absorption + reflections)
  - Triplanar mapping for detail textures
  - Verified with `test_raymarch.py`

- ✅ **Seamless tiling** - Implemented domain repetition for tileable terrain (Nov 23, 2025)
  - GLSL shader modifications (hashSeamless, noisedSeamless, erosionSeamless)
  - Python API integration (seamless parameter)
  - Automated verification script (verify_seamless.py)
  - Full documentation (SEAMLESS_FIX_REPORT.md, SEAMLESS_IMPLEMENTATION_SUMMARY.md)
  - Edge continuity verified (< 0.003 max difference)
  - 14 presets tested successfully
  - Added seamless toggle to Streamlit UI

- ✅ **Configuration UI** - Streamlit-based interface for terrain generation (Nov 23, 2025)
  - Full-featured web UI at localhost:8502
  - TerrainConfig unified model with YAML serialization
  - Sidebar with presets and quick controls
  - Tabbed settings panels (Terrain, Lighting, Rendering, Post-processing)
  - Real-time preview generation
  - Comprehensive documentation (docs/config_ui.md)

- ✅ **Sample render generation** - Automated testing across presets (Nov 23, 2025)
  - generate_sample_renders.py script
  - 14 render presets tested
  - Performance analysis and quality metrics
  - Comprehensive report (SAMPLE_RENDER_REPORT.md)
  - Comparison grid generation

- ✅ **Hydraulic Erosion Simulation** - Implemented Pipe Model erosion (Nov 24, 2025)
  - Multi-pass ping-pong rendering (Flux, Water, Erosion, Advection, Evaporation)
  - Added **Thermal Erosion** (Talus Deposition) pass for realistic slope stabilization
  - GLSL shaders for physical simulation
  - Python backend (HydraulicErosionGenerator)
  - Integrated into CLI and Streamlit UI

- ✅ **Seamless CLI flag** - Added `--seamless` to gpu_terrain.py (Nov 24, 2025)

- ✅ **HDR export fallback** - np.save fallback for EXR when imageio unavailable

- ✅ **Diagnostic GPU Viz** - Added Slope and Curvature visualization modes (Nov 24, 2025)
  - Mode 4: Slope (Gradient magnitude)
  - Mode 5: Curvature (Laplacian)

- ✅ **Advanced Raymarching** - Implemented Soft Shadows and Ambient Occlusion (Nov 24, 2025)
  - Soft shadows using distance field marching
  - Horizon-Based Ambient Occlusion (HBAO) approximation
  - Improved atmospheric scattering model

- ✅ **Marimo Dashboard** - Interactive UI refactor and enhancements (Nov 24, 2025)
  - Converted `terrain_gen.ipynb` to fully reactive `terrain_gen_marimo.py`
  - Added sidebar controls, collapsible settings, and real-time preview
  - Implemented file exports (16-bit PNG, OBJ mesh, NPY raw)
  - Added `postprocessing_marimo.py` for interactive FX tuning
  - Added `advanced_rendering_marimo.py` for lighting studies
  - **Added Interactive 3D Camera** (Yaw/Pitch/Zoom) to main dashboard

- ✅ **Artifact Reduction** - Fixed grid patterns in erosion generator (Nov 24, 2025)
  - Replaced GLSL hash function with high-quality sine-dot implementation
  - Added Domain Warp parameter to break up directional artifacts

- ✅ **Biome System** - Implemented Moisture, Temperature, and Biome classification (Nov 24, 2025)
  - Added Multiple Render Target (MRT) support to `ErosionTerrainGenerator`
  - Implemented `generateMoisture` and `getBiome` in GLSL
  - Exposed `moisture_map`, `temperature_map`, `biome_map` in `TerrainMaps`

### High Priority 🔴

- Visual inspection of seamless 2×2 tiled outputs (human verification)
- Migrate technical documentation to docs/ folder structure

### Medium Priority 🟡

- Add sun/sky controls to raymarcher (exposure, fog color/density/height, sun intensity) for consistent outputs
- Add raymarch tunables for max steps and minimum step size to balance quality vs. speed
- Add HDR (EXR) export path for shaded/ray renders (proper imageio implementation)
- Add multi-seed comparison grid render for quick visual QA
- Add regression/smoke tests: shader compilation, normal packing, ray render non-black variance

### Low Priority 🟢

- Optimize seamless modulo operations (pre-compute wrapped coordinates)
- Hybrid seamless mode (low-frequency seamless, high-frequency non-seamless)
- Automated visual regression tests for seamless tiling
- Performance profiling at different resolutions (256, 512, 1024, 2048)
- GPU texture wrapping modes investigation
