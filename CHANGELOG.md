# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added glTF 2.0 mesh export with embedded textures and PBR materials (`export_gltf_mesh`)
- Added comprehensive batch export function (`export_all_formats`) for exporting all formats at once
- Added RAW heightmap export (16-bit little-endian) for Unity/Unreal Engine compatibility
- Added R32 heightmap export (32-bit float) for maximum precision
- Added CLI arguments: `--gltf-out`, `--export-all`, `--export-formats`, `--mesh-scale`, `--mesh-height-scale`
- Created `docs/EXPORT_FORMATS.md` - Complete guide to all export formats with engine-specific workflows
- Created `docs/EXPORT_CLI_REFERENCE.md` - Quick CLI reference for export commands
- Created `examples/export_formats_demo.py` - Comprehensive export demonstration script
- Added procedural vegetation/biome system ("Scatter Map") generating density maps for trees, rocks, and grass.
- Added `scatter_density.frag` shader for GPU-accelerated biome placement based on height, slope, and erosion.
- Added `--scatter-out` CLI argument to `gpu_terrain.py` for exporting biome density maps.
- Added "Scatter Map" export option to the Streamlit UI.
- Added `u_time` uniform to `erosion_viz.frag` and `erosion_raymarch.frag` for time-based animations.
- Implemented animated water waves and cloud shadows in terrain shaders.
- Added `time` parameter to `render_visualization` and `render_raymarch` in `ErosionTerrainGenerator`.
- Added Context Manager support (`__enter__`/`__exit__`) to `ErosionTerrainGenerator`, `HydraulicErosionGenerator`, and `MorphologicalTerrainGPU` for automatic resource cleanup.
- Created `docs/architecture/rendering_pipeline.md` documentation.
- Created `docs/architecture/patterns_and_decisions.md` documentation.
- Created `docs/architecture/dependency_graph.md` with visual maps and coupling analysis.
- Created `docs/architecture/knowledge_graph.md` and `tools/build_knowledge_graph.py` for automated codebase analysis.
- Created `docs/index.md` documentation entry point.

### Changed
- Enhanced mesh export functions with configurable scale parameters
- Updated README.md with comprehensive export format examples
- Refactored `src/utils/shader_loader.py` to use `importlib.resources` for robust package-based resource loading.
- Updated `README.md`, `ADVANCED_RENDERING.md`, and `HYDRAULIC_EROSION.md` with improved documentation.
- Optimized `ErosionTerrainGenerator` rendering pipeline by caching Vertex Array Objects (VAOs) instead of recreating them per frame, significantly improving render loop performance.

