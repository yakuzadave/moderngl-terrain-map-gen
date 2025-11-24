# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added procedural vegetation/biome system ("Scatter Map") generating density maps for trees, rocks, and grass.
- Added `scatter_density.frag` shader for GPU-accelerated biome placement based on height, slope, and erosion.
- Added `--scatter-out` CLI argument to `gpu_terrain.py` for exporting biome density maps.
- Added "Scatter Map" export option to the Streamlit UI.
- Added `u_time` uniform to `erosion_viz.frag` and `erosion_raymarch.frag` for time-based animations.
- Implemented animated water waves and cloud shadows in terrain shaders.
- Added `time` parameter to `render_visualization` and `render_raymarch` in `ErosionTerrainGenerator`.
- Created `docs/architecture/rendering_pipeline.md` documentation.
- Created `docs/architecture/patterns_and_decisions.md` documentation.
- Created `docs/architecture/dependency_graph.md` with visual maps and coupling analysis.
- Created `docs/architecture/knowledge_graph.md` and `tools/build_knowledge_graph.py` for automated codebase analysis.
- Created `docs/index.md` documentation entry point.

### Changed
- Updated `README.md`, `ADVANCED_RENDERING.md`, and `HYDRAULIC_EROSION.md` with improved documentation.
- Optimized `ErosionTerrainGenerator` rendering pipeline by caching Vertex Array Objects (VAOs) instead of recreating them per frame, significantly improving render loop performance.

