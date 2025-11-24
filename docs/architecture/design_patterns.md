# Design Patterns & Architectural Decisions

This document documents the key design patterns, architectural decisions (ADRs), and potential areas for refactoring identified in the `map_gen` codebase.

## Design Patterns

### 1. Pipeline Pattern
**Context:** Terrain generation involves multiple sequential steps (Noise -> Erosion -> Thermal -> Rendering).
**Implementation:** The `ErosionTerrainGenerator` and `HydraulicErosionGenerator` orchestrate a linear pipeline of operations.
- **Example:** `generate_heightmap` calls `height_program` -> `apply_thermal_erosion` -> `normal_program`.

### 2. Strategy Pattern
**Context:** The system supports multiple generation algorithms that can be swapped at runtime.
**Implementation:** The CLI (`gpu_terrain.py`) selects a generator strategy (`erosion`, `hydraulic`, `morph`) based on user input. All generators produce a compatible `TerrainMaps` output.

### 3. Data Transfer Object (DTO)
**Context:** Complex terrain data (height, normals, masks) needs to be passed between generators, exporters, and visualizers.
**Implementation:** `TerrainMaps` (in `src/utils/artifacts.py`) is a frozen dataclass that acts as a standard contract for terrain data. It includes helper methods for type conversion (`height_u16`, `normal_map_u8`).

### 4. Parameter Object
**Context:** Generators require many configuration parameters.
**Implementation:** `ErosionParams` and `HydraulicParams` encapsulate these settings, providing defaults, presets (`canyon`, `plains`), and serialization methods (`uniforms`).

### 5. Ping-Pong Buffering
**Context:** Iterative simulations (hydraulic erosion, thermal erosion) need to read from the previous state and write to a new state.
**Implementation:** `HydraulicErosionGenerator` and `ErosionTerrainGenerator` (thermal step) use pairs of textures/FBOs (`_a` and `_b`) and swap them after each iteration.

### 6. Factory Method
**Context:** Creating OpenGL resources requires complex setup.
**Implementation:** `create_context` and `create_detail_texture` in `src/utils/gl_context.py` abstract away the initialization logic.

## Architectural Decisions (ADRs)

### ADR-001: Fragment Shaders for GPGPU
**Decision:** Use OpenGL Fragment Shaders (rendering a full-screen quad) for general-purpose computation (erosion simulation) instead of Compute Shaders.
**Rationale:**
- **Compatibility:** Fragment shaders (OpenGL 3.3) are supported on a wider range of hardware and drivers (including older integrated graphics) compared to Compute Shaders (OpenGL 4.3+).
- **Simplicity:** The "render to texture" workflow is well-understood and integrates easily with the visualization pipeline.

### ADR-002: Headless Context
**Decision:** Use `moderngl.create_standalone_context` for generation.
**Rationale:** Allows the tool to run on servers or CI/CD environments without a display server.

### ADR-003: CLI-First Design
**Decision:** The core application is a Command Line Interface (`gpu_terrain.py`). The UI (`app/ui_streamlit.py`) is a wrapper around the core logic.
**Rationale:** Enables batch processing, automation, and easy integration into other pipelines.

## Anti-Patterns & Refactoring Opportunities

### 1. God Class (Visualization in Generator)
**Observation:** `ErosionTerrainGenerator` is responsible for:
1.  Generating heightmaps (Compute).
2.  Simulating thermal erosion (Compute).
3.  Rendering 2D visualizations (Graphics).
4.  Rendering 3D raymarched views (Graphics).
**Impact:** The class is large and violates the Single Responsibility Principle.
**Refactoring:** Extract visualization logic into a dedicated `TerrainRenderer` class that accepts a `TerrainMaps` object and a Context.

### 2. Mixed Concerns in CLI
**Observation:** `gpu_terrain.py` handles argument parsing, generator orchestration, file I/O, and Matplotlib plotting.
**Impact:** The file is long and hard to test.
**Refactoring:** Move plotting logic to `src/utils/visualization.py` and file I/O to `src/utils/export.py` (mostly done, but some logic remains in CLI).

### 3. Hardcoded Shader Paths
**Observation:** `load_shader` relies on relative paths from its own file location.
**Impact:** Moving the `utils` module could break shader loading.
**Refactoring:** Use a resource loader (like `importlib.resources`) or a config-driven path system.
