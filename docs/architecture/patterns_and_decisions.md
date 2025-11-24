# Design Patterns, Anti-Patterns, and Architectural Decisions

This document outlines the key architectural patterns, design choices, and potential anti-patterns identified in the codebase.

## Architectural Decisions

### 1. GPU-First Computation
**Decision:** The core terrain generation logic is implemented in GLSL shaders and executed via ModernGL, rather than using CPU-based NumPy operations.
**Rationale:** Terrain generation (noise, erosion) is highly parallelizable. GPU execution offers orders-of-magnitude performance improvements, enabling real-time parameter tuning and high-resolution generation (up to 4k/8k) that would be prohibitively slow on CPU.
**Implications:**
- Requires an OpenGL context (headless or windowed).
- Debugging logic is harder (shader debugging vs Python debugging).
- Data must be marshaled between CPU (NumPy) and GPU (Textures/Buffers).

### 2. Hybrid Pipeline
**Decision:** Python acts as the orchestrator, managing state, configuration, and I/O, while GLSL handles the heavy computation.
**Rationale:** Python provides a flexible, high-level interface for configuration and UI (Streamlit), while GLSL provides raw performance.
**Implications:**
- Clear separation of concerns: `src/generators/*.py` manages the pipeline, `src/shaders/*.frag` implements the algorithms.

### 3. Stateless Generators (Mostly)
**Decision:** Generator classes (`ErosionTerrainGenerator`) are designed to be instantiated, used to generate a map, and then cleaned up. They hold the GL context and resources but don't persist terrain state between generations (unless explicitly managed by the caller).
**Rationale:** Simplifies resource management. Each generation request is an atomic operation.

## Design Patterns

### 1. Generator Pattern
**Implementation:** `ErosionTerrainGenerator`, `HydraulicErosionGenerator`.
**Description:** Encapsulates the complex logic of creating a terrain map. The client simply provides configuration (`ErosionParams`) and receives a result (`TerrainMaps`).
**Benefit:** Hides the complexity of ModernGL context creation, shader compilation, and framebuffer management from the user.

### 2. Data Transfer Object (DTO)
**Implementation:** `TerrainMaps`, `ErosionParams`.
**Description:**
- `ErosionParams`: A `@dataclass` that carries all configuration data for the generator. It includes factory methods for presets (`canyon()`, `mountains()`).
- `TerrainMaps`: A container for the output data (height, normals, erosion mask) with helper methods for format conversion (`height_u16`, `normal_map_u8`).
**Benefit:** Keeps method signatures clean and ensures type safety. Decouples data structure from logic.

### 3. Factory Pattern (Implicit)
**Implementation:** `create_context`, `load_shader`.
**Description:** Helper functions that abstract the creation of complex objects (OpenGL contexts) or resources (shader strings from files).
**Benefit:** Centralizes resource creation logic, making it easier to handle platform-specific quirks (e.g., EGL vs Windowed context).

### 4. Pipeline Pattern
**Implementation:** The `generate_heightmap` method in `ErosionTerrainGenerator`.
**Description:** The generation process is a linear pipeline:
1.  **Setup**: Configure uniforms.
2.  **Base Pass**: Render noise/heightmap.
3.  **Erosion Pass**: (Optional) Apply thermal/hydraulic erosion via ping-pong rendering.
4.  **Readback**: Transfer data from GPU to CPU.
**Benefit:** Makes the generation flow easy to follow and modify.

### 5. Ping-Pong Rendering
**Implementation:** `apply_thermal_erosion` in `ErosionTerrainGenerator`.
**Description:** Uses two framebuffers (`heightmap_fbo`, `pingpong_fbo`) to simulate iterative processes. The output of step `N` becomes the input of step `N+1`.
**Benefit:** Essential for simulation-based effects (erosion, cellular automata) on the GPU.

## Anti-Patterns & Technical Debt

### 1. God Class Potential (`ErosionTerrainGenerator`)
**Observation:** `ErosionTerrainGenerator` handles context creation, shader compilation, geometry setup, rendering, *and* visualization.
**Risk:** It violates the Single Responsibility Principle. As more features (e.g., new erosion types) are added, this class could become unmaintainable.
**Refactoring Opportunity:** Extract the visualization logic into a separate `TerrainRenderer` class.

### 2. Hardcoded Shader Paths
**Observation:** `load_shader` relies on a relative path structure (`_SHADER_ROOT = Path(__file__).resolve().parents[1] / "shaders"`).
**Risk:** If the project structure changes (e.g., moving `src/utils`), this will break.
**Mitigation:** Use a resource loader or package-based resource access (e.g., `importlib.resources`).

### 3. Magic Numbers in Shaders
**Observation:** Shaders often contain hardcoded constants (e.g., `#define CLIFF_COLOR vec3(...)`).
**Risk:** Makes it hard to tune visuals without editing code.
**Mitigation:** Move these constants to uniforms passed from the Python config.

### 4. Resource Management Complexity
**Observation:** The user must manually call `cleanup()` on generators to release ModernGL resources. Python's GC won't automatically release GPU handles.
**Risk:** Memory leaks if `cleanup()` is forgotten, especially in a long-running app like the Streamlit UI.
**Mitigation:** Implement `__enter__` and `__exit__` for Context Manager support (`with ErosionTerrainGenerator(...) as gen:`).

### 5. Uniform Naming Convention Coupling
**Observation:** The `_uniform_name` method automatically converts Python snake_case to GLSL camelCase (e.g., `erosion_strength` -> `u_ErosionStrength`).
**Risk:** Implicit coupling. Renaming a Python variable breaks the shader silently.
**Mitigation:** Explicit mapping or strict validation during shader compilation (checking active uniforms).
