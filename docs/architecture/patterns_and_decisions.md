# Design Patterns, Anti-Patterns, and Architectural Decisions

This document outlines the key architectural patterns, design choices, and potential anti-patterns identified in the codebase. Last updated: 2025-11-24.

---

## Table of Contents
1. [Architectural Decisions](#architectural-decisions)
2. [Design Patterns](#design-patterns)
3. [Anti-Patterns & Technical Debt](#anti-patterns--technical-debt)
4. [Code Smells & Refactoring Opportunities](#code-smells--refactoring-opportunities)

---

## Architectural Decisions

### ADR-001: GPU-First Computation
**Decision:** The core terrain generation logic is implemented in GLSL fragment shaders executed via ModernGL, rather than using CPU-based NumPy operations.

**Rationale:** 
- Terrain generation (noise, erosion) is embarrassingly parallel - each pixel/texel can be computed independently
- GPU execution provides 10-100x performance improvements over CPU for resolutions > 512x512
- Enables real-time parameter tuning and high-resolution generation (2k-8k) with subsecond feedback
- Modern GPUs (even integrated) have sufficient compute power for this workload

**Implications:**
- **Positive**: Near-instant generation allows for interactive exploration of parameter space
- **Negative**: Requires an OpenGL context (headless or windowed) - limits deployment to GPU-equipped machines
- **Negative**: Debugging is harder - shader errors are cryptic, no Python debugger access
- **Negative**: Data marshaling overhead between CPU (NumPy) and GPU (Textures/Buffers)
- **Mitigation**: Comprehensive logging, visualization modes for debugging, CPU fallback for CI/testing

**Status**: ✅ Stable, core to the architecture

---

### ADR-002: Hybrid Python/GLSL Pipeline
**Decision:** Python acts as the orchestrator for state management, configuration, and I/O, while GLSL handles compute-intensive operations.

**Rationale:**
- Python provides excellent ecosystem for configuration (dataclasses, YAML/JSON), UI (Streamlit), and file I/O (NumPy, PIL, Matplotlib)
- GLSL provides raw performance for parallel computation
- Clear separation allows independent optimization of both layers

**Implications:**
- **Positive**: Leverages strengths of both languages - Python's flexibility + GLSL's performance
- **Positive**: Clear interface boundary at shader uniform passing and framebuffer readback
- **Negative**: Debugging requires context-switching between Python and shader mindset
- **Negative**: Type safety breaks at the Python/GLSL boundary (uniforms are stringly-typed)

**Implementation**:
- `src/generators/*.py`: Pipeline orchestration, context management, uniform passing
- `src/shaders/*.frag`: Algorithm implementation (noise, erosion, lighting)

**Status**: ✅ Stable, well-established pattern

---

### ADR-003: Fragment Shaders as GPGPU (Not Compute Shaders)
**Decision:** Use OpenGL Fragment Shaders (full-screen quad rendering) for general-purpose GPU computation instead of Compute Shaders.

**Rationale:**
- **Compatibility**: Fragment shaders require OpenGL 3.3 (2010), Compute Shaders require OpenGL 4.3 (2012)
- **Wider hardware support**: Works on older integrated graphics, macOS (legacy OpenGL 4.1), budget laptops
- **Simplicity**: "Render to texture" pattern is well-understood and integrates naturally with visualization
- **Sufficient performance**: For embarrassingly parallel tasks like noise generation, fragment shaders are equally fast

**Tradeoffs**:
- Compute shaders offer better control over work group sizes and shared memory
- Fragment shader approach wastes minimal overhead on rasterization

**Status**: ✅ Stable, unlikely to change (compatibility is critical)

---

### ADR-004: Headless Context as Default
**Decision:** Use `moderngl.create_standalone_context()` with `backend="egl"` by default.

**Rationale:**
- Enables running on headless servers, Docker containers, CI/CD pipelines
- No dependency on X11/Wayland/Windows display server
- Batch generation workflows can run in background

**Implications:**
- **Positive**: Server deployments, automation-friendly
- **Negative**: EGL backend may not be available on all systems (Windows, macOS)
- **Mitigation**: Fallback to default backend if EGL unavailable (see `gl_context.py`)

**Status**: ✅ Stable, with robust fallback

---

### ADR-005: Stateless Generator Design
**Decision:** Generator classes (`ErosionTerrainGenerator`, `HydraulicErosionGenerator`) are designed to be instantiated, used to generate one or more maps, then explicitly cleaned up. They hold OpenGL resources but don't persist terrain state between generation calls.

**Rationale:**
- Simplifies resource management - explicit `cleanup()` prevents GPU memory leaks
- Atomic generation operations - each `generate_heightmap()` call is independent
- Allows reuse of expensive resources (compiled shaders, framebuffers) across multiple generations with different seeds

**Implications:**
- **Positive**: Predictable resource lifecycle
- **Positive**: Easy to parallelize - multiple generator instances in different threads/processes
- **Negative**: User must remember to call `cleanup()` or use context manager pattern
- **Pattern**: `with ErosionTerrainGenerator(...) as gen:` supported for automatic cleanup

**Status**: ✅ Stable, context manager pattern available

---

### ADR-006: Configuration as Serializable Data
**Decision:** All configuration is defined in typed `@dataclass` objects (`TerrainConfig`, `ErosionParams`, `HydraulicParams`, `RenderConfig`) that can be serialized to YAML/JSON.

**Rationale:**
- **Reproducibility**: Save exact configuration that produced a terrain
- **Preset system**: Ship predefined configurations (`canyon`, `mountains`, `plains`)
- **UI integration**: Streamlit UI binds directly to config dataclass fields
- **Batch workflows**: Generate terrain sets with config variations
- **Version control**: Config files are text-based, git-friendly

**Implementation**:
```python
@dataclass
class ErosionParams:
    height_tiles: float = 3.0
    erosion_strength: float = 0.04
    # ... 20+ parameters
    
    @classmethod
    def canyon(cls) -> "ErosionParams":
        return cls(erosion_strength=0.06, ...)
    
    def override(self, overrides: Dict) -> "ErosionParams":
        return ErosionParams(**{**self.__dict__, **overrides})
```

**Benefits**:
- Type safety with Python type hints
- Auto-completion in IDEs
- Factory methods for presets
- Easy extension with new parameters

**Status**: ✅ Stable, extensively used

---

### ADR-007: Dual Rendering Pipeline (CPU + GPU)
**Decision:** Maintain two distinct rendering paths for different use cases.

**Pipeline 1: CPU (Matplotlib LightSource)**
- **Use case**: High-quality 2D shaded relief for documentation, publications, quick previews
- **Strengths**: Superior colormap support, professional cartographic output, simple API
- **Implementation**: `src/utils/rendering.py::shade_heightmap()`

**Pipeline 2: GPU (ModernGL Shaders)**
- **Use case**: Real-time 3D visualization, raymarching, interactive parameter tuning
- **Strengths**: Fast iteration, dramatic lighting effects, 3D perspective views
- **Implementation**: `ErosionTerrainGenerator::render_visualization()`, `render_raymarch()`

**Rationale**:
- Matplotlib excellent for scientific visualization but slow for real-time
- GPU rendering enables interactive exploration but less control over cartographic conventions
- Different outputs serve different audiences (technical users vs artists/designers)

**Status**: ✅ Stable, both pipelines actively maintained

---

### ADR-008: CLI-First Design
**Decision:** The primary interface is a command-line tool (`gpu_terrain.py`). UI (`app/ui_streamlit.py`) is a separate layer built on top of core logic.

**Rationale:**
- **Automation**: Batch processing, CI/CD integration, scripting workflows
- **Composability**: Pipe outputs to other tools, integrate into larger pipelines
- **Simplicity**: Core logic has no UI framework dependencies
- **Performance**: No GUI overhead for batch generation

**Architecture**:
```
CLI (gpu_terrain.py)
    ↓ calls
Core Generators (src/generators/)
    ↓ produces
TerrainMaps (src/utils/artifacts.py)
    ↓ consumed by
Export/Visualization Utils
    ↓ writes
Output Files
```

**Status**: ✅ Stable, UI is optional enhancement

---

### ADR-009: Ping-Pong Rendering for Iterative Simulations
**Decision:** Use dual framebuffer/texture pairs for multi-pass simulations (hydraulic erosion, thermal erosion).

**Implementation**:
```python
# HydraulicErosionGenerator
for iteration in range(params.iterations):
    # Read from 'a', write to 'b'
    fbo_b.use()
    texture_a.use(location=0)
    render_pass()
    # Swap pointers
    ptr = 'b' if ptr == 'a' else 'a'
```

**Rationale**:
- **GPU Constraint**: Can't read from and write to same texture in single shader pass
- **Efficient**: Avoids CPU readback/upload each iteration - stays on GPU
- **Standard pattern**: Used in physics simulations, cellular automata, fluid dynamics

**Status**: ✅ Stable, essential for hydraulic erosion

---

## Design Patterns


### Pattern 1: Generator Pattern (Creational)
**Intent**: Encapsulate complex terrain generation logic behind a simple interface.

**Implementation**:
```python
class ErosionTerrainGenerator:
    def __init__(self, resolution: int, use_erosion: bool, defaults: ErosionParams):
        # Setup: Create context, compile shaders, allocate framebuffers
        
    def generate_heightmap(self, seed: int, **overrides) -> TerrainMaps:
        # Generate: Set uniforms, render, readback
        
    def cleanup(self):
        # Teardown: Release GPU resources
```

**Benefits**:
- **Encapsulation**: Hides ModernGL complexity (context creation, shader compilation, FBO management)
- **Separation of concerns**: Client code works with high-level `TerrainMaps` objects
- **Resource management**: Generator owns GPU resources lifecycle

**Used in**: `ErosionTerrainGenerator`, `HydraulicErosionGenerator`, `MorphologicalTerrainGPU`

---

### Pattern 2: Data Transfer Object (DTO)
**Intent**: Define a standard contract for passing complex terrain data between components.

**Implementation**:
```python
@dataclass(slots=True)
class TerrainMaps:
    height: np.ndarray           # Float32 heightmap (H×W)
    normals: np.ndarray          # RGB normals (H×W×3)
    erosion_mask: np.ndarray | None
    scatter_map: np.ndarray | None
    
    @classmethod
    def ensure(cls, data) -> "TerrainMaps":
        # Accept dict or TerrainMaps, normalize to TerrainMaps
    
    def height_u16(self) -> np.ndarray:
        # Convert float to 16-bit for PNG export
```

**Benefits**:
- **Type safety**: All components agree on terrain data structure
- **Flexibility**: Optional fields for advanced features
- **Convenience**: Helper methods for common conversions
- **Decoupling**: Generators and exporters don't depend on each other's internals

**Used in**: All generators produce `TerrainMaps`, all exporters/renderers consume it

---

### Pattern 3: Parameter Object (Behavioral)
**Intent**: Reduce method parameter lists by grouping related configuration into cohesive objects.

**Implementation**:
```python
@dataclass
class ErosionParams:
    height_tiles: float = 3.0
    height_octaves: int = 3
    erosion_strength: float = 0.04
    # ... 15+ parameters
    
    @classmethod
    def canyon(cls) -> "ErosionParams":
        """Preset: Deep erosion with branching valleys."""
        return cls(erosion_strength=0.06, erosion_octaves=6, ...)
    
    def override(self, overrides: Dict) -> "ErosionParams":
        """Create modified copy with overrides."""
        return ErosionParams(**{**self.__dict__, **overrides})
    
    def uniforms(self) -> Dict[str, Any]:
        """Export as dict for shader uniform passing."""
        return self.__dict__.copy()
```

**Benefits**:
- **Readability**: `generate(seed=42, params=ErosionParams.canyon())` vs 20-parameter method
- **Extensibility**: Add new parameters without breaking existing calls
- **Presets**: Factory methods provide curated configurations
- **Versioning**: Config files track parameter evolution

**Used in**: `ErosionParams`, `HydraulicParams`, `RenderConfig`, `TerrainConfig`

---

### Pattern 4: Strategy Pattern (Behavioral)
**Intent**: Allow swapping terrain generation algorithms at runtime.

**Implementation**:
```python
# CLI dispatcher (gpu_terrain.py)
if args.generator == "erosion":
    gen = ErosionTerrainGenerator(...)
elif args.generator == "hydraulic":
    gen = HydraulicErosionGenerator(...)
elif args.generator == "morph":
    gen = MorphologicalTerrainGPU(...)

terrain = gen.generate(...)  # Polymorphic interface
```

**Benefits**:
- **Open/Closed Principle**: Add new generators without modifying orchestration code
- **Consistent interface**: All generators produce `TerrainMaps`
- **User choice**: Let users select algorithm for their use case

**Variations**: Each generator has different parameters, but unified output format

**Used in**: CLI, Streamlit UI, batch generation

---

### Pattern 5: Factory Method (Creational)
**Intent**: Encapsulate complex resource creation logic.

**Implementation**:
```python
def create_context(require: int = 330, backend: str | None = "egl") -> moderngl.Context:
    """Create standalone ModernGL context with fallback."""
    try:
        return moderngl.create_standalone_context(require=require, backend=backend)
    except Exception:
        # Fallback if EGL unavailable (Windows, macOS)
        return moderngl.create_standalone_context(require=require)

def create_detail_texture(ctx: moderngl.Context, size: int = 256) -> moderngl.Texture:
    """Generate tiling noise texture for raymarch detail."""
    # Compile shader, render noise, build mipmaps, cleanup, return texture
```

**Benefits**:
- **Abstraction**: Hide platform-specific initialization details
- **Error handling**: Graceful fallbacks for missing backends
- **Reusability**: Centralized logic for common resource creation

**Used in**: `src/utils/gl_context.py`, `create_detail_texture`

---

### Pattern 6: Template Method (Behavioral)
**Intent**: Define skeleton of algorithm in base, let subclasses override specific steps.

**Implementation** (Implicit in shader pipeline):
```python
# Common pattern across all generators
def generate_heightmap(self, seed: int, **kwargs) -> TerrainMaps:
    # 1. Setup (uniform setting) - varies by generator
    self._setup_uniforms(seed, **kwargs)
    
    # 2. Render (dispatch draw call) - standardized
    self._render_pass()
    
    # 3. Readback (GPU→CPU transfer) - standardized
    data = self._readback_framebuffer()
    
    # 4. Post-process (normal computation) - varies by generator
    return self._build_terrain_maps(data)
```

**Benefits**:
- Code reuse for common operations (readback, cleanup)
- Flexibility for algorithm-specific steps

**Note**: Not formally implemented as inheritance hierarchy, but pattern emerges in practice

---

### Pattern 7: Pipeline Pattern (Architectural)
**Intent**: Structure generation as sequence of transformations.

**Implementation**:
```
Input (Seed + Params)
    ↓
[1] Base Noise Generation (FBM/Perlin)
    ↓
[2] Domain Warping (Optional)
    ↓
[3] Erosion Simulation (Hydraulic/Thermal)
    ↓
[4] Normal Map Computation
    ↓
[5] Scatter Map Generation
    ↓
Output (TerrainMaps)
```

**Shader pipeline** (erosion generator):
```python
# Pass 1: Heightmap + erosion
heightmap_fbo.use()
height_program.render()

# Pass 2: Thermal erosion (iterative)
for i in range(iterations):
    pingpong_fbo.use()
    thermal_program.render()
    swap_fbos()

# Pass 3: Recompute normals
normal_fbo.use()
normal_program.render()

# Pass 4: Scatter density
scatter_fbo.use()
scatter_program.render()
```

**Benefits**:
- **Clarity**: Each stage has single responsibility
- **Debuggability**: Can visualize intermediate outputs
- **Modularity**: Easy to enable/disable stages

**Used in**: All generators follow this pattern

---

### Pattern 8: Context Manager (Resource Management)
**Intent**: Ensure GPU resources are released even if exceptions occur.

**Implementation**:
```python
class ErosionTerrainGenerator:
    def __enter__(self) -> "ErosionTerrainGenerator":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

# Usage
with ErosionTerrainGenerator(resolution=1024) as gen:
    terrain = gen.generate_heightmap(seed=42)
    # Auto-cleanup on exit, even if exception raised
```

**Benefits**:
- **Safety**: Prevents GPU memory leaks
- **Pythonic**: Follows RAII pattern
- **Exception-safe**: Cleanup happens in finally block

**Status**: ✅ Implemented for all generators

---

### Pattern 9: Adapter Pattern (Structural)
**Intent**: Convert different terrain data representations to unified format.

**Implementation**:
```python
@classmethod
def ensure(cls, data: "TerrainMaps | Mapping[str, np.ndarray]") -> "TerrainMaps":
    """Accept dict or TerrainMaps, return TerrainMaps."""
    if isinstance(data, TerrainMaps):
        return data
    if isinstance(data, Mapping):
        return cls(
            height=np.asarray(data["height"], dtype=np.float32),
            normals=np.asarray(data["normals"], dtype=np.float32),
            # ... optional fields
        )
    raise TypeError("Terrain data must be TerrainMaps or mapping")
```

**Benefits**:
- **Flexibility**: Accept both dict (legacy) and TerrainMaps (modern)
- **Migration**: Allows gradual refactoring
- **Type safety**: Always returns TerrainMaps

**Used in**: Export functions, rendering utilities

---

## Anti-Patterns & Technical Debt


### ⚠️ Anti-Pattern 1: God Class - `ErosionTerrainGenerator`
**Observation**: `ErosionTerrainGenerator` has accumulated too many responsibilities:
1. **Context management** (OpenGL context creation/cleanup)
2. **Shader compilation** (5+ different shader programs)
3. **Geometry setup** (VBOs, VAOs)
4. **Heightmap generation** (compute)
5. **Thermal erosion simulation** (iterative compute)
6. **2D visualization rendering** (graphics)
7. **3D raymarch rendering** (graphics)
8. **Scatter map generation** (additional compute)

**Current state**: ~500 lines, violates Single Responsibility Principle

**Impact**:
- **Maintenance burden**: Hard to modify one aspect without risk to others
- **Testing complexity**: Unit tests require full OpenGL context
- **Cognitive load**: New contributors overwhelmed by class scope

**Refactoring path**:
```python
# Current (monolithic)
gen = ErosionTerrainGenerator(...)
terrain = gen.generate_heightmap(seed=42)
viz = gen.render_visualization(mode=0)
ray = gen.render_raymarch(exposure=1.0)

# Proposed (separated concerns)
# Phase 1: Split rendering
gen = HeightmapGenerator(...)
terrain = gen.generate(seed=42)

renderer = TerrainRenderer(ctx=gen.ctx)
viz = renderer.render_2d(terrain, mode=0)
ray = renderer.render_3d(terrain, camera=Camera(...))

# Phase 2: Extract simulation
simulator = ThermalErosionSimulator(ctx)
eroded = simulator.apply(terrain, iterations=10)
```

**Recommendation**: 🔶 **High priority** - Extract rendering into separate `TerrainRenderer` class

---

### ⚠️ Anti-Pattern 2: God Script - `gpu_terrain.py`
**Observation**: The CLI entry point has grown to ~600+ lines handling:
- Argument parsing (100+ lines)
- Generator dispatch
- Rendering orchestration
- Export orchestration
- Matplotlib plotting
- Comparison grid generation
- Turntable animation
- File I/O and path management

**Impact**:
- **Testability**: Hard to unit test individual workflows
- **Readability**: Difficult to follow control flow
- **Duplication**: Similar logic repeated (export formats, path construction)

**Refactoring path**:
```python
# Current (procedural script)
args = parse_args()
if args.generator == "erosion":
    gen = ErosionTerrainGenerator(...)
# ... 50 more lines of conditionals

# Proposed (orchestration layer)
from src.orchestration import TerrainPipeline

pipeline = TerrainPipeline.from_cli_args(args)
result = pipeline.execute()
result.export_all(args.export_dir)
```

**Recommendation**: 🔶 **Medium priority** - Extract orchestration into `src/orchestration.py`

---

### ⚠️ Anti-Pattern 3: Stringly-Typed Shader Uniforms
**Observation**: Shader uniforms are set using string keys with no compile-time validation:

```python
# Python side (erosion.py)
program["u_erosionStrength"].value = 0.04  # type: ignore

# GLSL side (erosion_heightmap.frag)
uniform float u_erosionStrength;
```

**Problems**:
- **Silent failures**: Typo in uniform name → no error, just incorrect rendering
- **Refactoring danger**: Rename Python var → must remember to update shader
- **No IDE support**: No autocomplete, no type checking across boundary

**Current mitigation**: `_uniform_name()` method auto-converts snake_case → camelCase
```python
def _uniform_name(self, py_name: str) -> str:
    """erosion_strength → u_ErosionStrength"""
    # Implicit coupling - fragile
```

**Impact**:
- Medium - bugs are caught during testing but waste time

**Alternative approaches**:
1. **Explicit mapping** (verbose but safe):
   ```python
   UNIFORM_MAP = {
       "erosion_strength": "u_erosionStrength",
       "height_tiles": "u_heightTiles",
   }
   ```

2. **Validation on shader compile** (fail fast):
   ```python
   active_uniforms = {u.name for u in program._members.values()}
   for key in params.keys():
       gl_name = uniform_name(key)
       assert gl_name in active_uniforms, f"Missing uniform: {gl_name}"
   ```

3. **Code generation** (overkill but bulletproof):
   Generate Python binding from GLSL uniform declarations

**Recommendation**: 🔷 **Low priority** - Add validation assertions during development mode

---

### ⚠️ Anti-Pattern 4: Manual Resource Management Burden
**Observation**: Generators require explicit `cleanup()` call to release GPU resources:

```python
gen = ErosionTerrainGenerator(...)
terrain = gen.generate_heightmap(seed=42)
gen.cleanup()  # ⚠️ Easy to forget!
```

**Problem**: If `cleanup()` not called → GPU memory leak, especially in long-running apps (Streamlit UI, batch processing)

**Current mitigation**: Context manager support exists but not enforced:
```python
# Good (auto-cleanup)
with ErosionTerrainGenerator(...) as gen:
    terrain = gen.generate(...)

# Bad (manual cleanup)
gen = ErosionTerrainGenerator(...)
terrain = gen.generate(...)
gen.cleanup()  # May forget, may skip on exception
```

**Impact**:
- **Moderate**: Streamlit UI had memory leak issues before adding explicit cleanup
- **Developer friction**: Must remember pattern, especially for new contributors

**Recommendations**:
1. ✅ **Context manager pattern documented** - encourage in examples
2. 🔷 **Consider**: Make `__del__` finalizer call cleanup (Python GC as fallback)
   ```python
   def __del__(self):
       if self._own_ctx and self.ctx is not None:
           self.cleanup()
   ```
3. 🔷 **Linting rule**: Detect generators created without `with` statement

**Status**: 🔷 **Low priority** - Mitigations in place, document best practices

---

### ⚠️ Anti-Pattern 5: Magic Numbers in Shaders
**Observation**: GLSL shaders contain hardcoded constants:

```glsl
// erosion_viz.frag
#define WATER_COLOR vec3(0.1, 0.3, 0.5)
#define CLIFF_COLOR vec3(0.4, 0.35, 0.3)
#define GRASS_COLOR vec3(0.3, 0.5, 0.2)
```

**Problems**:
- **Inflexible**: Changing colors requires editing shader source
- **No runtime control**: Can't tweak via UI/CLI without recompile
- **Duplicate definitions**: Same colors defined in multiple shaders

**Current workaround**: Some colors passed as uniforms, but not consistently

**Impact**:
- **Low-Medium**: Limits artistic control, but shaders are performant

**Refactoring path**:
```glsl
// Define as uniforms
uniform vec3 u_waterColor;
uniform vec3 u_cliffColor;
uniform vec3 u_grassColor;

// Pass from Python
viz_program["u_waterColor"].value = (0.1, 0.3, 0.5)
```

**Tradeoff**: More uniforms = slightly more overhead, but negligible for ~10 colors

**Recommendation**: 🔷 **Low priority** - Make colors configurable in next major refactor

---

### ⚠️ Anti-Pattern 6: Hardcoded Shader Paths
**Observation**: `load_shader()` uses relative path calculation:

```python
# src/utils/shader_loader.py
_SHADER_ROOT = Path(__file__).resolve().parents[1] / "shaders"

def load_shader(name: str) -> str:
    return (_SHADER_ROOT / name).read_text()
```

**Problems**:
- **Fragile**: Moving `utils/` folder breaks everything
- **Packaging issues**: Won't work if project installed as package (not in-place dev)

**Current state**: Works fine for development workflow, but brittle

**Impact**:
- **Low**: Hasn't caused issues yet, but will break if project packaged for PyPI

**Refactoring path**:
```python
# Use importlib.resources (Python 3.9+)
from importlib.resources import files

def load_shader(name: str) -> str:
    shader_files = files("src.shaders")
    return (shader_files / name).read_text()
```

**Recommendation**: 🔷 **Low priority** - Fix when preparing for package distribution

---

### ⚠️ Anti-Pattern 7: Inconsistent Error Handling
**Observation**: Some functions fail silently, others raise exceptions inconsistently:

```python
# Silent failure
if not shader_path.exists():
    return ""  # Returns empty string, causes cryptic GL error later

# Loud failure
if resolution < 64:
    raise ValueError("Resolution must be >= 64")  # Good!

# Inconsistent
try:
    ctx = create_context(backend="egl")
except:
    ctx = create_context()  # Bare except, swallows all errors including KeyboardInterrupt
```

**Problems**:
- **Hard to debug**: Silent failures manifest as mysterious rendering bugs
- **Inconsistent UX**: Some errors crash immediately, others show broken output

**Recommendations**:
1. **Fail fast**: Validate inputs early, raise clear exceptions
   ```python
   if not shader_path.exists():
       raise FileNotFoundError(f"Shader not found: {shader_path}")
   ```

2. **Specific exceptions**: Catch specific errors, let critical ones propagate
   ```python
   try:
       ctx = create_context(backend="egl")
   except moderngl.Error as e:
       # EGL unavailable, fallback
       ctx = create_context()
   # Don't catch KeyboardInterrupt, SystemExit, etc
   ```

3. **Logging**: Use Python `logging` for non-critical issues
   ```python
   if params.thermal_iterations < 0:
       logger.warning("Negative thermal iterations, clamping to 0")
       params.thermal_iterations = 0
   ```

**Recommendation**: 🔶 **Medium priority** - Audit error paths, add validation

---

## Code Smells & Refactoring Opportunities
