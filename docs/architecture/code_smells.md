# Code Smells & Refactoring Opportunities

This document catalogs specific code smells, areas for improvement, and refactoring suggestions. Last updated: 2025-11-24.

---

## 1. Duplicate Code

### Smell: Framebuffer Setup Duplication
**Location**: `ErosionTerrainGenerator.__init__()`, `HydraulicErosionGenerator.__init__()`

**Observation**: Both generators have nearly identical code for:
- Creating fullscreen quad vertices
- Creating VBO/IBO buffers
- Setting up vertex arrays

```python
# Repeated in erosion.py and hydraulic.py
vertices = np.array([
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,
], dtype="f4")
indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
self.vbo = self.ctx.buffer(vertices.tobytes())
self.ibo = self.ctx.buffer(indices.tobytes())
```

**Impact**: Maintenance burden - fix bug in one place, must remember to fix elsewhere

**Refactoring**:
```python
# src/utils/gl_helpers.py
def create_fullscreen_quad(ctx: moderngl.Context) -> tuple[Buffer, Buffer]:
    """Create VBO/IBO for fullscreen quad (0,0) to (1,1)."""
    vertices = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype="f4")
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
    return ctx.buffer(vertices.tobytes()), ctx.buffer(indices.tobytes())

# Usage
self.vbo, self.ibo = create_fullscreen_quad(self.ctx)
```

**Priority**: 🔷 Low - Works fine, consolidate during next refactor

---

### Smell: Export Path Construction Duplication
**Location**: `gpu_terrain.py` main function

**Observation**: Similar logic repeated for each export format:
```python
if args.heightmap_out:
    save_heightmap_png(args.heightmap_out, terrain)
if args.normals_out:
    save_normal_map_png(args.normals_out, terrain)
if args.obj_out:
    export_obj_mesh(args.obj_out, terrain, ...)
# ... 10+ more formats
```

**Refactoring**:
```python
EXPORT_HANDLERS = {
    "heightmap": (lambda p, t, a: save_heightmap_png(p, t)),
    "normals": (lambda p, t, a: save_normal_map_png(p, t)),
    "obj": (lambda p, t, a: export_obj_mesh(p, t, scale=a.mesh_scale)),
    # ...
}

def dispatch_exports(terrain: TerrainMaps, args: argparse.Namespace):
    """Execute all requested exports based on CLI args."""
    for format_name, handler in EXPORT_HANDLERS.items():
        output_path = getattr(args, f"{format_name}_out", None)
        if output_path:
            handler(output_path, terrain, args)
```

**Priority**: 🔶 Medium - Would significantly simplify `gpu_terrain.py`

---

## 2. Long Methods

### Smell: 100+ Line `main()` Function
**Location**: `gpu_terrain.py::main()`

**Observation**: Single function handles entire workflow:
- Generator selection (20 lines)
- Generation (30 lines)
- Comparison grid logic (40 lines)
- Export dispatch (60+ lines)
- Cleanup (10 lines)

**Impact**: Hard to follow control flow, difficult to unit test individual steps

**Refactoring**:
```python
def main():
    args = _parse_args()
    
    # Step 1: Create generator
    generator = create_generator(args)
    
    # Step 2: Generate terrain(s)
    if args.compare_seeds:
        results = generate_comparison_set(generator, args)
    else:
        results = [generate_single(generator, args)]
    
    # Step 3: Export results
    for terrain in results:
        export_terrain(terrain, args)
    
    # Step 4: Cleanup
    generator.cleanup()

def create_generator(args) -> TerrainGenerator:
    """Factory for generator instances."""
    ...

def generate_single(gen, args) -> TerrainMaps:
    """Generate single terrain from args."""
    ...

def export_terrain(terrain, args):
    """Dispatch all exports for a terrain."""
    ...
```

**Priority**: 🔶 Medium - Improves testability and readability

---

## 3. Large Classes

### Smell: `ErosionTerrainGenerator` Complexity
**Metrics**:
- **Lines of code**: ~500
- **Methods**: 15+
- **Shader programs**: 6
- **Responsibilities**: 8+ (see Anti-Pattern 1 in main doc)

**Cyclomatic complexity**: High in `generate_heightmap()` method

**Refactoring** (see Anti-Pattern 1 for full plan):
- Extract rendering: `TerrainRenderer`
- Extract simulation: `ThermalErosionSimulator`
- Core generator focuses only on base heightmap generation

**Priority**: 🔶 High - Blocking future maintainability

---

## 4. Primitive Obsession

### Smell: Resolution as `int` Everywhere
**Observation**: Resolution passed as bare `int` throughout codebase:

```python
def generate_heightmap(self, resolution: int = 512, ...):
    # What are the constraints?
    # Must be power of 2? Multiple of 64?
    # Min/max values?
```

**Problems**:
- No validation at creation
- Easy to pass invalid values
- Unclear constraints

**Refactoring**:
```python
@dataclass
class Resolution:
    """Valid texture resolution (power of 2, 64-8192)."""
    value: int
    
    def __post_init__(self):
        if not (64 <= self.value <= 8192):
            raise ValueError(f"Resolution must be 64-8192, got {self.value}")
        if self.value & (self.value - 1) != 0:
            raise ValueError(f"Resolution must be power of 2, got {self.value}")
    
    @classmethod
    def from_int(cls, value: int) -> "Resolution":
        """Round to nearest valid resolution."""
        clamped = max(64, min(8192, value))
        power = 2 ** round(math.log2(clamped))
        return cls(power)

# Usage
res = Resolution(1024)  # Validated
res = Resolution.from_int(1000)  # Auto-rounds to 1024
```

**Tradeoff**: More code for simple concept, but catches bugs early

**Priority**: 🔷 Low - Current validation works, apply to next major version

---

## 5. Feature Envy

### Smell: `TerrainMaps` Methods Operating on `np.ndarray`
**Location**: `src/utils/artifacts.py::TerrainMaps`

**Observation**: Methods like `height_u16()` just call NumPy operations:

```python
def height_u16(self) -> np.ndarray:
    return (np.clip(self.height, 0.0, 1.0) * 65535).astype(np.uint16)
```

**Is this actually a problem?** 🤔

**Argument FOR current design**:
- Convenience - users don't need to remember conversion formulas
- Consistency - all conversions in one place
- Testability - easy to test conversions

**Argument AGAINST**:
- Violates "Tell, Don't Ask" - operates on internal data
- Could be free functions: `to_u16(terrain.height)`

**Verdict**: 🟢 Not actually a smell - convenience methods are fine for DTOs

**Priority**: ✅ No action needed

---

## 6. Shotgun Surgery

### Smell: Adding New Generator Requires Many File Changes
**Observation**: To add new generator type, must modify:
1. `src/generators/new_generator.py` (new file)
2. `src/generators/__init__.py` (export)
3. `src/config.py` (add params to TerrainConfig)
4. `gpu_terrain.py` (add CLI args, dispatch logic)
5. `app/ui_streamlit.py` (add UI controls)
6. Tests (add test cases)

**Impact**: High friction for adding new algorithms

**Mitigation** (plugin architecture):
```python
# src/generators/registry.py
class GeneratorRegistry:
    _generators: Dict[str, Type[TerrainGenerator]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(generator_cls):
            cls._generators[name] = generator_cls
            return generator_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> TerrainGenerator:
        return cls._generators[name](**kwargs)

# Usage
@GeneratorRegistry.register("erosion")
class ErosionTerrainGenerator:
    ...

@GeneratorRegistry.register("my_new_algorithm")
class MyNewGenerator:
    ...

# Automatic CLI support
gen = GeneratorRegistry.create(args.generator, resolution=512)
```

**Priority**: 🔷 Low - Current system works, apply if more generators added

---

## 7. Data Clumps

### Smell: (azimuth, altitude, vert_exag) Always Together
**Location**: Rendering functions throughout codebase

**Observation**: These three parameters always travel together:
```python
def shade_heightmap(
    terrain,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 2.0,
    ...
):
```

**Current solution**: Already addressed with `RenderConfig` dataclass ✅

**Example**:
```python
@dataclass
class LightingParams:
    azimuth: float = 315.0
    altitude: float = 45.0
    intensity: float = 2.0

@dataclass  
class RenderConfig:
    lighting: LightingParams = field(default_factory=LightingParams)
    vert_exag: float = 2.0
    # ...
```

**Priority**: ✅ Already fixed in current architecture

---

## 8. Speculative Generality

### Smell: Unused Abstraction Layers?
**Question**: Are there over-engineered abstractions that add complexity without benefit?

**Audit**:
- `TerrainMaps.ensure()` - ✅ **Used**: Allows dict/TerrainMaps flexibility
- `ErosionParams.override()` - ✅ **Used**: CLI parameter overrides
- `create_context(backend=...)` - ✅ **Used**: EGL fallback
- Seamless tiling mode - ✅ **Used**: Requested feature

**Verdict**: 🟢 No significant over-engineering detected

---

## 9. Comments Instead of Code

### Smell: Unclear Variable Names Requiring Comments
**Location**: Various shaders

**Example**:
```glsl
float v = 0.5;  // Visibility factor for ambient occlusion
```

**Better**:
```glsl
float ambientVisibility = 0.5;
```

**In Python code**:
```python
# ptr points to current valid buffer ('a' or 'b')
ptr_h = 'a'
```

**Better**:
```python
active_height_buffer = 'a'  # or use enum
```

**Priority**: 🔷 Low - Gradual cleanup during refactoring

---

## 10. Mutable Default Arguments (Avoided ✅)

**Audit**: Check for dangerous pattern `def func(param=[]):`

**Result**: All generators use immutable defaults or `None`:
```python
def generate_heightmap(
    self,
    seed: int = 0,
    seamless: bool = False,
    params: ErosionParams | None = None,  # ✅ Correct
):
    params = params or ErosionParams()
```

**Verdict**: 🟢 No instances found - good practice followed

---

## Summary & Prioritized Action Items

### High Priority 🔶
1. **Extract rendering from `ErosionTerrainGenerator`** - Major refactor
2. **Refactor `gpu_terrain.py` main()** - Break into functions

### Medium Priority 🔷
1. **Audit error handling** - Add validation, specific exceptions
2. **Export dispatch pattern** - Simplify CLI export logic
3. **Consolidate quad creation** - DRY principle

### Low Priority 🔷
1. **Shader path using importlib.resources** - Future packaging
2. **Resolution value object** - Type safety
3. **Make shader colors configurable** - Artistic control
4. **Plugin architecture for generators** - Extensibility

### No Action Needed ✅
- DTOs with convenience methods
- Dataclass-based configuration
- Context manager support
- Immutable default arguments

---

## Testing Recommendations

To address code smells, expand test coverage for:

1. **Resource cleanup** - Test memory leaks with repeated generation
2. **Error paths** - Test invalid inputs, missing files, context failures
3. **Edge cases** - Test minimum/maximum resolutions, extreme parameters
4. **Integration** - Test full CLI workflows programmatically

**Current test status**: Basic generator tests exist, needs expansion

---

## Documentation Needs

Areas needing better documentation:

1. **Contributor guide** - How to add new generator types
2. **Shader conventions** - Uniform naming, output channel meanings
3. **Performance tuning** - Resolution vs quality tradeoffs
4. **Error troubleshooting** - Common issues and solutions

---

*This document should be reviewed and updated quarterly or when major refactoring occurs.*
