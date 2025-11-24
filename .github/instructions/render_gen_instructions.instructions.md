---
applyTo: '**'
---

# Project context

- This workspace is a **Python-based procedural generation and rendering project**.
- Core responsibilities:
  - GPU-accelerated terrain / map generation
  - Rendering via **ModernGL** (desktop OpenGL)
  - Optional browser-facing / WebGL-adjacent workflows (for example PyScript/Pyodide + JS, or image outputs consumed by a web client)
  - A configuration-driven **UI layer** for tweaking generation and render settings

Think of the codebase as three main layers:

1. **Procedural generation core**  
   - Pure Python (NumPy-heavy) logic for terrain, heightmaps, masks, and biome data.
2. **Rendering engine**  
   - ModernGL/OpenGL pipeline (contexts, shaders, buffers, framebuffers, textures).
3. **Configuration & UI**  
   - Config models (YAML/JSON-serializable), presets, and a UI to edit configs and trigger renders.

---

# Coding guidelines

## General

- Use **Python 3.11+** features and **type hints everywhere**.
- Prefer **small, focused functions** and **modular modules** instead of giant scripts.
- Avoid `global` state. Prefer explicit parameters, dependency injection, or clearly scoped objects.
- Keep **procedural generation, rendering, and UI** concerns separated:
  - Generation modules should not know about windowing or UI frameworks.
  - Rendering modules should consume structured data (heightmaps, configs), not hard-coded constants.
  - UI code should call into public APIs instead of importing deep internals.

## Configuration and data modeling

- Represent configuration as **typed models**:
  - Prefer `@dataclass` or Pydantic models for:
    - Render settings (resolution, MSAA, color/depth formats).
    - Camera (projection type, FOV, near/far, orbit parameters).
    - Lighting (direction, intensity, color, fog parameters).
    - Terrain / biome generation (noise types, seeds, octaves, erosion iterations, thresholds).
- Ensure configs are **serializable to YAML/JSON** and versioned (for example `schema_version` field).
- New parameters should:
  - Have sensible defaults.
  - Be added to the config model, any preset files, and the UI layer in a consistent way.

## Procedural generation

- Prefer **functional-style, deterministic** generation based on:
  - explicit seeds
  - explicit configuration objects
- Use standard building blocks:
  - Perlin/Simplex noise, FBM stacks, domain warping
  - Diamond–square / midpoint displacement
  - Post-processing passes (smoothing, erosion, normal generation)
- Make sure functions are **resolution-agnostic** and operate on arrays, not fixed sizes.
- For new algorithms:
  - Provide a small docstring describing the technique and its parameters.
  - Add simple tests that validate shape, ranges, and determinism given a fixed seed.

## Rendering (ModernGL / OpenGL)

- Treat the ModernGL layer as an **engine**:
  - Keep context creation, program compilation, VAO/VBO setup, and framebuffer management in dedicated modules.
  - Avoid mixing heavy math, config parsing, and rendering calls in one place.
- Shaders:
  - Keep GLSL in separate files when possible or in clearly labeled multi-line strings.
  - Use meaningful uniform names that correspond to config fields.
  - Keep vertex/fragment shaders simple and focused; complex logic belongs in the generation step if possible.
- Headless / offscreen rendering:
  - Prefer rendering to textures or framebuffers and saving to images (for example PNG/EXR) for test runs.
  - Use configuration objects to control resolution and formats; avoid hard-coded values.

## UI & interaction

- The UI (web or desktop) is **a client of the config and render APIs**:
  - Do not embed generation logic in event handlers.
  - Use “load config -> edit -> apply -> render” as the mental model.
- Provide:
  - Preset selection (for example coastline / mountains / marsh).
  - Controls for key config groups (render, camera, lighting, terrain, biomes).
  - Buttons for “Preview / Generate”, “Reset to defaults”, “Save preset”, and “Load preset”.
- Keep UI logic thin and predictable; surface errors clearly rather than swallowing them.

---

# Style, quality, and tooling

- Follow **PEP 8** and use **clear, descriptive names**:
  - `generate_heightmap()`, `build_terrain_mesh()`, `create_render_context()`, etc.
- Use **docstrings** for public functions, classes, and modules:
  - Brief summary
  - Important parameters and expected shapes/ranges
- Prefer **NumPy** for heavy numerical work; avoid unnecessary Python loops in hot paths.
- When introducing non-trivial behavior:
  - Add at least a **smoke test** or example script that exercises it.
  - Keep example scripts in a dedicated directory (for example `examples/` or `scripts/`).

---

# How AI should behave

- When generating code:
  - Respect the existing layering:
    - generation core  
    - rendering engine  
    - configuration & UI
  - Use type hints and follow the established module boundaries.
  - Avoid adding new third-party dependencies unless necessary; prefer `numpy`, `moderngl`, and whichever UI framework is already present.
  - Make new functions and classes **config-driven**, not hard-coded.
- When answering questions:
  - Explain **tradeoffs** (quality vs performance vs complexity) for new algorithms or rendering features.
  - Prefer patterns that make it easy to reproduce terrain from a seed and config.
- When reviewing changes:
  - Call out:
    - cross-layer coupling (for example UI importing deep rendering internals)
    - hidden global state or side effects
    - duplicated logic that should live in shared helpers
  - Suggest refactors that:
    - clarify responsibilities
    - improve testability and reproducibility

---

# Safety and performance considerations

- Assume renders may be run on **limited GPU/CPU**:
  - Avoid unbounded resolution or iteration counts; validate user inputs.
  - For very heavy operations, prefer batch or offline scripts.
- Be cautious with:
  - large allocations (heightmaps, multiple FBOs) in tight loops
  - blocking the UI thread during long renders (recommend async/background patterns where appropriate)

