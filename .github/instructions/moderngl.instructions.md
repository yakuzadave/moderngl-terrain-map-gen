---
applyTo: '**'
---

# ModernGL usage guidelines

These instructions specialize GitHub Copilot for working with **ModernGL** in this workspace.  
They sit on top of the general project rules (procedural generation core + rendering engine + config & UI).

---

## ModernGL mental model for this project

Treat ModernGL as a **thin, explicit rendering layer**:

- Python / NumPy: generates data (heightmaps, meshes, textures).
- ModernGL: owns **Context → Program → Buffer/Texture → VertexArray → Framebuffer** pipeline. :contentReference[oaicite:0]{index=0}  
- Everything should be **config driven** (resolutions, formats, samples, etc).

Avoid mixing:

- Terrain algorithms or business logic inside rendering functions.
- Windowing / UI specifics inside core ModernGL helpers.

---

## Context creation and lifecycle

When generating or editing code that touches contexts:

- Prefer using **`glcontext` / `moderngl.create_context()`** or **`moderngl-window`** depending on the project’s existing pattern. :contentReference[oaicite:1]{index=1}  
- Do **not** sprinkle `create_context()` calls everywhere:
  - one context per application / render job
  - pass it into helpers or hide it inside a small `Renderer` / `RenderBackend` class
- For headless renders:
  - use the **headless extras** (`pip install "moderngl[headless]"`) where appropriate :contentReference[oaicite:2]{index=2}  
  - render into offscreen framebuffers and save images

AI should **never** assume a global context. Either:

```python
def render_scene(ctx: moderngl.Context, config: RenderConfig) -> np.ndarray:
    ...
````

or use a small class:

```python
class TerrainRenderer:
    def __init__(self, ctx: moderngl.Context, config: RenderConfig):
        ...
```

---

## Core ModernGL pipeline patterns

When Copilot generates rendering code, encourage this canonical flow:

1. **Create buffers / textures** in GPU memory

   * Use `ctx.buffer(...)` for vertex/index data.
   * Use `ctx.texture(...)`, `ctx.depth_texture(...)`, etc for images. ([ModernGL][1])

2. **Compile shader programs**

   * `ctx.program(vertex_shader=..., fragment_shader=...)`
   * Keep GLSL in separate files or clearly separated multi-line strings.

3. **Bind data via VertexArray**

   * Use `ctx.vertex_array(program, [(vbo, "3f 2f", "in_position", "in_uv")], ibo)` or `ctx.simple_vertex_array(...)`. ([Scribd][2])

4. **Render into a Framebuffer**

   * For offscreen: `ctx.simple_framebuffer(size=(w, h), components=4, samples=0)` or `ctx.framebuffer(...)`. ([ModernGL][3])
   * Clear explicitly with `fbo.clear(...)`.

5. **Read back results when needed**

   * Use `fbo.read(components=4, dtype="f1"/"f4")` and convert via `np.frombuffer(...)` when you need CPU-side arrays. ([Stack Overflow][4])

6. **Release resources explicitly where appropriate**

   * When creating lots of transient objects in loops, call `.release()` or let them fall out of scope and be GC’d with care.

Generated code that deviates from this pattern should have a clear reason.

---

## Framebuffers and render targets

AI should:

* Prefer **`simple_framebuffer`** for basic offscreen RGBA + depth renders.
* Use full `ctx.framebuffer(color_attachments=[...], depth_attachment=...)` when:

  * multiple color attachments (G-buffers)
  * custom depth textures / renderbuffers are needed. ([ModernGL][5])

Guidelines:

* Size is always `(width, height)` and must match textures/renderbuffers attached.
* Use `fbo.viewport = (0, 0, width, height)` consistently before rendering.
* Clear before use:

```python
fbo.use()
fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
```

* For tiled/procedural outputs, avoid re-creating framebuffers in tight loops; reuse where possible and only change uniforms / bound textures.

---

## Shaders and uniforms

When Copilot writes shaders:

* Stick to **modern GLSL** style (no ancient fixed-function stuff).
* Match attribute and uniform names with Python code clearly:

  * `in_position`, `in_normal`, `in_uv`
  * `u_model`, `u_view`, `u_proj`, `u_time`, `u_heightmap`
* For terrain and heightmaps:

  * Use a vertex shader that takes positions and optionally height values.
  * Use a fragment shader that:

    * samples height/normal/biome maps where appropriate
    * outputs linear-space color; apply tone mapping and gamma in a deliberate way.

In Python:

* Use `program["u_something"].value = ...` or `.write(...)` for matrices.
* Avoid magic numbers inside shaders; prefer config-driven uniforms.

---

## Data flow between NumPy and ModernGL

For generated code that moves terrain data to/from GPU:

* Upload heightmaps, normal maps, and masks via textures:

  * `ctx.texture(size, components=1 or 4, data=np_array.tobytes(), dtype='f1'/'f4')`
* For mesh data:

  * pack positions/normals/UVs into a structured `np.ndarray` and upload with `ctx.buffer(data.tobytes())`.
* When reading back images:

  * `raw = fbo.read(components=4, dtype='f1' or 'f4')`
  * `img = np.frombuffer(raw, dtype='f1' or 'f4').reshape(height, width, components)` ([Stack Overflow][4])

AI should avoid:

* Copying large arrays back-and-forth per frame without reason.
* Mixing Python loops with per-pixel operations that belong in fragment shaders.

---

## Error handling and debugging

When Copilot suggests code that might fail, encourage:

* **Context sanity checks**

  * Ensure a context exists before creating resources.
* **Shader compile / link checking**

  * Catch `moderngl.Error` and log shader sources if compilation fails.
* **Framebuffer completeness**

  * After `ctx.framebuffer(...)`, do a small test render or read to verify.

Engrave some defaults:

* Always check the sizes of arrays before sending them to GPU.
* Log (or assert) that configuration-driven sizes match framebuffer / texture sizes.

---

## Performance and memory

AI should:

* Reuse:

  * Programs, VAOs, and static buffers across frames / renders.
  * Framebuffers for repeated renders at the same resolution.
* Ensure textures use the correct `dtype` and component count for the use case (no 4-channel float textures when 1-channel `f1` is enough).
* Avoid:

  * Recompiling shaders inside render loops.
  * Creating / destroying large buffers every frame.

When adding new features, explain tradeoffs:

* Resolution vs memory vs render time.
* Multi-sampling (`samples>0`) vs performance. ([ModernGL][3])

---

## How AI should behave with ModernGL code

When generating code:

* Follow the **Context → Program → Buffer/Texture → VertexArray → Framebuffer → readback** pipeline.
* Keep rendering functions **pure-ish**: config in, images/arrays out (or clearly defined side effects).
* Use **type hints and clear naming** (`TerrainRenderer`, `create_heightmap_texture`, `build_terrain_vao`, etc).
* Integrate with the existing config models instead of hard-coding parameters.

When answering questions:

* Reference how ModernGL expects objects to be created from a `Context`. ([ModernGL][1])
* Suggest using textures + framebuffers for offscreen terrain baking.
* Recommend reading official docs for finer API details (Context / Framebuffer / Texture / Program reference). ([ModernGL][1])

When reviewing changes:

* Call out:

  * Global or hidden contexts.
  * Shader code that does too much “game logic”.
  * Duplicate or inconsistent buffer / VAO setup.
* Suggest refactors toward:

  * small ModernGL helper modules
  * explicit lifecycles for GPU resources
  * clear data flow from NumPy → GPU → image / analysis outputs


[1]: https://moderngl.readthedocs.io/en/latest/reference/context.html?utm_source=chatgpt.com "Context - ModernGL 5.12.0 documentation"
[2]: https://www.scribd.com/document/744806216/moderngl-readthedocs-io-en-5-6-1?utm_source=chatgpt.com "Moderngl Readthedocs Io en 5.6.1 | PDF"
[3]: https://moderngl.readthedocs.io/en/5.8.2/reference/framebuffer.html?utm_source=chatgpt.com "Framebuffer - ModernGL 5.8.2 documentation - Read the Docs"
[4]: https://stackoverflow.com/questions/56980266/how-do-i-read-a-moderngl-fboframe-buffer-object-back-into-a-numpy-array?utm_source=chatgpt.com "How do I read a moderngl fbo(frame buffer object) back ..."
[5]: https://moderngl.readthedocs.io/en/5.5.1/reference/context.html?utm_source=chatgpt.com "Context — ModernGL 5.5.0 documentation - Read the Docs"