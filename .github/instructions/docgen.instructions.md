---
applyTo: '**'
---


# Documentation guidelines (`docs/` folder)

These instructions specialize GitHub Copilot for maintaining **project documentation**.

All documentation files **except** the root `README.md` belong in the `docs/` folder.

- `README.md`  
  High level project intro, quickstart, and links into `docs/`.
- `docs/`  
  Everything else: architecture, how to guides, references, design notes, ADRs, etc.

---

## Folder and file structure

AI should encourage and preserve a **structured docs tree** instead of dumping random files.

Recommended pattern:

- `docs/index.md`  
  Main entry point / table of contents for the docs.
- `docs/architecture/`  
  High level design: rendering pipeline, config system, UI structure.
- `docs/howto/`  
  Task oriented guides (how to add a new biome, how to create a new render preset).
- `docs/reference/`  
  More exhaustive, API oriented notes, configuration field definitions, shader conventions.
- `docs/decisions/`  
  Architecture Decision Records and design rationale.
- `docs/notes/` or `docs/dev/`  
  Rougher internal notes, experiments, spike results.

Conventions:

- Use **kebab case** for filenames: `render-pipeline.md`, `config-ui.md`, `terrain-generation.md`.
- Keep filenames stable; rename only with a clear reason.

When adding a new doc, also:

- Link it from `docs/index.md`.
- If relevant, link it from a nearby parent document (for example architecture overview linking to detailed subpages).

---

## Markdown style and formatting

AI should maintain **clean, predictable Markdown** that plays nicely with linters and static site generators.

Headings:

- Start documents with a single `#` title.
- Use `##`, `###`, etc hierarchically.
- Avoid skipping levels (do not jump from `#` to `###` directly).

Code blocks:

- Always specify the language:

  ```python
  def example():
      ...
````

```bash
python -m app.render --config configs/mountains.yaml
```

Lists and tables:

* Use unordered lists for bullets, numbered lists for sequences of steps.
* Use tables sparingly for reference style data (config fields, options).

Links:

* Prefer relative links:

  * `[Render pipeline](./architecture/render-pipeline.md)`
  * `[How to add a biome](../howto/add-biome.md)`

Diagrams:

* Use **Mermaid** for diagrams by default:

  ```mermaid
  flowchart TD
      Config --> TerrainGen --> Renderer --> Image
  ```

* Keep diagrams in the same file as the explanation unless they are reused a lot.

---

## Documentation scope for this project

Docs should reflect the actual structure of the project:

1. **Procedural generation core**

   * Explain the main algorithms used (noise, erosion, heightmap post processing).
   * Document key modules and how they fit together.
   * Describe how seeds and configs affect determinism.

2. **Rendering engine (ModernGL and related tooling)**

   * High level pipeline (context, programs, buffers, framebuffers, readback).
   * How headless rendering works.
   * How textures and heightmaps flow from NumPy to GPU and back.

3. **Configuration and presets**

   * Schema of config models (render settings, camera, lighting, terrain, biomes).
   * Example configs and presets with explanations.
   * Versioning strategy for config files.

4. **UI and workflows**

   * How to start the UI.
   * How to edit and save presets.
   * Typical workflows (iterating on a new biome, testing render quality).

5. **Testing, debugging, and QA**

   * How to run tests.
   * How to run test renders and where outputs are stored.
   * Common issues and how to diagnose them.

AI should create or update docs in these areas rather than scattering knowledge only in code comments.

---

## Writing style and level of detail

Audience: other developers, technical artists, and future you who forgot what you were thinking.

Guidelines:

* Prefer concise technical writing, **but** include enough context that a new contributor can follow.
* Explain the **why** as well as the **what**, especially in architecture and decisions.
* Use short paragraphs, and break up long sections with subheadings.

In documentation:

* Use a neutral tone.
* Be explicit about tradeoffs (quality vs performance vs complexity).
* When introducing a concept, give a small example or snippet.

Example pattern:

```markdown
## Erosion step

We apply a simple hydraulic erosion pass after the initial heightmap generation.

Goals:
- Soften harsh noise artifacts
- Carve plausible valleys along gradient directions

Tradeoffs:
- Slightly more CPU cost
- Can erase fine details if overused

[Optional: code snippet or pseudo code]
```

---

## Keeping docs in sync with code

When AI generates or edits code that changes behavior or structure, it should:

* Look for an existing doc that covers the affected area.
* Update or extend that doc accordingly.
* If none exists, consider creating a short stub in the right subfolder.

Examples:

* New config field added
  Update the relevant config reference doc and at least one example config snippet.

* New rendering feature (for example shadow map pass)
  Add a section to the render pipeline doc and a short how to note if it changes usage.

* New CLI entrypoint or script
  Document usage in a `docs/howto/` page and link it from `docs/index.md` or the relevant section.

Avoid:

* Letting the docs drift out of date when making nontrivial changes.
* Copy pasting large chunks of code into docs without explanation.

---

## How AI should behave with documentation

When generating docs:

* Place new docs in `docs/`, not next to source files, unless the project uses a special pattern.
* Use clear headings, relative links, and language tagged code blocks.
* Summarize, then drill into details; do not dump huge walls of text with no structure.

When editing docs:

* Preserve existing structure and links.
* Improve clarity, fix inaccuracies, and update outdated sections.
* Avoid rewriting everything unless explicitly asked.

When reviewing docs:

* Call out:

  * inconsistencies with current code behavior
  * missing references to important modules or entrypoints
  * hand wavy descriptions for core architecture
* Suggest concrete improvements:

  * add diagrams where structure is unclear
  * add examples where usage is ambiguous
  * split overgrown documents into smaller focused pages within `docs/`
