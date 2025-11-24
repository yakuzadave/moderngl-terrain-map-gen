---
applyTo: '**'
---

# Troubleshooting, investigation, and remediation guidelines

These instructions specialize GitHub Copilot for **debugging and fixing problems** in:

- Procedural generation (terrain/maps/textures)
- ModernGL/OpenGL rendering
- Supporting infrastructure (Pillow, configs, UI, scripts)

This project is typically developed and run **on Windows**, often via Python virtual environments, with GPU drivers installed for OpenGL.

---

## General troubleshooting mindset

When suggesting fixes or edits, AI should:

- **Reproduce, then reduce**  
  - Aim to isolate a **minimal reproducible example** of the issue.
  - Prefer small test scripts over poking at everything at once.

- **Work from symptoms to root cause**  
  - Start from what is observable (error messages, corrupted images, artifacts).
  - Trace through config → generation → rendering step by step.

- **Change one variable at a time**  
  - Avoid sweeping, speculative edits.
  - Prefer small, targeted changes plus instrumentation or logging.

- **Document findings**  
  - Encourage adding brief notes or comments (and where appropriate, docs in `docs/`) when a tricky bug is fixed.

---

## Environment assumptions (Windows)

When generating debug steps or scripts, assume:

- Shell: **PowerShell** or **Command Prompt**.
- Paths: Windows-style (`C:\path\to\project`, `.\venv\Scripts\python.exe`).
- Relevant checks:
  - Python version: `python --version` or `py --version`.
  - Virtual env activation: `.\venv\Scripts\activate`.
  - GPU / driver issues might manifest as:
    - ModernGL failing to create context
    - black or empty renders
    - cryptic GL error codes

Copilot should:

- Use Windows compatible path examples.
- Avoid hard-coding Linux/macOS-specific commands in troubleshooting tips.

---

## Basic triage steps

For any reported issue, Copilot should guide or generate code that:

1. **Captures the error details**
   - Wrap critical sections in `try/except` and log:
     - exception type and message
     - relevant config values
     - current resolution, device, etc.
   - Avoid bare `except:`; always log or re-raise with context.

2. **Confirms environment and dependencies**
   - Python version.
   - Installed packages and versions (for example `pip list` or using `import pkg; print(pkg.__version__)`).
   - On Windows, confirm that:
     - GPU drivers are reasonably recent.
     - Any platform specific libraries used by ModernGL headless backends are installed.

3. **Checks configuration inputs**
   - Validate config files (YAML/JSON) against expected schemas:
     - required fields present
     - correct value ranges (resolution, iteration counts, etc).
   - Add defensive checks:
     ```python
     if config.width <= 0 or config.height <= 0:
         raise ValueError("Render resolution must be positive")
     ```

4. **Introduces minimal debug outputs**
   - For generation:
     - shapes and ranges of arrays (min, max, mean).
   - For rendering:
     - confirmation that framebuffers and textures were created with expected sizes.
     - logging shader compilation status.

---

## Troubleshooting procedural generation

For issues like bad terrain, artifacts, or unexpected patterns:

- **Validate array shapes and types**
  - Check that generation functions return arrays with expected shape `(H, W)` or `(H, W, C)`.
  - Ensure data types are consistent (`float32` vs `uint8`).

- **Check value ranges**
  - Clip and log min/max:
    ```python
    arr = np.asarray(arr)
    print("heightmap range:", arr.min(), arr.max())
    ```
  - Normalize only when necessary, and document the intention.

- **Isolate each stage**
  - Noise generation
  - Erosion / smoothing passes
  - Biome classification / masking
  - Save intermediate arrays as images using Pillow into a `debug/` folder.

- **Add small unit / smoke tests**
  - For a fixed seed, confirm:
    - stable shapes
    - stable summary stats (mean, std) within a tolerance.

Copilot should suggest new tests or debug scripts instead of only editing core logic blindly.

---

## Troubleshooting ModernGL rendering

Typical symptoms: black images, weird colors, tearing, crashes, or context errors.

AI should:

1. **Verify context creation**
   - Ensure context is created once and reused.
   - Check that context creation isn’t failing silently.
   - On Windows, call out driver or OpenGL version issues when context creation fails.

2. **Check framebuffer setup**
   - Ensure framebuffer size matches render resolution.
   - Confirm `fbo.viewport = (0, 0, width, height)` before drawing.
   - Add a simple test render:
     - clear with a solid color
     - read back and verify that pixels match the expected color.

3. **Validate shaders**
   - Check for compilation and link errors and log them.
   - Ensure attribute and uniform names match those used in Python (`in_position`, `u_heightmap`, etc).
   - Temporarily simplify shaders to:
     - output a constant color
     - output `uv` coordinates as color
     - output height as grayscale
   - If a simplified shader works, then problems are in the more complex logic.

4. **Confirm buffer and attribute bindings**
   - Validate formats passed to `vertex_array` or `simple_vertex_array`:
     - example: `"3f 2f"` for position + UV
   - Check that strides and offsets match the actual buffer layout.
   - Log the size of buffers.

5. **Check readback and conversions**
   - Confirm that `fbo.read(...)` uses correct `components` and `dtype`.
   - Reshape with the right dimensions:
     ```python
     raw = fbo.read(components=4, dtype="f1")
     img = np.frombuffer(raw, dtype="uint8").reshape(height, width, 4)
     ```
   - If image appears flipped, systematically test `Image.FLIP_TOP_BOTTOM` or equivalent.

---

## Troubleshooting Pillow and image IO

For issues with saved images, missing files, or wrong colors:

- **Check mode and range**
  - Ensure images are in `"RGBA"` or `"L"` as expected.
  - Confirm that float arrays are scaled to 0–255 before converting.

- **Confirm paths and directories**
  - On Windows, watch for:
    - wrong slashes
    - missing directory creation before `img.save(...)`.
  - Add:
    ```python
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ```

- **Visually inspect debug outputs**
  - Save intermediate images:
    - heightmaps
    - biome masks
    - final renders
  - Compare these visually to isolate at which stage things go wrong.

---

## Debug logging and instrumentation

Copilot should favor **structured, minimal logging** over print spam:

- Use Python `logging` with appropriate levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- Create helpers like:

```python
import logging

logger = logging.getLogger(__name__)

def log_array_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    logger.debug(
        "%s: shape=%s dtype=%s min=%.4f max=%.4f mean=%.4f",
        name,
        arr.shape,
        arr.dtype,
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
    )
````

* For tricky bugs, add concise debug images or small logs, then remove or downgrade once resolved.

---

## Remediation patterns

When proposing fixes, AI should:

* Prefer **small, reversible changes**:

  * parameter tweaks
  * adding validation
  * local refactors that clarify data flow.

* Only restructure heavily when:

  * the current structure clearly obscures the root cause
  * the change simplifies future debugging.

* Where possible, accompany fixes with:

  * a small test (unit, integration, or smoke)
  * a short note in relevant documentation (`docs/`) or a comment describing the pitfall.

Examples:

* After fixing a heightmap normalization bug, add:

  * a test asserting that generated heightmaps are in `[0, 1]` for a given config.
  * a brief comment or doc snippet explaining the normalization step.

* After fixing a framebuffer size mismatch:

  * add assertions that framebuffer dimensions equal config resolution.
  * log a clear error message when they do not.

---

## How AI should behave when debugging

When generating code or suggestions in a debugging context, AI should:

* Ask implicitly: “what is the **next smallest thing** we can verify?”
* Propose:

  * a minimal repro script
  * a targeted logging function
  * a simple test render or debug output
* Avoid:

  * sweeping rewrites of unrelated modules
  * speculative changes without instrumentation

When reviewing changes related to bug fixes:

* Check that:

  * new logs/tests follow existing style.
  * fixes are tightly scoped to the identified issue.
  * any new behavior that affects users is reflected in docs and (if relevant) the changelog.

