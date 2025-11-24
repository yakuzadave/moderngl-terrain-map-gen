---
applyTo: '**'
---

# Pillow (PIL) usage guidelines

These instructions specialize GitHub Copilot for working with **Pillow** in this workspace.  
They sit on top of the general project rules (procedural generation core + rendering engine + config & UI).

---

## Pillow’s role in this project

Treat Pillow as the **CPU-side image toolkit** for:

- Loading textures and reference images from disk.
- Converting NumPy / ModernGL outputs into standard image formats (PNG, JPEG, EXR-if-elsewhere-handled, etc).
- Doing light, offline post-processing:
  - color grading “bakes”
  - compositing passes (overlays, masks, debug channels)
  - visualization of heightmaps, normal maps, and biome masks.
- Generating thumbnails, debug sheets, and quick previews for docs or UI.

Avoid using Pillow for things that belong on the GPU:

- Real-time effects in the render loop.
- Heavy per-pixel image operations that could live in shaders.

---

## Core patterns for using Pillow

When Copilot generates or edits Pillow code, it should lean on these idioms:

### Opening and saving images

```python
from pathlib import Path
from PIL import Image

def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGBA")

def save_image(img: Image.Image, path: str | Path, *, quality: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {}
    if quality is not None:
        save_kwargs["quality"] = quality
    img.save(path, **save_kwargs)
````

Guidelines:

* Call `convert("RGBA")` or another explicit mode when you know what you want (avoid “whatever the file was”).
* Always ensure parent directories exist before saving.
* Prefer **lossless formats (PNG)** for heightmaps, masks, and debug outputs.

### Creating images from scratch

```python
from PIL import Image

def create_blank_rgba(width: int, height: int, color=(0, 0, 0, 255)) -> Image.Image:
    return Image.new("RGBA", (width, height), color)
```

Use `Image.new(mode, size, color)` for synthetic debug layers, overlays, or test patterns.

---

## NumPy ↔ Pillow interop

This project already relies on NumPy for terrain and ModernGL readbacks. Copilot should use these patterns:

### NumPy → Pillow

```python
import numpy as np
from PIL import Image

def array_to_image(arr: np.ndarray, mode: str = "RGBA") -> Image.Image:
    """
    arr should be HxWxC with values in [0, 255] for uint8
    or [0.0, 1.0] for float which we remap.
    """
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype("uint8")
    elif arr.dtype != np.uint8:
        arr = arr.astype("uint8")
    return Image.fromarray(arr, mode=mode)
```

### Pillow → NumPy

```python
import numpy as np
from PIL import Image

def image_to_array(img: Image.Image, *, as_float: bool = False) -> np.ndarray:
    arr = np.array(img)
    if as_float:
        return arr.astype("float32") / 255.0
    return arr
```

Guidelines:

* Be **explicit about ranges**:

  * float arrays: 0.0–1.0
  * uint8 arrays: 0–255
* Document the expected shape (H, W, C) and mode in docstrings.

---

## Color modes and conversions

Copilot should:

* Default to `"RGBA"` when transparency matters (terrain overlays, masks, debug).
* Use `"L"` for single-channel data:

  * heightmaps
  * grayscale masks / weight maps
* Use `img.convert("RGB")` or `"RGBA"` when a consistent mode is required for subsequent operations.

Example:

```python
def save_heightmap(arr: np.ndarray, path: str | Path) -> None:
    """
    Save a heightmap array (H x W, float32 in [0, 1]) as an 8-bit grayscale PNG.
    """
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype("uint8")
    img = Image.fromarray(arr_u8, mode="L")
    save_image(img, path)
```

---

## Resizing, cropping, and transforms

For non-destructive image adjustments (thumbnails, debug composites, etc):

* Use `img.resize((w, h), resample=Image.BILINEAR or Image.BICUBIC)` for scaling.
* Use `img.crop((left, top, right, bottom))` for cropping.
* Use `img.transpose(Image.FLIP_TOP_BOTTOM)` when vertical flipping is needed (for example coordinate system differences with OpenGL).

Example:

```python
from PIL import Image

def to_thumbnail(img: Image.Image, max_size: int = 256) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail((max_size, max_size), resample=Image.BICUBIC)
    return thumb
```

Guidelines:

* Avoid modifying the original `Image` in-place unless clearly intended; prefer `.copy()` when in doubt.
* For heightmaps and mask-like images, be careful with interpolation; nearest-neighbor preserves discrete values.

---

## Compositing, alpha, and debug overlays

When Copilot writes compositing logic, it should use:

* `Image.alpha_composite()` for combining two RGBA images of the same size.
* `Image.blend(img1, img2, alpha)` for simple linear mixes.
* Mode-aware operations (for example convert to `"RGBA"` before alpha operations).

Example:

```python
from PIL import Image

def overlay_debug_layer(base: Image.Image, overlay: Image.Image, opacity: float = 0.5) -> Image.Image:
    base_rgba = base.convert("RGBA")
    overlay_rgba = overlay.convert("RGBA")
    # Apply opacity
    r, g, b, a = overlay_rgba.split()
    a = a.point(lambda v: int(v * opacity))
    overlay_rgba = Image.merge("RGBA", (r, g, b, a))
    return Image.alpha_composite(base_rgba, overlay_rgba)
```

Use this for:

* Highlighting biome regions on top of terrain.
* Showing error masks or out-of-range areas.

---

## File formats and naming

AI should enforce some conventions:

* **PNG** for:

  * heightmaps
  * normal maps
  * biome masks
  * debug / QA images
* **JPEG** only for:

  * final “pretty” renders where size > lossless fidelity.
* Prefer explicit suffixes:

  * `*_height.png`
  * `*_normal.png`
  * `*_biome.png`
  * `*_preview.png`
* Group outputs logically:

  * `renders/{date}/{config_name}_preview.png`
  * `renders/{date}/{config_name}_height.png`

---

## Error handling and robustness

Generated Pillow code should:

* Check that paths exist or create parent directories when saving.
* Fail clearly on invalid input types or shapes:

  * raise `ValueError` with a concise message.
* Avoid silently discarding alpha or channels; call out conversions in code and docstrings.

Example:

```python
def ensure_rgba(img: Image.Image) -> Image.Image:
    """
    Return an RGBA version of the image. If conversion drops information,
    document that at the call site.
    """
    if img.mode == "RGBA":
        return img
    return img.convert("RGBA")
```

---

## Performance considerations

Pillow is fine for:

* Offline / batched processing.
* A handful of post-process steps after GPU renders.

AI should avoid:

* Pixel-by-pixel Python loops over `Image.load()` unless absolutely necessary.
* Repeatedly re-opening the same image; cache when appropriate in higher level code.

When needing lower-level operations, prefer:

* Converting to NumPy (`np.array(img)`) for vectorized operations.
* Converting back to Pillow only once at the end.

---

## How AI should behave with Pillow code

When generating code:

* Use the canonical `Image.open`, `Image.new`, `Image.fromarray`, and `img.save` patterns.
* Be explicit about modes, shapes, and ranges.
* Integrate with existing configuration and naming conventions for outputs.

When answering questions:

* Explain when Pillow is the right tool vs when something belongs in the shader / ModernGL stage.
* Provide small, composable helper functions instead of giant monolithic scripts.

When reviewing changes:

* Call out:

  * magic constants for paths or sizes
  * lossy conversions that might surprise users
  * loops that should be vectorized via NumPy
* Suggest:

  * shared helper modules for image IO and conversions
  * consistent naming and directory structure for outputs
