"""Generate sample renders using different render configurations."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src import ErosionTerrainGenerator, ErosionParams
from src.utils import (
    shade_heightmap,
    save_shaded_relief_png,
    PRESET_CONFIGS,
)
import time
from PIL import Image
import numpy as np

# Output directory
OUTPUT_DIR = Path("sample_renders")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed seed for reproducible results
SEED = 42
RESOLUTION = 512
SEAMLESS = True  # Enable seamless/tileable terrain

# Select diverse presets to showcase different styles
PRESETS_TO_TEST = [
    "classic",
    "dramatic",
    "sunrise",
    "sunset",
    "noon",
    "flat",
    "contour",
    "vibrant",
    "monochrome",
    "underwater",
    "alpine",
    "desert",
    "debug_normals",
    "debug_erosion",
]

def generate_samples():
    """Generate sample renders for each preset configuration."""
    print(f"Generating sample renders at {RESOLUTION}x{RESOLUTION} with seed {SEED}")
    print(f"Seamless mode: {'ENABLED' if SEAMLESS else 'DISABLED'}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Testing {len(PRESETS_TO_TEST)} presets\n")
    
    # Generate terrain once (reuse for all renders)
    print("Generating base terrain...")
    gen = ErosionTerrainGenerator(resolution=RESOLUTION)
    try:
        terrain = gen.generate_heightmap(seed=SEED, seamless=SEAMLESS)
        print(f"✓ Terrain generated: {terrain.height.shape}")
    finally:
        gen.cleanup()
    
    results = []
    
    # Render with each preset
    for i, preset_name in enumerate(PRESETS_TO_TEST, 1):
        print(f"\n[{i}/{len(PRESETS_TO_TEST)}] Rendering '{preset_name}'...")
        
        try:
            # Get preset config
            if preset_name not in PRESET_CONFIGS:
                print(f"  ⚠ Preset '{preset_name}' not found, skipping")
                continue
            
            # Call the preset method to get the config instance
            config = PRESET_CONFIGS[preset_name]()
            
            # Time the rendering
            start_time = time.perf_counter()
            shaded = shade_heightmap(
                terrain=terrain,
                azimuth=config.azimuth,
                altitude=config.altitude,
                vert_exag=config.vert_exag,
                blend_mode=config.blend_mode,
                colormap=config.colormap,
            )
            render_time = time.perf_counter() - start_time
            
            # Save output
            output_path = OUTPUT_DIR / f"{preset_name}.png"
            Image.fromarray(shaded).save(output_path)
            
            # Calculate basic statistics
            gray = np.mean(shaded[:, :, :3], axis=2)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            print(f"  ✓ Saved to {output_path.name}")
            print(f"    Render time: {render_time*1000:.1f}ms")
            print(f"    Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
            
            results.append({
                "preset": preset_name,
                "render_time": render_time,
                "brightness": brightness,
                "contrast": contrast,
                "output_path": output_path,
            })
            
        except Exception as e:
            print(f"  ✗ Error rendering '{preset_name}': {e}")
            import traceback
            traceback.print_exc()
    
    return results

def create_comparison_grid(results):
    """Create a comparison grid of all renders."""
    print("\n\nCreating comparison grid...")
    
    if not results:
        print("  ⚠ No results to create grid from")
        return
    
    # Load all images
    images = []
    labels = []
    for result in results:
        try:
            img = Image.open(result["output_path"])
            images.append(img)
            labels.append(result["preset"])
        except Exception as e:
            print(f"  ⚠ Could not load {result['preset']}: {e}")
    
    if not images:
        print("  ⚠ No images loaded")
        return
    
    # Calculate grid dimensions
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    # Create grid
    img_width, img_height = images[0].size
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    grid = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    # Save grid
    grid_path = OUTPUT_DIR / "comparison_grid.png"
    grid.save(grid_path)
    print(f"  ✓ Saved comparison grid to {grid_path.name}")
    print(f"    Grid size: {grid_width}x{grid_height} ({rows}x{cols} layout)")

if __name__ == "__main__":
    print("=" * 70)
    print("GPU Terrain Generator - Sample Render Generation")
    print("=" * 70)
    
    start_time = time.perf_counter()
    results = generate_samples()
    total_time = time.perf_counter() - start_time
    
    if results:
        create_comparison_grid(results)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total renders: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average render time: {np.mean([r['render_time'] for r in results])*1000:.1f}ms")
        print(f"Output directory: {OUTPUT_DIR.absolute()}")
        print("\nRender Statistics:")
        print(f"{'Preset':<20} {'Time (ms)':<12} {'Brightness':<12} {'Contrast':<12}")
        print("-" * 56)
        for result in results:
            print(f"{result['preset']:<20} "
                  f"{result['render_time']*1000:<12.1f} "
                  f"{result['brightness']:<12.1f} "
                  f"{result['contrast']:<12.1f}")
    else:
        print("\n⚠ No renders were generated successfully")
    
    print("\n✓ Done!")
