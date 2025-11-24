# API Reference

Complete Python API reference for the GPU Terrain Generator.

## Table of Contents

- [Core Modules](#core-modules)
- [Generators](#generators)
- [Configuration](#configuration)
- [Export Utilities](#export-utilities)
- [Rendering Utilities](#rendering-utilities)
- [Texture Utilities](#texture-utilities)
- [Batch Processing](#batch-processing)
- [Advanced Rendering](#advanced-rendering)
- [Data Structures](#data-structures)

---

## Core Modules

### `src` Package

Main package providing access to all terrain generation functionality.

```python
from src import (
    ErosionTerrainGenerator,
    ErosionParams,
    MorphologicalTerrainGPU,
    MorphologicalParams,
    HydraulicErosionGenerator,
    HydraulicParams,
    TerrainMaps,
    TerrainConfig,
    RenderConfig,
    utils,
)
```

---

## Generators

### ErosionTerrainGenerator

GPU-accelerated terrain generator using fractal noise with procedural erosion simulation.

#### Class: `ErosionTerrainGenerator`

```python
class ErosionTerrainGenerator:
    """
    ModernGL-based terrain generator using GLSL shaders for real-time
    heightmap generation with fractal noise and erosion effects.
    """
    
    def __init__(
        self,
        resolution: int = 512,
        ctx: moderngl.Context | None = None
    ) -> None:
        """
        Initialize the erosion terrain generator.
        
        Args:
            resolution: Output texture resolution (power of 2 recommended)
            ctx: Optional ModernGL context (creates new if None)
        """
```

**Methods:**

#### `generate_heightmap()`

```python
def generate_heightmap(
    self,
    seed: int = 42,
    params: ErosionParams | None = None,
    seamless: bool = False,
) -> TerrainMaps:
    """
    Generate a terrain heightmap with erosion features.
    
    Args:
        seed: Random seed for reproducible generation
        params: Erosion parameters (uses defaults if None)
        seamless: Enable tileable/seamless terrain generation
        
    Returns:
        TerrainMaps: Container with height, normals, and erosion mask
        
    Example:
        >>> gen = ErosionTerrainGenerator(resolution=1024)
        >>> terrain = gen.generate_heightmap(seed=12345)
        >>> print(terrain.height.shape)
        (1024, 1024)
    """
```

#### `cleanup()`

```python
def cleanup(self) -> None:
    """
    Release GPU resources. Must be called when done with generator.
    
    Example:
        >>> gen = ErosionTerrainGenerator()
        >>> try:
        ...     terrain = gen.generate_heightmap()
        ... finally:
        ...     gen.cleanup()
    """
```

---

### ErosionParams

Configuration parameters for erosion-based terrain generation.

```python
@dataclass
class ErosionParams:
    """
    Parameters controlling terrain height generation and erosion simulation.
    
    Attributes:
        height_tiles: Base noise frequency (default: 3.0)
        height_octaves: Number of noise octaves (default: 3)
        height_amp: Base height amplitude (default: 0.25)
        height_gain: Amplitude decay per octave (default: 0.1)
        height_lacunarity: Frequency multiplier per octave (default: 2.0)
        water_height: Water level threshold (default: 0.45)
        erosion_tiles: Erosion noise frequency (default: 3.0)
        erosion_octaves: Erosion noise octaves (default: 5)
        erosion_gain: Erosion amplitude gain (default: 0.5)
        erosion_lacunarity: Erosion frequency multiplier (default: 2.0)
        erosion_slope_strength: Slope-based erosion intensity (default: 3.0)
        erosion_branch_strength: Valley branching intensity (default: 3.0)
        erosion_strength: Overall erosion strength (default: 0.04)
        warp_strength: Domain warping intensity (default: 0.0)
        ridge_noise: Use ridge noise (0=off, 1=on) (default: 0)
        thermal_iterations: Thermal erosion passes (default: 0)
        thermal_threshold: Thermal erosion threshold (default: 0.001)
        thermal_strength: Thermal erosion strength (default: 0.5)
    """
```

**Preset Methods:**

```python
@classmethod
def canyon(cls) -> ErosionParams:
    """Deep erosion with branching valleys."""

@classmethod
def plains(cls) -> ErosionParams:
    """Gentle rolling hills with minimal erosion."""

@classmethod
def mountains(cls) -> ErosionParams:
    """Sharp peaks with moderate erosion."""

@classmethod
def natural(cls) -> ErosionParams:
    """Organic terrain with ridge noise and domain warping."""
```

**Example:**

```python
# Use preset
params = ErosionParams.canyon()

# Custom parameters
params = ErosionParams(
    height_amp=0.35,
    erosion_strength=0.08,
    warp_strength=0.6
)

gen = ErosionTerrainGenerator(resolution=512)
terrain = gen.generate_heightmap(seed=42, params=params)
```

---

### HydraulicErosionGenerator

GPU-accelerated hydraulic erosion simulator using the pipe model.

#### Class: `HydraulicErosionGenerator`

```python
class HydraulicErosionGenerator:
    """
    Physically-based hydraulic erosion using multi-stage simulation:
    1. Flux computation (water movement)
    2. Water velocity & depth update
    3. Erosion & Deposition (sediment transport)
    4. Sediment Advection
    5. Evaporation
    6. Thermal Erosion
    """
    
    def __init__(
        self,
        resolution: int = 512,
        ctx: moderngl.Context | None = None
    ) -> None:
        """
        Initialize hydraulic erosion simulator.
        
        Args:
            resolution: Simulation grid resolution
            ctx: Optional ModernGL context
        """
```

**Methods:**

#### `simulate()`

```python
def simulate(
    self,
    initial_height: np.ndarray,
    params: HydraulicParams | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TerrainMaps:
    """
    Run hydraulic erosion simulation on initial heightmap.
    
    Args:
        initial_height: Input heightmap (HxW float32 array, range 0-1)
        params: Simulation parameters
        progress_callback: Optional callback(current_iter, total_iters)
        
    Returns:
        TerrainMaps with eroded terrain
        
    Example:
        >>> # Generate base terrain
        >>> gen_base = ErosionTerrainGenerator(resolution=512)
        >>> base = gen_base.generate_heightmap(seed=42)
        >>> 
        >>> # Apply hydraulic erosion
        >>> hydraulic = HydraulicErosionGenerator(resolution=512)
        >>> params = HydraulicParams(iterations=200, rain_rate=0.02)
        >>> eroded = hydraulic.simulate(base.height, params=params)
    """
```

---

### HydraulicParams

Parameters for hydraulic erosion simulation.

```python
@dataclass
class HydraulicParams:
    """
    Physical parameters for hydraulic erosion simulation.
    
    Attributes:
        iterations: Number of simulation steps (default: 100)
        dt: Time step per iteration (default: 0.02)
        pipe_length: Virtual pipe length (default: 1.0)
        sediment_capacity: Sediment capacity constant Kc (default: 1.0)
        soil_dissolving: Soil dissolving constant Ks (default: 0.5)
        sediment_deposition: Deposition constant Kd (default: 0.5)
        evaporation_rate: Water evaporation rate Ke (default: 0.01)
        rain_rate: Rain added per step (default: 0.01)
        thermal_erosion_rate: Thermal erosion rate (default: 0.15)
        talus_angle: Critical angle for thermal erosion (default: 0.008)
    """
```

---

### MorphologicalTerrainGPU

Alternative terrain generator using Voronoi patterns and distance fields.

#### Class: `MorphologicalTerrainGPU`

```python
class MorphologicalTerrainGPU:
    """
    GPU-based morphological terrain generator using Voronoi noise
    and distance field operations.
    """
    
    def __init__(self, ctx: moderngl.Context | None = None) -> None:
        """Initialize morphological generator."""
    
    def generate(
        self,
        resolution: int = 512,
        seed: int = 42,
        params: MorphologicalParams | None = None,
    ) -> TerrainMaps:
        """
        Generate morphological terrain.
        
        Args:
            resolution: Output resolution
            seed: Random seed
            params: Morphological parameters
            
        Returns:
            TerrainMaps with height, normals, and erosion mask
        """
    
    def cleanup(self) -> None:
        """Release GPU resources."""
```

---

### MorphologicalParams

```python
@dataclass
class MorphologicalParams:
    """
    Parameters for morphological terrain generation.
    
    Attributes:
        scale: Noise scale (default: 5.0)
        octaves: Fractal octaves (default: 8)
        persistence: Amplitude decay (default: 0.5)
        lacunarity: Frequency multiplier (default: 2.0)
        radius: Morphological operation radius (default: 2)
        strength: Effect strength (default: 0.5)
    """
```

---

## Configuration

### TerrainConfig

Unified configuration for terrain generation and rendering.

```python
@dataclass
class TerrainConfig:
    """
    Complete configuration combining generation, rendering, and export settings.
    
    Main Attributes:
        version: Config schema version (default: "1.0")
        resolution: Output resolution (default: 512)
        seed: Random seed (default: 42)
        output_dir: Output directory path (default: "output")
        generator_type: "erosion", "hydraulic", or "morph" (default: "erosion")
        terrain_preset: "canyon", "plains", "mountains", or "custom"
        seamless: Enable tileable generation (default: False)
        
    Terrain Parameters:
        height_tiles, height_octaves, height_amp, height_gain,
        height_lacunarity, water_height, erosion_tiles, erosion_octaves,
        erosion_gain, erosion_lacunarity, erosion_slope_strength,
        erosion_branch_strength, erosion_strength, use_erosion
        
    Render Parameters:
        render_preset: "classic", "dramatic", etc (default: "classic")
        azimuth, altitude, vert_exag, blend_mode, colormap,
        ambient_strength, sun_intensity, tonemap_enabled,
        gamma, exposure, contrast, saturation, viz_mode
        
    Export Flags:
        export_heightmap, export_normals, export_shaded,
        export_obj, export_scatter
    """
    
    def get_erosion_params(self) -> ErosionParams:
        """Extract ErosionParams from config."""
    
    def get_hydraulic_params(self) -> HydraulicParams:
        """Extract HydraulicParams from config."""
    
    def get_render_config(self) -> RenderConfig:
        """Extract RenderConfig from config."""
    
    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
    
    @classmethod
    def from_dict(cls, data: dict) -> TerrainConfig:
        """Deserialize config from dictionary."""
```

**Loading and Saving:**

```python
def load_config(path: str | Path) -> TerrainConfig:
    """Load config from YAML or JSON file."""

def save_config(config: TerrainConfig, path: str | Path) -> Path:
    """Save config to YAML or JSON file."""
```

**Example:**

```python
from src import TerrainConfig, load_config, save_config

# Create config
config = TerrainConfig(
    resolution=1024,
    seed=12345,
    generator_type="erosion",
    terrain_preset="canyon",
    export_obj=True
)

# Save to file
save_config(config, "configs/my_terrain.yaml")

# Load from file
loaded = load_config("configs/my_terrain.yaml")
```

---

### RenderConfig

Rendering-specific configuration.

```python
@dataclass
class RenderConfig:
    """
    Configuration for terrain rendering and visualization.
    
    Attributes:
        azimuth: Light azimuth angle in degrees (default: 315.0)
        altitude: Light altitude angle in degrees (default: 45.0)
        vert_exag: Vertical exaggeration (default: 2.0)
        blend_mode: "soft", "overlay", or "hsv" (default: "soft")
        colormap: Matplotlib colormap name (default: "terrain")
        ambient_strength: Ambient light intensity (default: 0.1)
        sun_intensity: Directional light intensity (default: 2.0)
        tonemap_enabled: Enable tonemapping (default: True)
        gamma: Gamma correction (default: 2.2)
        exposure: Exposure adjustment (default: 1.0)
        contrast: Contrast adjustment (default: 1.0)
        saturation: Saturation adjustment (default: 1.0)
        viz_mode: Visualization mode (0=full, 1=height, 2=normals, etc)
    """

# Preset configurations
PRESET_CONFIGS: dict[str, RenderConfig] = {
    "classic": RenderConfig(...),
    "dramatic": RenderConfig(...),
    "soft": RenderConfig(...),
    "topographic": RenderConfig(...),
}
```

---

## Export Utilities

Located in `src.utils.export`.

### Heightmap Exports

```python
def save_heightmap_png(path: str | Path, terrain) -> Path:
    """
    Save heightmap as 16-bit grayscale PNG (full precision).
    
    Args:
        path: Output file path
        terrain: TerrainMaps or compatible dict
        
    Returns:
        Path to saved file
    """

def save_heightmap_raw(path: str | Path, terrain) -> Path:
    """
    Save raw 16-bit little-endian heightmap data.
    Compatible with Unity, Unreal, and other game engines.
    """

def save_heightmap_r32(path: str | Path, terrain) -> Path:
    """
    Save raw 32-bit float little-endian heightmap data.
    Full floating-point precision for scientific applications.
    """
```

### Normal and Mask Exports

```python
def save_normal_map_png(path: str | Path, terrain) -> Path:
    """
    Save RGB tangent-space normal map as PNG.
    
    Normal vectors are encoded as:
    - R: Normal X component (remapped from [-1,1] to [0,255])
    - G: Normal Y component
    - B: Normal Z component
    """

def save_erosion_mask_png(path: str | Path, terrain) -> Path:
    """
    Save erosion intensity mask as 8-bit grayscale PNG.
    Useful for procedural texture blending.
    """
```

### Bundle Export

```python
def save_npz_bundle(path: str | Path, terrain) -> Path:
    """
    Save all terrain data as compressed NumPy archive.
    
    Contains:
        - height: Float32 heightmap
        - normals: Float32 normal vectors (HxWx3)
        - erosion_mask: Float32 erosion intensity
    
    Example:
        >>> save_npz_bundle("terrain.npz", terrain)
        >>> data = np.load("terrain.npz")
        >>> height = data["height"]
    """
```

### Mesh Exports

```python
def export_obj_mesh(
    path: str | Path,
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
) -> Path:
    """
    Export terrain as Wavefront OBJ mesh with UV coordinates.
    
    Args:
        path: Output .obj file path
        terrain: TerrainMaps instance
        scale: XZ plane scale factor
        height_scale: Vertical scale multiplier
        
    Returns:
        Path to saved .obj file
        
    Note:
        OBJ format is widely supported by 3D software (Blender, Maya, etc).
    """

def export_stl_mesh(
    path: str | Path,
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
) -> Path:
    """
    Export terrain as binary STL mesh for 3D printing.
    
    Args:
        path: Output .stl file path
        terrain: TerrainMaps instance
        scale: XZ plane scale factor
        height_scale: Vertical scale multiplier
    """

def export_gltf_mesh(
    path: str | Path,
    terrain,
    scale: float = 10.0,
    height_scale: float = 2.0,
    embed_textures: bool = False,
) -> Path:
    """
    Export terrain as glTF 2.0 mesh.
    
    Args:
        path: Output .gltf or .glb file path
        terrain: TerrainMaps instance
        scale: XZ plane scale factor
        height_scale: Vertical scale multiplier
        embed_textures: Include heightmap texture in glTF
        
    Note:
        Requires pygltflib: pip install pygltflib
    """
```

### Batch Export

```python
def export_all_formats(
    base_path: str | Path,
    terrain,
    formats: list[str] | None = None,
    **export_kwargs,
) -> dict[str, Path]:
    """
    Export terrain in multiple formats at once.
    
    Args:
        base_path: Base filename (without extension)
        terrain: TerrainMaps instance
        formats: List of format names:
            - "png": 16-bit heightmap PNG
            - "normals": Normal map PNG
            - "mask": Erosion mask PNG
            - "raw": Raw heightmap data
            - "r32": Float32 raw data
            - "npz": Compressed bundle
            - "obj": Wavefront mesh
            - "stl": STL mesh
            - "gltf": glTF mesh
        **export_kwargs: Additional arguments for mesh exports (scale, etc)
        
    Returns:
        Dictionary mapping format names to output paths
        
    Example:
        >>> paths = export_all_formats(
        ...     "output/terrain_001",
        ...     terrain,
        ...     formats=["png", "normals", "obj"],
        ...     scale=15.0
        ... )
        >>> print(paths["obj"])
        output/terrain_001.obj
    """
```

---

## Rendering Utilities

Located in `src.utils.rendering`.

### Shaded Relief

```python
def shade_heightmap(
    terrain,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 2.0,
    colormap: str = "terrain",
    blend_mode: Literal["soft", "overlay", "hsv"] = "soft",
) -> np.ndarray:
    """
    Render shaded relief visualization using matplotlib LightSource.
    
    Args:
        terrain: TerrainMaps or compatible dict
        azimuth: Light direction in degrees (0=N, 90=E, 180=S, 270=W)
        altitude: Light elevation angle in degrees (0=horizon, 90=overhead)
        vert_exag: Vertical exaggeration factor for shading
        colormap: Matplotlib colormap name ("terrain", "gist_earth", etc)
        blend_mode: Shading blend mode
        
    Returns:
        RGB uint8 array (HxWx3)
        
    Example:
        >>> rgb = shade_heightmap(
        ...     terrain,
        ...     azimuth=45.0,
        ...     altitude=60.0,
        ...     colormap="gist_earth"
        ... )
        >>> from PIL import Image
        >>> Image.fromarray(rgb).save("shaded.png")
    """

def save_shaded_relief_png(
    path: str | Path,
    terrain,
    **kwargs,
) -> Path:
    """
    Render and save shaded relief as PNG.
    Convenience wrapper for shade_heightmap() + save.
    """
```

### Slope Analysis

```python
def slope_intensity(
    terrain,
    *,
    scale: float = 1.0,
    gamma: float = 0.8,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate grayscale slope intensity map.
    
    Args:
        terrain: TerrainMaps instance
        scale: Gradient scale multiplier
        gamma: Gamma correction (< 1 = brighten slopes)
        normalize: Normalize to 0-255 range
        
    Returns:
        Grayscale uint8 array (HxW)
        
    Uses gradient magnitude: sqrt(dx² + dy²)
    """

def save_slope_map_png(
    path: str | Path,
    terrain,
    **kwargs,
) -> Path:
    """Save slope intensity map as 8-bit grayscale PNG."""
```

---

## Texture Utilities

Located in `src.utils.textures`.

### Splatmaps

```python
def save_splatmap_rgba(
    path: str | Path,
    terrain,
    height_thresholds: tuple[float, float, float] = (0.3, 0.5, 0.7),
    slope_threshold: float = 0.7,
) -> Path:
    """
    Export 4-channel splatmap for texture blending in game engines.
    
    Channels:
        R: Low terrain (beach/grass)
        G: Mid terrain (grass/dirt)
        B: High terrain (rock/snow)
        A: Steep slopes (cliffs)
    
    Args:
        path: Output PNG path
        terrain: TerrainMaps instance
        height_thresholds: (low, mid, high) height breakpoints
        slope_threshold: Normal.y threshold for steep detection
        
    Returns:
        Path to saved RGBA PNG
        
    Example:
        >>> # Custom biome thresholds
        >>> save_splatmap_rgba(
        ...     "textures/splatmap.png",
        ...     terrain,
        ...     height_thresholds=(0.2, 0.6, 0.8),
        ...     slope_threshold=0.6
        ... )
    """
```

### Ambient Occlusion

```python
def save_ao_map(
    path: str | Path,
    terrain,
    samples: int = 32,
    radius: float = 0.05,
    intensity: float = 1.5,
) -> Path:
    """
    Generate and save ambient occlusion map.
    
    Args:
        path: Output PNG path
        terrain: TerrainMaps instance
        samples: Number of AO samples per pixel (higher = better quality)
        radius: AO sample radius in normalized coordinates
        intensity: AO darkening intensity
        
    Returns:
        Path to saved grayscale PNG
        
    Note:
        AO computation is CPU-based and may be slow for high resolutions.
    """
```

### Curvature Maps

```python
def save_curvature_map(
    path: str | Path,
    terrain,
    scale: float = 1.0,
) -> Path:
    """
    Generate and save surface curvature map.
    
    Useful for:
        - Cavity/convexity masking
        - Weathering effects
        - Detail enhancement
    
    Args:
        path: Output PNG path
        terrain: TerrainMaps instance
        scale: Curvature scale multiplier
        
    Returns:
        Path to saved grayscale PNG
    """
```

### Packed Textures

```python
def save_packed_texture(
    path: str | Path,
    terrain,
    pack_mode: Literal["unity_mask", "ue_orm", "custom"] = "unity_mask",
    custom_channels: dict[str, str] | None = None,
) -> Path:
    """
    Export multi-channel packed texture for game engines.
    
    Packing Modes:
        "unity_mask": RGBA = (Metallic, AO, Detail, Smoothness)
        "ue_orm": RGB = (AO, Roughness, Metallic)
        "custom": User-defined channel mapping
    
    Args:
        path: Output PNG path
        terrain: TerrainMaps instance
        pack_mode: Predefined packing mode
        custom_channels: For "custom" mode, map channel to source:
            {"R": "ao", "G": "height", "B": "slope", "A": "erosion"}
    
    Example:
        >>> # Unity HDRP Mask Map
        >>> save_packed_texture(
        ...     "textures/mask.png",
        ...     terrain,
        ...     pack_mode="unity_mask"
        ... )
        >>> 
        >>> # Custom packing
        >>> save_packed_texture(
        ...     "textures/custom.png",
        ...     terrain,
        ...     pack_mode="custom",
        ...     custom_channels={"R": "ao", "G": "slope", "B": "erosion"}
        ... )
    """
```

### Scatter Maps

```python
def save_scatter_map(
    path: str | Path,
    terrain,
) -> Path:
    """
    Save procedural scatter density map (Trees, Rocks, Grass).
    
    Channels:
        R: Tree density
        G: Rock density
        B: Grass density
    
    Used for vegetation and object placement in game engines.
    """
```

---

## Batch Processing

Located in `src.utils.batch`.

### BatchGenerator

```python
class BatchGenerator:
    """
    Manages batch generation of terrain maps with varied parameters.
    
    Example:
        >>> batch = BatchGenerator(
        ...     generator_type="erosion",
        ...     resolution=512,
        ...     output_dir="batch_output"
        ... )
        >>> 
        >>> seeds = range(100, 150)
        >>> results = batch.generate_set(
        ...     seeds=list(seeds),
        ...     prefix="terrain",
        ...     export_formats=["png", "obj", "shaded"],
        ...     params=ErosionParams.canyon()
        ... )
    """
    
    def __init__(
        self,
        generator_type: str = "erosion",
        resolution: int = 512,
        output_dir: str | Path = "batch_output",
    ) -> None:
        """
        Initialize batch generator.
        
        Args:
            generator_type: "erosion" or "morph"
            resolution: Output resolution for all terrains
            output_dir: Base output directory
        """
    
    def generate_set(
        self,
        seeds: list[int],
        prefix: str = "terrain",
        export_formats: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        **generation_kwargs,
    ) -> list[TerrainMaps]:
        """
        Generate multiple terrains with different seeds.
        
        Args:
            seeds: List of random seeds
            prefix: Filename prefix for exports
            export_formats: ["png", "obj", "stl", "npz", "shaded"]
            progress_callback: Optional callback(current, total, status_msg)
            **generation_kwargs: Passed to generator (e.g., params=...)
            
        Returns:
            List of generated TerrainMaps
        """
    
    def cleanup(self) -> None:
        """Release GPU resources."""
```

### Convenience Function

```python
def generate_terrain_set(
    count: int,
    base_seed: int = 100,
    generator: str = "erosion",
    resolution: int = 512,
    output_dir: str = "output",
    formats: list[str] | None = None,
    **kwargs,
) -> list[TerrainMaps]:
    """
    Generate a set of terrains with sequential seeds.
    
    Convenience wrapper for BatchGenerator.
    
    Args:
        count: Number of terrains to generate
        base_seed: Starting seed value
        generator: "erosion", "hydraulic", or "morph"
        resolution: Output resolution
        output_dir: Output directory path
        formats: Export formats list
        **kwargs: Additional generator parameters
        
    Returns:
        List of TerrainMaps
        
    Example:
        >>> from src.utils import generate_terrain_set
        >>> from src import ErosionParams
        >>> 
        >>> terrains = generate_terrain_set(
        ...     count=20,
        ...     base_seed=1000,
        ...     generator="erosion",
        ...     resolution=1024,
        ...     output_dir="batch_renders",
        ...     formats=["png", "normals", "shaded"],
        ...     params=ErosionParams.mountains()
        ... )
    """
```

---

## Advanced Rendering

Located in `src.utils.advanced_rendering`.

### Turntable Animation

```python
def render_turntable_frames(
    terrain,
    frames: int = 36,
    *,
    altitude: float = 45.0,
    vert_exag: float = 2.0,
    colormap: str = "terrain",
    blend_mode: Literal["soft", "overlay", "hsv"] = "soft",
) -> list[np.ndarray]:
    """
    Render turntable sequence by rotating light around terrain.
    
    Args:
        terrain: TerrainMaps instance
        frames: Number of frames (one full 360° rotation)
        altitude: Sun altitude angle
        vert_exag: Vertical exaggeration
        colormap: Color scheme
        blend_mode: Shading mode
        
    Returns:
        List of RGB uint8 arrays (one per frame)
    """

def save_turntable_video(
    path: str | Path,
    terrain,
    frames: int = 36,
    fps: int = 12,
    **render_kwargs,
) -> Path:
    """
    Save turntable animation as MP4 video.
    
    Requires ffmpeg. Falls back to GIF if unavailable.
    
    Args:
        path: Output .mp4 or .gif path
        terrain: TerrainMaps instance
        frames: Number of frames
        fps: Frames per second
        **render_kwargs: Passed to render_turntable_frames()
        
    Returns:
        Path to saved video
        
    Example:
        >>> save_turntable_video(
        ...     "animations/turntable.mp4",
        ...     terrain,
        ...     frames=60,
        ...     fps=30,
        ...     altitude=60.0,
        ...     colormap="gist_earth"
        ... )
    """
```

### Multi-Angle Rendering

```python
def render_multi_angle(
    terrain,
    angles: Sequence[tuple[float, float]] | None = None,
    **render_kwargs,
) -> list[np.ndarray]:
    """
    Render terrain from multiple lighting angles.
    
    Args:
        terrain: TerrainMaps instance
        angles: List of (azimuth, altitude) tuples
               Default: [(0,45), (90,45), (180,45), (270,45)]
        **render_kwargs: Additional rendering parameters
        
    Returns:
        List of RGB arrays (one per angle)
        
    Example:
        >>> angles = [(45, 30), (135, 45), (225, 60), (315, 30)]
        >>> renders = render_multi_angle(terrain, angles=angles)
    """
```

### Lighting Study

```python
def render_lighting_study(
    terrain,
    azimuth_steps: int = 4,
    altitude_steps: int = 3,
    **render_kwargs,
) -> np.ndarray:
    """
    Render grid of lighting variations for comparison.
    
    Args:
        terrain: TerrainMaps instance
        azimuth_steps: Number of azimuth angles to test
        altitude_steps: Number of altitude angles to test
        **render_kwargs: Rendering parameters
        
    Returns:
        Single RGB image containing grid of renders
        
    Example:
        >>> study = render_lighting_study(
        ...     terrain,
        ...     azimuth_steps=6,
        ...     altitude_steps=4,
        ...     colormap="terrain"
        ... )
        >>> from PIL import Image
        >>> Image.fromarray(study).save("lighting_study.png")
    """
```

### Comparison Grid

```python
def create_comparison_grid(
    terrains: Sequence,
    labels: Sequence[str] | None = None,
    **render_kwargs,
) -> np.ndarray:
    """
    Create side-by-side comparison of multiple terrains.
    
    Args:
        terrains: List of TerrainMaps instances
        labels: Optional labels for each terrain
        **render_kwargs: Rendering parameters applied to all
        
    Returns:
        Single RGB image with terrains arranged in grid
        
    Example:
        >>> terrain_a = gen.generate_heightmap(seed=1, params=ErosionParams.canyon())
        >>> terrain_b = gen.generate_heightmap(seed=2, params=ErosionParams.mountains())
        >>> 
        >>> comparison = create_comparison_grid(
        ...     [terrain_a, terrain_b],
        ...     labels=["Canyon", "Mountains"],
        ...     colormap="terrain"
        ... )
    """
```

### Animation Sequences

```python
def save_animation_sequence(
    path: str | Path,
    frames: Sequence[np.ndarray],
    fps: int = 12,
    format: Literal["mp4", "gif", "frames"] = "mp4",
) -> Path:
    """
    Save sequence of frames as video or image sequence.
    
    Args:
        path: Output path (directory for "frames" mode)
        frames: List of RGB uint8 arrays
        fps: Frames per second
        format: "mp4", "gif", or "frames" (PNG sequence)
        
    Returns:
        Path to saved output
        
    Example:
        >>> frames = render_turntable_frames(terrain, frames=60)
        >>> save_animation_sequence(
        ...     "output/anim.mp4",
        ...     frames,
        ...     fps=30,
        ...     format="mp4"
        ... )
    """
```

---

## Data Structures

### TerrainMaps

Container for terrain generation outputs.

```python
@dataclass
class TerrainMaps:
    """
    Container for GPU terrain outputs with conversion utilities.
    
    Attributes:
        height: Float32 heightmap (HxW, range 0-1)
        normals: Float32 normal vectors (HxWx3, range -1 to 1)
        erosion_mask: Optional erosion intensity (HxW, range 0-1)
        scatter_map: Optional vegetation scatter map (HxWx3, range 0-1)
    """
    
    height: np.ndarray
    normals: np.ndarray
    erosion_mask: np.ndarray | None = None
    scatter_map: np.ndarray | None = None
    
    @classmethod
    def ensure(cls, data: TerrainMaps | Mapping[str, np.ndarray]) -> TerrainMaps:
        """
        Convert dict or other terrain data to TerrainMaps.
        
        Args:
            data: TerrainMaps instance or dict with keys:
                  "height", "normals", optional "erosion_mask"
        
        Returns:
            TerrainMaps instance
        """
    
    @property
    def resolution(self) -> tuple[int, int]:
        """Get (height, width) dimensions."""
    
    def as_dict(self, include_optional: bool = True) -> dict[str, np.ndarray]:
        """Convert to dictionary for serialization."""
    
    # Conversion methods
    def height_u16(self) -> np.ndarray:
        """Height as uint16 (0-65535 range)."""
    
    def normal_map_u8(self) -> np.ndarray:
        """Normals as RGB uint8 (remapped from -1..1 to 0..255)."""
    
    def erosion_mask_u8(self) -> np.ndarray:
        """Erosion mask as uint8 (0-255 range)."""
    
    def scatter_map_u8(self) -> np.ndarray:
        """Scatter map as RGB uint8."""
```

---

## Complete Usage Example

```python
import numpy as np
from pathlib import Path
from src import (
    ErosionTerrainGenerator,
    ErosionParams,
    HydraulicErosionGenerator,
    HydraulicParams,
    TerrainConfig,
    utils,
)

# Initialize generator
gen = ErosionTerrainGenerator(resolution=1024)

try:
    # Generate base terrain
    params = ErosionParams.canyon()
    terrain = gen.generate_heightmap(
        seed=42,
        params=params,
        seamless=True
    )
    
    # Optional: Apply hydraulic erosion
    hydraulic = HydraulicErosionGenerator(resolution=1024)
    hydraulic_params = HydraulicParams(
        iterations=200,
        rain_rate=0.02,
        sediment_capacity=1.5
    )
    terrain = hydraulic.simulate(terrain.height, params=hydraulic_params)
    hydraulic.cleanup()
    
    # Export in multiple formats
    output_dir = Path("output/canyon_terrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    utils.export_all_formats(
        output_dir / "terrain",
        terrain,
        formats=["png", "normals", "mask", "obj", "npz"],
        scale=20.0,
        height_scale=5.0
    )
    
    # Generate textures
    utils.save_splatmap_rgba(
        output_dir / "splatmap.png",
        terrain,
        height_thresholds=(0.25, 0.55, 0.75)
    )
    
    utils.save_ao_map(
        output_dir / "ao.png",
        terrain,
        samples=64,
        intensity=2.0
    )
    
    utils.save_packed_texture(
        output_dir / "unity_mask.png",
        terrain,
        pack_mode="unity_mask"
    )
    
    # Create visualizations
    utils.save_shaded_relief_png(
        output_dir / "shaded.png",
        terrain,
        azimuth=315.0,
        altitude=45.0,
        colormap="terrain"
    )
    
    # Generate turntable animation
    utils.save_turntable_video(
        output_dir / "turntable.mp4",
        terrain,
        frames=60,
        fps=30,
        altitude=50.0
    )
    
    # Create lighting study
    study = utils.render_lighting_study(
        terrain,
        azimuth_steps=8,
        altitude_steps=4
    )
    from PIL import Image
    Image.fromarray(study).save(output_dir / "lighting_study.png")
    
    print(f"✓ Terrain exported to {output_dir}")
    print(f"  Resolution: {terrain.resolution}")
    print(f"  Height range: {terrain.height.min():.3f} - {terrain.height.max():.3f}")

finally:
    gen.cleanup()
```

---

## See Also

- [Architecture Documentation](architecture/index.md)
- [How-To Guides](howto/index.md)
- [Export Formats Reference](EXPORT_FORMATS.md)
- [Configuration Guide](config_ui.md)
- [CLI Reference](../README.md#command-line-interface)
