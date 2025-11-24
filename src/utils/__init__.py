"""Utility namespace for terrain generation helpers."""
from .gl_context import create_context, create_detail_texture
from .shader_loader import load_shader
from .visualization import (
    plot_terrain_panels,
    create_turntable_animation,
    save_panel_overview,
    save_turntable_gif,
)
from .export import (
    save_heightmap_png,
    save_normal_map_png,
    save_erosion_mask_png,
    save_heightmap_raw,
    save_heightmap_r32,
    save_npz_bundle,
    export_obj_mesh,
    export_stl_mesh,
)
from .rendering import (
    shade_heightmap,
    save_shaded_relief_png,
    slope_intensity,
    save_slope_map_png,
)
from .batch import (
    BatchGenerator,
    generate_terrain_set,
)
from .artifacts import TerrainMaps
from .textures import (
    save_ao_map,
    save_curvature_map,
    save_packed_texture,
    save_splatmap_rgba,
    save_scatter_map,
)
from .advanced_rendering import (
    render_turntable_frames,
    save_turntable_video,
    render_multi_angle,
    create_comparison_grid,
    render_lighting_study,
    save_animation_sequence,
)
from .render_configs import (
    RenderConfig,
    PRESET_CONFIGS,
)

__all__ = [
    "TerrainMaps",
    "create_context",
    "create_detail_texture",
    "load_shader",
    "plot_terrain_panels",
    "create_turntable_animation",
    "save_panel_overview",
    "save_turntable_gif",
    "shade_heightmap",
    "save_shaded_relief_png",
    "slope_intensity",
    "save_slope_map_png",
    "save_heightmap_png",
    "save_normal_map_png",
    "save_erosion_mask_png",
    "save_heightmap_raw",
    "save_heightmap_r32",
    "save_npz_bundle",
    "export_obj_mesh",
    "export_stl_mesh",
    "BatchGenerator",
    "generate_terrain_set",
    "save_splatmap_rgba",
    "save_ao_map",
    "save_curvature_map",
    "save_packed_texture",
    "save_scatter_map",
    "render_turntable_frames",
    "save_turntable_video",
    "render_multi_angle",
    "create_comparison_grid",
    "render_lighting_study",
    "save_animation_sequence",
    "RenderConfig",
    "PRESET_CONFIGS",
]
