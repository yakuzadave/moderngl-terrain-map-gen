"""Streamlit UI for GPU Terrain Generator configuration and preview."""
from __future__ import annotations
from PIL import Image
import numpy as np
import streamlit as st
import traceback
import time
from src.config import TerrainConfig, save_config, load_config
from src import (
    ErosionTerrainGenerator,
    HydraulicErosionGenerator,
    HydraulicParams,
    ErosionParams,
)
from src.utils import (
    shade_heightmap,
    save_heightmap_png,
    save_normal_map_png,
    save_shaded_relief_png,
    export_obj_mesh,
    save_scatter_map,
    PRESET_CONFIGS,
)

# Add src to path BEFORE any src imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Page configuration
st.set_page_config(
    page_title="GPU Terrain Generator",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "config" not in st.session_state:
    st.session_state.config = TerrainConfig()
if "terrain" not in st.session_state:
    st.session_state.terrain = None
if "preview_image" not in st.session_state:
    st.session_state.preview_image = None
if "generation_time" not in st.session_state:
    st.session_state.generation_time = 0.0
if "render_time" not in st.session_state:
    st.session_state.render_time = 0.0


def generate_terrain() -> None:
    """Generate terrain based on current configuration."""
    try:
        config = st.session_state.config

        with st.spinner("Generating terrain..."):
            start = time.perf_counter()

            if config.generator_type == "hydraulic":
                # 1. Generate base heightmap (no erosion)
                base_gen = ErosionTerrainGenerator(
                    resolution=config.resolution,
                    use_erosion=False,  # Disable noise erosion
                    defaults=config.get_erosion_params(),
                )
                try:
                    base_terrain = base_gen.generate_heightmap(
                        seed=config.seed,
                        seamless=config.seamless
                    )
                finally:
                    base_gen.cleanup()

                # 2. Apply hydraulic erosion
                hydro_gen = HydraulicErosionGenerator(
                    resolution=config.resolution
                )
                try:
                    terrain = hydro_gen.simulate(
                        base_terrain.height,
                        config.get_hydraulic_params()
                    )
                finally:
                    hydro_gen.cleanup()
            else:
                # Standard erosion generator
                gen = ErosionTerrainGenerator(
                    resolution=config.resolution,
                    use_erosion=config.use_erosion,
                    defaults=config.get_erosion_params(),
                )

                try:
                    terrain = gen.generate_heightmap(
                        seed=config.seed,
                        seamless=config.seamless
                    )
                finally:
                    gen.cleanup()

            st.session_state.terrain = terrain
            st.session_state.generation_time = time.perf_counter() - start

        render_preview()
        st.success(
            f"✓ Terrain generated in {st.session_state.generation_time:.3f}s")

    except Exception as e:
        st.error(f"Error generating terrain: {e}")
        st.code(traceback.format_exc())


def render_preview() -> None:
    """Render preview image from current terrain."""
    if st.session_state.terrain is None:
        return

    try:
        config = st.session_state.config

        with st.spinner("Rendering preview..."):
            start = time.perf_counter()

            img_array = shade_heightmap(
                st.session_state.terrain,
                azimuth=config.azimuth,
                altitude=config.altitude,
                vert_exag=config.vert_exag,
                colormap=config.colormap,
                blend_mode=config.blend_mode,
            )

            st.session_state.preview_image = Image.fromarray(img_array)
            st.session_state.render_time = time.perf_counter() - start

    except Exception as e:
        st.error(f"Error rendering preview: {e}")
        st.code(traceback.format_exc())


def export_outputs() -> None:
    """Export terrain to files."""
    if st.session_state.terrain is None:
        st.warning("Generate terrain first!")
        return

    try:
        config = st.session_state.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Exporting files..."):
            terrain = st.session_state.terrain
            base_name = f"terrain_seed{config.seed}"

            if config.export_heightmap:
                save_heightmap_png(
                    str(output_dir / f"{base_name}_height.png"),
                    terrain
                )

            if config.export_normals:
                save_normal_map_png(
                    str(output_dir / f"{base_name}_normal.png"),
                    terrain
                )

            if config.export_shaded:
                save_shaded_relief_png(
                    str(output_dir / f"{base_name}_shaded.png"),
                    terrain,
                    azimuth=config.azimuth,
                    altitude=config.altitude,
                    vert_exag=config.vert_exag,
                    colormap=config.colormap,
                    blend_mode=config.blend_mode,
                )

            if config.export_obj:
                export_obj_mesh(
                    str(output_dir / f"{base_name}.obj"),
                    terrain,
                )

            if config.export_scatter:
                save_scatter_map(
                    str(output_dir / f"{base_name}_scatter.png"),
                    terrain,
                )

        st.success(f"✓ Files exported to {output_dir.absolute()}")

    except Exception as e:
        st.error(f"Error exporting: {e}")
        st.code(traceback.format_exc())


# Main UI Layout
st.title("🏔️ GPU Terrain Generator")
st.markdown(
    "Configure and generate GPU-accelerated terrain with real-time preview")

# Sidebar for quick controls
with st.sidebar:
    st.header("⚙️ Quick Settings")

    # Output settings
    st.subheader("Output")
    st.session_state.config.resolution = st.select_slider(
        "Resolution",
        options=[128, 256, 512, 1024, 2048],
        value=st.session_state.config.resolution,
        help="Higher resolution = more detail but slower",
    )

    st.session_state.config.seamless = st.checkbox(
        "Seamless / Tileable",
        value=st.session_state.config.seamless,
        help="Generate tileable terrain for infinite worlds",
    )

    st.session_state.config.seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=st.session_state.config.seed,
        help="Same seed = same terrain",
    )

    st.session_state.config.output_dir = st.text_input(
        "Output Directory",
        value=st.session_state.config.output_dir,
    )

    st.session_state.config.generator_type = st.selectbox(
        "Generator Type",
        options=["erosion", "hydraulic"],
        index=["erosion", "hydraulic"].index(
            st.session_state.config.generator_type),
        help="Choose the erosion simulation method",
    )

    st.divider()

    # Terrain preset
    st.subheader("🗻 Terrain Preset")
    terrain_presets = ["custom", "canyon", "plains", "mountains"]
    current_terrain = st.session_state.config.terrain_preset

    new_terrain_preset = st.selectbox(
        "Preset",
        options=terrain_presets,
        index=terrain_presets.index(
            current_terrain) if current_terrain in terrain_presets else 0,
        help="Choose a preset or 'custom' for manual control",
    )

    if new_terrain_preset != st.session_state.config.terrain_preset:
        st.session_state.config.apply_terrain_preset(new_terrain_preset)
        st.rerun()

    st.divider()

    # Render preset
    st.subheader("🎨 Render Style")
    render_presets = ["custom"] + list(PRESET_CONFIGS.keys())
    current_render = st.session_state.config.render_preset

    new_render_preset = st.selectbox(
        "Style",
        options=render_presets,
        index=render_presets.index(
            current_render) if current_render in render_presets else 0,
        help="Choose a render style or 'custom' for manual control",
    )

    if new_render_preset != st.session_state.config.render_preset:
        st.session_state.config.apply_render_preset(new_render_preset)
        if st.session_state.terrain is not None:
            render_preview()
        st.rerun()

    st.divider()

    # Actions
    st.subheader("🚀 Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate", type="primary", use_container_width=True):
            generate_terrain()

    with col2:
        if st.button("Export", use_container_width=True):
            export_outputs()

    # Config management
    st.divider()
    st.subheader("💾 Presets")

    preset_dir = Path("configs/presets")
    preset_dir.mkdir(parents=True, exist_ok=True)

    preset_files = list(preset_dir.glob("*.yaml"))
    preset_names = [f.stem for f in preset_files]

    if preset_names:
        selected_preset = st.selectbox(
            "Load Preset",
            options=[""] + preset_names,
            help="Load a saved configuration",
        )

        if selected_preset and st.button("Load"):
            try:
                loaded_config = load_config(
                    preset_dir / f"{selected_preset}.yaml")
                st.session_state.config = loaded_config
                st.success(f"✓ Loaded {selected_preset}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading preset: {e}")

    save_name = st.text_input("Preset Name", placeholder="my_terrain")
    if st.button("Save Preset") and save_name:
        try:
            save_config(
                st.session_state.config,
                preset_dir / f"{save_name}.yaml"
            )
            st.success(f"✓ Saved as {save_name}")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving preset: {e}")

# Main content area
col_preview, col_settings = st.columns([2, 1])

with col_preview:
    st.header("Preview")

    if st.session_state.preview_image is not None:
        st.image(
            st.session_state.preview_image,
            caption=f"Generation: {st.session_state.generation_time:.3f}s | Render: {st.session_state.render_time:.3f}s",
            use_container_width=True,
        )
    else:
        st.info("Click 'Generate' to create terrain")

    # Export options
    if st.session_state.terrain is not None:
        st.subheader("Export Options")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.session_state.config.export_heightmap = st.checkbox(
                "Heightmap", value=st.session_state.config.export_heightmap
            )
        with col2:
            st.session_state.config.export_normals = st.checkbox(
                "Normals", value=st.session_state.config.export_normals
            )
        with col3:
            st.session_state.config.export_shaded = st.checkbox(
                "Shaded", value=st.session_state.config.export_shaded
            )
        with col4:
            st.session_state.config.export_obj = st.checkbox(
                "OBJ Mesh", value=st.session_state.config.export_obj
            )
        with col5:
            st.session_state.config.export_scatter = st.checkbox(
                "Scatter Map", value=st.session_state.config.export_scatter
            )

with col_settings:
    st.header("Advanced Settings")

    tabs = st.tabs(["Terrain", "Lighting", "Rendering", "Post-FX"])

    with tabs[0]:  # Terrain
        st.subheader("Height Generation")

        st.session_state.config.height_amp = st.slider(
            "Amplitude",
            0.0, 1.0,
            st.session_state.config.height_amp,
            0.01,
            help="Overall terrain height",
        )

        st.session_state.config.height_octaves = st.slider(
            "Octaves",
            1, 8,
            st.session_state.config.height_octaves,
            help="Noise detail levels",
        )

        st.session_state.config.height_tiles = st.slider(
            "Tiling",
            0.1, 10.0,
            st.session_state.config.height_tiles,
            0.1,
            help="Noise frequency",
        )

        st.divider()
        st.subheader("Erosion")

        if st.session_state.config.generator_type == "hydraulic":
            st.info("Hydraulic erosion simulation active")

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.session_state.config.hydraulic_iterations = st.number_input(
                    "Iterations", 1, 1000, st.session_state.config.hydraulic_iterations, 10
                )
                st.session_state.config.hydraulic_dt = st.number_input(
                    "Time Step (dt)", 0.001, 0.1, st.session_state.config.hydraulic_dt, 0.001, format="%.4f"
                )
                st.session_state.config.hydraulic_rain_rate = st.slider(
                    "Rain Rate", 0.0, 0.1, st.session_state.config.hydraulic_rain_rate, 0.001
                )
                st.session_state.config.hydraulic_evaporation_rate = st.slider(
                    "Evaporation", 0.0, 0.1, st.session_state.config.hydraulic_evaporation_rate, 0.001
                )

            with col_h2:
                st.session_state.config.hydraulic_sediment_capacity = st.slider(
                    "Sediment Capacity", 0.1, 5.0, st.session_state.config.hydraulic_sediment_capacity, 0.1
                )
                st.session_state.config.hydraulic_soil_dissolving = st.slider(
                    "Soil Dissolving", 0.0, 1.0, st.session_state.config.hydraulic_soil_dissolving, 0.01
                )
                st.session_state.config.hydraulic_sediment_deposition = st.slider(
                    "Deposition Rate", 0.0, 1.0, st.session_state.config.hydraulic_sediment_deposition, 0.01
                )
                st.session_state.config.hydraulic_pipe_length = st.slider(
                    "Pipe Length", 0.1, 5.0, st.session_state.config.hydraulic_pipe_length, 0.1
                )

        else:
            st.session_state.config.use_erosion = st.checkbox(
                "Enable Erosion",
                value=st.session_state.config.use_erosion,
            )

            if st.session_state.config.use_erosion:
                st.session_state.config.erosion_strength = st.slider(
                    "Strength",
                    0.0, 0.2,
                    st.session_state.config.erosion_strength,
                    0.001,
                    help="Overall erosion intensity",
                )

                st.session_state.config.erosion_slope_strength = st.slider(
                    "Slope Strength",
                    0.0, 10.0,
                    st.session_state.config.erosion_slope_strength,
                    0.1,
                    help="Steeper slopes erode more",
                )

                st.session_state.config.erosion_branch_strength = st.slider(
                    "Branch Strength",
                    0.0, 10.0,
                    st.session_state.config.erosion_branch_strength,
                    0.1,
                    help="Valley branching",
                )

        st.divider()
        st.subheader("Water")

        st.session_state.config.water_height = st.slider(
            "Water Level",
            0.0, 1.0,
            st.session_state.config.water_height,
            0.01,
            help="Sea level threshold",
        )

    with tabs[1]:  # Lighting
        st.subheader("Sun Position")

        st.session_state.config.azimuth = st.slider(
            "Azimuth (degrees)",
            0.0, 360.0,
            st.session_state.config.azimuth,
            1.0,
            help="Sun direction (0=N, 90=E, 180=S, 270=W)",
        )

        st.session_state.config.altitude = st.slider(
            "Altitude (degrees)",
            1.0, 90.0,
            st.session_state.config.altitude,
            1.0,
            help="Sun height (90=overhead)",
        )

        st.session_state.config.sun_intensity = st.slider(
            "Intensity",
            0.1, 5.0,
            st.session_state.config.sun_intensity,
            0.1,
            help="Sun brightness",
        )

        st.session_state.config.ambient_strength = st.slider(
            "Ambient Light",
            0.0, 1.0,
            st.session_state.config.ambient_strength,
            0.01,
            help="Background illumination",
        )

        if st.button("Re-render Preview", use_container_width=True):
            render_preview()

    with tabs[2]:  # Rendering
        st.subheader("Shading")

        st.session_state.config.vert_exag = st.slider(
            "Vertical Exaggeration",
            0.1, 10.0,
            st.session_state.config.vert_exag,
            0.1,
            help="Emphasize height differences",
        )

        st.session_state.config.blend_mode = st.selectbox(
            "Blend Mode",
            options=["soft", "overlay", "hsv"],
            index=["soft", "overlay", "hsv"].index(
                st.session_state.config.blend_mode),
            help="How shading blends with colors",
        )

        st.session_state.config.colormap = st.selectbox(
            "Colormap",
            options=["terrain", "viridis", "gist_earth",
                     "copper", "rainbow", "gray", "ocean"],
            index=["terrain", "viridis", "gist_earth", "copper", "rainbow", "gray", "ocean"].index(
                st.session_state.config.colormap
            ) if st.session_state.config.colormap in ["terrain", "viridis", "gist_earth", "copper", "rainbow", "gray", "ocean"] else 0,
            help="Color scheme",
        )

        st.session_state.config.viz_mode = st.selectbox(
            "Visualization Mode",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: ["Full Render", "Height Only",
                                   "Normals", "Erosion", "Slope", "Curvature"][x],
            index=st.session_state.config.viz_mode,
            help="Debug visualization modes",
        )

        if st.button("Re-render Preview", key="render2", use_container_width=True):
            render_preview()

    with tabs[3]:  # Post-FX
        st.subheader("Post-Processing")

        st.session_state.config.tonemap_enabled = st.checkbox(
            "Tonemapping",
            value=st.session_state.config.tonemap_enabled,
            help="HDR to LDR conversion",
        )

        st.session_state.config.exposure = st.slider(
            "Exposure",
            0.1, 3.0,
            st.session_state.config.exposure,
            0.1,
            help="Overall brightness multiplier",
        )

        st.session_state.config.contrast = st.slider(
            "Contrast",
            0.5, 2.0,
            st.session_state.config.contrast,
            0.05,
            help="Contrast adjustment",
        )

        st.session_state.config.saturation = st.slider(
            "Saturation",
            0.0, 2.0,
            st.session_state.config.saturation,
            0.1,
            help="Color intensity (0=grayscale)",
        )

        st.session_state.config.gamma = st.slider(
            "Gamma",
            0.5, 3.0,
            st.session_state.config.gamma,
            0.1,
            help="Gamma correction",
        )

        if st.button("Re-render Preview", key="render3", use_container_width=True):
            render_preview()

# Footer
st.divider()
st.markdown("""
**Tips:**
- Use presets as starting points for your custom terrains
- Lower resolution for faster iteration, higher for final renders
- Same seed always generates the same terrain
- Export OBJ files for use in 3D software
""")
