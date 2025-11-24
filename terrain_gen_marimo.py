import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    import numpy as np
    import io
    import base64
    from PIL import Image
    import sys
    import os
    import time

    # Ensure we can import from src
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    try:
        from src.generators.erosion import ErosionTerrainGenerator, ErosionParams
        from src.generators.morphological import (
            MorphologicalTerrainGPU,
            MorphologicalParams,
        )
        from src.utils import TerrainMaps
    except ImportError:
        # Fallback or error message if src not found
        ErosionTerrainGenerator = None
        MorphologicalTerrainGPU = None
        ErosionParams = None
        MorphologicalParams = None
        TerrainMaps = None
    return (
        ErosionTerrainGenerator,
        Image,
        LightSource,
        MorphologicalParams,
        MorphologicalTerrainGPU,
        base64,
        io,
        mo,
        plt,
        time,
    )


@app.cell
def title(mo):
    mo.md("""
    # 🏔️ GPU Terrain Generator

    Interactive dashboard for procedural terrain generation using ModernGL.
    """)
    return


@app.cell
def ui_controls(mo):
    # Global Settings
    gen_type = mo.ui.dropdown(
        options=["Hydraulic Erosion", "Morphological (GPU)"],
        value="Hydraulic Erosion",
        label="Generator",
    )
    render_mode = mo.ui.dropdown(
        options=["2D Shaded Relief", "3D Raymarch"],
        value="2D Shaded Relief",
        label="Render Mode",
    )

    # Camera Controls (for 3D)
    cam_yaw = mo.ui.slider(0, 360, step=5, value=45, label="Camera Yaw")
    cam_pitch = mo.ui.slider(10, 89, step=1, value=30, label="Camera Pitch")
    cam_dist = mo.ui.slider(0.5, 3.0, step=0.1, value=1.5, label="Camera Distance")
    cam_fov = mo.ui.slider(30, 120, step=5, value=60, label="FOV")

    # Lighting Controls (for 3D)
    sun_yaw = mo.ui.slider(0, 360, step=5, value=225, label="Sun Azimuth")
    sun_pitch = mo.ui.slider(5, 85, step=5, value=30, label="Sun Altitude")
    sun_intensity = mo.ui.slider(
        0.5, 5.0, step=0.1, value=2.0, label="Sun Intensity"
    )
    exposure = mo.ui.slider(0.2, 3.0, step=0.1, value=1.0, label="Exposure")

    # Atmosphere Controls (for 3D)
    fog_density = mo.ui.slider(0.0, 0.5, step=0.01, value=0.1, label="Fog Density")
    fog_height = mo.ui.slider(0.0, 1.0, step=0.05, value=0.35, label="Fog Height")
    ao_strength = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="AO Strength")
    shadow_softness = mo.ui.slider(
        1.0, 32.0, step=1.0, value=16.0, label="Shadow Softness"
    )

    res_dropdown = mo.ui.dropdown(
        options={"256": 256, "512": 512, "1024": 1024, "2048": 2048},
        value="512",
        label="Resolution",
    )
    seed_input = mo.ui.number(value=42, label="Seed", step=1)
    erosion_switch = mo.ui.switch(value=True, label="Enable Erosion")

    # Hydraulic Params
    h_scale = mo.ui.slider(1.0, 10.0, step=0.5, value=3.0, label="Scale")
    h_octaves = mo.ui.slider(1, 12, step=1, value=8, label="Octaves")
    h_lacunarity = mo.ui.slider(1.0, 4.0, step=0.1, value=2.0, label="Lacunarity")
    h_gain = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="Gain")
    h_strength = mo.ui.slider(
        0.0, 0.2, step=0.005, value=0.04, label="Erosion Strength"
    )
    h_slope = mo.ui.slider(1.0, 5.0, step=0.1, value=3.0, label="Slope Strength")
    h_water = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="Water Level")
    h_warp = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="Domain Warp")

    # Morphological Params
    m_scale = mo.ui.slider(1.0, 10.0, step=0.5, value=5.0, label="Scale")
    m_octaves = mo.ui.slider(1, 12, step=1, value=8, label="Octaves")
    m_persist = mo.ui.slider(0.1, 1.0, step=0.05, value=0.5, label="Persistence")
    m_lacunarity = mo.ui.slider(1.0, 4.0, step=0.1, value=2.0, label="Lacunarity")
    m_radius = mo.ui.slider(0, 10, step=1, value=2, label="Erosion Radius")
    m_strength = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.5, label="Erosion Strength"
    )

    # Actions
    generate_btn = mo.ui.button("Generate Terrain")
    return (
        ao_strength,
        cam_dist,
        cam_fov,
        cam_pitch,
        cam_yaw,
        erosion_switch,
        exposure,
        fog_density,
        fog_height,
        gen_type,
        generate_btn,
        h_gain,
        h_lacunarity,
        h_octaves,
        h_scale,
        h_slope,
        h_strength,
        h_water,
        m_lacunarity,
        m_octaves,
        m_persist,
        m_radius,
        m_scale,
        m_strength,
        render_mode,
        res_dropdown,
        seed_input,
        shadow_softness,
        sun_intensity,
        sun_pitch,
        sun_yaw,
    )


@app.cell
def layout(
    ao_strength,
    cam_dist,
    cam_fov,
    cam_pitch,
    cam_yaw,
    erosion_switch,
    exposure,
    fog_density,
    fog_height,
    gen_type,
    generate_btn,
    h_gain,
    h_lacunarity,
    h_octaves,
    h_scale,
    h_slope,
    h_strength,
    h_water,
    m_lacunarity,
    m_octaves,
    m_persist,
    m_radius,
    m_scale,
    m_strength,
    mo,
    render_mode,
    res_dropdown,
    seed_input,
    shadow_softness,
    sun_intensity,
    sun_pitch,
    sun_yaw,
):
    # Organize controls into panels
    common_settings = mo.vstack(
        [
            mo.md("### ⚙️ Settings"),
            gen_type,
            render_mode,
            res_dropdown,
            seed_input,
            erosion_switch,
            generate_btn,
        ]
    )

    camera_settings = mo.vstack(
        [
            mo.md("### 📷 Camera"),
            cam_yaw,
            cam_pitch,
            cam_dist,
            cam_fov,
        ]
    )

    lighting_settings = mo.vstack(
        [
            mo.md("### ☀️ Lighting"),
            sun_yaw,
            sun_pitch,
            sun_intensity,
            exposure,
        ]
    )

    atmosphere_settings = mo.vstack(
        [
            mo.md("### 🌫️ Atmosphere"),
            fog_density,
            fog_height,
            ao_strength,
            shadow_softness,
        ]
    )

    hydraulic_settings = mo.vstack(
        [
            mo.md("### 🌊 Hydraulic Params"),
            h_scale,
            h_octaves,
            h_lacunarity,
            h_gain,
            h_strength,
            h_slope,
            h_water,
        ]
    )

    morph_settings = mo.vstack(
        [
            mo.md("### 🏔️ Morphological Params"),
            m_scale,
            m_octaves,
            m_persist,
            m_lacunarity,
            m_radius,
            m_strength,
        ]
    )

    # Show relevant settings based on generator type
    specific_settings = (
        hydraulic_settings
        if gen_type.value == "Hydraulic Erosion"
        else morph_settings
    )

    # Show 3D-specific controls only if 3D mode is active
    is_3d = render_mode.value == "3D Raymarch"
    cam_panel = camera_settings if is_3d else mo.md("")
    light_panel = lighting_settings if is_3d else mo.md("")
    atmo_panel = atmosphere_settings if is_3d else mo.md("")

    # Build sidebar with conditional 3D panels
    if is_3d:
        sidebar = mo.vstack(
            [
                common_settings,
                mo.md("---"),
                cam_panel,
                light_panel,
                atmo_panel,
                mo.md("---"),
                specific_settings,
            ]
        )
    else:
        sidebar = mo.vstack([common_settings, mo.md("---"), specific_settings])
    return (sidebar,)


@app.cell
def generation(
    ErosionTerrainGenerator,
    MorphologicalParams,
    MorphologicalTerrainGPU,
    ao_strength,
    cam_dist,
    cam_pitch,
    cam_yaw,
    erosion_switch,
    exposure,
    fog_density,
    fog_height,
    gen_type,
    h_gain,
    h_lacunarity,
    h_octaves,
    h_scale,
    h_slope,
    h_strength,
    h_water,
    m_lacunarity,
    m_octaves,
    m_persist,
    m_radius,
    m_scale,
    m_strength,
    render_mode,
    res_dropdown,
    seed_input,
    shadow_softness,
    sun_intensity,
    sun_pitch,
    sun_yaw,
    time,
):
    # Reactive generation
    # We depend on the button and the values.
    # To avoid auto-run on every slider change, we could check if generate_btn was clicked?
    # But Marimo is reactive.
    # Let's just let it run. It's a local GPU gen, should be fast enough.
    # If we wanted to gate it, we'd use mo.stop(not generate_btn.value) but that stops output.

    import math

    start_time = time.time()

    resolution = res_dropdown.value
    seed = seed_input.value

    terrain = None
    raymarch_img = None
    error_msg = None

    try:
        if gen_type.value == "Hydraulic Erosion":
            if ErosionTerrainGenerator is None:
                raise ImportError("ErosionTerrainGenerator not found in src")

            gen = ErosionTerrainGenerator(
                resolution=resolution, use_erosion=erosion_switch.value
            )

            # Map UI to params
            overrides = {
                "height_tiles": h_scale.value,
                "height_octaves": h_octaves.value,
                "height_lacunarity": h_lacunarity.value,
                "height_gain": h_gain.value,
                "water_height": h_water.value,
                "erosion_strength": h_strength.value,
                "erosion_slope_strength": h_slope.value,
                "erosion_branch_strength": h_slope.value,
            }

            terrain = gen.generate_heightmap(seed=seed, **overrides)

            # Render 3D if requested
            if render_mode.value == "3D Raymarch":
                # Calculate camera position from spherical coordinates
                yaw_rad = math.radians(cam_yaw.value)
                pitch_rad = math.radians(cam_pitch.value)
                dist = cam_dist.value

                cx = dist * math.cos(pitch_rad) * math.sin(yaw_rad)
                cy = dist * math.sin(pitch_rad)
                cz = dist * math.cos(pitch_rad) * math.cos(yaw_rad)

                # Calculate sun direction from spherical coordinates
                sun_yaw_rad = math.radians(sun_yaw.value)
                sun_pitch_rad = math.radians(sun_pitch.value)

                sx = -math.cos(sun_pitch_rad) * math.sin(sun_yaw_rad)
                sy = math.sin(sun_pitch_rad)
                sz = -math.cos(sun_pitch_rad) * math.cos(sun_yaw_rad)

                raymarch_img = gen.render_raymarch(
                    camera_pos=(cx, cy, cz),
                    look_at=(0, 0, 0),
                    water_height=h_water.value,
                    sun_dir=(sx, sy, sz),
                    time=0.0,
                    exposure=exposure.value,
                    fog_density=fog_density.value,
                    fog_height=fog_height.value,
                    sun_intensity=sun_intensity.value,
                    ao_strength=ao_strength.value,
                    shadow_softness=shadow_softness.value,
                )

            gen.cleanup()

        else:
            if MorphologicalTerrainGPU is None:
                raise ImportError("MorphologicalTerrainGPU not found in src")

            morph = MorphologicalTerrainGPU()

            params = MorphologicalParams(
                scale=m_scale.value,
                octaves=m_octaves.value,
                persistence=m_persist.value,
                lacunarity=m_lacunarity.value,
                radius=m_radius.value if erosion_switch.value else 0,
                strength=m_strength.value if erosion_switch.value else 0.0,
            )

            terrain = morph.generate(
                resolution=resolution, seed=seed, params=params
            )
            morph.cleanup()

            if render_mode.value == "3D Raymarch":
                error_msg = (
                    "3D Raymarch not supported for Morphological generator yet."
                )

    except Exception as e:
        error_msg = str(e)

    elapsed = time.time() - start_time
    return elapsed, error_msg, raymarch_img, resolution, terrain


@app.cell
def visualization(
    Image,
    LightSource,
    base64,
    cam_dist,
    elapsed,
    error_msg,
    fog_density,
    io,
    mo,
    plt,
    raymarch_img,
    resolution,
    sidebar,
    sun_pitch,
    sun_yaw,
    terrain,
):
    # Layout the main area
    if error_msg:
        main_content = mo.md(f"### ❌ Error\n\n{error_msg}")
    elif terrain:
        # Create visualizations
        height = terrain.height
        erosion = terrain.erosion_mask

        def plot_to_img(arr, cmap=None, title=""):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.axis("off")
            if cmap:
                ax.imshow(arr, cmap=cmap, origin="lower")
            else:
                ax.imshow(arr, origin="lower")
            ax.set_title(title)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        stats = f"""
        **Stats:**
        - Min Height: {height.min():.3f}
        - Max Height: {height.max():.3f}
        - Generation Time: {elapsed:.3f}s
        """

        if raymarch_img is not None:
            # 3D Raymarch View
            img_pil = Image.fromarray(raymarch_img)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            stats_3d = f"""
    ### 📊 Render Stats
    | Property | Value |
    |----------|-------|
    | Resolution | {resolution}×{resolution} |
    | Generation Time | {elapsed:.3f}s |
    | Height Range | {height.min():.3f} – {height.max():.3f} |
    | Camera Distance | {cam_dist.value:.1f} |
    | Sun Direction | ({sun_yaw.value}°, {sun_pitch.value}°) |
    | Fog Density | {fog_density.value:.2f} |
    """

            main_content = mo.vstack(
                [
                    mo.md(stats_3d),
                    mo.md(
                        f'<img src="data:image/png;base64,{b64_str}" style="width:100%; border-radius:8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"/>'
                    ),
                    mo.md(
                        "**🎬 3D Raymarch Render** — *Adjust camera, lighting, and atmosphere controls*"
                    ),
                ]
            )
        else:
            # 2D Shaded Relief View
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(
                height, cmap=plt.cm.terrain, blend_mode="soft", vert_exag=2.0
            )

            img_shaded = plot_to_img(rgb, title="Shaded Relief")
            img_height = plot_to_img(height, cmap="terrain", title="Height Map")
            img_erosion = plot_to_img(
                erosion, cmap="RdYlBu_r", title="Erosion Mask"
            )

            gallery = mo.md(
                f"""
                {stats}

                | | | |
                |:---:|:---:|:---:|
                | <img src="data:image/png;base64,{img_shaded}" width="100%"/> | <img src="data:image/png;base64,{img_height}" width="100%"/> | <img src="data:image/png;base64,{img_erosion}" width="100%"/> |
                | **Shaded Relief** | **Height Map** | **Erosion Mask** |
                """
            )
            main_content = gallery
    else:
        main_content = mo.md("Generating...")

    mo.hstack([sidebar, main_content], gap="2rem", align="start")
    return


@app.function
def exports(mo, np, terrain):
    import io
    from PIL import Image

    def get_png_data():
        if terrain is None:
            return b""
        # 16-bit PNG
        height_16 = (terrain.height * 65535).astype(np.uint16)
        img = Image.fromarray(height_16, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def get_obj_data():
        if terrain is None:
            return b""
        h, w = terrain.height.shape
        # Downsample for OBJ export if too large to prevent browser crash
        step = 1
        if w > 256:
            step = w // 256

        scale = 10.0
        height_scale = 2.0

        obj_content = io.StringIO()
        obj_content.write(f"# Terrain OBJ (Downsampled step={step})\n")

        # Vertices
        vertices = []
        # Use numpy for faster string generation?
        # For now, simple loop with step

        # Re-index map
        idx_map = {}
        curr_idx = 1

        for y in range(0, h, step):
            for x in range(0, w, step):
                vx = (x / (w - 1)) * scale
                vz = (y / (h - 1)) * scale
                vy = terrain.height[y, x] * height_scale
                vertices.append(f"v {vx:.4f} {vy:.4f} {vz:.4f}")
                idx_map[(y, x)] = curr_idx
                curr_idx += 1

        obj_content.write("\n".join(vertices) + "\n")

        # Faces
        faces = []
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                # indices
                i1 = idx_map.get((y, x))
                i2 = idx_map.get((y, x + step))
                i3 = idx_map.get((y + step, x + step))
                i4 = idx_map.get((y + step, x))

                if i1 and i2 and i3 and i4:
                    faces.append(f"f {i1} {i2} {i3} {i4}")

        obj_content.write("\n".join(faces))
        return obj_content.getvalue().encode("utf-8")

    def get_npy_data():
        if terrain is None:
            return b""
        buf = io.BytesIO()
        np.save(buf, terrain.height)
        buf.seek(0)
        return buf

    if terrain:
        download_section = mo.vstack(
            [
                mo.md("### Exports"),
                mo.hstack(
                    [
                        mo.download(
                            data=get_png_data,
                            filename="terrain.png",
                            label="Download PNG (16-bit)",
                        ),
                        mo.download(
                            data=get_obj_data,
                            filename="terrain.obj",
                            label="Download OBJ (Mesh)",
                        ),
                        mo.download(
                            data=get_npy_data,
                            filename="terrain.npy",
                            label="Download NPY (Raw)",
                        ),
                    ]
                ),
            ]
        )
    else:
        download_section = mo.md("")
    return download_section, get_npy_data, get_obj_data, get_png_data


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
