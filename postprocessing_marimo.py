import marimo as mo

__generated_with = "0.1.0"
app = mo.App()

@app.cell
def imports():
    import marimo as mo
    import numpy as np
    from PIL import Image
    import io
    import sys
    import os
    import time
    import matplotlib.pyplot as plt

    # Ensure we can import from src
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    try:
        from src import ErosionTerrainGenerator, ErosionParams
        from src.utils import shade_heightmap, RenderConfig
        from src.utils.postprocessing import (
            apply_tonemapping,
            apply_color_grading,
            apply_bloom_effect,
            apply_sharpening,
            apply_atmospheric_perspective,
        )
    except ImportError:
        pass
    return (
        ErosionParams,
        ErosionTerrainGenerator,
        Image,
        RenderConfig,
        apply_atmospheric_perspective,
        apply_bloom_effect,
        apply_color_grading,
        apply_sharpening,
        apply_tonemapping,
        io,
        mo,
        np,
        os,
        plt,
        shade_heightmap,
        sys,
        time,
    )


@app.cell
def header(mo):
    mo.md(
        """
        # 🎨 Advanced Post-Processing Dashboard

        Interactive playground for applying post-processing effects to generated terrain.
        """
    )
    return


@app.cell
def generation_controls(mo):
    # Terrain Generation Controls
    seed_input = mo.ui.number(value=12345, label="Seed", step=1)
    gen_btn = mo.ui.button(label="Generate New Terrain")

    mo.vstack([
        mo.md("### 1. Terrain Generation"),
        mo.hstack([seed_input, gen_btn], justify="start")
    ])
    return gen_btn, seed_input


@app.cell
def generate_terrain(
    ErosionParams,
    ErosionTerrainGenerator,
    gen_btn,
    mo,
    seed_input,
):
    # Generate or load terrain
    terrain_state = mo.state(None)

    # We use a simple check to trigger generation
    # In Marimo, cells run when dependencies change.
    # We want to run this when gen_btn is clicked OR initially.
    
    # However, gen_btn.value is just a boolean that might toggle or increment?
    # Actually mo.ui.button returns a value that updates on click.
    
    # Let's just generate if it's None or button clicked
    if gen_btn.value or terrain_state.value is None:
        # Generate a standard mountain terrain
        gen = ErosionTerrainGenerator(resolution=512, defaults=ErosionParams.mountains())
        try:
            t = gen.generate_heightmap(seed=seed_input.value)
            terrain_state.value = t
        finally:
            gen.cleanup()

    terrain = terrain_state.value
    return gen, t, terrain, terrain_state


@app.cell
def base_render_controls(mo):
    # Base Render Controls
    azimuth = mo.ui.slider(0, 360, value=315, label="Sun Azimuth")
    altitude = mo.ui.slider(0, 90, value=45, label="Sun Altitude")
    vert_exag = mo.ui.slider(0.1, 5.0, value=2.0, label="Vertical Exaggeration")
    colormap = mo.ui.dropdown(
        options=["terrain", "gist_earth", "ocean", "viridis", "magma"],
        value="terrain",
        label="Colormap"
    )

    mo.vstack([
        mo.md("### 2. Base Rendering"),
        mo.hstack([azimuth, altitude]),
        mo.hstack([vert_exag, colormap])
    ])
    return altitude, azimuth, colormap, vert_exag


@app.cell
def post_processing_controls(mo):
    # Post-Processing Controls

    # Tonemapping
    tm_method = mo.ui.dropdown(
        options=["none", "reinhard", "filmic", "aces", "uncharted2"],
        value="filmic",
        label="Method"
    )
    exposure = mo.ui.slider(0.1, 5.0, value=1.0, label="Exposure")

    # Color Grading
    temp = mo.ui.slider(-1.0, 1.0, value=0.0, label="Temperature")
    tint = mo.ui.slider(-1.0, 1.0, value=0.0, label="Tint")
    saturation = mo.ui.slider(0.0, 2.0, value=1.0, label="Saturation")
    contrast = mo.ui.slider(0.0, 2.0, value=1.0, label="Contrast")
    brightness = mo.ui.slider(-1.0, 1.0, value=0.0, label="Brightness")
    gamma = mo.ui.slider(0.1, 3.0, value=1.0, label="Gamma")

    # Bloom
    bloom_enabled = mo.ui.switch(label="Enable Bloom")
    bloom_thresh = mo.ui.slider(0.0, 1.0, value=0.7, label="Threshold")
    bloom_intensity = mo.ui.slider(0.0, 2.0, value=0.5, label="Intensity")
    bloom_radius = mo.ui.slider(1.0, 50.0, value=15.0, label="Blur Radius")

    # Sharpening
    sharp_amount = mo.ui.slider(0.0, 2.0, value=0.0, label="Amount")

    # Atmosphere
    fog_enabled = mo.ui.switch(label="Enable Fog")
    fog_density = mo.ui.slider(0.0, 1.0, value=0.4, label="Density")
    fog_falloff = mo.ui.slider(0.1, 10.0, value=3.0, label="Height Falloff")
    fog_color_picker = mo.ui.text(value="#bfd1e5", label="Fog Color (Hex)")

    # Layout
    tabs = mo.ui.tabs({
        "Tonemapping": mo.vstack([tm_method, exposure]),
        "Color Grading": mo.vstack([temp, tint, saturation, contrast, brightness, gamma]),
        "Bloom": mo.vstack([bloom_enabled, bloom_thresh, bloom_intensity, bloom_radius]),
        "Sharpening": mo.vstack([sharp_amount]),
        "Atmosphere": mo.vstack([fog_enabled, fog_density, fog_falloff, fog_color_picker]),
    })

    mo.vstack([
        mo.md("### 3. Post-Processing Effects"),
        tabs
    ])
    return (
        bloom_enabled,
        bloom_intensity,
        bloom_radius,
        bloom_thresh,
        brightness,
        contrast,
        exposure,
        fog_color_picker,
        fog_density,
        fog_enabled,
        fog_falloff,
        gamma,
        saturation,
        sharp_amount,
        tabs,
        temp,
        tint,
        tm_method,
    )


@app.cell
def process_pipeline(
    altitude,
    apply_atmospheric_perspective,
    apply_bloom_effect,
    apply_color_grading,
    apply_sharpening,
    apply_tonemapping,
    azimuth,
    bloom_enabled,
    bloom_intensity,
    bloom_radius,
    bloom_thresh,
    brightness,
    colormap,
    contrast,
    exposure,
    fog_color_picker,
    fog_density,
    fog_enabled,
    fog_falloff,
    gamma,
    saturation,
    shade_heightmap,
    sharp_amount,
    temp,
    terrain,
    time,
    tint,
    tm_method,
    vert_exag,
):
    # Apply the pipeline
    start_time = time.time()

    if terrain is None:
        base_img = None
        elapsed = 0
        current_img = None
        fog_rgb = None
    else:
        # 1. Base Render
        base_img = shade_heightmap(
            terrain,
            azimuth=azimuth.value,
            altitude=altitude.value,
            vert_exag=vert_exag.value,
            colormap=colormap.value,
            blend_mode="overlay",
        ).astype(float) / 255.0

        current_img = base_img.copy()

        # 2. Atmospheric Perspective (applied early usually, or before grading)
        if fog_enabled.value:
            # Convert hex to rgb
            h = fog_color_picker.value.lstrip('#')
            try:
                fog_rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
            except:
                fog_rgb = (0.75, 0.82, 0.9)
                
            current_img = apply_atmospheric_perspective(
                current_img,
                terrain.height,
                fog_color=fog_rgb,
                fog_density=fog_density.value,
                fog_height_falloff=fog_falloff.value
            )
        else:
            fog_rgb = None

        # 3. Tonemapping
        if tm_method.value != "none":
            current_img = apply_tonemapping(
                current_img, 
                method=tm_method.value, 
                exposure=exposure.value
            )

        # 4. Color Grading
        current_img = apply_color_grading(
            current_img,
            temperature=temp.value,
            tint=tint.value,
            saturation=saturation.value,
            contrast=contrast.value,
            brightness=brightness.value,
            gamma=gamma.value
        )

        # 5. Bloom
        if bloom_enabled.value:
            current_img = apply_bloom_effect(
                current_img,
                threshold=bloom_thresh.value,
                intensity=bloom_intensity.value,
                blur_radius=bloom_radius.value
            )

        # 6. Sharpening
        if sharp_amount.value > 0:
            current_img = apply_sharpening(
                current_img,
                amount=sharp_amount.value
            )

        elapsed = time.time() - start_time
    return base_img, current_img, elapsed, fog_rgb, start_time


@app.cell
def display_result(Image, current_img, elapsed, io, mo, np):
    # Display
    import base64

    def img_to_b64(arr):
        if arr is None:
            return ""
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    if current_img is not None:
        b64_img = img_to_b64(current_img)
        display_content = mo.vstack([
            mo.md(f"**Render Time:** {elapsed*1000:.1f} ms"),
            mo.md(f'<img src="data:image/png;base64,{b64_img}" width="100%"/>')
        ])
    else:
        display_content = mo.md("No terrain generated.")

    display_content
    return base64, b64_img, display_content, img_to_b64


if __name__ == "__main__":
    app.run()
