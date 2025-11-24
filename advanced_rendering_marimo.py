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

    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    try:
        from src import ErosionTerrainGenerator, ErosionParams
        from src.utils import shade_heightmap, RenderConfig
        from src.utils import advanced_rendering as ar
    except ImportError:
        pass
    return (
        ErosionParams,
        ErosionTerrainGenerator,
        Image,
        RenderConfig,
        ar,
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
        # 🎥 Advanced Rendering & Animation

        Tools for lighting studies, multi-angle rendering, and animations.
        """
    )
    return


@app.cell
def controls(mo):
    seed_input = mo.ui.number(value=42, label="Seed")
    preset_dropdown = mo.ui.dropdown(
        options=["canyon", "mountains", "plains", "natural"],
        value="canyon",
        label="Preset"
    )
    gen_btn = mo.ui.button(label="Generate Terrain")

    mo.vstack([
        mo.md("### Terrain Setup"),
        mo.hstack([seed_input, preset_dropdown, gen_btn])
    ])
    return gen_btn, preset_dropdown, seed_input


@app.cell
def generate(ErosionParams, ErosionTerrainGenerator, gen_btn, mo, preset_dropdown, seed_input):
    terrain_state = mo.state(None)

    if gen_btn.value or terrain_state.value is None:
        # Get preset
        if preset_dropdown.value == "canyon":
            params = ErosionParams.canyon()
        elif preset_dropdown.value == "mountains":
            params = ErosionParams.mountains()
        elif preset_dropdown.value == "plains":
            params = ErosionParams.plains()
        else:
            params = ErosionParams.natural()

        gen = ErosionTerrainGenerator(resolution=512, defaults=params)
        try:
            t = gen.generate_heightmap(seed=seed_input.value)
            terrain_state.value = t
        finally:
            gen.cleanup()

    terrain = terrain_state.value
    return gen, params, t, terrain, terrain_state


@app.cell
def lighting_study_ui(ar, mo, terrain):
    # Lighting Study
    study_btn = mo.ui.button(label="Generate Lighting Study")

    study_output = mo.md("")

    if study_btn.value and terrain is not None:
        fig = ar.render_lighting_study(
            terrain,
            azimuth_steps=4,
            altitude_steps=3,
            colormap="terrain",
            vert_exag=2.0
        )
        # Convert plot to image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        import base64
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        study_output = mo.md(
            f'<img src="data:image/png;base64,{b64}" width="100%"/>')

    mo.vstack([
        mo.md("### Lighting Study"),
        mo.md("Generates a grid of renders with varying sun angles."),
        study_btn,
        study_output
    ])
    return b64, buf, fig, study_btn, study_output


@app.cell
def multi_angle_ui(ar, mo, terrain):
    # Multi-Angle Interactive
    azimuth_slider = mo.ui.slider(0, 360, value=315, label="Azimuth")
    altitude_slider = mo.ui.slider(0, 90, value=45, label="Altitude")

    preview = mo.md("No terrain")

    if terrain is not None:
        # We can use shade_heightmap directly for fast preview
        from src.utils import shade_heightmap
        img = shade_heightmap(
            terrain,
            azimuth=azimuth_slider.value,
            altitude=altitude_slider.value,
            vert_exag=2.0,
            colormap="terrain"
        )

        import io
        from PIL import Image
        import numpy as np
        import base64

        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        preview = mo.md(
            f'<img src="data:image/png;base64,{b64}" width="100%"/>')

    mo.vstack([
        mo.md("### Interactive Lighting"),
        azimuth_slider,
        altitude_slider,
        preview
    ])
    return (
        Image,
        altitude_slider,
        azimuth_slider,
        b64,
        buf,
        img,
        np,
        pil_img,
        preview,
        shade_heightmap,
    )


if __name__ == "__main__":
    app.run()
