import moderngl
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from ..utils import load_shader, gl_context
from ..utils.artifacts import TerrainMaps


@dataclass
class HydraulicParams:
    """
    Parameters for hydraulic erosion simulation.

    Attributes:
        iterations: Number of simulation steps.
        dt: Time step per iteration.
        pipe_length: Length of virtual pipes (L).
        sediment_capacity: Sediment capacity constant (Kc).
        soil_dissolving: Soil dissolving constant (Ks).
        sediment_deposition: Sediment deposition constant (Kd).
        evaporation_rate: Water evaporation rate (Ke).
        rain_rate: Amount of rain added per step (optional).
        thermal_erosion_rate: Rate of thermal erosion (talus deposition).
        talus_angle: Critical angle for thermal erosion (tangent of angle).
    """
    iterations: int = 100
    dt: float = 0.02
    pipe_length: float = 1.0
    sediment_capacity: float = 1.0  # Kc
    soil_dissolving: float = 0.5    # Ks
    sediment_deposition: float = 0.5  # Kd
    evaporation_rate: float = 0.01  # Ke
    rain_rate: float = 0.01         # Amount of rain added per step (optional)
    thermal_erosion_rate: float = 0.15  # Rate of thermal erosion
    talus_angle: float = 0.008      # Critical angle for thermal erosion


class HydraulicErosionGenerator:
    """
    GPU-accelerated hydraulic erosion simulator using the pipe model.
    Implements a multi-stage simulation pipeline:
    1. Flux computation (water movement)
    2. Water velocity & depth update
    3. Erosion & Deposition (sediment transport)
    4. Sediment Advection (movement with flow)
    5. Evaporation
    6. Thermal Erosion (slope stabilization)
    """
    def __init__(self, resolution: int = 512, ctx: moderngl.Context | None = None):
        self.resolution = resolution
        self.ctx = ctx if ctx else gl_context.create_context()
        self._own_ctx = ctx is None

        # Load shaders - use hydraulic-specific vertex shader
        self.quad_vert = load_shader("hydraulic/quad.vert")

        self.prog_flux = self.ctx.program(
            vertex_shader=self.quad_vert,
            fragment_shader=load_shader("hydraulic/flux.frag")
        )
        self.prog_water = self.ctx.program(
            vertex_shader=self.quad_vert,
            fragment_shader=load_shader("hydraulic/water_velocity.frag")
        )
        self.prog_erosion = self.ctx.program(
            vertex_shader=self.quad_vert,
            fragment_shader=load_shader("hydraulic/erosion_deposition.frag")
        )
        self.prog_advection = self.ctx.program(
            vertex_shader=self.quad_vert,
            fragment_shader=load_shader("hydraulic/sediment_advection.frag")
        )
        self.prog_evaporation = self.ctx.program(
            vertex_shader=self.quad_vert,
            fragment_shader=load_shader("hydraulic/evaporation.frag")
        )
        self.prog_thermal = self.ctx.program(
            vertex_shader=self.quad_vert,
            fragment_shader=load_shader("hydraulic/thermal_erosion.frag")
        )

        # Create textures (Ping-Pong pairs)
        self.textures: Dict[str, moderngl.Texture] = {}
        for name in ['height', 'water', 'sediment', 'flux']:
            # Float32 textures
            # Height: R
            # Water: RGBA (R=Height, GB=Vel)
            # Sediment: R
            # Flux: RGBA
            components = 4 if name in ['water', 'flux'] else 1
            self.textures[f'{name}_a'] = self.ctx.texture(
                (resolution, resolution), components, dtype='f4')
            self.textures[f'{name}_b'] = self.ctx.texture(
                (resolution, resolution), components, dtype='f4')

            # Clear them
            self.textures[f'{name}_a'].filter = (
                moderngl.NEAREST, moderngl.NEAREST)
            self.textures[f'{name}_b'].filter = (
                moderngl.NEAREST, moderngl.NEAREST)
            # Clamp to edge
            self.textures[f'{name}_a'].repeat_x = False
            self.textures[f'{name}_a'].repeat_y = False
            self.textures[f'{name}_b'].repeat_x = False
            self.textures[f'{name}_b'].repeat_y = False

        # Create FBOs
        self.fbos: Dict[str, moderngl.Framebuffer] = {}

        # Flux FBOs
        self.fbos['flux_a'] = self.ctx.framebuffer(
            color_attachments=[self.textures['flux_a']])
        self.fbos['flux_b'] = self.ctx.framebuffer(
            color_attachments=[self.textures['flux_b']])

        # Water FBOs
        self.fbos['water_a'] = self.ctx.framebuffer(
            color_attachments=[self.textures['water_a']])
        self.fbos['water_b'] = self.ctx.framebuffer(
            color_attachments=[self.textures['water_b']])

        # Erosion FBOs (MRT: Height, Sediment)
        self.fbos['erosion_a'] = self.ctx.framebuffer(
            color_attachments=[self.textures['height_a'], self.textures['sediment_a']])
        self.fbos['erosion_b'] = self.ctx.framebuffer(
            color_attachments=[self.textures['height_b'], self.textures['sediment_b']])

        # Advection FBOs (Sediment only)
        self.fbos['sediment_a'] = self.ctx.framebuffer(
            color_attachments=[self.textures['sediment_a']])
        self.fbos['sediment_b'] = self.ctx.framebuffer(
            color_attachments=[self.textures['sediment_b']])

        # Evaporation FBOs (Water only)
        # Reuse water FBOs? Yes, they target the water texture.

        # Thermal FBOs (Height only)
        self.fbos['thermal_a'] = self.ctx.framebuffer(
            color_attachments=[self.textures['height_a']])
        self.fbos['thermal_b'] = self.ctx.framebuffer(
            color_attachments=[self.textures['height_b']])

        # Quad VBO
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
            1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())

        # VAOs - bind both position (2f) and UV (2f) attributes
        self.vaos = {}
        for prog_name, prog in [
            ('flux', self.prog_flux),
            ('water', self.prog_water),
            ('erosion', self.prog_erosion),
            ('advection', self.prog_advection),
            ('evaporation', self.prog_evaporation),
            ('thermal', self.prog_thermal)
        ]:
            self.vaos[prog_name] = self.ctx.vertex_array(
                prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')])

    def cleanup(self):
        # Release resources
        for tex in self.textures.values():
            tex.release()
        for fbo in self.fbos.values():
            fbo.release()
        self.vbo.release()
        for vao in self.vaos.values():
            vao.release()
        self.prog_flux.release()
        self.prog_water.release()
        self.prog_erosion.release()
        self.prog_advection.release()
        self.prog_evaporation.release()
        self.prog_thermal.release()
        if self._own_ctx:
            self.ctx.release()

    def __enter__(self) -> "HydraulicErosionGenerator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def simulate(self, initial_heightmap: np.ndarray, params: HydraulicParams) -> TerrainMaps:
        """
        Runs the hydraulic erosion simulation on the given heightmap.

        Args:
            initial_heightmap: Initial terrain heightmap (2D numpy array).
            params: Simulation parameters.

        Returns:
            A TerrainMaps object containing the eroded heightmap, normals, and
            sediment mask.
        """
        # Upload initial heightmap
        self.textures['height_a'].write(
            initial_heightmap.astype('f4').tobytes())

        # Clear other maps
        self.fbos['water_a'].clear()
        self.fbos['water_b'].clear()
        self.fbos['flux_a'].clear()
        self.fbos['flux_b'].clear()
        self.fbos['sediment_a'].clear()
        self.fbos['sediment_b'].clear()

        # Pointers to current valid data
        # Start with everything in 'a' (except others are empty)
        ptr_h = 'a'
        ptr_w = 'a'
        ptr_s = 'a'
        ptr_f = 'a'

        def other(p): return 'b' if p == 'a' else 'a'

        # Simple rain: Fill water_a with 0.1 water everywhere.
        rain_data = np.zeros((self.resolution, self.resolution, 4), dtype='f4')
        rain_data[:, :, 0] = 0.1  # 0.1 units of water
        self.textures['water_a'].write(rain_data.tobytes())

        for i in range(params.iterations):
            # 1. Flux
            # In: Height(ptr_h), Water(ptr_w), Flux(ptr_f)
            # Out: Flux(other(ptr_f))
            next_f = other(ptr_f)

            self.textures[f'height_{ptr_h}'].use(location=0)
            self.textures[f'water_{ptr_w}'].use(location=1)
            self.textures[f'flux_{ptr_f}'].use(location=2)

            self.prog_flux['u_heightMap'] = 0
            self.prog_flux['u_waterMap'] = 1
            self.prog_flux['u_fluxMap'] = 2
            self.prog_flux['u_dt'] = params.dt
            self.prog_flux['u_pipeLength'] = params.pipe_length
            self.prog_flux['u_texelSize'] = (
                1/self.resolution, 1/self.resolution)

            self.fbos[f'flux_{next_f}'].use()
            self.vaos['flux'].render(moderngl.TRIANGLE_STRIP)
            ptr_f = next_f

            # 2. Water & Velocity
            # In: Water(ptr_w), Flux(ptr_f)
            # Out: Water(other(ptr_w))
            next_w = other(ptr_w)

            self.textures[f'water_{ptr_w}'].use(location=0)
            self.textures[f'flux_{ptr_f}'].use(location=1)

            self.prog_water['u_waterMap'] = 0
            self.prog_water['u_fluxMap'] = 1
            self.prog_water['u_dt'] = params.dt
            self.prog_water['u_texelSize'] = (
                1/self.resolution, 1/self.resolution)

            self.fbos[f'water_{next_w}'].use()
            self.vaos['water'].render(moderngl.TRIANGLE_STRIP)
            ptr_w = next_w

            # 3. Erosion & Deposition
            # In: Height(ptr_h), Water(ptr_w), Sediment(ptr_s)
            # Out: Height(other(ptr_h)), Sediment(other(ptr_s))
            next_h = other(ptr_h)
            next_s = other(ptr_s)

            self.textures[f'height_{ptr_h}'].use(location=0)
            self.textures[f'water_{ptr_w}'].use(location=1)
            self.textures[f'sediment_{ptr_s}'].use(location=2)

            self.prog_erosion['u_heightMap'] = 0
            self.prog_erosion['u_waterVelocityMap'] = 1
            self.prog_erosion['u_sedimentMap'] = 2
            self.prog_erosion['u_dt'] = params.dt
            self.prog_erosion['u_capacity'] = params.sediment_capacity
            self.prog_erosion['u_dissolving'] = params.soil_dissolving
            self.prog_erosion['u_deposition'] = params.sediment_deposition
            self.prog_erosion['u_texelSize'] = (
                1/self.resolution, 1/self.resolution)

            self.fbos[f'erosion_{next_h}'].use()
            self.vaos['erosion'].render(moderngl.TRIANGLE_STRIP)
            ptr_h = next_h
            ptr_s = next_s

            # 4. Sediment Advection
            # In: Sediment(ptr_s), Water(ptr_w)
            # Out: Sediment(other(ptr_s))
            next_s = other(ptr_s)

            self.textures[f'sediment_{ptr_s}'].use(location=0)
            self.textures[f'water_{ptr_w}'].use(location=1)

            self.prog_advection['u_sedimentMap'] = 0
            self.prog_advection['u_waterVelocityMap'] = 1
            self.prog_advection['u_dt'] = params.dt
            self.prog_advection['u_texelSize'] = (
                1/self.resolution, 1/self.resolution)

            self.fbos[f'sediment_{next_s}'].use()
            self.vaos['advection'].render(moderngl.TRIANGLE_STRIP)
            ptr_s = next_s

            # 5. Evaporation
            # In: Water(ptr_w)
            # Out: Water(other(ptr_w))
            next_w = other(ptr_w)

            self.textures[f'water_{ptr_w}'].use(location=0)

            self.prog_evaporation['u_waterMap'] = 0
            self.prog_evaporation['u_dt'] = params.dt
            self.prog_evaporation['u_evaporationRate'] = params.evaporation_rate

            self.fbos[f'water_{next_w}'].use()
            self.vaos['evaporation'].render(moderngl.TRIANGLE_STRIP)
            ptr_w = next_w

            # 6. Thermal Erosion
            # In: Height(ptr_h)
            # Out: Height(other(ptr_h))
            next_h = other(ptr_h)

            self.textures[f'height_{ptr_h}'].use(location=0)

            self.prog_thermal['u_heightMap'] = 0
            self.prog_thermal['u_dt'] = params.dt
            self.prog_thermal['u_talusAngle'] = params.talus_angle
            self.prog_thermal['u_thermalRate'] = params.thermal_erosion_rate
            self.prog_thermal['u_texelSize'] = (
                1/self.resolution, 1/self.resolution)

            self.fbos[f'thermal_{next_h}'].use()
            self.vaos['thermal'].render(moderngl.TRIANGLE_STRIP)
            ptr_h = next_h

        # Read back result
        # Use erosion FBO which has height as attachment 0
        raw = self.fbos[f'erosion_{ptr_h}'].read(
            components=1, attachment=0, dtype='f4')
        heightmap = np.frombuffer(raw, dtype='f4').reshape(
            self.resolution, self.resolution)

        # Compute normals
        dy, dx = np.gradient(heightmap)
        # Scale by resolution to account for 0-1 UV space vs 0-1 height
        dx *= self.resolution
        dy *= self.resolution

        nx = -dx
        ny = np.ones_like(dx)
        nz = -dy

        length = np.sqrt(nx**2 + ny**2 + nz**2)
        normals = np.stack([nx/length, ny/length, nz/length], axis=-1)

        # Optional: Read back erosion mask (sediment)
        # Use sediment FBO which has sediment as attachment 0
        raw_sediment = self.fbos[f'sediment_{ptr_s}'].read(
            components=1, dtype='f4')
        sediment_map = np.frombuffer(raw_sediment, dtype='f4').reshape(
            self.resolution, self.resolution)

        return TerrainMaps(height=heightmap, normals=normals, erosion_mask=sediment_map)
