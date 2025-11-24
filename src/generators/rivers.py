"""
River Network Generator
-----------------------
GPU-accelerated river network generation using flow accumulation algorithms.

This module provides tools for generating realistic river networks on terrain,
including:
- Flow direction calculation (D8 algorithm)
- Flow accumulation
- River channel carving
- Moisture/wetland maps
- Lake detection

The implementation uses iterative GPU passes to propagate flow across the terrain.
"""

from dataclasses import dataclass
from typing import Callable

import moderngl
import numpy as np

from ..utils.artifacts import TerrainMaps
from ..utils.shader_loader import load_shader


@dataclass
class RiverParams:
    """
    Parameters for river network generation.

    Attributes:
        iterations: Number of flow propagation iterations (more = longer rivers)
        flow_power: Exponent for slope-based flow (higher = more focused rivers)
        river_threshold: Flow accumulation needed to form visible river
        river_depth: Maximum depth of river carving (0-1 scale)
        river_width: Width multiplier for river channels
        bank_slope: Steepness of river banks (higher = steeper)
        meander: Meandering/wiggle strength (0 = straight, 1 = very curvy)
        min_slope: Minimum terrain slope for water flow
        carve_terrain: Whether to modify the heightmap with river channels
        lake_threshold: Flow threshold for lake/pond formation
        seed: Random seed for meander noise
    """

    iterations: int = 64
    flow_power: float = 2.0
    river_threshold: float = 50.0  # Flow accumulation needed to form visible river
    river_depth: float = 0.03
    river_width: float = 1.5
    bank_slope: float = 2.0
    meander: float = 0.3
    min_slope: float = 0.001
    carve_terrain: bool = True
    lake_threshold: float = 200.0
    seed: float = 42.0

    @classmethod
    def gentle_streams(cls) -> "RiverParams":
        """Preset for subtle, gentle stream networks."""
        return cls(
            iterations=32,
            flow_power=1.5,
            river_threshold=30.0,
            river_depth=0.01,
            river_width=0.8,
            bank_slope=1.5,
            meander=0.5,
        )

    @classmethod
    def major_rivers(cls) -> "RiverParams":
        """Preset for prominent, deep river valleys."""
        return cls(
            iterations=128,
            flow_power=2.5,
            river_threshold=100.0,
            river_depth=0.06,
            river_width=2.0,
            bank_slope=3.0,
            meander=0.4,
        )

    @classmethod
    def delta_wetlands(cls) -> "RiverParams":
        """Preset for river deltas and wetland areas."""
        return cls(
            iterations=48,
            flow_power=1.2,
            river_threshold=20.0,
            river_depth=0.02,
            river_width=2.5,
            bank_slope=1.0,
            meander=0.7,
            lake_threshold=50.0,
        )

    @classmethod
    def canyon_rivers(cls) -> "RiverParams":
        """Preset for rivers in canyon/mountain terrain."""
        return cls(
            iterations=96,
            flow_power=3.0,
            river_threshold=80.0,
            river_depth=0.08,
            river_width=1.0,
            bank_slope=4.0,
            meander=0.2,
        )


class RiverGenerator:
    """
    GPU-accelerated river network generator.

    Uses flow accumulation to determine where rivers form, then optionally
    carves river channels into the terrain.

    Example:
        >>> from src.generators.rivers import RiverGenerator, RiverParams
        >>> gen = RiverGenerator(resolution=512)
        >>> terrain = ...  # Your existing TerrainMaps
        >>> params = RiverParams.major_rivers()
        >>> result = gen.generate(terrain.height, params)
        >>> # result contains river_map, moisture_map, carved_height, etc.
        >>> gen.cleanup()
    """

    def __init__(
        self,
        resolution: int = 512,
        ctx: moderngl.Context | None = None,
    ) -> None:
        """
        Initialize river generator.

        Args:
            resolution: Size of terrain grid
            ctx: Optional ModernGL context (creates new if None)
        """
        self.resolution = resolution
        self._owns_context = ctx is None
        self.ctx = ctx or moderngl.create_context(standalone=True)

        self._setup_shaders()
        self._setup_buffers()

    def _setup_shaders(self) -> None:
        """Compile shaders for river generation."""
        # Load vertex shader
        vertex_src = load_shader("quad.vert")

        # Flow accumulation shader
        flow_src = load_shader("river_flow.frag")
        self.flow_program = self.ctx.program(
            vertex_shader=vertex_src,
            fragment_shader=flow_src,
        )

        # River carving shader
        carve_src = load_shader("river_carve.frag")
        self.carve_program = self.ctx.program(
            vertex_shader=vertex_src,
            fragment_shader=carve_src,
        )

        # Fullscreen quad
        vertices = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32
        )
        vbo = self.ctx.buffer(vertices.tobytes())
        self.vao_flow = self.ctx.vertex_array(
            self.flow_program, [(vbo, "2f", "in_position")]
        )
        self.vao_carve = self.ctx.vertex_array(
            self.carve_program, [(vbo, "2f", "in_position")]
        )

    def _setup_buffers(self) -> None:
        """Create framebuffers and textures for ping-pong rendering."""
        res = self.resolution

        # Heightmap texture (input)
        self.tex_height = self.ctx.texture((res, res), 1, dtype="f4")
        self.tex_height.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_height.repeat_x = False
        self.tex_height.repeat_y = False

        # Flow accumulation textures (ping-pong)
        self.tex_flow_a = self.ctx.texture((res, res), 4, dtype="f4")
        self.tex_flow_b = self.ctx.texture((res, res), 4, dtype="f4")
        for tex in [self.tex_flow_a, self.tex_flow_b]:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False

        # River carve output texture
        self.tex_carved = self.ctx.texture((res, res), 4, dtype="f4")
        self.tex_carved.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Framebuffers
        self.fbo_flow_a = self.ctx.framebuffer(
            color_attachments=[self.tex_flow_a])
        self.fbo_flow_b = self.ctx.framebuffer(
            color_attachments=[self.tex_flow_b])
        self.fbo_carved = self.ctx.framebuffer(
            color_attachments=[self.tex_carved])

    def generate(
        self,
        heightmap: np.ndarray,
        params: RiverParams | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """
        Generate river network from heightmap.

        Args:
            heightmap: Input height array (HxW, float32, 0-1 range)
            params: River generation parameters
            progress_callback: Optional callback(current_iter, total_iters)

        Returns:
            Dictionary containing:
                - river_map: Binary river mask (HxW)
                - flow_accumulation: Raw flow accumulation values
                - moisture_map: Moisture/wetness map
                - carved_height: Height with river channels carved
                - water_depth: Depth of water at each point
                - flow_direction: Encoded flow direction
        """
        params = params or RiverParams()
        res = self.resolution

        # Ensure heightmap is correct shape and type
        if heightmap.shape != (res, res):
            raise ValueError(
                f"Heightmap shape {heightmap.shape} doesn't match resolution {res}"
            )
        heightmap = heightmap.astype(np.float32)

        # Upload heightmap
        self.tex_height.write(heightmap.tobytes())

        # Initialize flow accumulation to zero
        zeros = np.zeros((res, res, 4), dtype=np.float32)
        self.tex_flow_a.write(zeros.tobytes())
        self.tex_flow_b.write(zeros.tobytes())

        texel = 1.0 / res

        # Iterative flow accumulation
        for i in range(params.iterations):
            # Determine ping-pong direction
            if i % 2 == 0:
                src_tex = self.tex_flow_a
                dst_fbo = self.fbo_flow_b
            else:
                src_tex = self.tex_flow_b
                dst_fbo = self.fbo_flow_a

            dst_fbo.use()
            dst_fbo.viewport = (0, 0, res, res)
            dst_fbo.clear(0.0, 0.0, 0.0, 0.0)

            # Bind textures
            self.tex_height.use(location=0)
            src_tex.use(location=1)

            # Set uniforms
            self.flow_program["u_heightmap"].value = 0  # type: ignore
            self.flow_program["u_flowAccum"].value = 1  # type: ignore
            self.flow_program["u_texelSize"].value = (
                texel, texel)  # type: ignore
            # type: ignore
            self.flow_program["u_riverThreshold"].value = params.river_threshold
            # Only set uniforms that exist in the compiled shader
            if "u_flowPower" in self.flow_program:
                # type: ignore
                self.flow_program["u_flowPower"].value = params.flow_power
            if "u_minSlope" in self.flow_program:
                # type: ignore
                self.flow_program["u_minSlope"].value = params.min_slope

            self.vao_flow.render(moderngl.TRIANGLE_STRIP)

            if progress_callback:
                progress_callback(i + 1, params.iterations)

        # Read final flow accumulation
        final_flow_tex = self.tex_flow_b if params.iterations % 2 == 0 else self.tex_flow_a
        flow_data = np.frombuffer(
            final_flow_tex.read(), dtype=np.float32
        ).reshape(res, res, 4)

        # Carve rivers into terrain
        if params.carve_terrain:
            self.fbo_carved.use()
            self.fbo_carved.viewport = (0, 0, res, res)
            self.fbo_carved.clear(0.0, 0.0, 0.0, 0.0)

            self.tex_height.use(location=0)
            final_flow_tex.use(location=1)

            self.carve_program["u_heightmap"].value = 0  # type: ignore
            self.carve_program["u_riverFlow"].value = 1  # type: ignore
            self.carve_program["u_texelSize"].value = (
                texel, texel)  # type: ignore
            # type: ignore
            self.carve_program["u_riverDepth"].value = params.river_depth
            # type: ignore
            self.carve_program["u_riverWidth"].value = params.river_width
            # type: ignore
            self.carve_program["u_bankSlope"].value = params.bank_slope
            # type: ignore
            self.carve_program["u_meander"].value = params.meander
            self.carve_program["u_seed"].value = params.seed  # type: ignore

            self.vao_carve.render(moderngl.TRIANGLE_STRIP)

            carved_data = np.frombuffer(
                self.tex_carved.read(), dtype=np.float32
            ).reshape(res, res, 4)

            carved_height = carved_data[:, :, 0]
            river_mask = carved_data[:, :, 1]
            moisture_map = carved_data[:, :, 2]
            water_depth = carved_data[:, :, 3]
        else:
            carved_height = heightmap.copy()
            river_mask = (flow_data[:, :, 1] > 0.5).astype(np.float32)
            moisture_map = flow_data[:, :, 1]
            water_depth = np.zeros((res, res), dtype=np.float32)

        return {
            "river_map": river_mask,
            "flow_accumulation": flow_data[:, :, 0],
            "moisture_map": moisture_map,
            "carved_height": carved_height,
            "water_depth": water_depth,
            "flow_direction": flow_data[:, :, 2],
        }

    def apply_to_terrain(
        self,
        terrain: TerrainMaps,
        params: RiverParams | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TerrainMaps:
        """
        Apply river generation to existing TerrainMaps.

        Args:
            terrain: Input terrain data
            params: River generation parameters
            progress_callback: Optional progress callback

        Returns:
            New TerrainMaps with river data added
        """
        result = self.generate(
            terrain.height, params, progress_callback
        )

        # Create new TerrainMaps with river data
        return TerrainMaps(
            height=result["carved_height"],
            normals=terrain.normals,  # Will need to recalculate
            erosion_mask=terrain.erosion_mask,
            scatter_map=terrain.scatter_map,
            river_map=result["river_map"],
            moisture_map=result["moisture_map"],
            water_depth=result["water_depth"],
        )

    def cleanup(self) -> None:
        """Release GPU resources."""
        if self._owns_context:
            self.ctx.release()

    def __enter__(self) -> "RiverGenerator":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()
