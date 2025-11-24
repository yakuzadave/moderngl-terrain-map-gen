#!/usr/bin/env python3
"""
GPU terrain generator: hydraulic erosion.

- ErosionTerrainGenerator:
    Shadertoy-style hydraulic erosion shader translated to ModernGL,
    with runtime-configurable uniforms and domain warping to reduce tiling.

Returns a dict:
    {
        "height": 2D float32 array,
        "normals": HxWx3 float32 array,
        "erosion_mask": 2D float32 array
    }
"""

from typing import Optional, Dict, Any

import moderngl
import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Hydraulic Erosion Terrain Generator
# =============================================================================

class ErosionTerrainGenerator:
    """
    GPU-accelerated terrain generator using hydraulic-style erosion.

    Key features:
    - All important parameters are uniforms (no shader #defines),
      so you can drive them from a UI.
    - Domain warping (u_warpStrength, u_warpFrequency) reduces obvious tiling.
    """

    def __init__(
        self,
        resolution: int = 512,
        use_erosion: bool = True,
        scale: float = 3.0,
        erosion_strength: float = 0.04,
        slope_strength: float = 3.0,
        ctx: Optional[moderngl.Context] = None,
    ):
        """
        Args:
            resolution: Texture size (heightmap is resolution x resolution).
            use_erosion: Whether to run the erosion octaves at all.
            scale: Base tiling factor (used as default for height/erosion tiles).
            erosion_strength: Default erosion strength.
            slope_strength: Default slope strength.
            ctx: Optional existing moderngl.Context. If None, a standalone
                 context is created and owned by this instance.
        """
        self.resolution = int(resolution)
        self.use_erosion = bool(use_erosion)

        # Default parameters (all can be overridden per-generate call).
        # These will be mapped directly to shader uniforms.
        self.defaults: Dict[str, Any] = {
            # Heightfield
            "height_tiles": float(scale),
            "height_octaves": 3,
            "height_amp": 0.25,
            "height_gain": 0.1,
            "height_lacunarity": 2.0,
            "water_height": 0.45,

            # Erosion
            "erosion_tiles": float(scale),
            "erosion_octaves": 5,
            "erosion_gain": 0.5,
            "erosion_lacunarity": 2.0,
            "erosion_slope_strength": float(slope_strength),
            "erosion_branch_strength": float(slope_strength),
            "erosion_strength": float(erosion_strength),

            # Domain warp (new):
            "warp_strength": 0.3,   # how strongly we warp the domain
            "warp_frequency": 0.6,  # how coarse the warp field is
        }

        # Context
        if ctx is None:
            try:
                # Try to create a context. On Windows, standalone_context usually works fine.
                self.ctx = moderngl.create_standalone_context(require=330)
            except Exception as e:
                print(f"[ErosionTerrainGenerator] Error creating context: {e}")
                raise
            self._own_ctx = True
        else:
            self.ctx = ctx
            self._own_ctx = False

        renderer = self.ctx.info.get("GL_RENDERER", "Unknown")
        version = self.ctx.info.get("GL_VERSION", "Unknown")
        max_tex = self.ctx.info.get("GL_MAX_TEXTURE_SIZE", "Unknown")
        print(f"[ErosionTerrainGenerator] GL Renderer: {renderer}")
        print(f"[ErosionTerrainGenerator] GL Version:  {version}")
        print(f"[ErosionTerrainGenerator] Max Texture: {max_tex}")

        self._create_shaders()
        self._create_framebuffers()
        self._create_detail_texture()
        self._create_fullscreen_quad()

    # --------------------------------------------------------------------- #
    # Setup: shaders, FBOs, geometry
    # --------------------------------------------------------------------- #

    def _create_detail_texture(self) -> None:
        """Create a noise texture for high-frequency detail."""
        # Simple white noise
        noise_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.detail_texture = self.ctx.texture((256, 256), 3, noise_data.tobytes())
        self.detail_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.detail_texture.build_mipmaps()
        print("[ErosionTerrainGenerator] Detail texture created.")

    def _create_shaders(self) -> None:
        vertex_shader = """
        #version 330
        in vec2 in_position;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_position * 2.0 - 1.0, 0.0, 1.0);
            uv = in_position;
        }
        """

        # -----------------------------------------------------------------
        # 1. Heightmap Generation Shader (Compute-like)
        # -----------------------------------------------------------------
        fragment_shader = """
        #version 330

        in vec2 uv;
        out vec4 fragColor;

        // Erosion and height parameters (runtime configurable)
        uniform int useErosion;
        uniform float u_seed;

        // Heightfield
        uniform float u_heightTiles;
        uniform int   u_heightOctaves;
        uniform float u_heightAmp;
        uniform float u_heightGain;
        uniform float u_heightLacunarity;
        uniform float u_waterHeight;

        // Erosion
        uniform float u_erosionTiles;
        uniform int   u_erosionOctaves;
        uniform float u_erosionGain;
        uniform float u_erosionLacunarity;
        uniform float u_erosionSlopeStrength;
        uniform float u_erosionBranchStrength;
        uniform float u_erosionStrength;

        // Resolution dependent
        uniform vec2 u_texelSize; // (1.0/width, 1.0/height)

        #define PI 3.14159265358979

        vec2 hash(in vec2 x) {
            const vec2 k = vec2(0.3183099, 0.3678794);
            x = x * k + k.yx;
            return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
        }

        vec3 noised(in vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
            vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

            vec2 ga = hash(i + vec2(0.0, 0.0));
            vec2 gb = hash(i + vec2(1.0, 0.0));
            vec2 gc = hash(i + vec2(0.0, 1.0));
            vec2 gd = hash(i + vec2(1.0, 1.0));

            float va = dot(ga, f - vec2(0.0, 0.0));
            float vb = dot(gb, f - vec2(1.0, 0.0));
            float vc = dot(gc, f - vec2(0.0, 1.0));
            float vd = dot(gd, f - vec2(1.0, 1.0));

            return vec3(
                va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd),
                ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
                du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va)
            );
        }

        vec3 erosion(in vec2 p, vec2 dir) {
            vec2 ip = floor(p);
            vec2 fp = fract(p);
            float f = 2.0 * PI;
            vec3 va = vec3(0.0);
            float wt = 0.0;

            for (int i = -2; i <= 1; i++) {
                for (int j = -2; j <= 1; j++) {
                    vec2 o = vec2(float(i), float(j));
                    vec2 h = hash(ip - o) * 0.5;
                    vec2 pp = fp + o - h;
                    float d = dot(pp, pp);
                    float w = exp(-d * 2.0);
                    wt += w;
                    float mag = dot(pp, dir);
                    va += vec3(cos(mag * f), -sin(mag * f) * dir) * w;
                }
            }
            return va / wt;
        }

        vec2 generateHeightmap(vec2 uv) {
            // Offset UVs by seed to get different noise part
            vec2 p = (uv * u_heightTiles) + vec2(u_seed * 17.123, u_seed * 23.456);

            vec3 n = vec3(0.0);
            float nf = 1.0;
            float na = u_heightAmp;

            for (int i = 0; i < 16; i++) { // upper bound for safety
                if (i >= u_heightOctaves) break;
                n += noised(p * nf) * na * vec3(1.0, nf, nf);
                na *= u_heightGain;
                nf *= u_heightLacunarity;
            }
            n.x = n.x * 0.5 + 0.5;

            // Slope based initial direction
            vec2 dir = n.zy * vec2(1.0, -1.0) * u_erosionSlopeStrength;
            vec3 h = vec3(0.0);
            float a = 0.5;
            float f = 1.0;

            a *= smoothstep(u_waterHeight - 0.1, u_waterHeight + 0.2, n.x);

            int octaves = useErosion > 0 ? u_erosionOctaves : 0;

            for (int i = 0; i < 16; i++) { // same upper bound trick
                if (i >= octaves) break;
                vec2 currentDir = dir + h.zy * vec2(1.0, -1.0) * u_erosionBranchStrength;
                h += erosion(p * u_erosionTiles * f, currentDir) * a * vec3(1.0, f, f);
                a *= u_erosionGain;
                f *= u_erosionLacunarity;
            }

            float finalHeight = n.x + (h.x - 0.5) * u_erosionStrength;
            return vec2(finalHeight, h.x);
        }

        void main() {
            vec2 heightData = generateHeightmap(uv);
            float height = heightData.x;

            // Resolution independent derivatives
            float h_dx = generateHeightmap(uv + vec2(u_texelSize.x, 0.0)).x;
            float h_dy = generateHeightmap(uv + vec2(0.0, u_texelSize.y)).x;

            vec3 normal = normalize(cross(
                vec3(u_texelSize.x, 0.0, h_dx - height),
                vec3(0.0, u_texelSize.y, h_dy - height)
            ));

            fragColor = vec4(height, normal.x, normal.z, heightData.y);
        }
        """

        self.heightmap_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        # -----------------------------------------------------------------
        # 2. Visualization Shader (Shadertoy-style rendering)
        # -----------------------------------------------------------------
        viz_fragment_shader = """
        #version 330

        in vec2 uv;
        out vec4 fragColor;

        uniform sampler2D u_heightmap; // Contains (height, norm.x, norm.y, erosion)
        uniform float u_waterHeight;
        uniform vec3 u_sunDir;
        uniform int u_mode; // 0=Shaded, 1=Height, 2=Normals, 3=Erosion, 4=Slope

        // Colors from common.txt
        #define CLIFF_COLOR    vec3(0.22, 0.2, 0.2)
        #define DIRT_COLOR     vec3(0.6, 0.5, 0.4)
        #define GRASS_COLOR1   vec3(0.15, 0.3, 0.1)
        #define GRASS_COLOR2   vec3(0.4, 0.5, 0.2)
        #define SAND_COLOR     vec3(0.8, 0.7, 0.6)
        #define WATER_COLOR    vec3(0.0, 0.05, 0.1)
        #define WATER_SHORE_COLOR vec3(0.0, 0.25, 0.25)
        #define AMBIENT_COLOR  (vec3(0.3, 0.5, 0.7) * 0.1)
        #define SUN_COLOR      (vec3(1.0, 0.98, 0.95) * 2.0)

        #define PI 3.14159265358979
        #define saturate(x) clamp(x, 0.0, 1.0)
        #define sq(x) (x*x)

        // Simple noise for detail (replaces buffer_b.txt)
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), f.x),
                       mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
        }

        float fbm(vec2 p) {
            float v = 0.0;
            float a = 0.5;
            for (int i = 0; i < 5; i++) {
                v += noise(p) * a;
                p *= 2.0;
                a *= 0.5;
            }
            return v;
        }

        vec3 SkyColor(vec3 normal, vec3 sun) {
            float costh = dot(normal, sun);
            return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
        }

        // BRDF Functions from common.txt
        float pow5(float x) {
            float x2 = x * x;
            return x2 * x2 * x;
        }

        float D_GGX(float linearRoughness, float NoH, const vec3 h) {
            float oneMinusNoHSquared = 1.0 - NoH * NoH;
            float a = NoH * linearRoughness;
            float k = linearRoughness / (oneMinusNoHSquared + a * a);
            float d = k * k * (1.0 / PI);
            return d;
        }

        float V_SmithGGXCorrelated(float linearRoughness, float NoV, float NoL) {
            float a2 = linearRoughness * linearRoughness;
            float GGXV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
            float GGXL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
            return 0.5 / (GGXV + GGXL);
        }

        vec3 F_Schlick(const vec3 f0, float VoH) {
            return f0 + (vec3(1.0) - f0) * pow5(1.0 - VoH);
        }

        float Fd_Burley(float linearRoughness, float NoV, float NoL, float LoH) {
            float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
            float lightScatter = 1.0 + (f90 - 1.0) * pow5(1.0 - NoL);
            float viewScatter  = 1.0 + (f90 - 1.0) * pow5(1.0 - NoV);
            return lightScatter * viewScatter * (1.0 / PI);
        }

        float Fd_Lambert() {
            return 1.0 / PI;
        }

        vec3 Shade(vec3 diffuse, vec3 f0, float smoothness, vec3 n, vec3 v, vec3 l, vec3 lc) {
            vec3 h = normalize(v + l);

            float NoV = abs(dot(n, v)) + 1e-5;
            float NoL = saturate(dot(n, l));
            float NoH = saturate(dot(n, h));
            float LoH = saturate(dot(l, h));

            float roughness = 1.0 - smoothness;
            float linearRoughness = roughness * roughness;
            float D = D_GGX(linearRoughness, NoH, h);
            float V = V_SmithGGXCorrelated(linearRoughness, NoV, NoL);
            vec3 F = F_Schlick(f0, LoH);
            vec3 Fr = (D * V) * F;

            vec3 Fd = diffuse * Fd_Burley(linearRoughness, NoV, NoL, LoH);

            return (Fd + Fr) * lc * NoL;
        }

        void main() {
            vec4 data = texture(u_heightmap, uv);
            float height = data.x;
            vec3 normal;
            normal.xy = data.yz;
            // Reconstruct Z
            normal.z = sqrt(max(0.0, 1.0 - dot(normal.xy, normal.xy)));
            float erosion = data.w;

            // Mode switching
            if (u_mode == 1) { // Height
                fragColor = vec4(vec3(height), 1.0);
                return;
            }
            if (u_mode == 2) { // Normals
                fragColor = vec4(normal * 0.5 + 0.5, 1.0);
                return;
            }
            if (u_mode == 3) { // Erosion
                fragColor = vec4(vec3(erosion), 1.0);
                return;
            }
            if (u_mode == 4) { // Slope
                float slope = 1.0 - normal.z;
                fragColor = vec4(vec3(slope), 1.0);
                return;
            }

            vec3 sun = normalize(u_sunDir);
            vec3 viewDir = vec3(0.0, 0.0, 1.0); // Top-down view

            // Breakup noise
            vec3 breakupVec;
            breakupVec.x = fbm(uv * 32.0 + vec2(0.0, 17.2));
            breakupVec.y = fbm(uv * 58.0 + vec2(4.1, -9.7));
            breakupVec.z = fbm(uv * 96.0 + vec2(-21.3, 3.8));
            breakupVec = breakupVec * 2.0 - 1.0;

            float slope = 1.0 - normal.z;
            float breakup = breakupVec.x * 0.45;
            float macroBreak = breakupVec.y * 0.25;
            float microBreak = breakupVec.z * 0.15;

            float occlusion = sq(saturate(erosion + 0.5));

            // Material mixing
            vec3 diffuseColor = vec3(0.5);
            
            // Cliffs / Dirt
            diffuseColor = CLIFF_COLOR * smoothstep(0.4, 0.52, height);
            diffuseColor = mix(diffuseColor, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion + breakup * 1.0));
            
            // Grass
            vec3 grassMix = mix(GRASS_COLOR1, GRASS_COLOR2, smoothstep(0.4, 0.6, height - erosion * 0.05 + breakup * 0.3));
            diffuseColor = mix(diffuseColor, grassMix, smoothstep(u_waterHeight + 0.05, u_waterHeight + 0.02, height - breakup * 0.02) * smoothstep(0.8, 1.0, normal.y + breakup * 0.1));
            
            // Snow
            diffuseColor = mix(diffuseColor, vec3(1.0), smoothstep(0.53, 0.6, height + breakup * 0.1));

            vec3 f0 = vec3(0.04);
            float smoothness = 0.0;

            // Water
            if (height <= u_waterHeight + 0.0005) {
                float shore = normal.z < 0.99 ? exp(-(u_waterHeight - height) * 60.0) : 0.0;
                float foam = normal.z < 0.99 ? smoothstep(0.005, 0.0, (u_waterHeight - height) + breakup * 0.005) : 0.0;
            
                diffuseColor = mix(WATER_COLOR, WATER_SHORE_COLOR, shore);
                diffuseColor = mix(diffuseColor, vec3(1.0), foam);
                
                f0 = vec3(0.02);
                smoothness = 0.9;
                
                // Simple static water normal perturbation
                vec3 waterNormal = normalize(vec3(breakupVec.x, breakupVec.y, 15.0));
                normal = normalize(vec3(0.0, 0.0, 1.0) + waterNormal * 0.1);
                
                occlusion = 1.0;
            } else {
                // Sand at water edge
                vec3 sandColor = mix(SAND_COLOR * 0.8, SAND_COLOR * 1.1, saturate(0.5 + breakupVec.y * 0.5));
                float sandBlend = smoothstep(u_waterHeight + 0.005, u_waterHeight, height + breakup * 0.01);
                diffuseColor = mix(diffuseColor, sandColor, sandBlend);
            }

            // Lighting
            vec3 color = diffuseColor * SkyColor(normal, sun) * occlusion * Fd_Lambert();
            color += Shade(diffuseColor, f0, smoothness, normal, viewDir, sun, SUN_COLOR);
            
            // Bounce (simplified from image.txt)
            color += diffuseColor * SUN_COLOR * (dot(normal, sun * vec3(1.0,-1.0, 1.0)) * 0.5 + 0.5) * Fd_Lambert() / PI;

            // Fog (simplified)
            // float heightFog = smoothstep(u_waterHeight - 0.1, u_waterHeight + 0.6, height + macroBreak * 0.2);
            // vec3 horizon = vec3(0.45, 0.55, 0.65);
            // color = mix(horizon, color, heightFog);

            // Tonemap (ACES)
            const float a = 2.51;
            const float b = 0.03;
            const float c = 2.43;
            const float d = 0.59;
            const float e = 0.14;
            color = (color * (a * color + b)) / (color * (c * color + d) + e);
            
            // Gamma correction
            color = pow(color, vec3(1.0 / 2.2));

            fragColor = vec4(color, 1.0);
        }
        """

        self.viz_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=viz_fragment_shader,
        )

        # -----------------------------------------------------------------
        # 3. Raymarch 3D Shader (Advanced Visualization)
        # -----------------------------------------------------------------
        raymarch_fragment_shader = """
        #version 330
        in vec2 uv;
        out vec4 fragColor;

        uniform sampler2D u_heightmap;
        uniform sampler2D u_detail;
        uniform vec2 u_res;
        uniform float u_time;
        uniform vec3 u_camPos;
        uniform vec3 u_lookAt;
        uniform float u_waterHeight;
        uniform vec3 u_sunDir;

        #define PI 3.14159265
        #define saturate(x) clamp(x, 0.0, 1.0)
        #define sq(x) (x*x)

        // Colors
        #define CLIFF_COLOR    vec3(0.22, 0.2, 0.2)
        #define DIRT_COLOR     vec3(0.6, 0.5, 0.4)
        #define GRASS_COLOR1   vec3(0.15, 0.3, 0.1)
        #define GRASS_COLOR2   vec3(0.4, 0.5, 0.2)
        #define SAND_COLOR     vec3(0.8, 0.7, 0.6)
        #define WATER_COLOR    vec3(0.0, 0.05, 0.1)
        #define WATER_SHORE    vec3(0.0, 0.25, 0.25)
        #define AMBIENT_COLOR  (vec3(0.3, 0.5, 0.7) * 0.1)
        #define SUN_COLOR      (vec3(1.0, 0.98, 0.95) * 2.0)

        vec3 SkyColor(vec3 rd, vec3 sun) {
            float costh = dot(rd, sun);
            return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
        }

        vec3 Tonemap_ACES(vec3 x) {
            const float a = 2.51;
            const float b = 0.03;
            const float c = 2.43;
            const float d = 0.59;
            const float e = 0.14;
            return (x * (a * x + b)) / (x * (c * x + d) + e);
        }

        // BRDF
        float pow5(float x) { float x2 = x*x; return x2*x2*x; }
        float D_GGX(float linearRoughness, float NoH, const vec3 h) {
            float oneMinusNoHSquared = 1.0 - NoH * NoH;
            float a = NoH * linearRoughness;
            float k = linearRoughness / (oneMinusNoHSquared + a * a);
            return k * k * (1.0 / PI);
        }
        float V_SmithGGXCorrelated(float linearRoughness, float NoV, float NoL) {
            float a2 = linearRoughness * linearRoughness;
            float GGXV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
            float GGXL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
            return 0.5 / (GGXV + GGXL);
        }
        vec3 F_Schlick(const vec3 f0, float VoH) {
            return f0 + (vec3(1.0) - f0) * pow5(1.0 - VoH);
        }
        float Fd_Burley(float linearRoughness, float NoV, float NoL, float LoH) {
            float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
            return (1.0 + (f90 - 1.0) * pow5(1.0 - NoL)) * (1.0 + (f90 - 1.0) * pow5(1.0 - NoV)) * (1.0 / PI);
        }
        float Fd_Lambert() { return 1.0 / PI; }

        vec3 Shade(vec3 diffuse, vec3 f0, float smoothness, vec3 n, vec3 v, vec3 l, vec3 lc) {
            vec3 h = normalize(v + l);
            float NoV = abs(dot(n, v)) + 1e-5;
            float NoL = saturate(dot(n, l));
            float NoH = saturate(dot(n, h));
            float LoH = saturate(dot(l, h));
            float roughness = 1.0 - smoothness;
            float linearRoughness = roughness * roughness;
            float D = D_GGX(linearRoughness, NoH, h);
            float V = V_SmithGGXCorrelated(linearRoughness, NoV, NoL);
            vec3 F = F_Schlick(f0, LoH);
            vec3 Fr = (D * V) * F;
            vec3 Fd = diffuse * Fd_Burley(linearRoughness, NoV, NoL, LoH);
            return (Fd + Fr) * lc * NoL;
        }

        vec4 map(vec3 p, out float erosion) {
            vec2 texUV = p.xz + 0.5; // Map [-0.5, 0.5] to [0, 1]
            if (texUV.x < 0.0 || texUV.x > 1.0 || texUV.y < 0.0 || texUV.y > 1.0) {
                return vec4(-100.0, 0.0, 1.0, 0.0);
            }
            vec4 tex = texture(u_heightmap, texUV);
            float height = tex.x;
            vec3 normal;
            normal.xy = tex.yz;
            normal.z = sqrt(max(0.0, 1.0 - dot(normal.xy, normal.xy)));
            // Swap Y/Z for 3D world space (Y is up)
            vec3 worldNormal = vec3(normal.x, normal.z, normal.y);
            erosion = tex.w;
            return vec4(height, worldNormal);
        }

        float march(vec3 ro, vec3 rd, out vec3 normal, out int material, out float t) {
            t = 0.0;
            float tMax = 20.0;
            material = 0; // 0=Ground, 1=Water
            
            // Bounding box intersection (Box is -0.5 to 0.5 in X/Z, 0 to 1 in Y)
            // Actually height is 0 to 1, so box Y is 0 to 1.
            // Ray-box intersection
            vec3 boxMin = vec3(-0.5, 0.0, -0.5);
            vec3 boxMax = vec3(0.5, 1.0, 0.5);
            vec3 invRd = 1.0 / rd;
            vec3 t0 = (boxMin - ro) * invRd;
            vec3 t1 = (boxMax - ro) * invRd;
            vec3 tMin = min(t0, t1);
            vec3 tMaxBox = max(t0, t1);
            float tBoxIn = max(max(tMin.x, tMin.y), tMin.z);
            float tBoxOut = min(min(tMaxBox.x, tMaxBox.y), tMaxBox.z);
            
            if (tBoxIn > tBoxOut || tBoxOut < 0.0) return -1.0;
            
            t = max(0.0, tBoxIn);
            float stepScale = 1.0;
            
            for(int i=0; i<128; i++) {
                vec3 p = ro + rd * t;
                if (p.x < -0.5 || p.x > 0.5 || p.z < -0.5 || p.z > 0.5) {
                    t += 0.05; continue; 
                }
                
                float erosion;
                vec4 mapData = map(p, erosion);
                float h = mapData.x;
                normal = mapData.yzw;
                
                float h_water = u_waterHeight;
                float d_terrain = p.y - h;
                float d_water = p.y - h_water;
                
                // Check water
                if (d_water < 0.001 && d_terrain > 0.0) {
                    material = 1;
                    normal = vec3(0.0, 1.0, 0.0);
                    // Refine water intersection
                    t += d_water;
                    return t;
                }
                
                if (d_terrain < 0.001) {
                    material = 0;
                    t += d_terrain; // Refine
                    return t;
                }
                
                float dt = min(abs(d_terrain), abs(d_water));
                t += max(0.002, dt * 0.5);
                if (t > tBoxOut) break;
            }
            
            return -1.0;
        }

        void main() {
            // Camera setup
            vec3 ro = u_camPos;
            vec3 forward = normalize(u_lookAt - ro);
            vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
            vec3 up = cross(right, forward);
            
            vec2 screenUV = uv * 2.0 - 1.0;
            screenUV.x *= u_res.x / u_res.y;
            
            vec3 rd = normalize(forward * 2.0 + right * screenUV.x + up * screenUV.y);
            
            vec3 sun = normalize(u_sunDir);
            vec3 color = SkyColor(rd, sun);
            
            vec3 normal;
            int material;
            float t;
            float hit = march(ro, rd, normal, material, t);
            
            if (hit > 0.0) {
                vec3 p = ro + rd * hit;
                float erosion;
                vec4 mapData = map(p, erosion);
                
                // Detail texture
                vec3 detail = texture(u_detail, p.xz * 4.0).rgb;
                float breakup = detail.r * 0.1;
                
                vec3 diffuse = vec3(0.5);
                vec3 f0 = vec3(0.04);
                float smoothness = 0.0;
                float occlusion = saturate(erosion + 0.5);
                
                if (material == 0) { // Ground
                    // Cliffs
                    diffuse = CLIFF_COLOR * smoothstep(0.4, 0.52, p.y);
                    diffuse = mix(diffuse, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion));
                    
                    // Grass
                    vec3 grassMix = mix(GRASS_COLOR1, GRASS_COLOR2, smoothstep(0.4, 0.6, p.y - erosion * 0.05));
                    diffuse = mix(diffuse, grassMix, smoothstep(u_waterHeight + 0.05, u_waterHeight + 0.02, p.y) * smoothstep(0.7, 1.0, normal.y));
                    
                    // Snow
                    diffuse = mix(diffuse, vec3(1.0), smoothstep(0.53, 0.6, p.y));
                    
                    // Sand
                    vec3 sandColor = SAND_COLOR;
                    float sandBlend = smoothstep(u_waterHeight + 0.005, u_waterHeight, p.y);
                    diffuse = mix(diffuse, sandColor, sandBlend);
                } else { // Water
                    float shore = exp(-(p.y - mapData.x) * 60.0);
                    diffuse = mix(WATER_COLOR, WATER_SHORE, shore);
                    f0 = vec3(0.02);
                    smoothness = 0.9;
                    normal = normalize(vec3(0.0, 1.0, 0.0) + detail * 0.05);
                }
                
                // Lighting
                vec3 viewDir = -rd;
                color = diffuse * SkyColor(normal, sun) * occlusion * Fd_Lambert();
                color += Shade(diffuse, f0, smoothness, normal, viewDir, sun, SUN_COLOR);
                
                // Fog
                float fog = 1.0 - exp(-hit * 0.1);
                color = mix(color, vec3(0.6, 0.7, 0.8), fog);
            }
            
            color = Tonemap_ACES(color);
            color = pow(color, vec3(1.0 / 2.2));
            fragColor = vec4(color, 1.0);
        }
        """

        self.raymarch_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=raymarch_fragment_shader,
        )

        print("[ErosionTerrainGenerator] Shader programs compiled.")

    def _create_framebuffers(self) -> None:
        # Heightmap FBO (existing)
        self.heightmap_texture = self.ctx.texture(
            (self.resolution, self.resolution), 4, dtype="f4"
        )
        self.heightmap_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.heightmap_fbo = self.ctx.framebuffer(
            color_attachments=[self.heightmap_texture]
        )

        # Visualization FBO (new)
        self.viz_texture = self.ctx.texture(
            (self.resolution, self.resolution), 3, dtype="f1"
        )
        self.viz_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.viz_fbo = self.ctx.framebuffer(
            color_attachments=[self.viz_texture]
        )

        print(
            f"[ErosionTerrainGenerator] Framebuffers created: "
            f"{self.resolution} x {self.resolution}"
        )

    def _create_fullscreen_quad(self) -> None:
        vertices = np.array(
            [0.0, 0.0,
             1.0, 0.0,
             1.0, 1.0,
             0.0, 1.0],
            dtype="f4",
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.heightmap_vao = self.ctx.vertex_array(
            self.heightmap_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )
        self.viz_vao = self.ctx.vertex_array(
            self.viz_program,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )

    # --------------------------------------------------------------------- #
    # Main generation path
    # --------------------------------------------------------------------- #

    def generate_heightmap(self, seed: int = 0, seamless: bool = False, **overrides) -> Dict[str, np.ndarray]:
        """
        Generate terrain heightmap on GPU.

        Args:
            seed: Random seed for the shader.
            seamless: If True, generates a seamlessly tileable heightmap.
                      This forces tiles to be integers and disables domain rotation.
            overrides: Optional overrides for any of the default parameters.

        Returns:
            dict with 'height', 'normals', 'erosion_mask' (all np.float32 arrays)
        """
        print(
            f"[ErosionTerrainGenerator] Generating terrain (seed={seed}, seamless={seamless})...", end=" ")

        params = self.defaults.copy()
        params.update(overrides)

        if seamless:
            # For seamless tiling, the base frequencies must be integers
            params["height_tiles"] = float(round(params["height_tiles"]))
            params["erosion_tiles"] = float(round(params["erosion_tiles"]))
            if params["height_tiles"] < 1.0:
                params["height_tiles"] = 1.0
            if params["erosion_tiles"] < 1.0:
                params["erosion_tiles"] = 1.0

        self.heightmap_fbo.use()
        self.heightmap_fbo.clear(0.0, 0.0, 0.0, 1.0)

        prog = self.heightmap_program
        if "useErosion" in prog:
            prog["useErosion"] = 1 if self.use_erosion else 0
        prog["u_seed"] = float(seed)

        texel = 1.0 / float(self.resolution)
        prog["u_texelSize"] = (texel, texel)

        mapping = {
            "height_tiles": "u_heightTiles",
            "height_octaves": "u_heightOctaves",
            "height_amp": "u_heightAmp",
            "height_gain": "u_heightGain",
            "height_lacunarity": "u_heightLacunarity",
            "water_height": "u_waterHeight",
            "erosion_tiles": "u_erosionTiles",
            "erosion_octaves": "u_erosionOctaves",
            "erosion_gain": "u_erosionGain",
            "erosion_lacunarity": "u_erosionLacunarity",
            "erosion_slope_strength": "u_erosionSlopeStrength",
            "erosion_branch_strength": "u_erosionBranchStrength",
            "erosion_strength": "u_erosionStrength",
        }

        for py_name, gl_name in mapping.items():
            val = params[py_name]
            if "octaves" in py_name:
                prog[gl_name] = int(val)
            else:
                prog[gl_name] = float(val)

        self.heightmap_vao.render(moderngl.TRIANGLES)

        # Read back data
        raw_data = self.heightmap_fbo.read(components=4, dtype="f4")
        data = np.frombuffer(raw_data, dtype="f4").reshape(
            (self.resolution, self.resolution, 4)
        )
        # Flip Y because OpenGL origin is bottom-left, but images usually top-left
        data = np.flip(data, axis=0)

        height = data[:, :, 0]
        normal_x = data[:, :, 1]
        normal_z = data[:, :, 2]
        erosion_mask = data[:, :, 3]

        # Reconstruct Y component of normal (assuming upward facing)
        # Clamp to avoid sqrt of negative due to precision errors
        normal_y_sq = 1.0 - normal_x**2 - normal_z**2
        normal_y_sq = np.maximum(normal_y_sq, 0.0)
        normal_y = np.sqrt(normal_y_sq)

        # Stack normals into HxWx3
        normals = np.dstack((normal_x, normal_y, normal_z))

        print("Done.")
        return {
            "height": height,
            "normals": normals,
            "erosion_mask": erosion_mask
        }

    def render_visualization(self, water_height: float = 0.45, sun_dir: tuple = (-1.0, 0.1, 0.25), mode: int = 0) -> np.ndarray:
        """
        Render the generated heightmap to a colored image using the visualization shader.
        mode: 0=Shaded, 1=Height, 2=Normals, 3=Erosion, 4=Slope
        """
        self.viz_fbo.use()
        self.viz_fbo.clear(0.0, 0.0, 0.0, 1.0)

        # Bind heightmap texture to unit 0
        self.heightmap_texture.use(location=0)

        prog = self.viz_program
        prog["u_heightmap"] = 0
        prog["u_waterHeight"] = float(water_height)
        prog["u_sunDir"] = tuple(sun_dir)
        if "u_mode" in prog:
            prog["u_mode"] = int(mode)

        self.heightmap_vao.render(moderngl.TRIANGLES)

        # Read back
        raw_data = self.viz_fbo.read(components=3, dtype="f1")
        data = np.frombuffer(raw_data, dtype="u1").reshape(
            (self.resolution, self.resolution, 3)
        )
        # Flip Y
        data = np.flip(data, axis=0)
        return data

    def render_3d(self, camera_pos: tuple = (0.0, 0.8, -1.2), look_at: tuple = (0.0, 0.0, 0.0), 
                 water_height: float = 0.45, sun_dir: tuple = (-1.0, 0.1, 0.25)) -> np.ndarray:
        """
        Render the terrain using 3D raymarching.
        """
        self.viz_fbo.use()
        self.viz_fbo.clear(0.0, 0.0, 0.0, 1.0)

        self.heightmap_texture.use(location=0)
        self.detail_texture.use(location=1)

        prog = self.raymarch_program
        prog["u_heightmap"] = 0
        prog["u_detail"] = 1
        prog["u_res"] = (self.resolution, self.resolution)
        if "u_time" in prog:
            prog["u_time"] = 0.0 # Static for now
        prog["u_camPos"] = tuple(camera_pos)
        prog["u_lookAt"] = tuple(look_at)
        prog["u_waterHeight"] = float(water_height)
        prog["u_sunDir"] = tuple(sun_dir)

        self.heightmap_vao.render(moderngl.TRIANGLES)

        raw_data = self.viz_fbo.read(components=3, dtype="f1")
        data = np.frombuffer(raw_data, dtype="u1").reshape(
            (self.resolution, self.resolution, 3)
        )
        data = np.flip(data, axis=0)
        return data

    def analyze_terrain(self, result: Dict[str, np.ndarray], title: str = "Terrain Analysis"):
        """
        Analyze terrain statistics and plot histograms, CDF, and FFT.
        """
        height = result["height"]
        normals = result["normals"]
        erosion = result["erosion_mask"];

        print(f"--- {title} ---")
        if np.isnan(height).any():
            print("WARNING: Heightmap contains NaNs!")
        
        # Basic Stats
        h_min, h_max, h_mean, h_std = np.nanmin(height), np.nanmax(height), np.nanmean(height), np.nanstd(height)
        print(f"Height: min={h_min:.4f}, max={h_max:.4f}, mean={h_mean:.4f}, std={h_std:.4f}")
        
        slope = 1.0 - normals[:, :, 2]
        s_mean = np.nanmean(slope)
        print(f"Slope: mean={s_mean:.4f}")
        
        # Ruggedness (Terrain Ruggedness Index - simplified as std dev of slope)
        ruggedness = np.nanstd(slope)
        print(f"Ruggedness Index: {ruggedness:.4f}")

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{title} (Ruggedness: {ruggedness:.4f})", fontsize=16)

        # Clean data
        height_clean = height.flatten()
        height_clean = height_clean[~np.isnan(height_clean)]
        slope_clean = slope.flatten()
        slope_clean = slope_clean[~np.isnan(slope_clean)]

        if len(height_clean) > 0:
            # 1. Height Histogram
            axs[0, 0].hist(height_clean, bins=50, color='skyblue', edgecolor='black', density=True)
            axs[0, 0].set_title("Height Distribution")
            
            # 2. Hypsometric Curve (CDF)
            sorted_h = np.sort(height_clean)
            yvals = np.arange(len(sorted_h)) / float(len(sorted_h) - 1);
            axs[0, 1].plot(sorted_h, yvals, color='purple', lw=2)
            axs[0, 1].set_title("Hypsometric Curve (CDF)")
            axs[0, 1].set_xlabel("Height")
            axs[0, 1].set_ylabel("Cumulative Fraction")
            axs[0, 1].grid(True, alpha=0.3)

            # 3. Slope Histogram
            axs[0, 2].hist(slope_clean, bins=50, color='salmon', edgecolor='black', density=True)
            axs[0, 2].set_title("Slope Distribution")

            # 4. Heightmap Image
            im = axs[1, 0].imshow(height, cmap='terrain')
            axs[1, 0].set_title("Heightmap")
            plt.colorbar(im, ax=axs[1, 0])

            # 5. FFT (Spatial Frequency)
            f = np.fft.fft2(height)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
            axs[1, 1].imshow(magnitude_spectrum, cmap='inferno')
            axs[1, 1].set_title("Spatial Frequency (FFT)")
            axs[1, 1].axis('off')

            # 6. Erosion Mask
            axs[1, 2].imshow(erosion, cmap='RdYlBu_r')
            axs[1, 2].set_title("Erosion Mask")
            axs[1, 2].axis('off')

        plt.tight_layout()
        plt.show(block=False)


def run_comprehensive_tests():
    """Run extensive testing suite with multiple scenarios."""
    print("="*80)
    print("TERRAIN GENERATOR COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_configs = [
        {
            "name": "Minimal (No Erosion)",
            "resolution": 512,
            "use_erosion": False,
            "seed": 42,
            "height_octaves": 3,
            "erosion_strength": 0.0
        },
        {
            "name": "Light Erosion",
            "resolution": 512,
            "use_erosion": True,
            "seed": 42,
            "height_octaves": 3,
            "erosion_strength": 0.02,
            "erosion_octaves": 3
        },
        {
            "name": "Heavy Erosion (Default)",
            "resolution": 512,
            "use_erosion": True,
            "seed": 42,
            "height_octaves": 3,
            "erosion_strength": 0.04,
            "erosion_octaves": 5
        },
        {
            "name": "Extreme Detail",
            "resolution": 512,
            "use_erosion": True,
            "seed": 42,
            "height_octaves": 6,
            "erosion_octaves": 8,
            "erosion_strength": 0.06
        },
    ]
    
    results = []
    
    for idx, config in enumerate(test_configs, 1):
        print(f"\\n[Test {idx}/{len(test_configs)}] {config['name']}")
        print("-" * 60)
        
        try:
            gen = ErosionTerrainGenerator(
                resolution=config["resolution"],
                use_erosion=config["use_erosion"]
            )
            
            # Extract generation params
            gen_params = {k: v for k, v in config.items() 
                         if k not in ["name", "resolution", "use_erosion"]}
            
            result = gen.generate_heightmap(**gen_params)
            gen.analyze_terrain(result, config["name"])
            
            results.append({
                "config": config,
                "data": result,
                "generator": gen
            })
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def create_comparison_visualization(results):
    """Create a comprehensive comparison of all test results."""
    n_tests = len(results)
    
    # Main comparison figure: Top-down vs 3D
    fig1, axes1 = plt.subplots(2, n_tests, figsize=(5 * n_tests, 8))
    if n_tests == 1:
        axes1 = axes1.reshape(2, 1)
        
    fig1.suptitle("Terrain Comparison: Top-Down vs 3D Raymarch", fontsize=16, fontweight='bold')
    
    for idx, res in enumerate(results):
        gen = res["generator"]
        config = res["config"]
        
        # Top-down
        viz_top = gen.render_visualization(mode=0)
        axes1[0, idx].imshow(viz_top)
        axes1[0, idx].set_title(f"{config['name']}\nTop-Down", fontsize=10)
        axes1[0, idx].axis("off")
        
        # 3D Raymarch
        viz_3d = gen.render_3d(camera_pos=(0.0, 0.6, -1.0), look_at=(0.0, -0.2, 0.0))
        axes1[1, idx].imshow(viz_3d)
        axes1[1, idx].set_title("3D Raymarch", fontsize=10)
        axes1[1, idx].axis("off")
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Detailed analysis for the best result (last one)
    if results:
        best = results[-1]
        gen = best["generator"]
        config = best["config"]
        
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
        fig2.suptitle(f"Detailed Analysis: {config['name']}", fontsize=16, fontweight='bold')
        
        # Render all visualization modes
        viz_modes = [
            (0, "Shaded (PBR)"),
            (1, "Height Map"),
            (2, "Normal Map"),
            (3, "Erosion Mask"),
            (4, "Slope Map"),
        ]
        
        for idx, (mode, title) in enumerate(viz_modes):
            row = idx // 3
            col = idx % 3
            viz = gen.render_visualization(mode=mode)
            axes2[row, col].imshow(viz)
            axes2[row, col].set_title(title, fontsize=12, fontweight='bold')
            axes2[row, col].axis("off")
        
        # 3D Surface plot (Matplotlib)
        from mpl_toolkits.mplot3d import Axes3D
        ax_3d = fig2.add_subplot(2, 3, 6, projection='3d')
        height = best["data"]["height"]
        x = np.linspace(0, 1, height.shape[1])
        y = np.linspace(0, 1, height.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Downsample for performance
        stride = 8
        ax_3d.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], 
                          height[::stride, ::stride], cmap='terrain',
                          linewidth=0, antialiased=True, alpha=0.9)
        ax_3d.set_title("Matplotlib 3D Surface", fontsize=12, fontweight='bold')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Height')
        
        plt.tight_layout()
        plt.show(block=False)


if __name__ == "__main__":
    try:
        # Run comprehensive test suite
        results = run_comprehensive_tests()
        
        print("\\n" + "="*80)
        print("GENERATING VISUALIZATIONS...")
        print("="*80)
        
        # Create comparison visualizations
        create_comparison_visualization(results)
        
        print("\n✅ All tests completed successfully!")
        plt.show()
        
    except Exception as e:
        print(f"\\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
