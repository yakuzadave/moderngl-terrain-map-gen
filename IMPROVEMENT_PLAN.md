# Terrain Generation & Rendering Improvement Plan

Based on research into modern GPU terrain techniques, the following improvements are proposed for the project.

## 1. Rendering Improvements

### A. Tri-Planar Texture Mapping [DONE]
**Concept:** Projects textures from three axes (X, Y, Z) and blends them based on surface normal to prevent stretching on steep slopes.
**Implementation:**
- **GLSL:** Sample texture 3 times using world coordinates.
- **Blending:** Weight samples by `abs(normal)`.
- **Action:** Add `u_useTriplanar` to `erosion_viz.frag`.
- **Status:** Implemented in `erosion_viz.frag` and `ErosionTerrainGenerator`.

### B. Atmospheric Scattering (Fog) [DONE]
**Concept:** Physically based scattering (Rayleigh/Mie) for realistic depth.
**Implementation:**
- **GLSL:** Exponential fog based on distance and sun direction.
- **Action:** Update `erosion_viz.frag` to replace linear fog.
- **Status:** Added height fog to `erosion_viz.frag` and upgraded `erosion_raymarch.frag` to Exp2 fog.

### C. PBR Terrain Shading [DONE]
**Concept:** Use roughness/metallic maps for material properties (e.g., shiny water, matte rock).
**Implementation:**
- **Python:** Generate material masks.
- **GLSL:** Implement Cook-Torrance BRDF in `erosion_viz.frag`.
- **Status:** Refined `erosion_viz.frag` to vary smoothness based on terrain type (sand, wet sand, rock, grass).

### D. Raymarched Shadows in Viz [DONE]
**Concept:** Calculate real-time shadows in the 2D visualization shader by raymarching the heightmap.
**Implementation:**
- **GLSL:** `CalculateShadow` function in `erosion_viz.frag` marches towards the sun.
- **Status:** Implemented in `erosion_viz.frag`.

## 2. Generation Improvements

### A. Thermal Erosion (Talus Deposition) [DONE]
**Concept:** Simulates gravity moving loose material down steep slopes, creating realistic talus slopes.
**Implementation:**
- **Algorithm:** Move material from pixel to neighbor if height difference > threshold.
- **Action:** Add a new compute shader pass `thermal_erosion.frag`.
- **Status:** Implemented `thermal_erosion.frag` and `recompute_normals.frag`. Integrated into `ErosionTerrainGenerator`.

### B. Domain Warping [DONE]
**Concept:** Distort noise coordinates to create swirling, fluid-like patterns.
**Implementation:**
- **GLSL:** `fbm(p + strength * fbm(p))`.
- **Action:** Add `u_warpStrength` to `erosion_heightmap.frag`.
- **Status:** Implemented in `erosion_heightmap.frag` and `ErosionParams`.

### C. Ridge Noise [DONE]
**Concept:** Generates sharp peaks instead of lumpy hills.
**Implementation:**
- **Formula:** `(1.0 - abs(noise)) ^ 2`.
- **Action:** Add `u_useRidgeNoise` option to `erosion_heightmap.frag`.
- **Status:** Implemented in `erosion_heightmap.frag` and `ErosionParams`.
