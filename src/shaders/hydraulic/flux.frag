#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_heightMap; // R=Terrain Height
uniform sampler2D u_waterMap;  // R=Water Height
uniform sampler2D u_fluxMap;   // RGBA=Flux(Left, Right, Top, Bottom)

uniform float u_dt;
uniform float u_pipeLength; // Virtual pipe length (usually 1.0)
uniform vec2 u_texelSize;

// Gravity constant scaled for simulation
const float GRAVITY = 9.81;
const float MIN_WATER = 0.0001;

void main() {
    float terrainHeight = texture(u_heightMap, v_uv).r;
    float waterHeight = texture(u_waterMap, v_uv).r;
    float totalHeight = terrainHeight + waterHeight;

    vec4 flux = texture(u_fluxMap, v_uv);

    // Neighbor heights (terrain + water)
    float hL = texture(u_heightMap, v_uv + vec2(-u_texelSize.x, 0.0)).r + 
               texture(u_waterMap, v_uv + vec2(-u_texelSize.x, 0.0)).r;
    float hR = texture(u_heightMap, v_uv + vec2(u_texelSize.x, 0.0)).r + 
               texture(u_waterMap, v_uv + vec2(u_texelSize.x, 0.0)).r;
    float hT = texture(u_heightMap, v_uv + vec2(0.0, u_texelSize.y)).r + 
               texture(u_waterMap, v_uv + vec2(0.0, u_texelSize.y)).r;
    float hB = texture(u_heightMap, v_uv + vec2(0.0, -u_texelSize.y)).r + 
               texture(u_waterMap, v_uv + vec2(0.0, -u_texelSize.y)).r;

    // Height differences (positive means we are higher -> water flows out)
    float dL = totalHeight - hL;
    float dR = totalHeight - hR;
    float dT = totalHeight - hT;
    float dB = totalHeight - hB;

    // Cross-sectional area of pipe (A) and pipe length (l)
    // Flux update: f_new = max(0, f_old + dt * A * g * dh / l)
    // We use A = l * l (square pipe cross-section)
    float area = u_pipeLength * u_pipeLength;
    float fluxFactor = u_dt * area * GRAVITY / u_pipeLength;

    vec4 newFlux;
    newFlux.x = max(0.0, flux.x + fluxFactor * dL);
    newFlux.y = max(0.0, flux.y + fluxFactor * dR);
    newFlux.z = max(0.0, flux.z + fluxFactor * dT);
    newFlux.w = max(0.0, flux.w + fluxFactor * dB);

    // Rescale flux if total outflow exceeds available water volume
    // This prevents creating water from nothing
    float totalOutflow = newFlux.x + newFlux.y + newFlux.z + newFlux.w;
    float availableVolume = waterHeight * u_pipeLength * u_pipeLength;
    
    if (totalOutflow > MIN_WATER && availableVolume > MIN_WATER) {
        float K = min(1.0, availableVolume / (totalOutflow * u_dt));
        newFlux *= K;
    } else if (waterHeight < MIN_WATER) {
        // No water, no outflow
        newFlux = vec4(0.0);
    }

    // Apply damping to prevent oscillations
    newFlux *= 0.995;

    fragColor = newFlux;
}
