#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightMap; // R=Terrain Height
uniform sampler2D u_waterMap;  // R=Water Height
uniform sampler2D u_fluxMap;   // RGBA=Flux(Left, Right, Top, Bottom)

uniform float u_dt;
uniform float u_pipeLength; // Virtual pipe length (usually 1.0)
uniform vec2 u_texelSize;

void main() {
    float terrainHeight = texture(u_heightMap, uv).r;
    float waterHeight = texture(u_waterMap, uv).r;
    float totalHeight = terrainHeight + waterHeight;

    vec4 flux = texture(u_fluxMap, uv);

    // Neighbor heights (total)
    float hL = texture(u_heightMap, uv + vec2(-u_texelSize.x, 0.0)).r + texture(u_waterMap, uv + vec2(-u_texelSize.x, 0.0)).r;
    float hR = texture(u_heightMap, uv + vec2(u_texelSize.x, 0.0)).r + texture(u_waterMap, uv + vec2(u_texelSize.x, 0.0)).r;
    float hT = texture(u_heightMap, uv + vec2(0.0, u_texelSize.y)).r + texture(u_waterMap, uv + vec2(0.0, u_texelSize.y)).r;
    float hB = texture(u_heightMap, uv + vec2(0.0, -u_texelSize.y)).r + texture(u_waterMap, uv + vec2(0.0, -u_texelSize.y)).r;

    // Height differences (positive means we are higher)
    float dL = totalHeight - hL;
    float dR = totalHeight - hR;
    float dT = totalHeight - hT;
    float dB = totalHeight - hB;

    // Flux is proportional to height difference
    // f_new = max(0, f_old + dt * A * g * dh / l)
    // Assuming A*g/l is a constant factor
    float fluxFactor = u_dt * 9.81; // Gravity constant-ish

    vec4 newFlux;
    newFlux.x = max(0.0, flux.x + fluxFactor * dL);
    newFlux.y = max(0.0, flux.y + fluxFactor * dR);
    newFlux.z = max(0.0, flux.z + fluxFactor * dT);
    newFlux.w = max(0.0, flux.w + fluxFactor * dB);

    // Rescale flux if total outflow exceeds water volume
    float totalOutflow = newFlux.x + newFlux.y + newFlux.z + newFlux.w;
    
    if (totalOutflow > 0.0001) {
        float K = min(1.0, (waterHeight * u_pipeLength * u_pipeLength) / (totalOutflow * u_dt));
        newFlux *= K;
    }

    fragColor = newFlux;
}
