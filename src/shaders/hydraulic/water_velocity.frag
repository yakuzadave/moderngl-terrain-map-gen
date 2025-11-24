#version 330

in vec2 uv;
out vec4 fragColor; // R=New Water Height, G=Velocity X, B=Velocity Y, A=Unused

uniform sampler2D u_waterMap;
uniform sampler2D u_fluxMap;
uniform sampler2D u_heightMap; // Needed? Maybe not for this step, but good to have context
uniform vec2 u_texelSize;
uniform float u_dt;
uniform float u_pipeLength;

void main() {
    float waterHeight = texture(u_waterMap, uv).r;
    vec4 flux = texture(u_fluxMap, uv); // Outflow from this cell: L, R, T, B

    // Inflow from neighbors
    // Neighbor's Right flux flows into our Left side
    // Neighbor's Left flux flows into our Right side
    // Neighbor's Top flux flows into our Bottom side
    // Neighbor's Bottom flux flows into our Top side
    
    float inL = texture(u_fluxMap, uv + vec2(-u_texelSize.x, 0.0)).y; // Neighbor Left's Right flux
    float inR = texture(u_fluxMap, uv + vec2(u_texelSize.x, 0.0)).x;  // Neighbor Right's Left flux
    float inT = texture(u_fluxMap, uv + vec2(0.0, u_texelSize.y)).w;  // Neighbor Top's Bottom flux
    float inB = texture(u_fluxMap, uv + vec2(0.0, -u_texelSize.y)).z; // Neighbor Bottom's Top flux

    float totalInflow = inL + inR + inT + inB;
    float totalOutflow = flux.x + flux.y + flux.z + flux.w;

    float volumeChange = u_dt * (totalInflow - totalOutflow);
    float newWaterHeight = max(0.0, waterHeight + volumeChange);

    // Calculate Velocity Field
    // Average flux through the cell
    // Flux is amount of water passing through boundary per time
    // Velocity = Flux / (Area * Depth)
    // Area is 1x1 (virtual). Depth is average water height.
    
    float flowX = (inL - flux.x + flux.y - inR) * 0.5;
    float flowY = (inB - flux.w + flux.z - inT) * 0.5;
    
    // Use average water height to avoid singularities
    float avgWater = (waterHeight + newWaterHeight) * 0.5;
    
    vec2 velocity = vec2(0.0);
    if (avgWater > 0.0001) {
        velocity = vec2(flowX, flowY) / avgWater; // simplified
    }

    fragColor = vec4(newWaterHeight, velocity.x, velocity.y, 1.0);
}
