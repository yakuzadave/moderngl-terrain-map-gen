#version 330

in vec2 v_uv;
out vec4 fragColor; // R=New Water Height, G=Velocity X, B=Velocity Y, A=Unused

uniform sampler2D u_waterMap;
uniform sampler2D u_fluxMap;
uniform vec2 u_texelSize;
uniform float u_dt;
uniform float u_pipeLength;

const float MIN_WATER = 0.0001;

void main() {
    vec4 waterData = texture(u_waterMap, v_uv);
    float waterHeight = waterData.r;
    vec4 flux = texture(u_fluxMap, v_uv); // Outflow from this cell: L, R, T, B

    // Inflow from neighbors
    // Neighbor's Right flux flows into our Left side
    // Neighbor's Left flux flows into our Right side
    // Neighbor's Top flux flows into our Bottom side
    // Neighbor's Bottom flux flows into our Top side
    
    float inL = texture(u_fluxMap, v_uv + vec2(-u_texelSize.x, 0.0)).y; // Neighbor Left's Right flux
    float inR = texture(u_fluxMap, v_uv + vec2(u_texelSize.x, 0.0)).x;  // Neighbor Right's Left flux
    float inT = texture(u_fluxMap, v_uv + vec2(0.0, u_texelSize.y)).w;  // Neighbor Top's Bottom flux
    float inB = texture(u_fluxMap, v_uv + vec2(0.0, -u_texelSize.y)).z; // Neighbor Bottom's Top flux

    float totalInflow = inL + inR + inT + inB;
    float totalOutflow = flux.x + flux.y + flux.z + flux.w;

    // Volume change = dt * (inflow - outflow)
    // Cell area is pipeLength^2
    float cellArea = u_pipeLength * u_pipeLength;
    float volumeChange = u_dt * (totalInflow - totalOutflow);
    float newWaterHeight = max(0.0, waterHeight + volumeChange / cellArea);

    // Calculate Velocity Field using average flux
    // Velocity in X direction: (inflow from left - outflow left + outflow right - inflow from right) / 2
    // This gives us the net flow direction
    float flowX = (inL - flux.x) + (flux.y - inR);
    float flowY = (inB - flux.w) + (flux.z - inT);
    
    // Average water height for velocity calculation (avoid divide by zero)
    float avgWater = max(MIN_WATER, (waterHeight + newWaterHeight) * 0.5);
    
    // Velocity = flux / (area * depth)
    // Scale velocity to be in grid units per time step
    vec2 velocity = vec2(flowX, flowY) / (avgWater * u_pipeLength * 2.0);
    
    // Clamp velocity to prevent instabilities
    float maxVel = 10.0;
    velocity = clamp(velocity, vec2(-maxVel), vec2(maxVel));

    fragColor = vec4(newWaterHeight, velocity.x, velocity.y, 1.0);
}
