#version 330

in vec2 uv;
layout(location = 0) out vec4 outHeight;   // R=Height
layout(location = 1) out vec4 outSediment; // R=Sediment

uniform sampler2D u_heightMap;
uniform sampler2D u_waterVelocityMap; // R=Water, G=VelX, B=VelY
uniform sampler2D u_sedimentMap;      // R=Suspended Sediment

uniform float u_dt;
uniform float u_capacity;   // Kc - Sediment capacity constant
uniform float u_dissolving; // Ks - Soil dissolving rate
uniform float u_deposition; // Kd - Sediment deposition rate
uniform vec2 u_texelSize;

void main() {
    float terrainHeight = texture(u_heightMap, uv).r;
    vec3 waterInfo = texture(u_waterVelocityMap, uv).rgb;
    float waterHeight = waterInfo.r;
    vec2 velocity = waterInfo.gb;
    float sediment = texture(u_sedimentMap, uv).r;

    // Tilt angle (slope) calculation
    // Calculate local gradient
    float hL = texture(u_heightMap, uv + vec2(-u_texelSize.x, 0.0)).r;
    float hR = texture(u_heightMap, uv + vec2(u_texelSize.x, 0.0)).r;
    float hT = texture(u_heightMap, uv + vec2(0.0, u_texelSize.y)).r;
    float hB = texture(u_heightMap, uv + vec2(0.0, -u_texelSize.y)).r;
    
    vec2 gradient = vec2(hR - hL, hT - hB) * 0.5; // Central difference
    float slope = length(gradient); // Approximation of sin(tilt)

    float speed = length(velocity);

    // Sediment Transport Capacity
    // C = Kc * sin(tilt) * |v|
    float transportCapacity = u_capacity * max(0.01, slope) * speed; 

    float newTerrainHeight = terrainHeight;
    float newSediment = sediment;

    if (transportCapacity > sediment) {
        // Erode
        float amountToErode = u_dissolving * (transportCapacity - sediment) * u_dt;
        newTerrainHeight -= amountToErode;
        newSediment += amountToErode;
    } else {
        // Deposit
        float amountToDeposit = u_deposition * (sediment - transportCapacity) * u_dt;
        newTerrainHeight += amountToDeposit;
        newSediment -= amountToDeposit;
    }

    outHeight = vec4(newTerrainHeight, 0.0, 0.0, 1.0);
    outSediment = vec4(newSediment, 0.0, 0.0, 1.0);
}
