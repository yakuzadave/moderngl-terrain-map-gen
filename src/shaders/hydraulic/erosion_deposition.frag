#version 330

in vec2 v_uv;
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

const float MIN_SLOPE = 0.001;
const float MIN_SPEED = 0.0001;
const float MAX_EROSION_PER_STEP = 0.01; // Limit erosion per step for stability

void main() {
    float terrainHeight = texture(u_heightMap, v_uv).r;
    vec4 waterInfo = texture(u_waterVelocityMap, v_uv);
    float waterHeight = waterInfo.r;
    vec2 velocity = waterInfo.gb;
    float sediment = texture(u_sedimentMap, v_uv).r;

    // Calculate local terrain gradient using central differences
    float hL = texture(u_heightMap, v_uv + vec2(-u_texelSize.x, 0.0)).r;
    float hR = texture(u_heightMap, v_uv + vec2(u_texelSize.x, 0.0)).r;
    float hT = texture(u_heightMap, v_uv + vec2(0.0, u_texelSize.y)).r;
    float hB = texture(u_heightMap, v_uv + vec2(0.0, -u_texelSize.y)).r;
    
    // Gradient in each direction
    vec2 gradient = vec2(hR - hL, hT - hB) * 0.5 / u_texelSize.x;
    
    // Slope magnitude (approximation of sin(tilt angle))
    float slope = length(gradient);
    slope = max(MIN_SLOPE, slope);
    
    // Water velocity magnitude
    float speed = length(velocity);
    
    // Sediment Transport Capacity
    // C = Kc * sin(tilt) * |velocity|
    // Higher slope and faster water = more sediment can be carried
    float transportCapacity = u_capacity * slope * max(MIN_SPEED, speed);

    float newTerrainHeight = terrainHeight;
    float newSediment = sediment;

    // Only erode/deposit where there is water
    if (waterHeight > 0.0001) {
        if (transportCapacity > sediment) {
            // Water can carry more sediment -> Erode terrain
            // Amount to erode is proportional to the difference
            float amountToErode = u_dissolving * (transportCapacity - sediment) * u_dt;
            
            // Limit erosion for stability
            amountToErode = min(amountToErode, MAX_EROSION_PER_STEP);
            amountToErode = min(amountToErode, terrainHeight * 0.1); // Don't erode more than 10% of height
            
            newTerrainHeight -= amountToErode;
            newSediment += amountToErode;
        } else {
            // Water is over-saturated -> Deposit sediment
            float amountToDeposit = u_deposition * (sediment - transportCapacity) * u_dt;
            
            // Can't deposit more than we have
            amountToDeposit = min(amountToDeposit, sediment);
            amountToDeposit = min(amountToDeposit, MAX_EROSION_PER_STEP);
            
            newTerrainHeight += amountToDeposit;
            newSediment -= amountToDeposit;
        }
    } else {
        // No water - deposit all sediment
        newTerrainHeight += sediment * u_deposition * u_dt;
        newSediment = max(0.0, sediment - sediment * u_deposition * u_dt);
    }

    // Clamp values to valid ranges
    newTerrainHeight = max(0.0, newTerrainHeight);
    newSediment = max(0.0, newSediment);

    outHeight = vec4(newTerrainHeight, 0.0, 0.0, 1.0);
    outSediment = vec4(newSediment, 0.0, 0.0, 1.0);
}
