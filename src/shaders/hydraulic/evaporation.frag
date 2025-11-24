#version 330

in vec2 v_uv;
out vec4 fragColor; // R=New Water Height, G=VelX, B=VelY (preserved), A=Unused

uniform sampler2D u_waterMap; // R=Water, G=VelX, B=VelY
uniform float u_dt;
uniform float u_evaporationRate; // Ke

void main() {
    vec4 waterData = texture(u_waterMap, v_uv);
    float waterHeight = waterData.r;
    vec2 velocity = waterData.gb;
    
    // Exponential decay evaporation
    // h_new = h_old * (1 - Ke * dt)
    float evapFactor = 1.0 - u_evaporationRate * u_dt;
    evapFactor = clamp(evapFactor, 0.0, 1.0);
    
    float newWaterHeight = waterHeight * evapFactor;
    newWaterHeight = max(0.0, newWaterHeight);
    
    // Reduce velocity as water evaporates (momentum conservation-ish)
    vec2 newVelocity = velocity * evapFactor;
    
    // Output: preserve velocity for next advection step
    fragColor = vec4(newWaterHeight, newVelocity.x, newVelocity.y, 1.0);
}
