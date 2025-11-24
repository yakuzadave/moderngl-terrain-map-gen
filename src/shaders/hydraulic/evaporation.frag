#version 330

in vec2 uv;
out vec4 fragColor; // R=New Water Height

uniform sampler2D u_waterMap;
uniform float u_dt;
uniform float u_evaporationRate; // Ke

void main() {
    float waterHeight = texture(u_waterMap, uv).r;
    
    // Simple exponential decay or linear subtraction
    // h_new = h_old * (1 - Ke * dt)
    
    float newWaterHeight = waterHeight * (1.0 - u_evaporationRate * u_dt);
    newWaterHeight = max(0.0, newWaterHeight);
    
    fragColor = vec4(newWaterHeight, 0.0, 0.0, 1.0);
}
