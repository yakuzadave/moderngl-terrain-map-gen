#version 330

in vec2 v_uv;
out vec4 fragColor; // R=Advected Sediment

uniform sampler2D u_sedimentMap;
uniform sampler2D u_waterVelocityMap; // R=Water, G=VelX, B=VelY
uniform float u_dt;
uniform vec2 u_texelSize;

void main() {
    vec4 waterInfo = texture(u_waterVelocityMap, v_uv);
    vec2 velocity = waterInfo.gb;
    
    // Semi-Lagrangian Advection
    // Backtrace: find where the sediment came from
    // Velocity is in grid units per time, convert to UV space
    // UV offset = velocity * dt * texelSize
    vec2 offset = velocity * u_dt * u_texelSize;
    
    // Clamp offset to prevent sampling too far (stability)
    float maxOffset = 5.0 * max(u_texelSize.x, u_texelSize.y);
    offset = clamp(offset, vec2(-maxOffset), vec2(maxOffset));
    
    vec2 sourcePos = v_uv - offset;
    
    // Clamp to valid UV range
    sourcePos = clamp(sourcePos, vec2(0.0), vec2(1.0));
    
    // Sample sediment from source position (bilinear interpolation)
    float advectedSediment = texture(u_sedimentMap, sourcePos).r;
    
    // Small diffusion to prevent sharp artifacts
    float sedimentHere = texture(u_sedimentMap, v_uv).r;
    advectedSediment = mix(advectedSediment, sedimentHere, 0.05);
    
    fragColor = vec4(max(0.0, advectedSediment), 0.0, 0.0, 1.0);
}
