#version 330

in vec2 uv;
out vec4 fragColor; // R=Advected Sediment

uniform sampler2D u_sedimentMap;
uniform sampler2D u_waterVelocityMap; // G=VelX, B=VelY
uniform float u_dt;
uniform vec2 u_texelSize;

void main() {
    vec2 velocity = texture(u_waterVelocityMap, uv).gb;
    
    // Semi-Lagrangian Advection
    // Backtrace position: pos - velocity * dt
    // Velocity is in grid units per second? Or UV units?
    // If velocity was calculated as flux/water, flux is volume/time.
    // If cell size is 1x1, then velocity is cells/time.
    // So we need to scale by texelSize to get UV offset.
    
    vec2 offset = velocity * u_dt * u_texelSize;
    vec2 oldPos = uv - offset;
    
    // Bilinear interpolation is handled by texture()
    float advectedSediment = texture(u_sedimentMap, oldPos).r;
    
    fragColor = vec4(advectedSediment, 0.0, 0.0, 1.0);
}
