#version 330

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
uniform sampler2D u_texture_0;

in vec2 uv;
out vec4 fragColor;

void main() {
    // Tile the texture 2x2
    vec2 st = uv * 2.0;
    
    // Optional: Add slow movement to check seams in motion
    st += vec2(u_time * 0.1, u_time * 0.05);
    
    // Wrap UVs
    st = fract(st);
    
    vec3 color = texture(u_texture_0, st).rgb;
    
    fragColor = vec4(color, 1.0);
}
