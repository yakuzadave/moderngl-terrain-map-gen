#version 300 es
precision mediump float;

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

// This shader is designed to be used with the vscode-glsl-canvas extension
// to visually verify seamless tiling of generated textures.
//
// Instructions:
// 1. Generate a seamless heightmap or texture using gpu_terrain.py:
//    python gpu_terrain.py --seamless --resolution 512 --heightmap-out seamless_test.png
// 2. Update the texture path in .vscode/settings.json (or workspace settings) for glsl-canvas:
//    "glsl-canvas.textures": {
//        "0": "./seamless_test.png"
//    }
// 3. Open this file and run "Show glslCanvas" command.

uniform sampler2D u_texture_0;

void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    
    // Tile the texture 2x2
    vec2 uv = st * 2.0;
    
    // Optional: Add slow movement to check seams in motion
    uv += vec2(u_time * 0.1, u_time * 0.05);
    
    // Wrap UVs
    uv = fract(uv);
    
    vec3 color = texture(u_texture_0, uv).rgb;
    
    // Draw grid lines to show tile boundaries (optional, comment out to check seams)
    // if (st.x > 0.495 && st.x < 0.505) color = vec3(1.0, 0.0, 0.0);
    // if (st.y > 0.495 && st.y < 0.505) color = vec3(1.0, 0.0, 0.0);

    gl_FragColor = vec4(color, 1.0);
}
