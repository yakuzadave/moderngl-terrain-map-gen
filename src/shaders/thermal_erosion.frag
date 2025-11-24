#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;
uniform vec2 u_texelSize;
uniform float u_talusThreshold;
uniform float u_thermalStrength;

void main() {
    vec4 data = texture(u_heightmap, uv);
    float h = data.x;
    float erosion = data.w;

    float h_new = h;
    
    // Simple 4-neighbor thermal erosion
    // We gain material from higher neighbors and lose to lower ones
    
    vec2 offsets[4] = vec2[](
        vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1)
    );

    for (int i = 0; i < 4; i++) {
        float hn = texture(u_heightmap, uv + offsets[i] * u_texelSize).x;
        float d = h - hn;
        
        // If we are higher (d > 0), we lose to neighbor
        if (d > u_talusThreshold) {
            h_new -= (d - u_talusThreshold) * u_thermalStrength;
        }
        
        // If neighbor is higher (d < 0), we gain from neighbor
        if (-d > u_talusThreshold) {
            h_new += (-d - u_talusThreshold) * u_thermalStrength;
        }
    }

    // Output new height. Normals (y, z) are invalidated (set to 0).
    fragColor = vec4(h_new, 0.0, 0.0, erosion);
}
