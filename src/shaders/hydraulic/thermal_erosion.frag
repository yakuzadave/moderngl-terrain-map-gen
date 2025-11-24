#version 330

in vec2 v_uv;  // UV from vertex shader
out vec4 fragColor;  // vec4 for FBO compatibility

uniform sampler2D u_heightMap;
uniform vec2 u_texelSize;
uniform float u_dt;
uniform float u_talusAngle;
uniform float u_thermalRate;

void main() {
    float h = texture(u_heightMap, v_uv).r;
    float dH = 0.0;
    
    vec2 offsets[4] = vec2[](
        vec2(-u_texelSize.x, 0.0),
        vec2(u_texelSize.x, 0.0),
        vec2(0.0, -u_texelSize.y),
        vec2(0.0, u_texelSize.y)
    );
    
    for(int i = 0; i < 4; i++) {
        float hn = texture(u_heightMap, v_uv + offsets[i]).r;
        float diff = h - hn;
        if(diff > u_talusAngle) {
            // Material flows downhill from this cell
            dH -= (diff - u_talusAngle) * u_thermalRate * u_dt;
        }
        if(diff < -u_talusAngle) {
            // Material flows to this cell from neighbor
            dH += (-diff - u_talusAngle) * u_thermalRate * u_dt;
        }
    }
    
    // Output new height (GBA unused but available for extensions)
    fragColor = vec4(h + dH, 0.0, 0.0, 1.0);
}
