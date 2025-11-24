#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;
uniform vec2 u_texelSize;

void main() {
    vec4 data = texture(u_heightmap, uv);
    float h = data.x;
    float erosion = data.w;

    // Calculate derivatives for normal
    float h_dx = texture(u_heightmap, uv + vec2(u_texelSize.x, 0.0)).x;
    float h_dy = texture(u_heightmap, uv + vec2(0.0, u_texelSize.y)).x;

    vec3 normal = normalize(cross(
        vec3(u_texelSize.x, 0.0, h_dx - h),
        vec3(0.0, u_texelSize.y, h_dy - h)
    ));

    // Output height, new normal x/z, and erosion mask
    fragColor = vec4(h, normal.x, normal.z, erosion);
}
