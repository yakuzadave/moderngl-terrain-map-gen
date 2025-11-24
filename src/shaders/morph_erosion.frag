#version 330
uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform float u_radius;
uniform float u_strength;
in vec2 uv;
out vec4 f_color;

void main() {
    float center_h = texture(u_texture, uv).r;
    float min_h = center_h;
    float max_h = center_h;

    int r = int(u_radius);
    if (r > 0) {
        for (int y = -r; y <= r; y++) {
            for (int x = -r; x <= r; x++) {
                if (x*x + y*y <= r*r) {
                    vec2 offset = vec2(float(x), float(y)) / u_resolution;
                    float h = texture(u_texture, uv + offset).r;
                    min_h = min(min_h, h);
                    max_h = max(max_h, h);
                }
            }
        }
    }

    float gradient = max_h - min_h;
    float eroded = center_h - u_strength * gradient;
    eroded = clamp(eroded, 0.0, 1.0);
    f_color = vec4(vec3(eroded), 1.0);
}
