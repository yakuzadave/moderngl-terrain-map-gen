#version 330
uniform float u_seed;
uniform float u_scale;
uniform int u_octaves;
uniform float u_persistence;
uniform float u_lacunarity;
in vec2 uv;
out vec4 f_color;

float hash(vec2 p) {
    p = 50.0 * fract(p * 0.3183099 + vec2(0.71, 0.113));
    return -1.0 + 2.0 * fract(p.x * p.y * (p.x + p.y));
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

void main() {
    vec2 p = uv * u_scale;
    vec2 shift = vec2(u_seed * 12.34, u_seed * 56.78);

    float value = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float max_amp = 0.0;

    for (int i = 0; i < u_octaves; i++) {
        value += noise(p * freq + shift) * amp;
        max_amp += amp;
        amp *= u_persistence;
        freq *= u_lacunarity;
    }

    value = (value / max_amp) * 0.5 + 0.5;
    f_color = vec4(vec3(value), 1.0);
}
