#version 330

in vec2 in_position;  // Vertex position in NDC (-1 to 1)
in vec2 in_uv;        // Texture coordinates (0 to 1)

out vec2 v_uv;        // Pass UV to fragment shader

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_uv;
}
