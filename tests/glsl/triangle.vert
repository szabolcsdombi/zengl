#version 450 core

vec2 positions[3] = vec2[](
    vec2(0.0, 0.7),
    vec2(-0.85, -0.8),
    vec2(0.85, -0.8)
);

void main() {
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}
