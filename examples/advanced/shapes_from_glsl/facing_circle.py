import zengl


class Circle:
    def __init__(self, framebuffer, uniform_buffer):
        self.ctx = zengl.context()
        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec3 eye;
                    vec3 light;
                };

                vec2 vertices[16] = vec2[](
                    vec2(1.0000, 0.0000),
                    vec2(0.9239, 0.3827),
                    vec2(0.7071, 0.7071),
                    vec2(0.3827, 0.9239),
                    vec2(0.0000, 1.0000),
                    vec2(-0.3827, 0.9239),
                    vec2(-0.7071, 0.7071),
                    vec2(-0.9239, 0.3827),
                    vec2(-1.0000, 0.0000),
                    vec2(-0.9239, -0.3827),
                    vec2(-0.7071, -0.7071),
                    vec2(-0.3827, -0.9239),
                    vec2(0.0000, -1.0000),
                    vec2(0.3827, -0.9239),
                    vec2(0.7071, -0.7071),
                    vec2(0.9239, -0.3827)
                );

                vec3 position = vec3(0.0, 0.0, 0.0);
                vec3 up = vec3(0.0, 0.0, 1.0);

                out vec3 v_vertex;
                out vec3 v_normal;
                out vec2 v_texcoord;

                void main() {
                    v_normal = normalize(eye.xyz - position);
                    vec3 tangent = normalize(cross(up, v_normal));
                    vec3 bitangent = cross(v_normal, tangent);
                    v_vertex = position + tangent * vertices[gl_VertexID].x + bitangent * vertices[gl_VertexID].y;
                    v_texcoord = vertices[gl_VertexID] + 0.5;
                    gl_Position = mvp * vec4(v_vertex, 1.0);
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec3 eye;
                    vec3 light;
                };

                in vec3 v_normal;

                layout (location = 0) out vec4 out_color;

                void main() {
                    float lum = dot(normalize(light.xyz), normalize(v_normal)) * 0.7 + 0.3;
                    out_color = vec4(lum, lum, lum, 1.0);
                }
            ''',
            layout=[
                {
                    'name': 'Common',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': uniform_buffer,
                },
            ],
            framebuffer=framebuffer,
            topology='triangle_fan',
            cull_face='back',
            vertex_count=16,
        )

    def render(self):
        self.pipeline.render()


if __name__ == '__main__':
    import preview
    from grid import Grid

    preview.show([Grid, Circle])
