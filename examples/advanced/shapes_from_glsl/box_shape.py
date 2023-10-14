import zengl


class Box:
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

                vec3 vertices[36] = vec3[](
                    vec3(-0.5, -0.5, -0.5),
                    vec3(-0.5, 0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, -0.5, 0.5),
                    vec3(0.5, -0.5, -0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(-0.5, 0.5, -0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, 0.5),
                    vec3(0.5, 0.5, -0.5),
                    vec3(-0.5, 0.5, -0.5),
                    vec3(-0.5, -0.5, -0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, -0.5, 0.5),
                    vec3(-0.5, 0.5, 0.5),
                    vec3(-0.5, 0.5, -0.5)
                );

                vec3 normals[36] = vec3[](
                    vec3(0.0, 0.0, -1.0),
                    vec3(0.0, 0.0, -1.0),
                    vec3(0.0, 0.0, -1.0),
                    vec3(0.0, 0.0, -1.0),
                    vec3(0.0, 0.0, -1.0),
                    vec3(0.0, 0.0, -1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, -1.0, 0.0),
                    vec3(0.0, -1.0, 0.0),
                    vec3(0.0, -1.0, 0.0),
                    vec3(0.0, -1.0, 0.0),
                    vec3(0.0, -1.0, 0.0),
                    vec3(0.0, -1.0, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(-1.0, 0.0, 0.0),
                    vec3(-1.0, 0.0, 0.0),
                    vec3(-1.0, 0.0, 0.0),
                    vec3(-1.0, 0.0, 0.0),
                    vec3(-1.0, 0.0, 0.0),
                    vec3(-1.0, 0.0, 0.0)
                );

                vec2 texcoords[36] = vec2[](
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 0.0),
                    vec2(1.0, 0.0),
                    vec2(0.0, 0.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 0.0),
                    vec2(0.0, 0.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 0.0),
                    vec2(0.0, 0.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 0.0),
                    vec2(0.0, 0.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 0.0),
                    vec2(0.0, 0.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, 1.0),
                    vec2(0.0, 1.0),
                    vec2(0.0, 0.0)
                );

                out vec3 v_vertex;
                out vec3 v_normal;
                out vec2 v_texcoord;

                void main() {
                    v_vertex = vertices[gl_VertexID];
                    v_normal = normals[gl_VertexID];
                    v_texcoord = texcoords[gl_VertexID];
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
            topology='triangles',
            cull_face='back',
            vertex_count=36,
        )

    def render(self):
        self.pipeline.render()


if __name__ == '__main__':
    import preview
    from grid import Grid

    preview.show([Grid, Box])
