import zengl


class Grid:
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

                const int N = 17;

                void main() {
                    vec2 point = vec2(float(gl_VertexID % 2), float(gl_VertexID / 2 % N) / float(N - 1)) * 4.0 - 2.0;
                    if (gl_VertexID > N * 2) {
                        point = point.yx;
                    }
                    gl_Position = mvp * vec4(point, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(0.0, 0.0, 0.0, 1.0);
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
            topology='lines',
            vertex_count=17 * 4,
        )

    def render(self):
        self.pipeline.render()


if __name__ == '__main__':
    import preview

    preview.show([Grid])
