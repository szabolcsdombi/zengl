import zengl


def grid_pipeline(ctx: zengl.Pipeline, framebuffer):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            #include "defaults"

            const int N = 17;

            void main() {
                vec2 point = vec2(float(gl_VertexID % 2), float(gl_VertexID / 2 % N) / (N - 1)) * 4.0 - 2.0;
                if (gl_VertexID > N * 2) {
                    point = point.yx;
                }
                gl_Position = mvp * vec4(point, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        ''',
        framebuffer=framebuffer,
        topology='lines',
        vertex_count=17 * 4,
    )
