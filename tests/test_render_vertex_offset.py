import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    vertex_buffer = ctx.buffer(
        np.array(
            [
                [-100.0, -100.0, 0.0, 1.0, 0.0],
                [-100.0, 200.0, 0.0, 1.0, 0.0],
                [200.0, -100.0, 0.0, 1.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 1.0],
                [-0.05, 0.086, 0.0, 0.0, 1.0],
                [-0.05, -0.086, 0.0, 0.0, 1.0],
                [-100.0, -100.0, 0.0, 1.0, 0.0],
                [-100.0, 200.0, 0.0, 1.0, 0.0],
                [200.0, -100.0, 0.0, 1.0, 0.0],
            ],
            'f4',
        )
    )
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            layout (location = 0) in vec2 in_vertex;
            layout (location = 1) in vec3 in_color;

            out vec3 v_color;

            void main() {
                v_color = in_color;
                gl_Position = vec4(in_vertex + 0.5, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            in vec3 v_color;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(v_color, 1.0);
            }
        ''',
        framebuffer=[image],
        topology='triangles',
        vertex_buffers=zengl.bind(vertex_buffer, '2f 3f', 0, 1),
        first_vertex=3,
        vertex_count=3,
    )

    ctx.new_frame()
    image.clear()
    pipeline.render()
    ctx.end_frame()
    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 255, 255],
        ],
    )
