import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID] + 0.5, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(0.0, 0.0, 1.0, 1.0);
            }
        ''',
        framebuffer=None,
        viewport=(0, 0, 64, 64),
        topology='triangles',
        vertex_count=3,
    )

    assert ctx.screen == 0

    framebuffer = zengl.inspect(image.face(0))['framebuffer']
    assert framebuffer != 0

    ctx.screen = framebuffer
    assert ctx.screen == framebuffer

    ctx.new_frame()
    image.clear()
    pipeline.render()
    ctx.end_frame()

    ctx.screen = 0
    assert ctx.screen == 0

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
