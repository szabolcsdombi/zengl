import numpy as np
import zengl


def test_render_triangle(ctx: zengl.Context):
    img = ctx.image((256, 256), 'rgba8unorm')
    triangle = ctx.pipeline(
        vertex_shader='''
            #version 450 core
            vec2 positions[3] = vec2[](
                vec2(0.0, 0.7),
                vec2(-0.85, -0.8),
                vec2(0.85, -0.8)
            );
            void main() {
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 450 core
            layout (location = 0) out vec4 out_color;
            void main() {
                out_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ''',
        framebuffer=[img],
        vertex_count=3,
    )
    img.clear()
    triangle.render()
    pixels = np.frombuffer(img.read(), 'u1').reshape(256, 256, 4)
    x = np.repeat(np.arange(4) * 50 + 50, 4)
    y = np.tile(np.arange(4) * 50 + 50, 4)
    r = [255, 0, 0, 255]
    z = [0, 0, 0, 0]
    np.testing.assert_array_equal(pixels[x, y], [
        r, r, r, r,
        z, r, r, z,
        z, r, r, z,
        z, z, z, z,
    ])
