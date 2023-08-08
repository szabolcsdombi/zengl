import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    texture_1 = ctx.image((16, 16), 'rg32float', np.full((16, 16, 2), 0.5, 'f4'))
    texture_2 = ctx.image((16, 16), 'rgba8unorm', np.full((16, 16, 4), (0, 0, 255, 255), 'u1'))
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            uniform sampler2D Texture_1;

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID] + texture(Texture_1, vec2(0.5, 0.5)).rg, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            uniform sampler2D Texture_2;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = texture(Texture_2, vec2(0.5, 0.5));
            }
        ''',
        layout=[
            {
                'name': 'Texture_1',
                'binding': 1,
            },
            {
                'name': 'Texture_2',
                'binding': 3,
            },
        ],
        resources=[
            {
                'type': 'sampler',
                'binding': 1,
                'image': texture_1,
            },
            {
                'type': 'sampler',
                'binding': 3,
                'image': texture_2,
            },
        ],
        framebuffer=[image],
        topology='triangles',
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
