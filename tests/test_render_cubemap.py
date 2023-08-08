import struct

import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    temp = np.concatenate(
        [
            np.full((16, 16, 2), (-0.5, -0.5)),
            np.full((16, 16, 2), (0.5, -0.5)),
            np.full((16, 16, 2), (-0.5, 0.5)),
            np.full((16, 16, 2), (0.5, 0.5)),
            np.full((16, 16, 2), (0.0, 0.0)),
            np.full((16, 16, 2), (0.0, 0.0)),
        ],
        dtype='f4',
    )
    texture = ctx.image((16, 16), 'rg32float', temp, cubemap=True)
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            uniform samplerCube Texture;

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            uniform vec3 ray;

            void main() {
                gl_Position = vec4(positions[gl_VertexID] + texture(Texture, ray).rg, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(0.0, 0.0, 1.0, 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Texture',
                'binding': 5,
            },
        ],
        resources=[
            {
                'type': 'sampler',
                'binding': 5,
                'image': texture,
            },
        ],
        uniforms={
            'ray': (0.0, 0.0, 0.0),
        },
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )

    ctx.new_frame()
    image.clear()
    struct.pack_into('3f', pipeline.uniforms['ray'], 0, 0.9, 0.1, 0.2)
    pipeline.render()
    struct.pack_into('3f', pipeline.uniforms['ray'], 0, -0.9, 0.2, 0.1)
    pipeline.render()
    struct.pack_into('3f', pipeline.uniforms['ray'], 0, 0.1, 0.9, -0.2)
    pipeline.render()
    struct.pack_into('3f', pipeline.uniforms['ray'], 0, -0.2, -0.9, 0.1)
    pipeline.render()
    ctx.end_frame()
    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
    )
