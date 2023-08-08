import struct

import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            uniform vec2 offset;
            uniform vec2 scale;
            uniform int step;

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID * step] * scale + offset, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            uniform vec3 color;

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(color, 1.0);
            }
        ''',
        uniforms={
            'offset': (0.0, 0.0),
            'scale': (1.0, 1.0),
            'color': (0.0, 0.0, 0.0),
            'step': 0,
        },
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )

    ctx.new_frame()
    image.clear()
    struct.pack_into('2f', pipeline.uniforms['offset'], 0, 0.5, 0.5)
    struct.pack_into('3f', pipeline.uniforms['color'], 0, 0.0, 0.0, 1.0)
    struct.pack_into('i', pipeline.uniforms['step'], 0, 1)
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
