import struct

import numpy as np
import zengl


def test(ctx: zengl.Context):
    image = ctx.image((64, 64), 'rgba8unorm')
    uniform_buffer = ctx.buffer(size=64)
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core

            layout (std140) uniform Common {
                vec4 offset;
                vec4 scale;
                vec4 color;
                int step;
            };

            vec2 positions[3] = vec2[](
                vec2(0.1, 0.0),
                vec2(-0.05, 0.086),
                vec2(-0.05, -0.086)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID * step] * scale.xy + offset.xy, 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            layout (std140) uniform Common {
                vec4 offset;
                vec4 scale;
                vec4 color;
                int step;
            };

            layout (location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(color.rgb, 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Common',
                'binding': 2,
            },
        ],
        resources=[
            {
                'type': 'uniform_buffer',
                'binding': 2,
                'buffer': uniform_buffer,
            },
        ],
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )

    uniform_buffer_data = struct.pack(
        '4f4f4f4i',
        *(0.5, 0.5, -999.0, -999.0),
        *(1.0, 1.0, -999.0, -999.0),
        *(0.0, 0.0, 1.0, -999.0),
        *(1, -999, -999, -999),
    )

    ctx.new_frame()
    image.clear()
    uniform_buffer.write(uniform_buffer_data)
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
