import struct

import zengl

from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')

uniform_buffer = ctx.buffer(size=32)

scene = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            vec2 size;
            vec2 mouse;
            int iter;
        };

        layout (location = 0) out vec4 out_color;

        void main() {
            vec2 z = vec2(5.0, 3.0) * (gl_FragCoord.xy / size - 0.5);
            vec2 c = mouse / size * 2.0 - 1.0;
            int i;
            for(i = 0; i < iter; i++) {
                vec2 v = vec2(
                    (z.x * z.x - z.y * z.y) + c.x,
                    (z.y * z.x + z.x * z.y) + c.y
                );
                if (dot(v, v) > 4.0) break;
                z = v;
            }
            float cm = fract((i == iter ? 0.0 : float(i)) * 10.0 / float(iter));
            out_color = vec4(
                fract(cm + 0.0 / 3.0),
                fract(cm + 1.0 / 3.0),
                fract(cm + 2.0 / 3.0),
                1.0
            );
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while window.update():
    ctx.new_frame()
    uniform_buffer.write(struct.pack('=2f2fi', *window.size, *window.mouse, 100))
    image.clear()
    scene.render()
    image.blit()
    ctx.end_frame()
