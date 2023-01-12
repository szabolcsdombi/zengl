import struct

import numpy as np
import zengl

from window import Window

window = Window()
ctx = zengl.context()
image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

uniform_buffer = ctx.buffer(size=16)

vertex_buffer = ctx.buffer(np.array([
    1.0, 0.0,
    1.0, 0.0, 0.0, 0.5,

    -0.5, 0.86,
    0.0, 1.0, 0.0, 0.5,

    -0.5, -0.86,
    0.0, 0.0, 1.0, 0.5,
], 'f4'))

triangle = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            vec2 scale;
            float rotation;
        };

        layout (location = 0) in vec2 in_vert;
        layout (location = 1) in vec4 in_color;

        out vec4 v_color;

        void main() {
            float r = rotation * (0.5 + gl_InstanceID * 0.05);
            mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
            gl_Position = vec4((rot * in_vert) * scale, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec4 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    blend=[
        {
            'enable': True,
            'src_color': 'src_alpha',
            'dst_color': 'one_minus_src_alpha',
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '2f 4f', 0, 1),
    vertex_count=3,
    instance_count=10,
)

while window.update():
    image.clear()
    uniform_buffer.write(struct.pack('3f4x', 0.5, 0.5 * window.aspect, window.time))
    triangle.render()
    image.blit()
