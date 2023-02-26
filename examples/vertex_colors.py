import gzip
import struct

import numpy as np
import zengl

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = gzip.decompress(open(assets.get('colormonkey.mesh.gz'), 'rb').read())
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=80)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
            vec4 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec3 in_color;

        out vec3 v_normal;
        out vec3 v_color;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_normal = in_normal;
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
            vec4 light;
        };

        in vec3 v_normal;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = dot(normalize(light.xyz), normalize(v_normal)) * 0.7 + 0.3;
            if (lum < 0.6) {
                lum = 0.6;
            } else if (lum < 0.9) {
                lum = 0.9;
            } else {
                lum = 1.0;
            }
            out_color = vec4(v_color * lum, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 3f'),
)

while window.update():
    ctx.new_frame()
    eye = (np.cos(window.time * 0.5) * 5.0, np.sin(window.time * 0.5) * 5.0, 1.0)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera + struct.pack('fff4x', *eye))

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit(srgb=True)
    ctx.end_frame()
