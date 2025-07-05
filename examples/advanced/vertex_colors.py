import gzip
import struct
import sys

import assets
import numpy as np
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.005, 0.005, 0.005, 1.0)

model = gzip.decompress(open(assets.get('colormonkey.mesh.gz'), 'rb').read())
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=80)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
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
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
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
            out_color = vec4(pow(v_color * lum, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
    ],
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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now  = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    eye = (np.cos(now * 0.5) * 5.0, np.sin(now * 0.5) * 5.0, 1.0)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
    uniform_buffer.write(camera + struct.pack('fff4x', *eye))

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
