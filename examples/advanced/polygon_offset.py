import sys

import assets
import numpy as np
import pygame
import zengl
import zengl_extras
from objloader import Obj

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')

vertex_buffer = ctx.buffer(model)
vertex_count = vertex_buffer.size // 24

index_buffer = ctx.buffer(np.array([
    np.arange(0, vertex_count, 3),
    np.arange(1, vertex_count, 3),
    np.arange(2, vertex_count, 3),
    np.full(vertex_count // 3, -1),
], dtype='i4').T.tobytes())

uniform_buffer = ctx.buffer(size=80)

monkey = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
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
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_count,
)

monkey_wire = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        layout (location = 0) out vec4 out_color;

        void main() {
            gl_FragDepth = gl_FragCoord.z - 1e-4;
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
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
    topology='line_loop',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, -1),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4 - 124 * 4,
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window_aspect, fov=45.0)
uniform_buffer.write(camera)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    depth.clear()
    monkey.render()
    monkey_wire.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
