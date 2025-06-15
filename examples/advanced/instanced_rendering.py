import sys

import numpy as np
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()

ctx = zengl.context()
image = ctx.image(window_size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

vertex_buffer = ctx.buffer(np.array([
    0.0, 0.08, 1.0, 0.0, 0.0,
    -0.06, -0.08, 0.0, 1.0, 0.0,
    0.06, -0.08, 0.0, 0.0, 1.0,
], 'f4'))

angle = np.linspace(0.0, np.pi * 2.0, 32)
xy = np.array([np.cos(angle) * 0.42, np.sin(angle) * 0.75])

instance_buffer = ctx.buffer(xy.T.astype('f4').tobytes())

triangle = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (location = 0) in vec2 in_vert;
        layout (location = 1) in vec3 in_color;

        layout (location = 2) in vec2 in_pos;

        out vec3 v_color;

        void main() {
            gl_Position = vec4(in_pos + in_vert, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '2f 3f', 0, 1),
        *zengl.bind(instance_buffer, '2f /i', 2),
    ],
    vertex_count=3,
    instance_count=32,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    triangle.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
