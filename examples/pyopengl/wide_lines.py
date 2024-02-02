'''
    Line width is not managed by ZenGL because wide lines are not supported in core OpenGL profile.
    This example intends to demonstrate how to set the line width while rendering with ZenGL.
    To render nice lines please see the bezier_curves.py.
'''

import os

import numpy as np
import pygame
import zengl
from OpenGL import GL

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def create_helix(points, offset):
    t = np.linspace(0.0, 64.0, points)
    return np.array([np.sin(t) + offset, np.cos(t), t * 0.2 - 6.4]).T.astype('f4').tobytes()


def build_pipeline(vertex_buffer):
    return ctx.pipeline(
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
        topology='line_strip',
        vertex_buffers=zengl.bind(vertex_buffer, '3f', 0),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f'),
    )


size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

uniform_buffer = ctx.buffer(size=64)

vertex_buffers = [
    ctx.buffer(create_helix(1000, -8.0)),
    ctx.buffer(create_helix(1000, 0.0)),
    ctx.buffer(create_helix(1000, 8.0)),
]

pipelines = [build_pipeline(vbo) for vbo in vertex_buffers]

camera = zengl.camera((0.0, -20.0, 0.0), (0.0, 0.0, 0.0), aspect=1.777, fov=45.0)
uniform_buffer.write(camera)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    ctx.new_frame()
    image.clear()
    depth.clear()
    GL.glLineWidth(1.0)
    pipelines[0].render()
    GL.glLineWidth(3.0)
    pipelines[1].render()
    GL.glLineWidth(9.0)
    pipelines[2].render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
