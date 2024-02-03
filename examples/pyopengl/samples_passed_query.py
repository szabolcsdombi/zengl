import math
import os
import struct
import sys

import pygame
import zengl
from OpenGL import GL

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)

uniform_buffer = ctx.buffer(size=16)

pipeline = ctx.pipeline(
    vertex_shader="""
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            vec2 scale;
        };

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.08),
            vec2(-0.06, -0.08),
            vec2(0.06, -0.08)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID] * scale, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 300 es
        precision highp float;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(1.0, 1.0, 1.0, 1.0);
        }
    """,
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
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

query = GL.glGenQueries(1)[0]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    image.clear()
    uniform_buffer.write(struct.pack('ff8x', math.sin(now), math.cos(now)))
    GL.glBeginQuery(GL.GL_SAMPLES_PASSED, query)
    pipeline.render()
    GL.glEndQuery(GL.GL_SAMPLES_PASSED)
    query_result = GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT)
    pygame.display.set_caption(f'Samples Passed: {query_result}')
    image.blit()
    ctx.end_frame()

    pygame.display.flip()

