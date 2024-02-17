import os
import sys

import pygame
import zengl

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)

texture = ctx.image((8, 8), 'rgba8unorm', os.urandom(8 * 8 * 4))

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        out vec2 v_uv;

        vec2 vertices[4] = vec2[](
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0)
        );

        void main() {
            vec2 vertex = vertices[gl_VertexID] - 0.5;
            gl_Position = vec4(vertex, 0.0, 1.0);
            v_uv = vertices[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330 core

        uniform sampler2D Texture;

        in vec2 v_uv;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = texture(Texture, v_uv);
        }
    ''',
    layout=[
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image],
    topology='triangle_strip',
    vertex_count=4,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()

