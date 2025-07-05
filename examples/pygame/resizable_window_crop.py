import os
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()
pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=True)

ctx = zengl.context()

monitor_size = pygame.display.get_desktop_sizes()[0]
image = ctx.image(monitor_size, 'rgba8unorm', samples=4)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        out vec3 v_color;

        vec2 vertices[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

clock = pygame.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    width, height = pygame.display.get_window_size()
    viewport = (0, 0, width, height)

    ctx.new_frame()
    image.clear()
    pipeline.viewport = viewport
    pipeline.render()
    image.blit(crop=viewport)
    ctx.end_frame()

    pygame.display.flip()
    clock.tick(60)
