import os

import pygame
import zengl

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)

triangle = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        out vec3 v_color;

        vec2 positions[3] = vec2[](
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
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(pow(v_color, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    ctx.new_frame()
    image.clear()
    triangle.viewport = (0, 0, 640, 360)
    triangle.render()
    triangle.viewport = (0, 360, 640, 360)
    triangle.render()
    triangle.viewport = (640, 0, 640, 360)
    triangle.render()
    triangle.viewport = (640, 360, 640, 360)
    triangle.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
