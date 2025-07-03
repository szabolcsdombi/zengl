import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()

ctx = zengl.context()

image_size = (640, 720)
image1 = ctx.image(image_size, 'rgba8unorm', samples=4)
image2 = ctx.image(image_size, 'rgba8unorm', samples=4)
image1.clear_value = (1.0, 1.0, 1.0, 1.0)
image2.clear_value = (1.0, 1.0, 1.0, 1.0)

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

        layout (location = 0) out vec4 out_color1;
        layout (location = 1) out vec4 out_color2;

        void main() {
            out_color1 = vec4(v_color, 1.0);
            out_color2 = vec4(0.5, 0.5, 0.5, 1.0);
        }
    ''',
    framebuffer=[image1, image2],
    topology='triangles',
    vertex_count=3,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    ctx.new_frame()
    image1.clear()
    image2.clear()
    triangle.render()
    image1.blit(None, (0, 0))
    image2.blit(None, (640, 0))
    ctx.end_frame()

    pygame.display.flip()
