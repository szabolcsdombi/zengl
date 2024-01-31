import os

import pygame
import zengl

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', texture=False)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec2 vertices[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        out vec2 vertex;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            vertex = vertices[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec2 vertex;

        layout (location = 0) out vec4 out_color;

        float mandelbrot(vec2 uv) {
            vec2 z = vec2(0.0, 0.0);
            vec2 c = uv;
            int i;
            for (i = 0; i < 100; ++i) {
                vec2 v = vec2(
                    (z.x * z.x - z.y * z.y) + c.x,
                    (z.y * z.x + z.x * z.y) + c.y
                );
                if (dot(v, v) > 4.0) break;
                z = v;
            }
            if (i == 100) {
                i = 0;
            }
            return float(i) / 100.0;
        }

        void main() {
            float cm = mandelbrot(vertex * 1.75 - vec2(0.5, 0.0));
            vec3 color = vec3(0.2 + cm * 0.6, 0.4 + cm * 0.5, 1.0);
            out_color = vec4(color * cm, 1.0);
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
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
