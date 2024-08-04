# python -m pygbag --PYBUILD 3.12 --ume_block 0 --template noctx.tmpl .

import asyncio
import sys

import pygame
import zengl

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', texture=False)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec2 vertices[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.866, -0.7),
            vec2(0.866, -0.7)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        out vec3 v_color;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

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

async def main():
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
        await asyncio.sleep(0)

asyncio.run(main())
