import struct
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]
ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        uniform float time;

        layout (location = 0) out vec4 out_color;

        void main() {
            ivec2 at = ivec2(gl_FragCoord.xy);
            float dots = sin(float(at.x) * 0.1) + cos(float(at.y) * 0.1);
            float wave = sin(sqrt(float(at.x * at.x + at.y * at.y)) * 0.01 + time) * 0.5 + 0.5;
            out_color = vec4(dots * wave, wave, 1.0, 1.0) * 0.8 + 0.2;
        }
    ''',
    uniforms={
        'time': 0.0,
    },
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    now = pygame.time.get_ticks() / 1000.0
    image.clear()
    pipeline.uniforms['time'][:] = struct.pack('f', now)
    pipeline.render()
    image.blit()

    pygame.display.flip()
