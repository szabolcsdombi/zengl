import os
import struct

import pygame
import zengl

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def color(hex_color: str):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return r, g, b


size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec2 vertices[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        uniform vec2 size;
        uniform vec3 color1;
        uniform vec3 color2;
        uniform float noise;
        uniform float time;

        layout (location = 0) out vec4 out_color;

        float hash12(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        vec3 hash32(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yxz + 33.33);
            return fract((p3.xxy + p3.yzz) * p3.zyx);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / size;
            float c = distance(uv, vec2(0.5)) / sqrt(2.0);
            c += hash12(uv * size + time) * noise;
            vec3 color = mix(color1, color2, c);
            out_color = vec4(color, 1.0);
        }
    ''',
    uniforms={
        'size': size,
        'color1': color('#0C5291'),
        'color2': color('#06294F'),
        'noise': 0.03,
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
            quit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    pipeline.uniforms['time'][:] = struct.pack('f', now)
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()

