import os

import pygame
import zengl

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;
        vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        layout (location = 0) out vec4 out_color;

        uniform vec2 resolution;

        struct BrickTexture {
            float scale;
            vec2 brick_size;
            float offset;
            int offset_frequency;
            float squash;
            int squash_frequency;
            float mortar_size;
            float mortar_smooth;
            float bias;
            vec3 color1;
            vec3 color2;
            vec3 color3;
        };

        float hash12(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        vec2 brick_coordinates(vec2 uv, BrickTexture mt) {
            vec2 b = uv * mt.scale / mt.brick_size;
            if (int(floor(b.y) + 1.0) % mt.offset_frequency == 0) {
                b.x += mt.offset;
            }
            if (int(floor(b.y) + 1.0) % mt.squash_frequency == 0) {
                b.x *= mt.squash;
            }
            return b;
        }

        float brick_factor(vec2 uv, BrickTexture mt) {
            vec2 b = brick_coordinates(uv, mt);
            vec2 a = (0.5 - abs(fract(b) - 0.5)) * mt.brick_size;
            return 1.0 - smoothstep(mt.mortar_size * (1.0 - mt.mortar_smooth), mt.mortar_size, min(a.x, a.y));
        }

        vec3 brick_color(vec2 uv, BrickTexture mt) {
            vec2 b = brick_coordinates(uv, mt);
            float f1 = hash12(floor(b));
            float f2 = brick_factor(uv, mt);
            vec3 color = mix(mt.color1, mt.color2, f1);
            return mix(color, mt.color3, f2);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution;

            BrickTexture mt;
            mt.offset = 0.5;
            mt.offset_frequency = 2;
            mt.squash = 1.0;
            mt.squash_frequency = 2;
            mt.scale = 5.0;
            mt.mortar_size = 0.02;
            mt.mortar_smooth = 0.1;
            mt.bias = 0.0;
            mt.brick_size = vec2(0.5, 0.25);
            mt.color1 = vec3(0.2, 0.2, 0.2);
            mt.color2 = vec3(0.8, 0.8, 0.8);
            mt.color3 = vec3(0.0, 0.0, 0.0);

            vec3 color = brick_color(uv, mt);
            out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    uniforms={
        'resolution': size,
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
    image.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
