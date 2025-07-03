import sys

import numpy as np
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()


def noise(width, height, red, green, blue):
    img = np.ones((height, width, 4), 'u1')
    img[:, :, 0:3] += np.random.randint(0, 50, (height, width, 3), 'u1')
    img[:, :, 0] += red
    img[:, :, 1] += green
    img[:, :, 2] += blue
    return img


def quad(x, y, width, height, layer, level):
    return np.array([
        x - width / 2.0, y - height / 2.0, 0.0, 0.0, layer, level,
        x + width / 2.0, y - height / 2.0, 1.0, 0.0, layer, level,
        x + width / 2.0, y + height / 2.0, 1.0, 1.0, layer, level,
        x - width / 2.0, y - height / 2.0, 0.0, 0.0, layer, level,
        x - width / 2.0, y + height / 2.0, 0.0, 1.0, layer, level,
        x + width / 2.0, y + height / 2.0, 1.0, 1.0, layer, level,
    ], 'f4')


ctx = zengl.context()
image = ctx.image(window_size, 'rgba8unorm', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

texture = ctx.image((64, 64), 'rgba8unorm', array=4, levels=3)
texture.mipmaps()

texture.write(noise(256, 64, 50, 50, 50))
texture.write(noise(128, 32, 50, 50, 50), level=1)

texture.write(noise(64, 64, 200, 50, 50), layer=0)
texture.write(noise(64, 64, 50, 200, 50), layer=1)
texture.write(noise(64, 64, 50, 50, 200), layer=2)
texture.write(noise(64, 64, 200, 200, 200), layer=3)

texture.write(noise(32, 32, 200, 50, 50), layer=0, level=1)
texture.write(noise(32, 32, 50, 200, 50), layer=1, level=1)
texture.write(noise(32, 32, 50, 50, 200), layer=2, level=1)
texture.write(noise(32, 32, 200, 200, 200), layer=3, level=1)

texture.write(noise(16, 16, 200, 50, 50), layer=0, level=2)
texture.write(noise(16, 16, 50, 200, 50), layer=1, level=2)
texture.write(noise(16, 16, 50, 50, 200), layer=2, level=2)
texture.write(noise(16, 16, 200, 200, 200), layer=3, level=2)

vertex_buffer = ctx.buffer(np.concatenate([
    quad(130.0, 130.0, 200.0, 200.0, 0.0, 0.0),
    quad(360.0, 130.0, 200.0, 200.0, 1.0, 0.0),
    quad(590.0, 130.0, 200.0, 200.0, 2.0, 0.0),
    quad(820.0, 130.0, 200.0, 200.0, 3.0, 0.0),
    quad(130.0, 360.0, 200.0, 200.0, 0.0, 1.0),
    quad(360.0, 360.0, 200.0, 200.0, 1.0, 1.0),
    quad(590.0, 360.0, 200.0, 200.0, 2.0, 1.0),
    quad(820.0, 360.0, 200.0, 200.0, 3.0, 1.0),
    quad(130.0, 590.0, 200.0, 200.0, 0.0, 2.0),
    quad(360.0, 590.0, 200.0, 200.0, 1.0, 2.0),
    quad(590.0, 590.0, 200.0, 200.0, 2.0, 2.0),
    quad(820.0, 590.0, 200.0, 200.0, 3.0, 2.0),
]))

width, height = window_size
ctx.includes['screen_size'] = f'const vec2 screen_size = vec2({width}, {height});'

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "screen_size"

        layout (location = 0) in vec2 in_vertex;
        layout (location = 1) in vec2 in_texcoord;
        layout (location = 2) in float in_layer;
        layout (location = 3) in float in_level;

        out vec2 v_vertex;
        out vec2 v_texcoord;
        out float v_layer;
        out float v_level;

        void main() {
            v_vertex = in_vertex;
            v_texcoord = in_texcoord;
            v_layer = in_layer;
            v_level = in_level;
            gl_Position = vec4((v_vertex / screen_size) * 2.0 - 1.0, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        uniform sampler2DArray Texture;

        in vec2 v_vertex;
        in vec2 v_texcoord;
        in float v_layer;
        in float v_level;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = textureLod(Texture, vec3(v_texcoord, v_layer), v_level);
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
            'min_filter': 'nearest_mipmap_nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '2f 2f 1f 1f', 0, 1, 2, 3),
    vertex_count=vertex_buffer.size // zengl.calcsize('2f 2f 1f 1f'),
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
