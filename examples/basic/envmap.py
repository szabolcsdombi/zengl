import math
import os
import struct
import sys
import zipfile

import pygame
import zengl
from meshtools import obj
from zengl_extras import assets

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def load_texture(buf):
    img = pygame.image.load(buf)
    pixels = pygame.image.tobytes(img, 'RGBA')
    return ctx.image(img.get_size(), 'rgba8unorm', pixels)


def load_model(name):
    with open(assets.get(name)) as f:
        model = obj.parse_obj(f.read(), 'vn')
    return ctx.buffer(model)


size = pygame.display.get_window_size()

image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

pack = zipfile.ZipFile(assets.get('forest-panorama.zip'))

texture = load_texture(pack.open('forest.jpg'))

vertex_buffer = load_model('blob.obj')

uniform_buffer = ctx.buffer(size=80)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 eye;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_vert;
        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 eye;
        };

        uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        float atan2(float y, float x) {
            return x == 0.0 ? sign(y) * 3.1415 / 2.0 : atan(y, x);
        }

        void main() {
            vec3 ray = reflect(normalize(v_vert - eye), normalize(v_norm));
            vec2 tex = vec2(atan2(ray.y, ray.x) / 3.1415, -ray.z);
            float lum = dot(normalize(eye - v_vert), normalize(v_norm)) * 0.3 + 0.7;
            vec3 color = texture(Texture, tex).rgb;
            out_color = vec4(color * lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0

    x, y = math.sin(now * 0.5) * 5.0, math.cos(now * 0.5) * 5.0
    camera = zengl.camera((x, y, 2.0), (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)
    uniform_buffer.write(camera)
    uniform_buffer.write(struct.pack('3f4x', x, y, 2.0), offset=64)

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
