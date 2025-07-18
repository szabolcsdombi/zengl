import colorsys
import struct
import sys
import zipfile

import assets
import numpy as np
import pygame
import zengl
import zengl_extras
from objloader import Obj
from PIL import Image, ImageDraw, ImageFont

pack = zipfile.ZipFile(assets.get('Roboto.zip'))

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

instance_buffer = ctx.buffer(np.array([
    np.repeat(np.linspace(-20.0, 20.0, 30), 30),
    np.tile(np.linspace(-20.0, 20.0, 30), 30),
    np.random.normal(0.0, 0.2, 30 * 30),
    np.random.randint(0, 10, 30 * 30),
]).T.astype('f4').tobytes())

texture = ctx.image((128, 128), 'rgba8unorm', array=10)

for i in range(10):
    img = Image.new('RGBA', (128, 128), '#fff')
    draw = ImageDraw.Draw(img)
    draw.font = ImageFont.truetype(pack.open('Roboto-Bold.ttf'), size=64)
    rgb = (np.array(colorsys.hls_to_rgb(i / 10, 0.6, 0.6)) * 255).astype('u1')
    draw.rectangle((0, 0, 128, 128), tuple(rgb))
    aabb = draw.textbbox((0, 0), f'{i + 1}')
    draw.text((64 - (aabb[2] - aabb[0]) // 2, 25), f'{i + 1}', '#000')
    texture.write(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes(), layer=i)

texture.mipmaps()

uniform_buffer = ctx.buffer(size=80)

crate = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec2 in_text;
        layout (location = 3) in vec3 in_pos;
        layout (location = 4) in float in_map;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec3 v_text;

        void main() {
            v_vert = in_pos + in_vert;
            gl_Position = mvp * vec4(v_vert, 1.0);
            v_norm = in_norm;
            v_text = vec3(in_text, in_map);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        uniform sampler2DArray Texture;

        in vec3 v_vert;
        in vec3 v_norm;
        in vec3 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = clamp(dot(normalize(light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.6 + 0.4;
            out_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
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
            'min_filter': 'linear_mipmap_linear',
            'mag_filter': 'linear',
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
        *zengl.bind(instance_buffer, '3f 1f /i', 3, 4),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
    instance_count=30 * 30,
)

camera = zengl.camera((3.0, 2.0, 1.5), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)

uniform_buffer.write(camera)
uniform_buffer.write(struct.pack('3f4x', 3.0, 2.0, 1.5), offset=64)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    depth.clear()
    crate.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
