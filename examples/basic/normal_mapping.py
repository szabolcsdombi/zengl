import os
import struct
import sys
import zipfile

import numpy as np
import pygame
import zengl
from zengl_extras import assets

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def load_texture(buf):
    img = pygame.image.load(buf)
    pixels = pygame.image.tobytes(img, 'RGBA')
    return ctx.image(img.get_size(), 'rgba8unorm', pixels)


size = pygame.display.get_window_size()

pack = zipfile.ZipFile(assets.get('metal_plate_1k.gltf.zip'))
texture1 = load_texture(pack.open('textures/metal_plate_diff_1k.jpg'))
texture2 = load_texture(pack.open('textures/metal_plate_rough_1k.jpg'))
texture3 = load_texture(pack.open('textures/metal_plate_nor_gl_1k.jpg'))

image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

vertex_buffer = ctx.buffer(np.array([
    -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
], 'f4'))

index_buffer = ctx.buffer(np.array([0, 2, 1, 1, 2, 3], 'i4'))

uniform_buffer = ctx.buffer(size=144)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 eye_pos;
            vec3 light_pos;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec2 in_text;
        layout (location = 3) in vec3 in_tangent;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec2 v_text;
        out vec3 v_tangent;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
            v_text = in_text;
            v_tangent = in_tangent;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 eye_pos;
            vec3 light_pos;
        };

        uniform sampler2D Texture1;
        uniform sampler2D Texture2;
        uniform sampler2D Texture3;

        in vec3 v_vert;
        in vec3 v_norm;
        in vec2 v_text;
        in vec3 v_tangent;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 bitangent = cross(v_tangent, v_norm);
            mat3 btn = mat3(v_tangent, bitangent, v_norm);
            vec3 texture_normal = texture(Texture3, v_text).rgb - 0.5;
            vec3 normal = normalize(btn * texture_normal);
            float shininess = 32.0;
            vec3 light_dir = normalize(light_pos - v_vert);
            vec3 eye_dir = normalize(eye_pos - v_vert);
            vec3 halfway_dir = normalize(light_dir + eye_dir);
            vec3 surface_normal = texture(Texture3, v_text).rgb;
            float rought = texture(Texture2, v_text).r;
            float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess) * rought;
            vec3 color = pow(texture(Texture1, v_text).rgb, vec3(2.2)) + vec3(1.0, 1.0, 1.0) * spec;
            out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Texture1',
            'binding': 0,
        },
        {
            'name': 'Texture2',
            'binding': 1,
        },
        {
            'name': 'Texture3',
            'binding': 2,
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
            'image': texture1,
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': texture2,
        },
        {
            'type': 'sampler',
            'binding': 2,
            'image': texture3,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    index_buffer=index_buffer,
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f 3f', 0, 1, 2, 3),
    vertex_count=index_buffer.size // 4,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0
    mouse = pygame.mouse.get_pos()

    ctx.new_frame()
    image.clear()
    depth.clear()

    mx, my = mouse[0] / size[0] - 0.5, 0.5 - mouse[1] / size[1]
    eye_pos = (mx * 2.0, my * 2.0, 3.0)
    light_pos = (mx * 2.0, my * 2.0, 1.0)
    camera = zengl.camera(eye_pos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), fov=45.0)
    uniform_buffer.write(struct.pack('=64s3f4x3f4x', camera, *eye_pos, *light_pos))

    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
