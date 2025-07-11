import os
import sys

import numpy as np
import pygame
import zengl
import zengl_extras
from PIL import Image

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

uniform_buffer = ctx.buffer(size=80)

if not os.path.isfile('downloads/skybox_0.png'):
    raise Exception('run panorama_to_cubemap.py first!')

size = Image.open('downloads/skybox_0.png').size

faces = b''.join([
    Image.open('downloads/skybox_0.png').convert('RGBA').tobytes('raw', 'RGBA', 0, -1),
    Image.open('downloads/skybox_1.png').convert('RGBA').tobytes('raw', 'RGBA', 0, -1),
    Image.open('downloads/skybox_2.png').convert('RGBA').tobytes('raw', 'RGBA', 0, -1),
    Image.open('downloads/skybox_3.png').convert('RGBA').tobytes('raw', 'RGBA', 0, -1),
    Image.open('downloads/skybox_4.png').convert('RGBA').tobytes('raw', 'RGBA', 0, -1),
    Image.open('downloads/skybox_5.png').convert('RGBA').tobytes('raw', 'RGBA', 0, -1),
])

texture = ctx.image(size, 'rgba8unorm', faces, cubemap=True)

shape = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 eye;
        };

        vec3 vertices[36] = vec3[](
            vec3(-1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, -1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(-1.0, 1.0, -1.0)
        );

        out vec3 v_text;

        void main() {
            gl_Position = mvp * vec4(vertices[gl_VertexID] * 500.0 + eye.xyz, 1.0);
            v_text = vertices[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        uniform samplerCube Texture;
        in vec3 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(texture(Texture, v_text).rgb, 1.0);
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
    vertex_count=36,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now  = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    t = now * 0.5
    eye = (np.cos(t) * 5.0, np.sin(t) * 5.0, np.sin(t * 0.7) * 2.0)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
    uniform_buffer.write(camera + np.array(eye, 'f4').tobytes())

    image.clear()
    depth.clear()
    shape.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
