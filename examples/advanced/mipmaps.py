import sys

import numpy as np
import pygame
import zengl
import zengl_extras
from skimage.data import gravel

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

texture = ctx.image((512, 512), 'rgba8unorm', np.repeat(gravel(), 4).tobytes(), levels=5)
texture.mipmaps()

uniform_buffer = ctx.buffer(size=80)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        out vec2 v_text;

        vec3 positions[3] = vec3[](
            vec3(0.0, 0.0, 0.0),
            vec3(100.0, 0.0, 0.0),
            vec3(0.0, 100.0, 0.0)
        );

        vec2 texcoords[3] = vec2[](
            vec2(0.0, 0.0),
            vec2(30.0, 0.0),
            vec2(0.0, 30.0)
        );

        void main() {
            gl_Position = mvp * vec4(positions[gl_VertexID], 1.0);
            v_text = texcoords[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        uniform sampler2D Texture;

        in vec2 v_text;

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
            'min_filter': 'linear_mipmap_linear',
            'mag_filter': 'linear',
            'max_anisotropy': 16.0,
            'lod_bias': -1.0,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_count=3,
)

camera = zengl.camera((0, 0, 1.0), (2.0, 2.0, 0.25), aspect=window_aspect, fov=45.0)
uniform_buffer.write(camera)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
