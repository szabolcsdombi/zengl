import math
import struct
import sys

import pygame
import zengl
import zengl_extras
from OpenGL.GL.ARB.bindless_texture import glGetTextureHandleARB, glMakeTextureHandleResidentARB

zengl_extras.init()
zengl_extras.download('crate.zip')

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

model = open('downloads/crate/crate.bin', 'rb').read()
vertex_buffer = ctx.buffer(model)

img = pygame.image.load('downloads/crate/crate.png')
img_size, img_data = img.get_size(), pygame.image.tobytes(img, 'RGBA', True)
texture = ctx.image(img_size, 'rgba8unorm', img_data)

texture_obj = zengl.inspect(texture)['texture']
texture_handle = glGetTextureHandleARB(texture_obj)
glMakeTextureHandleResidentARB(texture_handle)

uniform_buffer = ctx.buffer(size=96, uniform=True)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core
        #extension GL_ARB_bindless_texture : enable

        layout (std140) uniform Common {
            mat4 camera;
            vec3 light;
            layout (bindless_sampler) sampler2D Texture;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec2 in_uv;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec2 v_uv;

        void main() {
            gl_Position = camera * vec4(in_vertex, 1.0);
            v_vertex = in_vertex;
            v_normal = in_normal;
            v_uv = in_uv;
        }
    ''',
    fragment_shader='''
        #version 330 core
        #extension GL_ARB_bindless_texture : enable

        layout (std140) uniform Common {
            mat4 camera;
            vec3 light;
            layout (bindless_sampler) sampler2D Texture;
        };

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec2 v_uv;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = clamp(dot(normalize(light - v_vertex), normalize(v_normal)), 0.0, 1.0) * 0.6 + 0.4;
            out_color = vec4(texture(Texture, v_uv).rgb * lum, 1.0);
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
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    time = pygame.time.get_ticks() / 1000.0
    eye = (math.cos(time * 0.6) * 3.0, math.sin(time * 0.6) * 3.0, 1.5)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)
    uniform_buffer.write(struct.pack('64s3f4xQ', camera, *eye, texture_handle))
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
