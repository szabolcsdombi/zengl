import sys

import assets
import numpy as np
import pygame
import zengl
import zengl_extras
from objloader import Obj
from PIL import Image
from progress.bar import Bar
from skimage.filters import gaussian

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

size = 1024
samples = 512
temp_color = ctx.image((size, size), 'r32sint')
temp_depth = ctx.image((size, size), 'depth24plus')
temp_color.clear_value = -1

model = Obj.open(assets.get('ao-map-target.obj')).pack('vx vy vz nx ny nz tx ty')
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=64)

ctx.includes['size'] = f'const int size = {size};'

texcoord_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec2 in_texcoord;

        out vec2 v_texcoord;

        void main() {
            gl_Position = mvp * vec4(in_vertex, 1.0);
            v_texcoord = in_texcoord;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;
        precision highp int;

        #include "size"

        in vec2 v_texcoord;

        layout (location = 0) out int out_address;

        void main() {
            int tx = int(v_texcoord.x * float(size) + 0.5);
            int ty = int(v_texcoord.y * float(size) + 0.5);
            out_address = ty * size + tx;
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[temp_color, temp_depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, -1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

bar = Bar('Progress', fill='-', suffix='%(percent)d%%', max=samples)

ao = np.zeros(size * size, 'f4')

for i in range(samples):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (i / (samples - 1.0)) * 2.0
    x = np.cos(phi * i) * np.sqrt(1.0 - y * y)
    z = np.sin(phi * i) * np.sqrt(1.0 - y * y)

    camera = zengl.camera((x * 5.0, y * 5.0, z * 5.0), (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)
    uniform_buffer.write(camera)
    temp_color.clear()
    temp_depth.clear()
    texcoord_pipeline.render()
    t = np.frombuffer(temp_color.read(), 'i4').reshape((size, size))
    ao[np.unique(t[t >= 0])] += 1.0
    bar.next()

ao -= ao.min()
ao /= ao.max()
ao = gaussian(ao, 1.0)

texture = ctx.image((size, size), 'r32float', ao)

Image.fromarray((ao.reshape(size, size) * 255.0).astype('u1'), 'L').save('generated-ao-map.png')

render_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec2 in_texcoord;

        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            gl_Position = mvp * vec4(in_vertex, 1.0);
            v_normal = in_normal;
            v_texcoord = in_texcoord;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        uniform sampler2D Texture;

        in vec2 v_texcoord;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = texture(Texture, v_texcoord).r;
            vec3 color = vec3(1.0, 1.0, 1.0);
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
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, -1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now =  pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    x, y = np.cos(now * 0.5) * 5.0, np.sin(now * 0.5) * 5.0
    camera = zengl.camera((x, y, 1.0), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
    uniform_buffer.write(camera)
    image.clear()
    depth.clear()
    render_pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
