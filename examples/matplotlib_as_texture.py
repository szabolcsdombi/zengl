import io
import math
import struct
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import zengl
from objloader import Obj

import assets
from window import Window

figure_size = (256, 256)
temp = io.BytesIO()


def plot(offset):
    t = np.arange(0.0, 2.0, 0.01) + offset
    s = 1 + np.sin(2 * np.pi * t) * 0.9 ** offset
    plt.clf()
    plt.figure(0, figsize=(figure_size[0] / 72, figure_size[1] / 72))
    ax = plt.gca()
    ax.plot(t, s)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
    ax.grid()
    plt.savefig(temp, format='raw', dpi=72)


window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

model = Obj.open(assets.get('box.obj')).to_array()[:, 0:8].copy()
model[:, 7] = 1.0 - model[:, 7]
vertex_buffer = ctx.buffer(model)

texture = ctx.image(figure_size, 'rgba8unorm')

uniform_buffer = ctx.buffer(size=80)

crate = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec2 in_text;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec2 v_text;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
            v_text = in_text;
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
            vec3 light;
        };

        uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;
        in vec2 v_text;

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
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

redraw_plot = cycle([True] + [False] * 15)

while window.update():
    ctx.new_frame()
    if next(redraw_plot):
        temp.seek(0)
        plot(window.time)
        texture.write(temp.getvalue())

    x, y = math.sin(window.time * 0.5) * 3.0, math.cos(window.time * 0.5) * 3.0
    camera = zengl.camera((x, y, 1.5), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)
    uniform_buffer.write(struct.pack('3f4x', x, y, 1.5), offset=64)

    image.clear()
    depth.clear()
    crate.render()
    image.blit()
    ctx.end_frame()
