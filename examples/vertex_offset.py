from itertools import cycle

import chull
import numpy as np
import zengl

from window import Window


def gen_sphere(radius, res=100):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (np.arange(res) / (res - 1.0)) * 2.0
    x = np.cos(phi * np.arange(res)) * np.sqrt(1.0 - y * y)
    z = np.sin(phi * np.arange(res)) * np.sqrt(1.0 - y * y)
    return np.array([x, y, z]).T * radius


window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

table = {}
temp = bytearray()
for i in range(50, 400):
    points = gen_sphere(1.0, i)
    vert, norm = chull.make_hull(points)
    mesh = np.array([vert[:, 0], vert[:, 1], vert[:, 2], norm[:, 0], norm[:, 1], norm[:, 2]]).T.astype('f4').tobytes()
    table[i] = len(temp), len(mesh)
    temp.extend(mesh)

vertex_buffer = ctx.buffer(temp)
uniform_buffer = ctx.buffer(size=64)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
)

vertex_size = zengl.calcsize('3f 3f')

camera = zengl.camera((4.0, 3.0, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

it = iter(cycle(np.clip(np.sin(np.linspace(0.0, 2.0 * np.pi, 180)) * 175 + 225, 50, 400).astype(int)))

while window.update():
    ctx.new_frame()
    image.clear()
    depth.clear()
    pipeline.render()

    offset, size = table[next(it)]
    pipeline.first_vertex = offset // vertex_size
    pipeline.vertex_count = size // vertex_size
    image.blit()
    ctx.end_frame()
