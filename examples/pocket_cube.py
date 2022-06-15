import json
import struct

import numpy as np
import zengl

import assets
from window import Window


def rgb(r, g, b):
    return (pow(r / 255.0, 2.2), pow(g / 255.0, 2.2), pow(b / 255.0, 2.2), 1.0)


def read_gltf(filename):
    with open(filename, 'rb') as f:
        sig, ver, _ = struct.unpack('III', f.read(12))
        assert sig == 0x46546c67 and ver == 2
        size, code = struct.unpack('II', f.read(8))
        assert code == 0x4e4f534a
        info = json.loads(f.read(size))
        size, code = struct.unpack('II', f.read(8))
        assert code == 0x4e4942
        data = f.read(size)
        return info, data


def read_cube(filename):
    data = read_gltf(filename)[1]
    v = np.frombuffer(data[0:10392], 'f4').reshape(-1, 3).T
    n = np.frombuffer(data[10392:20784], 'f4').reshape(-1, 3).T
    t = np.frombuffer(data[20784:27712], 'f4').reshape(-1, 2).T
    idx = np.frombuffer(data[27712:36352], 'u2')
    model = np.array([v[0], v[1], v[2], n[0], n[1], n[2], t[0]], 'f4').T[idx]
    return model


def make_colors(N):
    colors = []
    for z in range(N):
        for y in range(N):
            for x in range(N):
                for i in range(7):
                    color = (0.01, 0.01, 0.01, 1.0)
                    if i == 1 and x == N - 1:
                        color = rgb(183, 18, 52)
                    if i == 2 and x == 0:
                        color = rgb(255, 88, 0)
                    if i == 3 and y == N - 1:
                        color = rgb(0, 70, 173)
                    if i == 4 and y == 0:
                        color = rgb(0, 155, 72)
                    if i == 5 and z == N - 1:
                        color = rgb(255, 255, 255)
                    if i == 6 and z == 0:
                        color = rgb(255, 213, 0)
                    colors.append(color)
    return np.array(colors, 'f4')


N = 2

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.03, 0.03, 0.03, 1.0)

model = read_cube(assets.get('rounded-cube.glb'))
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=64)
color_buffer = ctx.buffer(make_colors(N))

ctx.includes['N'] = f'const int N = {N};'

cube = ctx.pipeline(
    vertex_shader='''
        #version 330

        #include "N"

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (std140) uniform Colors {
            vec4 colors[N * N * N * 7];
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in float in_color;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec3 v_color;

        void main() {
            float x = float(gl_InstanceID % N - 1);
            float y = float(gl_InstanceID / N % N - 1);
            float z = float(gl_InstanceID / N / N % N - 1);
            v_vertex = (in_vertex + vec3(x, y, z)) / N;
            v_normal = in_normal;
            v_color = colors[gl_InstanceID * 7 + int(in_color)].rgb;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_normal)) * 0.7 + 0.3;
            out_color = vec4(v_color * lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Colors',
            'binding': 1,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'uniform_buffer',
            'binding': 1,
            'buffer': color_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 1f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 1f'),
    instance_count=N * N * N,
)


camera = zengl.camera((4.0, 3.0, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    image.clear()
    depth.clear()
    cube.render()
    image.blit()
