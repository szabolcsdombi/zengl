import json
import re
import struct

import numpy as np
import zengl

import assets
from window import Window

N = 2


def quatmul(A, B):
    ax, ay, az, aw = A.T
    bx, by, bz, bw = B.T
    return np.array([
        ax * bw + ay * bz - az * by + aw * bx,
        -ax * bz + ay * bw + az * bx + aw * by,
        ax * by - ay * bx + az * bw + aw * bz,
        -ax * bx - ay * by - az * bz + aw * bw,
    ]).T


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


def get_layer(rotations, axis, level, N):
    i = np.arange(N * N * N)
    v = np.array([i % N, i // N % N, i // N // N % N]).T - (N - 1) / 2.0
    v = v + np.cross(np.cross(v, rotations[:, :3]) - v * rotations[:, [3, 3, 3]], rotations[:, :3]) * 2.0
    v = np.round(v + (N - 1) / 2.0).astype(int)
    return i[v[:, 'xyz'.index(axis)] == level]


def make_rotations(axis, angle, sign, count):
    quat = [0.0, 0.0, 0.0, np.cos(angle / 2.0)]
    quat['xyz'.index(axis)] = np.sin(angle / 2.0) * (1.0 if sign == '+' else -1.0)
    return np.full((count, 4), quat)


window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.03, 0.03, 0.03, 1.0)

model = read_cube(assets.get('rounded-cube.glb'))
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=64)
color_buffer = ctx.buffer(make_colors(N))

rotations = np.full((N * N * N, 4), (0.0, 0.0, 0.0, 1.0), 'f4')
rotation_buffer = ctx.buffer(rotations)

ctx.includes['N'] = f'const int N = {N};'

ctx.includes['qtransform'] = '''
    vec3 qtransform(vec4 q, vec3 v) {
        return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
    }
'''

cube = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "N"
        #include "qtransform"

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (std140) uniform Colors {
            vec4 colors[N * N * N * 7];
        };

        layout (std140) uniform Rotations {
            vec4 rotations[N * N * N];
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in float in_color;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec3 v_color;

        void main() {
            float x = float(gl_InstanceID % N) - float(N - 1) / 2.0;
            float y = float(gl_InstanceID / N % N) - float(N - 1) / 2.0;
            float z = float(gl_InstanceID / N / N % N) - float(N - 1) / 2.0;
            v_vertex = qtransform(rotations[gl_InstanceID], (in_vertex + vec3(x, y, z)) / float(N));
            v_normal = qtransform(rotations[gl_InstanceID], in_normal);
            v_color = colors[gl_InstanceID * 7 + int(in_color)].rgb;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_normal)) * 0.7 + 0.3;
            out_color = vec4(pow(v_color * lum, vec3(1.0 / 2.2)), 1.0);
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
        {
            'name': 'Rotations',
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
            'type': 'uniform_buffer',
            'binding': 1,
            'buffer': color_buffer,
        },
        {
            'type': 'uniform_buffer',
            'binding': 2,
            'buffer': rotation_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 1f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 1f'),
    instance_count=N * N * N,
)


class Rotate:
    def __init__(self, rotations, rotate, steps=30):
        sign, axis, level = re.match(r'([+-])([xyz])(\d)', rotate).groups()
        self.idx = get_layer(rotations, axis, int(level), N)
        self.final = quatmul(make_rotations(axis, np.pi / 2.0, sign, len(self.idx)), rotations[self.idx])
        self.rotate = make_rotations(axis, np.pi / 2.0 / steps, sign, len(self.idx))
        self.rotations = rotations
        self.steps = steps

    def update(self):
        if self.steps > 0:
            self.rotations[self.idx] = quatmul(self.rotate, self.rotations[self.idx])
            self.steps -= 1
            return False
        else:
            self.rotations[self.idx] = self.final
            return True


class Rotateions:
    def __init__(self, rotations, sequence):
        self.rotations = rotations
        self.it = iter(sequence)
        self.rotate = None

    def update(self):
        try:
            if self.rotate is None:
                self.rotate = Rotate(rotations, next(self.it))
        except StopIteration:
            return
        if self.rotate.update():
            self.rotate = None


camera = zengl.camera((4.0, 3.0, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

moves = [f'{sign}{axis}{level}' for sign in '+-' for axis in 'xyz' for level in range(N)]
animation = Rotateions(rotations, np.random.choice(moves, 1000))

while window.update():
    ctx.new_frame()
    animation.update()
    rotation_buffer.write(rotations)

    image.clear()
    depth.clear()
    cube.render()
    image.blit()
    ctx.end_frame()
