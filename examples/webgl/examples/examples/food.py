import random
import struct
from math import cos, pi, sin, sqrt

import zengl

from . import assets, webgl

window = webgl.Canvas()
ctx = zengl.context(window)


def random_rotation():
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()
    return (
        sqrt(1.0 - u1) * sin(2.0 * pi * u2),
        sqrt(1.0 - u1) * cos(2.0 * pi * u2),
        sqrt(u1) * sin(2.0 * pi * u3),
        sqrt(u1) * cos(2.0 * pi * u3),
    )


def quatmul(a, b):
    rx = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    ry = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
    rz = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
    rw = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    return rx, ry, rz, rw


def qtransform(a, b):
    tx = b[1] * a[2] - a[1] * b[2] - a[3] * b[0]
    ty = b[2] * a[0] - a[2] * b[0] - a[3] * b[1]
    tz = b[0] * a[1] - a[0] * b[1] - a[3] * b[2]
    return (
        b[0] + (ty * a[2] - a[1] * tz) * 2.0,
        b[1] + (tz * a[0] - a[2] * tx) * 2.0,
        b[2] + (tx * a[1] - a[0] * ty) * 2.0,
    )


def axisangle(axis, angle):
    s = sin(angle / 2.0) / sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    return axis[0] * s, axis[1] * s, axis[2] * s, cos(angle / 2.0)


image = ctx.image(window.size, 'rgba8unorm', texture=False)
depth = ctx.image(window.size, 'depth24plus', texture=False)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

vertex_buffer = ctx.buffer(open(assets.get('food-model.bin'), 'rb').read())
uniform_buffer = ctx.buffer(size=64)
bone_buffer = ctx.buffer(size=256 * 32)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec3 qtransform(vec4 q, vec3 v) {
            return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
        }

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (std140) uniform Bones {
            vec4 bone[256 * 2];
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec3 in_color;
        layout (location = 3) in int in_bone;

        out vec3 v_vert;
        out vec3 v_norm;
        out vec3 v_color;

        void main() {
            float scale = bone[in_bone * 2 + 0].w;
            vec3 position = bone[in_bone * 2 + 0].xyz;
            vec4 rotation = bone[in_bone * 2 + 1];
            v_vert = position + qtransform(rotation, in_vert * scale);
            v_norm = normalize(qtransform(rotation, in_norm));
            gl_Position = mvp * vec4(v_vert, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_norm;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(1.0, -4.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(v_color * lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Bones',
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
            'buffer': bone_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 4nu1 1i', 0, 1, 2, 3),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 4nu1 1i'),
)

camera = zengl.camera((1.0, -4.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

packer = struct.Struct('8f')
bones = bytearray(8192)

packer.pack_into(bones, 0 * 32, -0.1, 0.12, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
packer.pack_into(bones, 1 * 32, -0.42, -0.225, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
packer.pack_into(bones, 2 * 32, 0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0)
packer.pack_into(bones, 3 * 32, 0.0, 0.0, 0.325, 0.245, 0.0, 0.0, 0.0, 1.0)
packer.pack_into(bones, 4 * 32, 0.14, 0.0, 0.335, 0.45, 0.0, 0.0, -0.707, 0.707)
packer.pack_into(bones, 5 * 32, -0.19, 0.0, 0.335, 0.45, 0.0, 0.0, -0.707, 0.707)
packer.pack_into(bones, 6 * 32, -0.14, 0.0, 0.335, 0.4, 0.0, 0.0, -0.707, 0.707)

size = [random.uniform(0.9, 1.0) for _ in range(256)]
for i in (147, 149, 150, 151, 152, 153, 154, 184, 185):
    size[i] = 0.5

for i in range(7, 207):
    packer.pack_into(bones, i * 32, 0.0, 0.0, 0.0, 0.0, *random_rotation())

offset = [random.uniform(0.0, pi * 2.0) for _ in range(256)]
speed = [random.uniform(0.5, 0.7) for _ in range(256)]
radius = [random.uniform(2.0, 3.0) for _ in range(256)]

vertical_offset = [random.uniform(0.0, pi * 2.0) for _ in range(256)]
vertical_speed = [random.uniform(0.2, 0.4) for _ in range(256)]
vertical_radius = [random.uniform(0.1, 0.2) for _ in range(256)]

rotation_speed = [random.uniform(0.6, 1.2) for _ in range(256)]
rotation_base = [random_rotation() for _ in range(256)]
rotation_axis = [qtransform(random_rotation(), (0.0, 0.0, 1.0)) for _ in range(256)]


def render():
    t = window.time
    packer.pack_into(bones, 2 * 32, 0.0, 0.0, 0.8, 1.0, *axisangle((0.0, 0.0, 1.0), t * 20.0))
    for i in range(7, 207):
        a = offset[i] + t * speed[i]
        x = cos(a) * radius[i]
        y = sin(a) * radius[i]
        z = sin(vertical_offset[i] + vertical_speed[i] * t) * vertical_radius[i] + 0.5
        rotation = quatmul(rotation_base[i], axisangle(rotation_axis[i], rotation_speed[i] * t))
        packer.pack_into(bones, i * 32, x, y, z, size[i], *rotation)

    ctx.new_frame()
    bone_buffer.write(bones)
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()
    window.update()