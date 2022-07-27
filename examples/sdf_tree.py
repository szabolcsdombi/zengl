# pip install https://github.com/fogleman/sdf/archive/refs/heads/main.zip
import math

import numpy as np
import sdf
import zengl

from window import Window


def tree(depth=0):
    f = sdf.rounded_cone(0.25 * 0.7 ** depth, 0.15 * 0.7 ** depth, 2)
    if depth < 3:
        for i in range(5):
            a = i * np.pi * 2 / 5 + np.random.normal(0.0, 0.3)
            if np.random.uniform(0.0, 1.0) < 0.7:
                f |= tree(depth + 1).orient((np.cos(a), np.sin(a), 2.0)).translate(sdf.Z * 2.0)
    return f


np.random.seed(1234567)
f = tree().translate(-sdf.Z * 4.0)

points = np.array(f.generate(samples=2**25))
tmp = points.reshape(-1, 3, 3)
normals = np.repeat(np.cross(tmp[:, 1] - tmp[:, 0], tmp[:, 2] - tmp[:, 0]), 3, axis=0)
radius = np.max(np.sqrt(np.sum(points * points, axis=1)))

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

mesh = np.concatenate([points, normals], axis=1).astype('f4').tobytes()

vertex_buffer = ctx.buffer(mesh)

uniform_buffer = ctx.buffer(size=80)

model = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
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
        #version 330

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 color = vec3(0.1, 0.7, 1.0);
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.4 + 0.6;
            out_color = vec4(color * lum, 1.0);
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
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // 24,
)

while window.update():
    x, y = math.sin(window.time * 0.5) * 2.5 * radius, math.cos(window.time * 0.5) * 2.5 * radius
    camera = zengl.camera((x, y, 1.5 * radius), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    model.render()
    image.blit()
