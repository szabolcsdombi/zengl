# pip install https://github.com/fogleman/sdf/archive/refs/heads/main.zip
import math
import sys

import numpy as np
import pygame
import sdf
import zengl
import zengl_extras

c = sdf.cylinder(0.5)
f = sdf.sphere(1) & sdf.box(1.5)
f -= c.orient(sdf.X) | c.orient(sdf.Y) | c.orient(sdf.Z)

# f = sdf.sphere(2) & sdf.slab(z0=-0.5, z1=0.5).k(0.1)
# f -= sdf.cylinder(1).k(0.1)
# f -= sdf.cylinder(0.25).circular_array(16, 2).k(0.1)

# s = sdf.sphere(0.75)
# s = s.translate(sdf.Z * -3) | s.translate(sdf.Z * 3)
# s = s.union(sdf.capsule(sdf.Z * -3, sdf.Z * 3, 0.5), k=1)
# f = sdf.sphere(1.5).union(s.orient(sdf.X), s.orient(sdf.Y), s.orient(sdf.Z), k=1)

# f = sdf.rounded_cylinder(1, 0.1, 5)
# x = sdf.box((1, 1, 4)).rotate(sdf.pi / 4)
# x = x.circular_array(24, 1.6)
# x = x.twist(0.75) | x.twist(-0.75)
# f -= x.k(0.1)
# f -= sdf.cylinder(0.5).k(0.1)
# c = sdf.cylinder(0.25).orient(sdf.X)
# f -= c.translate(sdf.Z * -2.5).k(0.1)
# f -= c.translate(sdf.Z * 2.5).k(0.1)

# f = sdf.rounded_box([3.2, 1, 0.25], 0.1).translate((1.5, 0, 0.0625))
# f = f.bend_linear(sdf.X * 0.75, sdf.X * 2.25, sdf.Z * -0.1875, sdf.ease.in_out_quad)
# f = f.circular_array(3, 0)
# f = f.repeat((2.7, 5.4, 0), padding=1)
# f |= f.translate((2.7 / 2, 2.7, 0))
# f &= sdf.cylinder(10)
# f |= (sdf.cylinder(12) - sdf.cylinder(10)) & sdf.slab(z0=-0.5, z1=0.5).k(0.25)

points = np.array(f.generate())
tmp = points.reshape(-1, 3, 3)
normals = np.repeat(np.cross(tmp[:, 1] - tmp[:, 0], tmp[:, 2] - tmp[:, 0]), 3, axis=0)
radius = np.max(np.sqrt(np.sum(points * points, axis=1)))

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

mesh = np.concatenate([points, normals], axis=1).astype('f4').tobytes()

vertex_buffer = ctx.buffer(mesh)

uniform_buffer = ctx.buffer(size=80)

model = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

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
        #version 300 es
        precision highp float;

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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now  = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    x, y = math.sin(now * 0.5) * 2.5 * radius, math.cos(now * 0.5) * 2.5 * radius
    camera = zengl.camera((x, y, 1.5 * radius), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)

    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    model.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
