import sys

import assets
import pygame
import vmath
import zengl
import zengl_extras
from objloader import Obj

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]
ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('blob.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)


class Blob:
    def __init__(self, x, y):
        self.position = vmath.vec(x * 4.0 - 36.0, y * 4.0 - 36.0, 0.0)
        self.rotation = vmath.random_rotation()
        self.axis = vmath.random_axis()

    def update(self):
        self.rotation = vmath.rotate(self.axis, 0.01) * self.rotation

    def pack(self):
        return self.position.pack() + self.rotation.pack()


blobs = [Blob(i // 20, i % 20) for i in range(400)]
instance_buffer = ctx.buffer(b''.join(blob.pack() for blob in blobs))

uniform_buffer = ctx.buffer(size=64)

ctx.includes['qtransform'] = '''
    vec3 qtransform(vec4 q, vec3 v) {
        return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
    }
'''

shape = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "qtransform"

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;
        layout (location = 2) in vec3 in_position;
        layout (location = 3) in vec4 in_rotation;

        out vec3 v_norm;

        void main() {
            vec3 v_vert = in_position + qtransform(in_rotation, in_vert);
            gl_Position = mvp * vec4(v_vert, 1.0);
            v_norm = qtransform(in_rotation, in_norm);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(0.0, 0.0, 1.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.5 + 0.5;
            out_color = vec4(lum, lum, lum, 1.0);
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
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        *zengl.bind(instance_buffer, '3f 4f /i', 2, 3),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    instance_count=len(blobs),
)

camera = zengl.camera((60.0, 0.0, 15.0), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
uniform_buffer.write(camera)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    for blob in blobs:
        blob.update()

    ctx.new_frame()
    instance_buffer.write(b''.join(blob.pack() for blob in blobs))

    image.clear()
    depth.clear()
    shape.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
