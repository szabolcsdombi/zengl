import vmath
import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
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
        #version 450 core

        #include "qtransform"

        layout (std140, binding = 0) uniform Common {
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
        #version 450 core

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(0.0, 0.0, 1.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.5 + 0.5;
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
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        *zengl.bind(instance_buffer, '3f 4f /i', 2, 3),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    instance_count=len(blobs),
)

camera = zengl.camera((60.0, 0.0, 15.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    for blob in blobs:
        blob.update()

    instance_buffer.write(b''.join(blob.pack() for blob in blobs))

    image.clear()
    depth.clear()
    shape.run()
    image.blit()
