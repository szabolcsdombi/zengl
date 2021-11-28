import math

import numpy as np
import zengl
from skimage import measure
from skimage.draw import ellipsoid

from window import Window

# Generate a level set about zero of two identical ellipsoids in 3D
ellip_base = ellipsoid(6, 10, 16, levelset=True)
ellip_double = np.concatenate((ellip_base[:-1, ...],
                               ellip_base[2:, ...]), axis=0)

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(ellip_double, 0)

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

mesh = np.concatenate([verts, -normals], axis=1).astype('f4').tobytes()
index = faces.astype('i4').tobytes()

vertex_buffer = ctx.buffer(mesh)
index_buffer = ctx.buffer(index)

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
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
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
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
)

while window.update():
    x, y = math.sin(window.time * 0.5) * 80.0, math.cos(window.time * 0.5) * 80.0
    camera = zengl.camera((12.0 + x, 10.0 + y, 40.0), (12.0, 10.0, 16.0), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    model.render()
    image.blit()
