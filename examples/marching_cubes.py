import math

import numpy as np
import zengl
from skimage import measure
from skimage.draw import ellipsoid
from skimage.filters import gaussian

from window import Window

volume = np.full((50, 50, 50), 1.0)
volume[15:35, 15:35, 15:35] = -1.0
sphere = ellipsoid(8.0, 8.0, 8.0, levelset=True)
volume[27:46, 27:46, 27:46] = np.min([volume[27:46, 27:46, 27:46], sphere], axis=0)
volume[4:23, 4:23, 4:23] = np.min([volume[4:23, 4:23, 4:23], sphere], axis=0)
volume = gaussian(volume, 1.5)

verts, faces, normals, values = measure.marching_cubes(volume, 0.0)
verts -= (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2.0

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
    x, y = np.sin(window.time * 0.5) * 80.0, np.cos(window.time * 0.5) * 80.0
    camera = zengl.camera((x, y, 40.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    model.render()
    image.blit()
