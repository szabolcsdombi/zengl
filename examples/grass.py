import math
import struct

import numpy as np
import zengl

from window import Window


def grass_mesh():
    a = np.linspace(0.0, 1.0, 8)
    b = np.square(a)
    c = np.sin(b * (np.pi - 1.0) + 1.0)
    verts = []
    for i in range(7):
        verts.append((-c[i] * 0.03, b[i] * 0.2, a[i]))
        verts.append((c[i] * 0.03, b[i] * 0.2, a[i]))
    verts.append((0.0, 0.2, 1.0))
    verts = ','.join('vec3(%.8f, %.8f, %.8f)' % x for x in verts)
    return f'vec3 grass[15] = vec3[]({verts});'


window = Window()
ctx = zengl.context()
image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

uniform_buffer = ctx.buffer(size=80)

N = 200

ctx.includes['N'] = f'const int N = {N};'
ctx.includes['grass'] = grass_mesh()

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        #include "N"
        #include "grass"

        vec4 hash41(float p) {
            vec4 p4 = fract(vec4(p) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
            p4 += dot(p4, p4.wzxy + 33.33);
            return fract((p4.xxyz + p4.yzzw) * p4.zywx);
        }

        float hash11(float p) {
            p = fract(p * 0.1031);
            p *= p + 33.33;
            p *= p + p;
            return fract(p);
        }

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
        };

        out vec2 v_data;

        void main() {
            vec3 v = grass[gl_VertexID];
            vec4 data = hash41(float(gl_InstanceID));
            vec2 cell = vec2(gl_InstanceID % N, gl_InstanceID / N);
            float height = (sin(cell.x * 0.1) + cos(cell.y * 0.1)) * 0.2;
            float scale = 0.9 + hash11(gl_InstanceID) * 0.2;
            data.xy = (data.xy + cell - N / 2) * 0.1;
            data.z *= 6.283184;
            vec3 vert = vec3(
                data.x + cos(data.z) * v.x + sin(data.z) * v.y,
                data.y + cos(data.z) * v.y - sin(data.z) * v.x,
                height + v.z
            );
            vert *= scale;
            gl_Position = mvp * vec4(vert, 1.0);
            v_data = vec2(data.w, v.z);
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec2 v_data;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 yl = vec3(0.63, 1.0, 0.3);
            vec3 gn = vec3(0.15, 0.83, 0.3);
            out_color = vec4((yl + (gn - yl) * v_data.x) * v_data.y, 1.0);
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
    topology='triangle_strip',
    cull_face='back',
    instance_count=N * N,
    vertex_count=15,
)

while window.update():
    ctx.new_frame()
    x, y = math.sin(window.time * 0.2) * 12.0, math.cos(window.time * 0.2) * 12.0
    camera = zengl.camera((x, y, 4.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)
    uniform_buffer.write(struct.pack('3f4x', x, y, 2.0), offset=64)

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()
