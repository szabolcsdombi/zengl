import numpy as np
import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')

vertex_buffer = ctx.buffer(model)
vertex_count = vertex_buffer.size // 24

index_buffer = ctx.buffer(np.array([
    np.arange(0, vertex_count, 3),
    np.arange(1, vertex_count, 3),
    np.arange(2, vertex_count, 3),
    np.full(vertex_count // 3, -1),
], dtype='i4').T.tobytes())

uniform_buffer = ctx.buffer(size=80)

monkey = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
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
        #version 450 core

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
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
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_count,
)

monkey_wire = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core

        layout (location = 0) out vec4 out_color;

        void main() {
            gl_FragDepth = gl_FragCoord.z - 1e-4;
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
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
    topology='line_loop',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, -1),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4 - 124 * 4,
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    image.clear()
    depth.clear()
    monkey.run()
    monkey_wire.run()
    image.blit()
