import numpy as np
import zengl
from objloader import Obj

import assets
from window import Window


def kernel(s):
    x = np.arange(-s, s + 1)
    y = np.exp(-x * x / (s * s / 4))
    y /= y.sum()
    v = ', '.join(f'{t:.8f}' for t in y)
    # from matplotlib import pyplot as plt
    # plt.plot(x, y)
    # plt.show()
    return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[]({v});'


window = Window()
ctx = zengl.context()
ctx.includes['kernel'] = kernel(19)

image = ctx.image(window.size, 'rgba8unorm')
depth = ctx.image(window.size, 'depth24plus')

temp = ctx.image(window.size, 'rgba8unorm')
output = ctx.image(window.size, 'rgba8unorm')

image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

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
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)


def make_blur(src, dst, mode):
    return ctx.pipeline(
        vertex_shader='''
            #version 450 core

            vec2 positions[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(3.0, -1.0),
                vec2(-1.0, 3.0)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader='''
            #version 450 core

            uniform int mode;

            layout (binding = 0) uniform sampler2D Texture;

            layout (location = 0) out vec4 out_color;

            #include "kernel"

            void main() {
                vec3 color = vec3(0.0, 0.0, 0.0);
                if (mode == 0) {
                    for (int i = 0; i < N; ++i) {
                        color += texelFetch(Texture, ivec2(gl_FragCoord.xy) + ivec2(i - N / 2, 0), 0).rgb * coeff[i];
                    }
                } else {
                    for (int i = 0; i < N; ++i) {
                        color += texelFetch(Texture, ivec2(gl_FragCoord.xy) + ivec2(0, i - N / 2), 0).rgb * coeff[i];
                    }
                }
                out_color = vec4(color, 1.0);
            }
        ''',
        resources=[
            {
                'type': 'sampler',
                'binding': 0,
                'image': src,
            },
        ],
        uniforms={
            'mode': ['x', 'y'].index(mode),
        },
        framebuffer=[dst],
        topology='triangles',
        vertex_count=3,
    )


blur_x = make_blur(image, temp, 'x')
blur_y = make_blur(temp, output, 'y')

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    ctx.new_frame()
    image.clear()
    depth.clear()
    monkey.render()
    blur_x.render()
    blur_y.render()
    output.blit()
    ctx.end_frame()
