import struct
from colorsys import hls_to_rgb

import numpy as np
import zengl

from window import Window

window = Window()
ctx = zengl.context()
image = ctx.image(window.size, 'rgba8unorm-srgb', samples=4)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

N, M = 16, 128
t = np.linspace(0.0, np.pi / 2.0, N, endpoint=False)

sx = np.repeat(-np.cos(t), 2)[1:]
sy = np.array([np.sin(t), -np.sin(t)]).T.flatten()[1:]
sz = np.zeros(N * 2 - 1)

vx = np.zeros(M * 2)
vy = np.tile([1.0, -1.0], M)
vz = np.repeat(np.linspace(0.0, 1.0, M), 2)

x = np.concatenate([sx, vx, -sx[::-1]])
y = np.concatenate([sy, vy, -sy[::-1]])
z = np.concatenate([sz, vz, sz + 1.0])

instance_count = 32
curves = []

for i in range(instance_count):
    w, h = window.size
    a = np.random.uniform(0.0, np.pi * 2.0)
    b = a + np.pi + np.random.uniform(-0.5, 0.5)
    c = np.random.uniform(0.0, np.pi * 2.0)
    d = np.random.uniform(0.0, np.pi * 2.0)
    x1, y1 = np.cos(a) * 300.0 + w / 2.0, np.sin(a) * 300.0 + h / 2.0
    x2, y2 = np.cos(b) * 300.0 + w / 2.0, np.sin(b) * 300.0 + h / 2.0
    x3, y3 = np.cos(c) * 150.0 + w / 2.0 - x1, np.sin(c) * 150.0 + h / 2.0 - y1
    x4, y4 = x2 - (np.cos(d) * 150.0 + w / 2.0), y2 - (np.sin(d) * 150.0 + h / 2.0)
    r, g, b = hls_to_rgb(np.random.uniform(0.0, 1.0), 0.3, 1.0)
    s = np.random.uniform(5.0, 15.0)
    curves.append([
        x1, y1, x3, y3,
        x2, y2, x4, y4,
        r, g, b, s,
    ])

offset = np.random.uniform(0.0, np.pi * 2.0, (4, instance_count))
curves = np.array(curves, 'f4')

vertex_buffer = ctx.buffer(np.array([x, y, z]).T.astype('f4').tobytes())
instance_buffer = ctx.buffer(curves)
uniform_buffer = ctx.buffer(size=16)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            vec2 screen_size;
        };

        layout (location = 0) in vec3 in_vert;

        layout (location = 1) in vec4 in_a;
        layout (location = 2) in vec4 in_b;
        layout (location = 3) in vec4 in_color_and_size;

        out vec3 v_color;

        void main() {
            float t = in_vert.z;
            vec2 A = in_a.xy;
            vec2 B = in_a.xy + in_a.zw;
            vec2 C = in_b.xy - in_b.zw;
            vec2 D = in_b.xy;
            vec2 E = B - A;
            vec2 F = C - B;
            vec2 G = D - C;
            vec2 H = A + E * t;
            vec2 I = B + F * t;
            vec2 J = C + G * t;
            vec2 K = H + (I - H) * t;
            vec2 L = I + (J - I) * t;
            vec2 M = K + (L - K) * t;
            vec2 N = E + (F - E) * t;
            vec2 O = F + (G - F) * t;
            vec2 P = N + (O - N) * t;
            vec2 n = normalize(P);
            mat2 basis = mat2(n, vec2(-n.y, n.x));
            vec2 vert = M + basis * in_vert.xy * in_color_and_size.w;
            gl_Position = vec4((vert / screen_size) * 2.0 - 1.0, 0.0, 1.0);
            v_color = in_color_and_size.rgb;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
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
    framebuffer=[image],
    topology='triangle_strip',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f', 0),
        *zengl.bind(instance_buffer, '4f 4f 4f /i', 1, 2, 3),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f'),
    instance_count=instance_count,
)

uniform_buffer.write(struct.pack('=ff8x', *window.size))

initial = curves.copy()

while window.update():
    t = window.time
    curves[:, 0] = initial[:, 0] + np.sin(offset[0] + t) * 10.0
    curves[:, 1] = initial[:, 1] + np.cos(offset[0] + t) * 10.0
    curves[:, 2] = initial[:, 2] + np.sin(offset[1] + t) * 50.0
    curves[:, 3] = initial[:, 3] + np.cos(offset[1] + t) * 50.0
    curves[:, 4] = initial[:, 4] + np.sin(offset[2] + t) * 10.0
    curves[:, 5] = initial[:, 5] + np.cos(offset[2] + t) * 10.0
    curves[:, 6] = initial[:, 6] - np.sin(offset[3] + t) * 50.0
    curves[:, 7] = initial[:, 7] - np.cos(offset[3] + t) * 50.0
    ctx.new_frame()
    instance_buffer.write(curves)
    image.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()
