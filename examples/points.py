from colorsys import hls_to_rgb

import numpy as np
import zengl

from window import Window


class ParticleSystem:
    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.position = np.zeros((N, 2), 'f4')
        self.velocity = np.zeros((N, 2), 'f4')
        self.position[:, 1] = -1.0
        self.sweep = 0

    def update(self):
        self.position[self.sweep:self.sweep + self.K] = 0.0
        self.velocity[self.sweep:self.sweep + self.K] = np.random.normal((0.003, 0.01), 0.001, (self.K, 2))
        self.sweep = (self.sweep + self.K) % self.N
        self.position += self.velocity
        self.velocity -= (0.0, 0.0001)

    def get_buffer(self):
        return self.position.astype('f4').tobytes()


window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)


ps = ParticleSystem(10000, 100)
vertex_buffer = ctx.buffer(size=ps.N * 8)

color_buffer = ctx.buffer(np.array([
    hls_to_rgb(np.random.uniform(0.0, 1.0), 0.5, 0.5)
    for _ in range(ps.N)
]).astype('f4').tobytes())

triangle = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (location = 0) in vec2 in_vert;
        layout (location = 1) in vec3 in_color;

        out vec3 v_color;

        void main() {
            gl_PointSize = 3.0;
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 0.7);
        }
    ''',
    blend=[
        {
            'enable': True,
            'src_color': 'src_alpha',
            'dst_color': 'one_minus_src_alpha',
        },
    ],
    framebuffer=[image],
    topology='points',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '2f', 0),
        *zengl.bind(color_buffer, '3f', 1),
    ],
    vertex_count=ps.N,
)

while window.update():
    image.clear()
    ps.update()
    vertex_buffer.write(ps.get_buffer())
    triangle.run()
    image.blit()
