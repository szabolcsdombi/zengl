import math

import glwindow
import numpy as np
import zengl


def create_terrain(size):
    V = np.random.normal(0.0, 0.005, size * size)

    for _ in range(128):
        k = np.random.randint(5, 15)
        x, y = np.random.randint(0, size, 2)
        sx, ex, sy, ey = np.clip([x - k, x + k + 1, y - k, y + k + 1], 0, size)
        IX, IY = np.meshgrid(np.arange(sx, ex), np.arange(sy, ey))
        D = np.clip(np.sqrt(np.square((IX - x) / k) + np.square((IY - y) / k)), 0.0, 1.0)
        V[IX * size + IY] += np.cos(D * np.pi / 2.0) * k * np.random.normal(0.0, 0.005)

    X, Y = np.meshgrid(np.linspace(-1.0, 1.0, size), np.linspace(-1.0, 1.0, size))
    P = np.array([X.flatten(), Y.flatten(), V]).T
    A, B = np.meshgrid(np.arange(size + 1), np.arange(size))
    Q = np.concatenate([A + B * size, A * size + B])
    Q[:, -1] = -1

    return P.astype('f4').tobytes(), Q.astype('i4').tobytes()


class WireframeTerrain:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', samples=samples)
        self.depth = self.ctx.image(size, 'depth24plus', samples=samples)

        vertex_data, index_data = create_terrain(64)
        self.vertex_buffer = self.ctx.buffer(vertex_data)
        self.index_buffer = self.ctx.buffer(index_data, index=True)
        self.uniform_buffer = self.ctx.buffer(size=64, uniform=True)

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                };

                layout (location = 0) in vec3 in_vert;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(1.0, 1.0, 1.0, 1.0);
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
                    'buffer': self.uniform_buffer,
                },
            ],
            framebuffer=[self.image, self.depth],
            topology='line_strip',
            vertex_buffers=zengl.bind(self.vertex_buffer, '3f', 0),
            index_buffer=self.index_buffer,
            vertex_count=self.index_buffer.size // 4,
        )

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time * 0.3) * 3.0, math.sin(self.time * 0.3) * 3.0, 1.5)
        camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=self.aspect, fov=45.0)
        self.uniform_buffer.write(camera)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene = WireframeTerrain(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
