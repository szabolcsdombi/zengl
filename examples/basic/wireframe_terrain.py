import sys

import numpy as np
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


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


size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

vertex_data, index_data = create_terrain(64)
vertex_buffer = ctx.buffer(vertex_data)
index_buffer = ctx.buffer(index_data, index=True)
uniform_buffer = ctx.buffer(size=64, uniform=True)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

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
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='line_strip',
    vertex_buffers=zengl.bind(vertex_buffer, '3f', 0),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    eye = (np.cos(now * 0.3) * 3.0, np.sin(now * 0.3) * 3.0, 1.5)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=1.777, fov=45.0)
    uniform_buffer.write(camera)
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
