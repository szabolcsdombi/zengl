import os
import struct

import pygame
import zengl

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)
uniform_buffer = ctx.buffer(size=80, uniform=True)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
        };

        vec3 vertices[8] = vec3[](
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, -0.5,  0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, 0.5)
        );

        int indices[24] = int[](
            0, 1, 1, 3, 3, 2, 2, 0,
            4, 5, 5, 7, 7, 6, 6, 4,
            0, 4, 1, 5, 2, 6, 3, 7
        );

        out vec3 v_vertex;

        void main() {
            gl_Position = camera_matrix * vec4(vertices[indices[gl_VertexID]], 1.0);
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
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='lines',
    vertex_count=24,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    image.clear()
    depth.clear()
    camera_position = (4.0, 3.0, 2.0)
    camera = zengl.camera(camera_position, aspect=1.0, fov=45.0)
    uniform_buffer.write(struct.pack('64s3f4x', camera, *camera_position))
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
