import os
import struct

import pygame
import zengl

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

pygame.init()
pygame.display.set_mode((800, 600), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=True)

ctx = zengl.context()

image = ctx.image((640, 360), 'rgba8unorm')
image.write(os.urandom(image.size[0] * image.size[1] * 4))

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec2 vertices[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        out vec2 vertex;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            vertex = vertices[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330 core

        uniform sampler2D Texture;

        uniform vec2 window_size;
        uniform vec2 render_size;

        in vec2 vertex;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec2 scale = window_size / render_size;
            vec2 uv = vertex * scale / min(scale.x, scale.y);
            uv = uv * 0.5 + 0.5;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                out_color = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }
            out_color = texture(Texture, uv);
        }
    ''',
    layout=[
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': image,
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    uniforms={
        'window_size': (0, 0),
        'render_size': image.size,
    },
    framebuffer=None,
    viewport=(0, 0, 0, 0),
    topology='triangles',
    vertex_count=3,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    size = pygame.display.get_window_size()
    pipeline.viewport = (0, 0, size[0], size[1])
    pipeline.uniforms['window_size'][:] = struct.pack('2f', size[0], size[1])

    ctx.new_frame()
    pipeline.render()
    ctx.end_frame()

    pygame.display.flip()

