import os

import pygame
import zengl
from meshtools import obj
from OpenGL import GL
from zengl_extras import assets

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def load_model(name):
    ctx = zengl.context()
    with open(assets.get(name)) as f:
        model = obj.parse_obj(f.read(), 'vn')
    return ctx.buffer(model)


size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

vertex_buffer = load_model('monkey.obj')
uniform_buffer = ctx.buffer(size=64)

pipeline = ctx.pipeline(
    vertex_shader="""
        #version 300 es
        precision highp float;

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
    """,
    fragment_shader="""
        #version 300 es
        precision highp float;

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    """,
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
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=1.777, fov=45.0)
uniform_buffer.write(camera)

space_down = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key == pygame.K_SPACE:
            space_down = event.type == pygame.KEYDOWN

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    image.clear()
    depth.clear()

    if space_down:
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
    else:
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
