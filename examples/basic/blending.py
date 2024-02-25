import array
import struct
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)

triangle = array.array('f', [
    1.0, 0.0, 1.0, 0.0, 0.0, 0.5,
    -0.5, 0.86, 0.0, 1.0, 0.0, 0.5,
    -0.5, -0.86, 0.0, 0.0, 1.0, 0.5,
])

vertex_buffer = ctx.buffer(triangle)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        uniform vec2 scale;
        uniform float rotation;

        layout (location = 0) in vec2 in_vertex;
        layout (location = 1) in vec4 in_color;

        out vec4 v_color;

        void main() {
            float r = rotation * (0.5 + float(gl_InstanceID) * 0.05);
            mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
            gl_Position = vec4((rot * in_vertex) * scale, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec4 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
        }
    ''',
    uniforms={
        'scale': (0.8, 0.8),
        'rotation': 0.0,
    },
    blend={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha',
    },
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '2f 4f', 0, 1),
    vertex_count=3,
    instance_count=10,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    image.clear()
    pipeline.uniforms['rotation'][:] = struct.pack('f', now)
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
