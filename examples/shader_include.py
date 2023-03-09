import numpy as np
import zengl

from window import Window

window = Window()
ctx = zengl.context()
image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

ctx.includes['vertex_attributes'] = '''
    layout (location = 0) in vec2 in_vert;
    layout (location = 1) in vec3 in_color;
'''

vertex_buffer = ctx.buffer(np.array([
    0.0, 0.8, 1.0, 0.0, 0.0,
    -0.6, -0.8, 0.0, 1.0, 0.0,
    0.6, -0.8, 0.0, 0.0, 1.0,
], 'f4'))

triangle = ctx.pipeline(
    # includes={
    #     'vertex_attributes': '''
    #         layout (location = 0) in vec2 in_vert;
    #         layout (location = 1) in vec3 in_color;
    #     ''',
    # },
    vertex_shader='''
        #version 330 core

        #include "vertex_attributes"

        out vec3 v_color;

        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '2f 3f', 0, 1),
    vertex_count=3,
)

while window.update():
    ctx.new_frame()
    image.clear()
    triangle.render()
    image.blit()
    ctx.end_frame()
