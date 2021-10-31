import numpy as np
import zengl

from window import Window

window = Window(1280, 720)
ctx = zengl.context(zengl.loader())
image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

vertex_buffer = ctx.buffer(np.array([
    0.0, 0.08, 1.0, 0.0, 0.0,
    -0.06, -0.08, 0.0, 1.0, 0.0,
    0.06, -0.08, 0.0, 0.0, 1.0,
], 'f4'))

angle = np.linspace(0.0, np.pi * 2.0, 32)
xy = np.array([np.cos(angle) * 0.42, np.sin(angle) * 0.75])

instance_buffer = ctx.buffer(xy.T.astype('f4').tobytes())

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (location = 0) in vec2 in_vert;
        layout (location = 1) in vec3 in_color;

        layout (location = 2) in vec2 in_pos;

        out vec3 v_color;

        void main() {
            gl_Position = vec4(in_pos + in_vert, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '2f 3f', 0, 1),
        *zengl.bind(instance_buffer, '2f /i', 2),
    ],
    vertex_count=3,
    instance_count=32,
)


@window.render
def render():
    image.clear()
    triangle.render()
    image.blit()


window.run()
