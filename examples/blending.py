import glwindow
import numpy as np
import zengl


class Blending:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()

        self.image = self.ctx.image(self.wnd.size, 'rgba8unorm-srgb', samples=4)
        self.depth = self.ctx.image(self.wnd.size, 'depth24plus', samples=4)
        self.image.clear_value = (0.0, 0.0, 0.0, 1.0)

        self.uniform_buffer = self.ctx.buffer(size=16)

        self.vertex_buffer = self.ctx.buffer(np.array([
            1.0, 0.0,
            1.0, 0.0, 0.0, 0.5,

            -0.5, 0.86,
            0.0, 1.0, 0.0, 0.5,

            -0.5, -0.86,
            0.0, 0.0, 1.0, 0.5,
        ], 'f4'))

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    vec2 scale;
                    float rotation;
                };

                layout (location = 0) in vec2 in_vert;
                layout (location = 1) in vec4 in_color;

                out vec4 v_color;

                void main() {
                    float r = rotation * (0.5 + float(gl_InstanceID) * 0.05);
                    mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
                    gl_Position = vec4((rot * in_vert) * scale, 0.0, 1.0);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec4 v_color;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(v_color);
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
            blend={
                'enable': True,
                'src_color': 'src_alpha',
                'dst_color': 'one_minus_src_alpha',
            },
            framebuffer=[self.image],
            topology='triangles',
            vertex_buffers=zengl.bind(self.vertex_buffer, '2f 4f', 0, 1),
            vertex_count=3,
            instance_count=10,
        )

        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        self.image.clear()
        self.uniform_buffer.write(np.array([0.5, 0.5 * self.wnd.aspect_ratio, self.time, 0.0], 'f4'))
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.logo = Blending()

    def update(self):
        self.ctx.new_frame()
        self.logo.render()
        self.logo.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
