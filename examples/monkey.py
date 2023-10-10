import math

import glwindow
import zengl
from objloader import Obj

import assets


class Monkey:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()

        self.image = self.ctx.image(size, 'rgba8unorm', samples=samples)
        self.depth = self.ctx.image(size, 'depth24plus', samples=samples)

        self.image.clear_value = (0.2, 0.2, 0.2, 1.0)

        model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')
        self.vertex_buffer = self.ctx.buffer(model)
        self.uniform_buffer = self.ctx.buffer(size=80, uniform=True)
        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
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
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec3 v_norm;

                layout (location = 0) out vec4 out_color;

                void main() {
                    vec3 light = vec3(4.0, 3.0, 10.0);
                    float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
                    out_color = vec4(lum, lum, lum, 1.0);
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
            topology='triangles',
            cull_face='back',
            vertex_buffers=zengl.bind(self.vertex_buffer, '3f 3f', 0, 1),
            vertex_count=self.vertex_buffer.size // zengl.calcsize('3f 3f'),
        )

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.sin(self.time) + 1.0, 3.0, 2.0)
        camera = zengl.camera(eye, (0.0, 0.0, 0.5), aspect=self.aspect, fov=45.0)
        self.uniform_buffer.write(camera)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene = Monkey(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
