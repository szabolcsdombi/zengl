import math
import struct

import glwindow
import objloader
import zengl
from PIL import Image

import assets


class Crate:
    def __init__(self, size, samples=4):
        self.ctx = zengl.context()
        self.image = self.ctx.image(size, 'rgba8unorm', samples=samples)
        self.depth = self.ctx.image(size, 'depth24plus', samples=samples)

        model = objloader.Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz tx ty')
        self.vertex_buffer = self.ctx.buffer(model)

        img = Image.open(assets.get('crate.png')).convert('RGBA')
        self.texture = self.ctx.image(img.size, 'rgba8unorm', img.tobytes())

        self.ubo = bytearray(80)
        self.uniform_buffer = self.ctx.buffer(self.ubo)

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec3 light;
                };

                layout (location = 0) in vec3 in_vert;
                layout (location = 1) in vec3 in_norm;
                layout (location = 2) in vec2 in_text;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    v_vert = in_vert;
                    v_norm = in_norm;
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec3 light;
                };

                uniform sampler2D Texture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                layout (location = 0) out vec4 out_color;

                void main() {
                    float lum = clamp(dot(normalize(light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.6 + 0.4;
                    out_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                }
            ''',
            layout=[
                {
                    'name': 'Common',
                    'binding': 0,
                },
                {
                    'name': 'Texture',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.uniform_buffer,
                },
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.texture,
                },
            ],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            cull_face='back',
            vertex_buffers=zengl.bind(self.vertex_buffer, '3f 3f 2f', 0, 1, 2),
            vertex_count=self.vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
        )

        self.aspect = size[0] / size[1]
        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        eye = (math.cos(self.time * 0.6) * 3.0, math.sin(self.time * 0.6) * 3.0, 1.5)
        self.ubo[:64] = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=self.aspect, fov=45.0)
        self.ubo[64:] = struct.pack('3f4x', *eye)
        self.uniform_buffer.write(self.ubo)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene = Crate(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
