import zipfile

import glwindow
import numpy as np
import zengl
from PIL import Image

import assets


class Sprites:
    def __init__(self, size, count):
        self.ctx = zengl.context()
        self.size = size

        pack = zipfile.ZipFile(assets.get('kenney_simplespace.zip'))
        files = [x for x in pack.namelist() if x.startswith('PNG/Default/') and x.endswith('.png')]
        pixels = b''.join(Image.open(pack.open(fn)).convert('RGBA').tobytes() for fn in files)
        self.turn = np.random.uniform(-0.002, 0.002, count)

        self.image = self.ctx.image(size, 'rgba8unorm')
        self.output = self.image

        self.texture = self.ctx.image((64, 64), 'rgba8unorm', pixels, array=48)
        self.instance_buffer = self.ctx.buffer(size=count * 16)

        self.instances = np.array([
            np.random.uniform(0.0, size[0], count),
            np.random.uniform(0.0, size[1], count),
            np.random.uniform(0.0, np.pi, count),
            np.random.uniform(2, 48, count),
        ], 'f4').T.copy()

        self.instance_buffer.write(self.instances)

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                #include "screen_size"

                vec2 vertices[4] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2(-1.0, 1.0),
                    vec2(1.0, -1.0),
                    vec2(1.0, 1.0)
                );

                layout (location = 0) in vec4 in_attributes;

                out vec3 v_texcoord;

                void main() {
                    vec2 position = in_attributes.xy;
                    float rotation = in_attributes.z;
                    float texture = in_attributes.w;
                    mat2 rot = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
                    vec2 vertex = position + rot * vertices[gl_VertexID] * 32.0;
                    gl_Position = vec4(vertex / screen_size * 2.0 - 1.0, 0.0, 1.0);
                    v_texcoord = vec3(vertices[gl_VertexID] * 0.5 + 0.5, texture);
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec3 v_texcoord;

                uniform sampler2DArray Texture;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = texture(Texture, v_texcoord);
                    if (out_color.a < 0.05) {
                        discard;
                    }
                }
            ''',
            includes={
                'screen_size': f'const vec2 screen_size = vec2({size[0]}, {size[1]});',
            },
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
                    'image': self.texture,
                    'wrap_x': 'clamp_to_edge',
                    'wrap_y': 'clamp_to_edge',
                },
            ],
            blend={
                'enable': True,
                'src_color': 'src_alpha',
                'dst_color': 'one_minus_src_alpha',
            },
            framebuffer=[self.image],
            topology='triangle_strip',
            vertex_buffers=zengl.bind(self.instance_buffer, '4f /i', 0),
            vertex_count=4,
            instance_count=count,
        )

    def render(self):
        self.instances[:, 0] = (self.instances[:, 0] - np.sin(self.instances[:, 2]) * 0.2) % self.size[0]
        self.instances[:, 1] = (self.instances[:, 1] - np.cos(self.instances[:, 2]) * 0.2) % self.size[1]
        self.instances[:, 2] += self.turn
        self.instance_buffer.write(self.instances)
        self.image.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())
        self.scene = Sprites(self.wnd.size, 100)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
