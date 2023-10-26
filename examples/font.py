import zipfile
import struct
import string

import glwindow
import zengl

import assets


class FontDemo:
    def __init__(self, size):
        self.ctx = zengl.context()

        self.image = self.ctx.image(size, 'rgba8unorm')
        self.output = self.image

        pack = zipfile.ZipFile(assets.get('Inconsolata.zip'))
        fonts = [
            pack.open('Inconsolata-Regular.ttf').read(),
            pack.open('Inconsolata-Bold.ttf').read(),
        ]
        self.fonts = ['regular', 'bold']
        self.font_sizes = [16.0, 24.0, 32.0]
        code_points = [ord(x) for x in string.printable]

        texture_size = (512, 512)
        pixels, glyphs = glwindow.load_font(texture_size, fonts, self.font_sizes, code_points)

        self.glyph_lookup = {x: i for i, x in enumerate(code_points)}
        self.glyph_struct = struct.Struct('Q3f')
        self.glyphs = glyphs

        self.texture = self.ctx.image(texture_size, 'rgba8unorm', pixels)

        self.instance = struct.Struct('2fQ1I')
        self.instances = bytearray(self.instance.size * 100000)
        self.instance_buffer = self.ctx.buffer(self.instances)
        self.instance_count = 0

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                #include "screen_size"
                #include "texture_size"

                vec2 vertices[4] = vec2[](
                    vec2(0.0, 0.0),
                    vec2(0.0, 1.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0)
                );

                layout (location = 0) in vec2 in_position;
                layout (location = 1) in ivec4 in_bbox;
                layout (location = 2) in vec4 in_color;

                out vec2 v_texcoord;
                out vec4 v_color;

                void main() {
                    v_color = in_color;
                    v_texcoord = mix(vec2(in_bbox.xy), vec2(in_bbox.zw), vertices[gl_VertexID]) / texture_size;
                    vec2 vertex = in_position + vertices[gl_VertexID] * vec2(in_bbox.zw - in_bbox.xy);
                    gl_Position = vec4(vertex / screen_size * 2.0 - 1.0, 0.0, 1.0);
                    gl_Position.y *= -1.0;
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec2 v_texcoord;
                in vec4 v_color;

                uniform sampler2D Texture;

                layout (location = 0) out vec4 out_color;

                void main() {
                    float alpha = texture(Texture, v_texcoord).r;
                    if (alpha < 0.001) {
                        discard;
                    }
                    out_color = vec4(v_color.rgb, v_color.a * alpha);
                }
            ''',
            includes={
                'screen_size': f'const vec2 screen_size = vec2({size[0]}, {size[1]});',
                'texture_size': f'const vec2 texture_size = vec2({texture_size[0]}, {texture_size[1]});',
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
                    'min_filter': 'nearest',
                    'mag_filter': 'nearest',
                },
            ],
            blend={
                'enable': True,
                'src_color': 'src_alpha',
                'dst_color': 'one_minus_src_alpha',
            },
            framebuffer=[self.image],
            topology='triangle_strip',
            vertex_buffers=zengl.bind(self.instance_buffer, '2f 4i2 4nu1 /i', 0, 1, 2),
            vertex_count=4,
        )

    def clear(self):
        self.instance_count = 0

    def text(self, x, y, text, font, size, color):
        col = int(color[5:7] + color[3:5] + color[1:3], 16) | 0xFF000000
        font_index = self.fonts.index(font)
        size_index = self.font_sizes.index(size)
        num_sizes = len(self.font_sizes)
        num_glyphs = len(self.glyph_lookup)
        cursor = x
        for c in text:
            glyph_index = self.glyph_lookup[ord(c)]
            idx = ((font_index * num_sizes + size_index) * num_glyphs + glyph_index) * 28
            bbox, xoff, yoff, xadvance = self.glyph_struct.unpack(self.glyphs[idx:idx + 20])
            at = self.instance_count * self.instance.size
            self.instance.pack_into(self.instances, at, cursor + xoff, y + yoff, bbox, col)
            self.instance_count += 1
            cursor += xadvance

    def render(self):
        instances_size = self.instance_count * self.instance.size
        self.instance_buffer.write(memoryview(self.instances)[:instances_size])
        self.pipeline.instance_count = self.instance_count
        self.image.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()
        self.scene = FontDemo(self.wnd.size)

        self.scene.clear()
        self.scene.text(100.0, 100.0, 'Hello World!', 'regular', 16.0, '#ff0000')
        self.scene.text(100.0, 130.0, 'Hello World!', 'bold', 16.0, '#ff0000')
        self.scene.text(200.0, 100.0, 'Hello World!', 'regular', 24.0, '#00ff00')
        self.scene.text(200.0, 130.0, 'Hello World!', 'bold', 24.0, '#00ff00')
        self.scene.text(350.0, 100.0, 'Hello World!', 'regular', 32.0, '#0000ff')
        self.scene.text(350.0, 130.0, 'Hello World!', 'bold', 32.0, '#0000ff')
        self.scene.text(100.0, 160.0, string.printable, 'regular', 16.0, '#ffffff')

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
