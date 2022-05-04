import zipfile

import numpy as np
import zengl
from PIL import Image, ImageDraw, ImageFont

import assets
from window import Window


def line(x, y, text):
    res = np.zeros((len(text), 3), 'f4')
    res[:, 0] = x + np.arange(len(text)) * 12
    res[:, 1] = y
    res[:, 2] = np.frombuffer(text.encode(), 'u1')
    return res


window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')
image.clear_value = (0.1, 0.1, 0.1, 1.0)

pack = zipfile.ZipFile(assets.get('Inconsolata.zip'))

img = Image.new('RGBA', (32, 32), '#fff')
draw = ImageDraw.Draw(img)
draw.font = ImageFont.truetype(pack.open('Inconsolata-Regular.ttf'), size=24)

pixels = bytearray()
for c in range(256):
    draw.rectangle((0, 0, 32, 32), (0, 0, 0, 0))
    draw.text((1, -2), chr(c), '#fff')
    pixels.extend(img.tobytes('raw', 'RGBA', 0, -1))

font_texture = ctx.image((32, 32), 'rgba8unorm', pixels, array=256)

instance_data = np.concatenate([
    line(100, 100, 'Hello World!'),
    line(400, 400, 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'),
    line(400, 376, 'Aenean commodo purus vel molestie pellentesque.'),
    line(400, 352, 'Curabitur non egestas mi.'),
    line(400, 328, 'Nulla tincidunt suscipit enim id tristique.'),
    line(400, 304, 'Aliquam at massa ultrices magna eleifend mollis vitae ut justo.'),
])

count = len(instance_data)
instance_buffer = ctx.buffer(size=count * 32)

instance_buffer.write(instance_data)

ctx.includes['screen_size'] = f'const vec2 screen_size = vec2({window.width}, {window.height});'
ctx.includes['font_size'] = 'const vec2 font_size = vec2(32.0, 32.0);'

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330

        #include "screen_size"
        #include "font_size"

        vec2 vertices[4] = vec2[](
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0)
        );

        layout (location = 0) in vec4 in_attributes;

        out vec3 v_texcoord;

        void main() {
            vec2 position = in_attributes.xy;
            float texture = in_attributes.z;
            vec2 vertex = position + vertices[gl_VertexID] * font_size;
            gl_Position = vec4(vertex / screen_size * 2.0 - 1.0, 0.0, 1.0);
            v_texcoord = vec3(vertices[gl_VertexID], texture);
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_texcoord;

        uniform sampler2DArray Texture;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = texture(Texture, v_texcoord);
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
            'image': font_texture,
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    blending={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha',
    },
    framebuffer=[image],
    topology='triangle_strip',
    vertex_buffers=zengl.bind(instance_buffer, '3f /i', 0),
    vertex_count=4,
    instance_count=count,
)


while window.update():
    image.clear()
    pipeline.render()
    image.blit()
