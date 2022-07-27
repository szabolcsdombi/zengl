import zipfile

import numpy as np
import zengl
from PIL import Image

import assets
from window import Window

pack = zipfile.ZipFile(assets.get('kenney_simplespace.zip'))
files = [x for x in pack.namelist() if x.startswith('PNG/Default/') and x.endswith('.png')]
pixels = b''.join(Image.open(pack.open(fn)).convert('RGBA').tobytes() for fn in files)

count = 100

window = Window()
ctx = zengl.context()

texture = ctx.image((64, 64), 'rgba8unorm', pixels, array=48)
image = ctx.image(window.size, 'rgba8unorm')
image.clear_value = (0.1, 0.1, 0.1, 1.0)

instance_buffer = ctx.buffer(size=count * 16)

instance_data = np.array([
    np.random.uniform(0.0, window.size[0], count),
    np.random.uniform(0.0, window.size[1], count),
    np.random.uniform(0.0, np.pi, count),
    np.random.uniform(2, 48, count),
], 'f4').T.copy()

instance_buffer.write(instance_data)

width, height = window.size
ctx.includes['screen_size'] = f'const vec2 screen_size = vec2({width}, {height});'

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330

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
        #version 330

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
            'image': texture,
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
        },
    ],
    blending={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha',
    },
    framebuffer=[image],
    topology='triangle_strip',
    vertex_buffers=zengl.bind(instance_buffer, '4f /i', 0),
    vertex_count=4,
    instance_count=count,
)

turn = np.random.uniform(-0.002, 0.002, count)

while window.update():
    instance_data[:, 0] = (instance_data[:, 0] - np.sin(instance_data[:, 2]) * 0.2) % window.size[0]
    instance_data[:, 1] = (instance_data[:, 1] - np.cos(instance_data[:, 2]) * 0.2) % window.size[1]
    instance_data[:, 2] += turn
    instance_buffer.write(instance_data)
    image.clear()
    triangle.render()
    image.blit()
