import os

import zengl
from PIL import Image

zengl.init(zengl.loader(headless=True))

ctx = zengl.context()

image = ctx.image((1280, 720), 'rgba8unorm', samples=4)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        out vec3 v_color;

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(pow(v_color, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

ctx.new_frame()
image.clear()
pipeline.render()
ctx.end_frame()

img = Image.frombuffer('RGBA', image.size, image.read(), 'raw', 'RGBA', 0, -1)
img.save('hello.png')
print('Created:', os.path.abspath('hello.png'))
