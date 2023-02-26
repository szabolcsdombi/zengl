import zengl
from PIL import Image

ctx = zengl.context(zengl.loader(headless=True))

size = (256, 256)
image = ctx.image(size, 'rgba8unorm', samples=1)

triangle = ctx.pipeline(
    vertex_shader='''
        #version 450 core

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
        #version 450 core

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

ctx.new_frame()
image.clear_value = (1.0, 1.0, 1.0, 1.0)
image.clear()
triangle.render()
ctx.end_frame()

Image.frombuffer('RGBA', size, image.read(), 'raw', 'RGBA', 0, -1).save('hello.png')
