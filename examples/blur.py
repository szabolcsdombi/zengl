import zengl

from window import Window

window = Window(1280, 720)
ctx = zengl.instance(zengl.context())

temp = ctx.image(window.size, 'rgba8unorm')
image = ctx.image(window.size, 'rgba8unorm')

triangle = ctx.renderer(
    vertex_shader='''
        #version 330

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
        #version 330

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
        }
    ''',
    framebuffer=[temp],
    topology='triangles',
    vertex_count=3,
)

blur = ctx.renderer(
    vertex_shader='''
        #version 330

        out vec2 v_text;

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        vec2 text[3] = vec2[](
            vec2(0.0, 0.0),
            vec2(2.0, 0.0),
            vec2(0.0, 2.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
            v_text = text[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330

        uniform sampler2D Texture;

        in vec2 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 color = vec3(0.0, 0.0, 0.0);
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    color += texture(Texture, v_text + vec2(i, j) * 0.01).rgb;
                }
            }
            out_color = vec4(color / 25.0, 1.0);
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
            'image': temp,
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)


@window.render
def render():
    temp.clear(1.0, 1.0, 1.0, 1.0)
    triangle.render()
    blur.render()
    temp.blit()
    image.blit()


window.run()
