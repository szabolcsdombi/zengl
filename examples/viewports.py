import zengl

from window import Window

window = Window(1280, 720)
ctx = zengl.instance(zengl.context())

window.set_caption('Hello World | Vendor: %s | Renderer: %s | Version: %s' % ctx.info)

image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

triangle = ctx.pipeline(
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
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)


@window.render
def render():
    image.clear()
    triangle.viewport = (0, 0, 640, 360)
    triangle.render()
    triangle.viewport = (0, 360, 640, 360)
    triangle.render()
    triangle.viewport = (640, 0, 640, 360)
    triangle.render()
    triangle.viewport = (640, 360, 640, 360)
    triangle.render()
    image.blit()


window.run()
