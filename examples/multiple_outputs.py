import zengl

from window import Window

window = Window()
ctx = zengl.context()

image_size = (640, 720)
image1 = ctx.image(image_size, 'rgba8unorm', samples=4)
image2 = ctx.image(image_size, 'rgba8unorm', samples=4)
image1.clear_value = (1.0, 1.0, 1.0, 1.0)
image2.clear_value = (1.0, 1.0, 1.0, 1.0)

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

        layout (location = 0) out vec4 out_color1;
        layout (location = 1) out vec4 out_color2;

        void main() {
            out_color1 = vec4(v_color, 1.0);
            out_color2 = vec4(0.5, 0.5, 0.5, 1.0);
        }
    ''',
    framebuffer=[image1, image2],
    topology='triangles',
    vertex_count=3,
)

while window.update():
    image1.clear()
    image2.clear()
    triangle.run()
    image1.blit(None, (0, 0, 640, 720), flush=False)
    image2.blit(None, (640, 0, 640, 720))
