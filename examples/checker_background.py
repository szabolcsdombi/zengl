import zengl

from window import Window

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)

background = ctx.pipeline(
    vertex_shader='''
        #version 330

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        layout (location = 0) out vec4 out_color;

        void main() {
            float gray = 0.7 + float((int(gl_FragCoord.x) / 40 + int(gl_FragCoord.y) / 40) % 2) * 0.2;
            out_color = vec4(gray, gray, gray, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while window.update():
    image.clear()
    background.render()
    image.blit()
