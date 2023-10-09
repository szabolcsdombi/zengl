import zengl
import glwindow

from monkey import Monkey


def gaussian_kernel(s):
    c = [2.718281828459045 ** (-x * x / (s * s / 4.0)) for x in range(-s, s + 1)]
    v = ', '.join(f'{x / sum(c):.8f}' for x in c)
    return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[]({v});'


class Blur1D:
    def __init__(self, src: zengl.Image, dst: zengl.Image, mode):
        self.ctx = zengl.context()

        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

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
                #version 300 es
                precision highp float;

                uniform int mode;

                uniform sampler2D Texture;

                layout (location = 0) out vec4 out_color;

                #include "kernel"

                void main() {
                    vec3 color = vec3(0.0, 0.0, 0.0);
                    if (mode == 0) {
                        for (int i = 0; i < N; ++i) {
                            ivec2 at = ivec2(gl_FragCoord.xy) + ivec2(i - N / 2, 0);
                            color += texelFetch(Texture, at, 0).rgb * coeff[i];
                        }
                    } else {
                        for (int i = 0; i < N; ++i) {
                            ivec2 at = ivec2(gl_FragCoord.xy) + ivec2(0, i - N / 2);
                            color += texelFetch(Texture, at, 0).rgb * coeff[i];
                        }
                    }
                    out_color = vec4(color, 1.0);
                }
            ''',
            includes={
                'kernel': gaussian_kernel(19),
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
                    'image': src,
                },
            ],
            uniforms={
                'mode': ['x', 'y'].index(mode),
            },
            framebuffer=[dst],
            topology='triangles',
            vertex_count=3,
        )

    def render(self):
        self.pipeline.render()


class Blur2D:
    def __init__(self, src: zengl.Image, dst: zengl.Image):
        self.ctx = zengl.context()

        self.temp = self.ctx.image(src.size, 'rgba8unorm')
        self.blur_x = Blur1D(src, self.temp, 'x')
        self.blur_y = Blur1D(self.temp, dst, 'y')

    def render(self):
        self.blur_x.render()
        self.blur_y.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context(glwindow.get_loader())

        self.scene = Monkey(self.wnd.size, samples=1)
        self.blur = Blur2D(self.scene.image, self.scene.image)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.blur.render()
        self.scene.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
