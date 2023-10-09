import zengl
import glwindow
from objloader import Obj

import assets

glwindow.init()


def gaussian_kernel(s):
    c = [2.718281828459045 ** (-x * x / (s * s / 4.0)) for x in range(-s, s + 1)]
    v = ', '.join(f'{x / sum(c):.8f}' for x in c)
    return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[]({v});'


class Blur1D:
    def __init__(self, ctx, src, dst, mode):
        self.pipeline = ctx.pipeline(
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
    def __init__(self, ctx, src, dst):
        self.temp = ctx.image(src.size, 'rgba8unorm')
        self.blur_x = Blur1D(ctx, src, self.temp, 'x')
        self.blur_y = Blur1D(ctx, self.temp, dst, 'y')

    def render(self):
        self.blur_x.render()
        self.blur_y.render()


class Scene:
    def __init__(self, ctx):
        self.ctx = ctx
        self.wnd = glwindow.get_window()
        self.image = self.ctx.image(self.wnd.size, 'rgba8unorm')
        self.depth = self.ctx.image(self.wnd.size, 'depth24plus')

        self.image.clear_value = (0.2, 0.2, 0.2, 1.0)

        model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')
        self.vertex_buffer = self.ctx.buffer(model)
        self.uniform_buffer = self.ctx.buffer(size=80)
        self.pipeline = self.ctx.pipeline(
            vertex_shader='''
                #version 300 es
                precision highp float;

                layout (std140) uniform Common {
                    mat4 mvp;
                };

                layout (location = 0) in vec3 in_vert;
                layout (location = 1) in vec3 in_norm;

                out vec3 v_norm;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    v_norm = in_norm;
                }
            ''',
            fragment_shader='''
                #version 300 es
                precision highp float;

                in vec3 v_norm;

                layout (location = 0) out vec4 out_color;

                void main() {
                    vec3 light = vec3(4.0, 3.0, 10.0);
                    float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
                    out_color = vec4(lum, lum, lum, 1.0);
                }
            ''',
            layout=[
                {
                    'name': 'Common',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.uniform_buffer,
                },
            ],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            cull_face='back',
            vertex_buffers=zengl.bind(self.vertex_buffer, '3f 3f', 0, 1),
            vertex_count=self.vertex_buffer.size // zengl.calcsize('3f 3f'),
        )

    def render(self):
        camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=self.wnd.aspect_ratio, fov=45.0)
        self.uniform_buffer.write(camera)
        self.image.clear()
        self.depth.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()

        self.scene = Scene(self.ctx)
        self.blur = Blur2D(self.ctx, self.scene.image, self.scene.image)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.blur.render()
        self.scene.image.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(app=App())
