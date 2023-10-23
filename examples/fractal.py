import math
import struct

import glwindow
import zengl


class Fractal:
    def __init__(self, size):
        self.ctx = zengl.context()
        self.size = size

        self.image = self.ctx.image(size, 'rgba8unorm')
        self.output = self.image

        self.uniform_buffer = self.ctx.buffer(size=32)
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

                layout (std140) uniform Common {
                    vec2 size;
                    vec2 center;
                    int iter;
                };

                layout (location = 0) out vec4 out_color;

                void main() {
                    vec2 z = vec2(5.0, 3.0) * (gl_FragCoord.xy / size - 0.5);
                    vec2 c = center;
                    int i;
                    for (i = 0; i < iter; ++i) {
                        vec2 v = vec2(
                            (z.x * z.x - z.y * z.y) + c.x,
                            (z.y * z.x + z.x * z.y) + c.y
                        );
                        if (dot(v, v) > 4.0) break;
                        z = v;
                    }
                    float cm = fract((i == iter ? 0.0 : float(i)) * 10.0 / float(iter));
                    out_color = vec4(
                        fract(cm + 0.0 / 3.0),
                        fract(cm + 1.0 / 3.0),
                        fract(cm + 2.0 / 3.0),
                        1.0
                    );
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
            framebuffer=[self.image],
            topology='triangles',
            vertex_count=3,
        )

        self.time = 0.0

    def render(self):
        self.time += 1.0 / 60.0
        center = 0.25 + math.cos(self.time * 1.3) * 0.04, 0.55 + math.sin(self.time * 1.3) * 0.04
        self.uniform_buffer.write(struct.pack('2f2fi', *self.size, *center, 100))
        self.image.clear()
        self.pipeline.render()


class App:
    def __init__(self):
        self.wnd = glwindow.get_window()
        self.ctx = zengl.context()
        self.scene = Fractal(self.wnd.size)

    def update(self):
        self.ctx.new_frame()
        self.scene.render()
        self.scene.output.blit()
        self.ctx.end_frame()


if __name__ == '__main__':
    glwindow.run(App)
