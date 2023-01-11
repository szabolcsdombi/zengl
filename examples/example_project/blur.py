import numpy as np
import zengl

from context import Context


def kernel(s):
    x = np.arange(-s, s + 1)
    y = np.exp(-x * x / (s * s / 4))
    y /= y.sum()
    v = ', '.join(f'{t:.8f}' for t in y)
    return f'const int N = {s * 2 + 1};\nfloat coeff[N] = float[]({v});'


class Blur:
    def __init__(self, image: zengl.Image, format: str = 'rgba8unorm', kernel_size: int = 19):
        ctx = Context.context
        self.input_image = image
        self.temp_image = ctx.image(image.size, format)
        self.output_image = ctx.image(image.size, format)

        ctx.includes['blur_kernel'] = kernel(kernel_size)

        self.blur_x = ctx.pipeline(
            vertex_shader='''
                #version 450 core

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
                #version 450 core

                layout (binding = 0) uniform sampler2D Texture;

                layout (location = 0) out vec4 out_color;

                #include "blur_kernel"

                void main() {
                    vec3 color = vec3(0.0, 0.0, 0.0);
                    for (int i = 0; i < N; ++i) {
                        color += texelFetch(Texture, ivec2(gl_FragCoord.xy) + ivec2(i - N / 2, 0), 0).rgb * coeff[i];
                    }
                    out_color = vec4(color, 1.0);
                }
            ''',
            resources=[
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.input_image,
                },
            ],
            framebuffer=[self.temp_image],
            topology='triangles',
            vertex_count=3,
        )

        self.blur_y = ctx.pipeline(
            vertex_shader='''
                #version 450 core

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
                #version 450 core

                layout (binding = 0) uniform sampler2D Texture;

                layout (location = 0) out vec4 out_color;

                #include "blur_kernel"

                void main() {
                    vec3 color = vec3(0.0, 0.0, 0.0);
                    for (int i = 0; i < N; ++i) {
                        color += texelFetch(Texture, ivec2(gl_FragCoord.xy) + ivec2(0, i - N / 2), 0).rgb * coeff[i];
                    }
                    out_color = vec4(color, 1.0);
                }
            ''',
            resources=[
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.temp_image,
                },
            ],
            framebuffer=[self.output_image],
            topology='triangles',
            vertex_count=3,
        )

    def render(self):
        self.blur_x.render()
        self.blur_y.render()
