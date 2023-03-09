import numpy as np
import zengl
from matplotlib import pyplot as plt

ctx = zengl.context(zengl.loader(headless=True))

size = (256, 256)
image = ctx.image(size, 'r32float')

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330 core

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
        #version 330 core

        layout (location = 0) out float out_value;

        void main() {
            vec2 pos = gl_FragCoord.xy;
            out_value = sin(pos.x * 0.07) + sin(pos.y * 0.04);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

ctx.new_frame()
image.clear()
triangle.render()
ctx.end_frame()

plt.imshow(np.frombuffer(image.read(), 'f4').reshape(size))
plt.show()
