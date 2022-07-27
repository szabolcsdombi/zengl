import ctypes
import struct

import numpy as np
import zengl
from OpenGL import GL
from progress.bar import Bar

from window import Window

window = Window()
ctx = zengl.context()

query = ctypes.c_uint32()
query_result = ctypes.c_uint32()
GL.glGenQueries(1, ctypes.byref(query))

image = ctx.image(window.size, 'rgba8unorm', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

uniform_buffer = ctx.buffer(size=16)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
            vec2 scale;
        };

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.08),
            vec2(-0.06, -0.08),
            vec2(0.06, -0.08)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID] * scale, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
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
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

bar = Bar('Samples Passed:', fill='-')

while window.update():
    image.clear()
    uniform_buffer.write(struct.pack('ff8x', np.sin(window.time), np.cos(window.time)))
    GL.glBeginQuery(GL.GL_SAMPLES_PASSED, query)
    pipeline.render()
    GL.glEndQuery(GL.GL_SAMPLES_PASSED)
    GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT, ctypes.byref(query_result))
    bar.max = max(bar.max, query_result.value)
    bar.goto(query_result.value)
    image.blit()
