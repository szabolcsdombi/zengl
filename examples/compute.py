import struct

import zengl

from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')

pipeline = ctx.pipeline(
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

        uniform float time;

        layout (location = 0) out vec4 out_color;

        void main() {
            ivec2 at = ivec2(gl_FragCoord.xy);
            float dots = sin(float(at.x) * 0.1) + cos(float(at.y) * 0.1);
            float wave = sin(sqrt(float(at.x * at.x + at.y * at.y)) * 0.01 + time) * 0.5 + 0.5;
            out_color = vec4(dots * wave, wave, 1.0, 1.0) * 0.8 + 0.2;
        }
    ''',
    uniforms={
        'time': 0.0,
    },
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while window.update():
    image.clear()
    pipeline.uniforms['time'][:] = struct.pack('f', window.time)
    pipeline.render()
    image.blit()
