import struct

import zengl

from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')

compute = ctx.compute(
    compute_shader='''
        #version 450 core

        layout (local_size_x = 16, local_size_y = 16) in;
        layout(rgba8, binding = 0) writeonly uniform image2D output_image;
        uniform float time;
        void main() {
            ivec2 at = ivec2(gl_GlobalInvocationID.xy);
            float dots = sin(float(at.x) * 0.1) + cos(float(at.y) * 0.1);
            float wave = sin(sqrt(float(at.x * at.x + at.y * at.y)) * 0.01 + time) * 0.5 + 0.5;
            imageStore(output_image, at, vec4(dots * wave, wave, 1.0, 1.0) * 0.8 + 0.2);
        }
    ''',
    resources=[
        {
            'type': 'image',
            'binding': 0,
            'image': image,
        },
    ],
    uniforms={
        'time': 0.0,
    },
    group_count=(80, 45, 1),
)

while window.update():
    compute.uniforms['time'][:] = struct.pack('f', window.time)
    compute.run()
    image.blit()
