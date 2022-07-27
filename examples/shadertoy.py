import struct

import zengl

from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm')
uniform_buffer = ctx.buffer(size=64)

# Tested with:
# Happy Jumping - https://www.shadertoy.com/view/3lsSzf
# Raymarching - Primitives - https://www.shadertoy.com/view/Xds3zN
# GLSL ray tracing test - https://www.shadertoy.com/view/3sc3z4
# Ray Marching: Part 6 - https://www.shadertoy.com/view/4tcGDr
# Seascape - https://www.shadertoy.com/view/Ms2SD1
# Mandelbulb - https://www.shadertoy.com/view/MdXSWn

# Paste your code below

shadertoy = '''
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}
'''

ctx.includes['shadertoy'] = shadertoy
ctx.includes['uniforms'] = '''
    layout (std140) uniform Uniforms {
        vec3 iResolution;
        float iTime;
        float iTimeDelta;
        int iFrame;
        vec4 iMouse;
        vec4 iDate;
    };
'''

canvas = ctx.pipeline(
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

        #include "uniforms"
        #include "shadertoy"

        layout (location = 0) out vec4 shader_color_output;

        void main() {
            mainImage(shader_color_output, gl_FragCoord.xy);
        }
    ''',
    layout=[
        {
            'name': 'Uniforms',
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

ubo = struct.Struct('=3f1f1f1i8x4f4f')
last_time = window.time
frame = 0

while window.update():
    image.clear()
    uniform_buffer.write(ubo.pack(
        window.size[0], window.size[1], 0.0,
        window.time,
        window.time - last_time,
        frame,
        window.mouse[0], window.mouse[1], 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ))
    canvas.render()
    image.blit()
    frame += 1
