# pip install https://github.com/szabolcsdombi/layered-window/archive/refs/heads/main.zip
import array
import struct
import time

import layered_window
import zengl
import zengl_extras

zengl_extras.init()
zengl.init(zengl.loader(headless=True))
ctx = zengl.context()

image = ctx.image((400, 400), 'rgba8unorm')

triangle = array.array('f', [
    1.0, 0.0, 1.0, 0.0, 0.0, 0.5,
    -0.5, 0.86, 0.0, 1.0, 0.0, 0.5,
    -0.5, -0.86, 0.0, 0.0, 1.0, 0.5,
])

vertex_buffer = ctx.buffer(triangle)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        uniform vec2 scale;
        uniform float rotation;

        layout (location = 0) in vec2 in_vertex;
        layout (location = 1) in vec4 in_color;

        out vec4 v_color;

        void main() {
            float r = rotation * (0.5 + float(gl_InstanceID) * 0.05);
            mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
            gl_Position = vec4((rot * in_vertex) * scale, 0.0, 1.0);
            v_color = in_color;
        }
    ''',
    fragment_shader='''
        #version 330 core

        in vec4 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
        }
    ''',
    uniforms={
        'scale': (0.8, 0.8),
        'rotation': 0.0,
    },
    blend={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha',
        'src_alpha': 'src_alpha',
        'dst_alpha': 'one',
    },
    framebuffer=[image],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '2f 4f', 0, 1),
    vertex_count=3,
    instance_count=10,
)

mem = layered_window.init((400, 400), title='Animation', always_on_top=True, tool_window=True)
start_time = time.perf_counter()

while True:
    now = time.perf_counter() - start_time
    ctx.new_frame()
    image.clear()
    pipeline.uniforms['rotation'][:] = struct.pack('f', now)
    pipeline.render()
    image.blit()
    ctx.end_frame()
    image.read(into=mem)

    layered_window.update()
