import zengl
from objloader import Obj

import assets
from window import Window

'''
    This is the NOT recommanded way to do it.
    Check out uniform buffers and per-instance attributes.

    TODO: add reference to examples using:
        - per object bound chunk of uniform buffer from a single large uniform buffer
        - per object bound per instance vertex attributes from a single larger vertex buffer
'''

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('box.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        uniform mat4 mvp;

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 450 core

        uniform vec3 color;

        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_norm)) * 0.7 + 0.3;
            out_color = vec4(color * lum, 1.0);
        }
    ''',
    uniforms={
        'mvp': [
            [-0.814797, -0.717293, -0.742929, -0.742781],
            [1.086396, -0.537969, -0.557197, -0.557085],
            [0.000000, 2.241540, -0.371464, -0.371390],
            [0.000000, 0.000000, 5.186222, 5.385164],
        ],
        'color': [0.0, 0.5, 1.0],
    },
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

while window.update():
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
