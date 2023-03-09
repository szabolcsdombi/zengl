import math

import zengl
from objloader import Obj

import assets
from window import Window

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus-stencil8', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)
depth.clear_value = (1.0, 1)

model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')[:-124 * 3 * 24]
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=80)

monkey = ctx.pipeline(
    vertex_shader='''
        #version 330 core

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
        #version 330 core

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
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

monkey_reflection = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_norm;

        void main() {
            gl_Position = mvp * vec4(in_vert * vec3(1.0, 1.0, -1.0), 1.0);
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330 core

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
            'buffer': uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='front',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

monkey_shadow = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;

        void main() {
            vec3 vert = vec3(in_vert.xy + vec2(-0.7, -0.4) * in_vert.z, 0.0);
            gl_Position = mvp * vec4(vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (location = 0) out vec4 out_color;

        void main() {
            gl_FragDepth = gl_FragCoord.z - 1e-4;
            out_color = vec4(0.0, 0.0, 0.0, 0.3);
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
    blend=[
        {
            'enable': True,
            'src_color': 'src_alpha',
            'dst_color': 'one_minus_src_alpha',
            'src_alpha': 'one',
            'dst_alpha': 'zero',
        },
    ],
    stencil={
        'test': True,
        'both': {
            'fail_op': 'zero',
            'pass_op': 'zero',
            'depth_fail_op': 'zero',
            'compare_op': 'equal',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
    },
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, -1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

plane = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        vec3 positions[4] = vec3[](
            vec3(-3.0, -3.0, 0.0),
            vec3(-3.0, 3.0, 0.0),
            vec3(3.0, -3.0, 0.0),
            vec3(3.0, 3.0, 0.0)
        );

        void main() {
            gl_Position = mvp * vec4(positions[gl_VertexID], 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(1.0, 1.0, 1.0, 0.7);
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
    blend=[
        {
            'enable': True,
            'src_color': 'src_alpha',
            'dst_color': 'one_minus_src_alpha',
            'src_alpha': 'one',
            'dst_alpha': 'zero',
        },
    ],
    framebuffer=[image, depth],
    topology='triangle_strip',
    vertex_count=4,
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    ctx.new_frame()
    x, y = math.sin(window.time * 0.5) * 5.0, math.cos(window.time * 0.5) * 5.0
    camera = zengl.camera((x, y, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    monkey.render()
    monkey_reflection.render()
    monkey_shadow.render()
    plane.render()
    image.blit()
    ctx.end_frame()
