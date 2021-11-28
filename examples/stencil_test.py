import zengl
from objloader import Obj

from window import Window

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth_stencil = ctx.image(window.size, 'depth24plus-stencil8', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)
depth_stencil.clear_value = (1.0, 0)

model = Obj.open('examples/data/monkey.obj').pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=80)

triangle = ctx.pipeline(
    vertex_shader='''
        #version 330

        vec2 positions[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.6, -0.8),
            vec2(0.6, -0.8)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        void main() {
        }
    ''',
    stencil={
        'test': True,
        'front': {
            'fail_op': 'replace',
            'pass_op': 'replace',
            'depth_fail_op': 'replace',
            'compare_op': 'always',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
        'back': {
            'fail_op': 'replace',
            'pass_op': 'replace',
            'depth_fail_op': 'replace',
            'compare_op': 'always',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
    },
    depth=False,
    framebuffer=[depth_stencil],
    topology='triangles',
    vertex_count=3,
)

monkey = ctx.pipeline(
    vertex_shader='''
        #version 330

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
        #version 330

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
    stencil={
        'test': True,
        'front': {
            'fail_op': 'keep',
            'pass_op': 'keep',
            'depth_fail_op': 'keep',
            'compare_op': 'equal',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
        'back': {
            'fail_op': 'keep',
            'pass_op': 'keep',
            'depth_fail_op': 'keep',
            'compare_op': 'equal',
            'compare_mask': 1,
            'write_mask': 1,
            'reference': 1,
        },
    },
    framebuffer=[image, depth_stencil],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

while window.update():
    depth_stencil.clear()
    image.clear()
    triangle.render()
    monkey.render()
    image.blit()
