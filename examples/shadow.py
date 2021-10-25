import numpy as np
import zengl
from objloader import Obj

from window import Window

window = Window(1280, 720)
ctx = zengl.instance(zengl.context())

shadow = ctx.image((512, 512), 'r32float')
depth1 = ctx.image((512, 512), 'depth24plus')
shadow.clear_value = 1.0

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth2 = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open('examples/data/monkey.obj').pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=160)

shadow_program = ctx.renderer(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp_camera;
            mat4 mvp_light;
            vec3 camera;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;

        out float v_depth;

        void main() {
            gl_Position = mvp_light * vec4(in_vert, 1.0);
            v_depth = gl_Position.z / gl_Position.w;
        }
    ''',
    fragment_shader='''
        #version 330

        in float v_depth;

        layout (location = 0) out float out_depth;

        void main() {
            out_depth = v_depth;
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
    framebuffer=[shadow, depth1],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 12x', 0),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 12x'),
)

monkey = ctx.renderer(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp_camera;
            mat4 mvp_light;
            vec3 camera;
            vec3 light;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_vert;
        out vec3 v_norm;

        void main() {
            gl_Position = mvp_camera * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp_camera;
            mat4 mvp_light;
            vec3 camera;
            vec3 light;
        };

        uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec4 temp = mvp_light * vec4(v_vert, 1.0);
            float d1 = texture(Texture, temp.xy * 0.5 + 0.5).r;
            float d2 = temp.z / temp.w;
            float lum = abs(dot(normalize(light - v_vert), normalize(v_norm))) * 0.9 + 0.1;
            if (d1 + 0.005 < d2) {
                lum = 0.1;
            }
            lum = lum * 0.8 + abs(dot(normalize(camera - v_vert), normalize(v_norm))) * 0.2;
            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'sampler',
            'binding': 0,
            'image': shadow,
        },
    ],
    framebuffer=[image, depth2],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

uniform_buffer.write(b''.join([
    zengl.camera((4.0, 3.0, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect),
    zengl.camera((4.0, 3.0, 12.0), (0.0, 0.0, 0.0), fov=0.0, size=3.0, near=1.0, far=16.0),
    np.array([3.0, 2.0, 2.0], 'f4').tobytes(),
    np.array([3.0, 2.0, 4.0], 'f4').tobytes(),
]))

@window.render
def render():
    image.clear()
    shadow.clear()
    depth1.clear()
    depth2.clear()
    shadow_program.render()
    monkey.render()
    image.blit()
    shadow.blit()


window.run()
