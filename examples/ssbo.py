import struct

import zengl

from window import Window
import assets

import numpy as np

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

uniform_buffer = ctx.buffer(size=64)
state_buffer = ctx.buffer(size=1048576)

model = open(assets.get('indirect.mesh'), 'rb').read()
vertex_buffer = ctx.buffer(model)

# vertex_offset = 0 vertex_count - vertex_offset = 564
# vertex_offset = 564 vertex_count - vertex_offset = 1524
# vertex_offset = 2088 vertex_count - vertex_offset = 960
# vertex_offset = 3048 vertex_count - vertex_offset = 11808

indirect_buffer = ctx.buffer(np.array([
    564, 1, 0, 0,
    1524, 1, 564, 1,
    960, 1, 2088, 2,
    11808, 1, 3048, 3,
], 'i4').tobytes())

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 460 core

        struct ObjectState {
            vec4 position;
            vec4 rotation;
            vec4 scale;
            vec4 ambient_color;
            vec4 diffuse_color;
            vec4 specular_color;
            float shininess;
        };

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
        };

        layout(std430, binding = 0) buffer ObjectStateBuffer {
            ObjectState states[];
        };

        vec3 qtransform(vec4 q, vec3 v) {
            return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
        }

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec3 v_color;

        void main() {
            ObjectState state = states[gl_BaseInstance + gl_InstanceID];
            v_vertex = state.position.xyz + qtransform(state.rotation, in_vertex);
            v_normal = qtransform(state.rotation, in_normal);
            v_color = state.diffuse_color.rgb;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(0.0, 0.0, 1.0);
            float lum = dot(normalize(light), normalize(v_normal)) * 0.5 + 0.5;
            out_color = vec4(v_color * lum, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'storage_buffer',
            'binding': 0,
            'buffer': state_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    indirect_buffer=indirect_buffer,
    indirect_count=4,
)

camera = zengl.camera((1.0, -3.0, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
uniform_buffer.write(camera)

state_buffer.write(b''.join(struct.pack(
    '3f4x4f3f4x3f4x3f4x3f4xf12x',
    -1.5 + i, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    i / 4, 1.0 - i / 4, 1.0,
    1.0, 1.0, 1.0,
    16.0,
) for i in range(4)))

while window.update():
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
