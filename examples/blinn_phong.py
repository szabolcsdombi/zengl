import struct

import numpy as np
import zengl
from objloader import Obj

from window import Window

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open('examples/data/blob.obj').pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=80)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 eye_pos;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;

        out vec3 v_vertex;
        out vec3 v_normal;

        void main() {
            v_vertex = in_vertex;
            v_normal = in_normal;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 eye_pos;
        };

        in vec3 v_vertex;
        in vec3 v_normal;

        layout (location = 0) out vec4 out_color;

        const vec3 light_pos = vec3(2.0, 3.0, 4.0);
        const vec3 light_color = vec3(1.0, 1.0, 1.0);
        const float light_power = 40.0;
        const vec3 ambient_color = vec3(0.0, 0.01, 0.05);
        const vec3 diffuse_color = vec3(0.0, 0.1, 0.5);
        const vec3 spec_color = vec3(1.0, 1.0, 1.0);
        const float shininess = 16.0;

        const float screen_gamma = 2.2;

        void main() {
            vec3 light_dir = light_pos - v_vertex;
            float light_distance = length(light_dir);
            light_distance = light_distance * light_distance;
            light_dir = normalize(light_dir);

            float lambertian = max(dot(light_dir, v_normal), 0.0);
            float specular = 0.0;

            if (lambertian > 0.0) {
                vec3 view_dir = normalize(eye_pos.xyz - v_vertex);
                vec3 half_dir = normalize(light_dir + view_dir);
                float spec_angle = max(dot(half_dir, v_normal), 0.0);
                specular = pow(spec_angle, shininess);
            }

            vec3 color_linear = ambient_color +
                diffuse_color * lambertian * light_color * light_power / light_distance +
                spec_color * specular * light_color * light_power / light_distance;

            vec3 color_gamma_corrected = pow(color_linear, vec3(1.0 / screen_gamma));
            out_color = vec4(color_gamma_corrected, 1.0);
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

while window.update():
    x, y = np.cos(window.time * 0.5) * 5.0, np.sin(window.time * 0.5) * 5.0
    camera = zengl.camera((x, y, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera + struct.pack('fff4x', x, y, 2.0))

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
