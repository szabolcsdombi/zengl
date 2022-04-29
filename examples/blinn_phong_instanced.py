import struct
from colorsys import hls_to_rgb

import numpy as np
import vmath
import zengl
from objloader import Obj

import assets
from window import Window

window = Window(1280, 720)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

model = Obj.open(assets.get('blob.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)

instance_count = 100
instance_buffer = ctx.buffer(size=instance_count * 68)

uniform_buffer = ctx.buffer(size=80)
light_uniform_buffer = ctx.buffer(size=48)

ctx.includes['qtransform'] = '''
    vec3 qtransform(vec4 q, vec3 v) {
        return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
    }
'''

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330

        #include "qtransform"

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 eye_pos;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;

        layout (location = 2) in vec3 in_position;
        layout (location = 3) in vec4 in_rotation;

        layout (location = 4) in vec3 in_ambient_color;
        layout (location = 5) in vec3 in_diffuse_color;
        layout (location = 6) in vec3 in_spec_color;
        layout (location = 7) in float in_shininess;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec3 v_ambient_color;
        out vec3 v_diffuse_color;
        out vec3 v_spec_color;
        out float v_shininess;

        void main() {
            v_vertex = in_position + qtransform(in_rotation, in_vertex);
            v_normal = qtransform(in_rotation, in_normal);
            v_ambient_color = in_ambient_color;
            v_diffuse_color = in_diffuse_color;
            v_spec_color = in_spec_color;
            v_shininess = in_shininess;
            gl_Position = mvp * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 eye_pos;
        };

        layout (std140) uniform Light {
            vec4 light_pos;
            vec4 light_color;
            float light_power;
        };

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec3 v_ambient_color;
        in vec3 v_diffuse_color;
        in vec3 v_spec_color;
        in float v_shininess;

        layout (location = 0) out vec4 out_color;

        const float screen_gamma = 2.2;

        void main() {
            vec3 light_dir = light_pos.xyz - v_vertex;
            float light_distance = length(light_dir);
            light_distance = light_distance * light_distance;
            light_dir = normalize(light_dir);

            float lambertian = max(dot(light_dir, v_normal), 0.0);
            float specular = 0.0;

            if (lambertian > 0.0) {
                vec3 view_dir = normalize(eye_pos.xyz - v_vertex);
                vec3 half_dir = normalize(light_dir + view_dir);
                float spec_angle = max(dot(half_dir, v_normal), 0.0);
                specular = pow(spec_angle, v_shininess);
            }

            vec3 color_linear = v_ambient_color +
                v_diffuse_color * lambertian * light_color.rgb * light_power / light_distance +
                v_spec_color * specular * light_color.rgb * light_power / light_distance;

            vec3 color_gamma_corrected = pow(color_linear, vec3(1.0 / screen_gamma));
            out_color = vec4(color_gamma_corrected, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Light',
            'binding': 1,
        },
    ],
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'uniform_buffer',
            'binding': 1,
            'buffer': light_uniform_buffer,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=[
        *zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        *zengl.bind(instance_buffer, '3f 4f 3f 3f 3f 1f /i', 2, 3, 4, 5, 6, 7),
    ],
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    instance_count=instance_count,
)

light_pos = (2.0, 3.0, 4.0)
light_color = (1.0, 1.0, 1.0)
light_power = 40.0

light_uniform_buffer.write(struct.pack('=3f4x3f4xf12x', *light_pos, *light_color, light_power))

instances = np.zeros((instance_count, 17), 'f4')

for i in range(instance_count):
    ambient_color = (0.0, 0.01, 0.05)
    diffuse_color = (0.0, 0.1, 0.5)
    spec_color = (1.0, 1.0, 1.0)
    shininess = 16.0

    instances[i][0:3] = (i % 10 * 4.0 - 18.0, i // 10 * 4.0, 0.0)
    instances[i][3:7] = vmath.random_rotation()
    instances[i][7:10] = ambient_color
    instances[i][10:13] = hls_to_rgb(np.random.uniform(0.0, 1.0), 0.5, 0.5)
    instances[i][13:16] = spec_color
    instances[i][16] = shininess

instance_buffer.write(instances)

while window.update():
    camera = zengl.camera((0.0, -15.0, 10.0), (0.0, 10.0, 0.0), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera + struct.pack('=3f4x', 0.0, -15.0, 10.0))

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
