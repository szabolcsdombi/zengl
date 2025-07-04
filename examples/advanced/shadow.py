import struct
import sys

import assets
import pygame
import zengl
import zengl_extras
from objloader import Obj

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.03, 0.03, 0.03, 1.0)

ctx.includes['ubo'] = '''
    layout (std140) uniform Common {
        mat4 mvp;
        vec4 eye;
        mat4 light_matrix[3];
        vec4 light_position[3];
        vec4 light_color[3];
    };
'''


def create_shadow_framebuffer(size):
    image = ctx.image(size, 'r32float')
    depth = ctx.image(size, 'depth24plus')
    image.clear_value = 1e6
    return [image, depth]


def create_shadow_pipeline(light_index, vertex_buffer, uniform_buffer, framebuffer):
    ctx.includes['light_index'] = f'const int light_index = {light_index};'
    return ctx.pipeline(
        vertex_shader='''
            #version 300 es
            precision highp float;

            #include "ubo"
            #include "light_index"

            layout (location = 0) in vec3 in_vertex;

            out vec3 v_vertex;

            void main() {
                v_vertex = in_vertex;
                gl_Position = light_matrix[light_index] * vec4(v_vertex, 1.0);
            }
        ''',
        fragment_shader='''
            #version 300 es
            precision highp float;

            #include "ubo"
            #include "light_index"

            in vec3 v_vertex;

            layout (location = 0) out float out_depth;

            void main() {
                out_depth = distance(light_position[light_index].xyz, v_vertex);
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
        framebuffer=framebuffer,
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, -1),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    )


def create_render_pipeline(vertex_buffer, uniform_buffer, framebuffer):
    return ctx.pipeline(
        vertex_shader='''
            #version 300 es
            precision highp float;

            #include "ubo"

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
            #version 300 es
            precision highp float;

            #include "ubo"

            uniform sampler2D Shadow[3];

            in vec3 v_vertex;
            in vec3 v_normal;

            layout (location = 0) out vec4 out_color;

            void main() {
                float ambient = 0.01;
                float facing = 0.03;
                float shininess = 32.0;

                vec3 view_dir = normalize(eye.xyz - v_vertex);
                float facing_view_dot = max(dot(view_dir, v_normal), 0.0);

                vec3 v_color = vec3(1.0);
                vec3 color = v_color * ambient + v_color * facing_view_dot * facing;

                for (int light_index = 0; light_index < 3; ++light_index) {
                    vec4 tmp = light_matrix[light_index] * vec4(v_vertex, 1.0);
                    float shadow = texture(Shadow[light_index], tmp.xy / tmp.w * 0.5 + 0.5).r;
                    if (shadow + 1e-3 > distance(light_position[light_index].xyz, v_vertex)) {
                        vec3 light_dir = light_position[light_index].xyz - v_vertex;
                        float light_distance = length(light_dir);
                        light_distance = light_distance * light_distance;
                        light_dir = normalize(light_dir);

                        float lambertian = max(dot(light_dir, v_normal), 0.0);
                        float specular = 0.0;

                        if (lambertian > 0.0) {
                            vec3 half_dir = normalize(light_dir + view_dir);
                            float spec_angle = max(dot(half_dir, v_normal), 0.0);
                            specular = pow(spec_angle, shininess);
                        }

                        float light_coef = light_color[light_index].a / light_distance;
                        color += v_color * lambertian * light_color[light_index].rgb * light_coef +
                            specular * light_color[light_index].rgb * light_coef;
                    }
                    out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
                }
            }
        ''',
        layout=[
            {
                'name': 'Common',
                'binding': 0,
            },
            {
                'name': 'Shadow[0]',
                'binding': 0,
            },
            {
                'name': 'Shadow[1]',
                'binding': 1,
            },
            {
                'name': 'Shadow[2]',
                'binding': 2,
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
                'image': shadow_framebuffers[0][0],
            },
            {
                'type': 'sampler',
                'binding': 1,
                'image': shadow_framebuffers[1][0],
            },
            {
                'type': 'sampler',
                'binding': 2,
                'image': shadow_framebuffers[2][0],
            },
        ],
        framebuffer=framebuffer,
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
        vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
    )


model = Obj.open(assets.get('monkey.obj')).pack('vx vy vz nx ny nz')
vertex_buffer = ctx.buffer(model)
uniform_buffer = ctx.buffer(size=368)

shadow_framebuffers = [create_shadow_framebuffer((1024, 1024)) for i in range(3)]
shadow_pipelines = [create_shadow_pipeline(i, vertex_buffer, uniform_buffer, shadow_framebuffers[i]) for i in range(3)]
scene = create_render_pipeline(vertex_buffer, uniform_buffer, [image, depth])

light_pos = [
    [0.0 * 3.0, 1.0 * 3.0, 5.0],
    [0.866 * 3.0, -0.5 * 3.0, 5.0],
    [-0.866 * 3.0, -0.5 * 3.0, 5.0],
]

light_matrix = [
    zengl.camera(light_pos[0], (0.0, 0.0, 0.0), fov=45.0),
    zengl.camera(light_pos[1], (0.0, 0.0, 0.0), fov=45.0),
    zengl.camera(light_pos[2], (0.0, 0.0, 0.0), fov=45.0),
]

light_color = [
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 10.0],
    [0.0, 0.0, 1.0, 10.0],
]

eye = (3.0, 2.0, 2.0)
camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
uniform_buffer.write(struct.pack(
    '=64s3f4x64s64s64s3f4x3f4x3f4x4f4f4f',
    camera, *eye,
    light_matrix[0], light_matrix[1], light_matrix[2],
    *light_pos[0], *light_pos[1], *light_pos[2],
    *light_color[0], *light_color[1], *light_color[2],
))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    image.clear()
    depth.clear()
    for fbo in shadow_framebuffers:
        fbo[0].clear()
        fbo[1].clear()

    for pipeline in shadow_pipelines:
        pipeline.render()

    scene.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
