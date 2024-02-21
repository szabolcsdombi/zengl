import array
import math
import os
import struct
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

vertex_buffer = ctx.buffer(array.array('f', [
    -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    -0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
]))
index_buffer = ctx.buffer(array.array('i', [0, 2, 1, 1, 2, 3]), index=True)
uniform_buffer = ctx.buffer(size=96, uniform=True)

texture = ctx.image((8, 8), 'rgba8unorm', os.urandom(8 * 8 * 4))

ctx.includes['blinn_phong'] = '''
    vec3 blinn_phong(
            vec3 vertex, vec3 normal, vec3 camera_position, vec3 light_position, float shininess, vec3 ambient_color,
            vec3 diffuse_color, vec3 light_color, vec3 spec_color, float light_power) {

        vec3 light_dir = light_position - vertex;
        float light_distance = length(light_dir);
        light_distance = light_distance * light_distance;
        light_dir = normalize(light_dir);

        float lambertian = max(dot(light_dir, normal), 0.0);
        float specular = 0.0;

        if (lambertian > 0.0) {
            vec3 view_dir = normalize(camera_position - vertex);
            vec3 half_dir = normalize(light_dir + view_dir);
            float spec_angle = max(dot(half_dir, normal), 0.0);
            specular = pow(spec_angle, shininess);
        }

        vec3 color_linear = ambient_color +
            diffuse_color * lambertian * light_color * light_power / light_distance +
            spec_color * specular * light_color * light_power / light_distance;

        return color_linear;
    }
'''

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
            vec4 light_position;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec2 in_uv;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec2 v_uv;

        void main() {
            v_vertex = in_vertex;
            v_normal = in_normal;
            v_uv = in_uv;
            gl_Position = camera_matrix * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        #include "blinn_phong"

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
            vec4 light_position;
        };

        uniform sampler2D Texture;

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec2 v_uv;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 diffuse_color = texture(Texture, v_uv).rgb;
            vec3 ambient_color = diffuse_color * 0.1;
            float light_power = 0.1;

            vec3 color = blinn_phong(
                v_vertex, v_normal, camera_position.xyz, light_position.xyz, 16.0,
                ambient_color, diffuse_color, vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), light_power
            );
            out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
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
            'image': texture,
            'wrap_x': 'clamp_to_edge',
            'wrap_y': 'clamp_to_edge',
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image, depth],
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
    cull_face='back',
    topology='triangles',
)

debug_light = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
            vec4 light_position;
        };

        vec3 vertices[36] = vec3[](
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, -0.5, -0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, -0.5, -0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(-0.5, 0.5, -0.5)
        );

        void main() {
            gl_Position = camera_matrix * vec4(light_position.xyz + vertices[gl_VertexID] * 0.025, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(1.0, 1.0, 1.0, 1.0);
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
    vertex_count=36,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000

    ctx.new_frame()
    image.clear()
    depth.clear()

    light = (math.cos(now * 0.6) * 0.4, 0.0, abs(math.sin(now * 0.6) * 0.4) + 0.01)
    eye = (math.cos(now * 0.6) * 1.0, math.sin(now * 0.6) * 1.0, 0.5)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)
    uniform_buffer.write(struct.pack('64s3f4x3f4x', camera, *eye, *light))

    pipeline.render()
    debug_light.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
