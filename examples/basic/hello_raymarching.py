import math
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
image = ctx.image(size, 'rgba8unorm', texture=False)
depth = ctx.image(size, 'depth24plus', texture=False)

uniform_buffer = ctx.buffer(size=128, uniform=True)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec2 vertices[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        out vec2 v_vertex;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            v_vertex = vertices[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
            vec4 light_position;
            float time;
        };

        float sdf_soft_min(float a, float b, float k) {
            float h = max(k - abs(a - b), 0.0) / k;
            return min(a, b) - h * h * k * (1.0 / 4.0);
        }

        float sdf_sphere(vec3 p, float r) {
            return length(p) - r;
        }

        float sdf_plane(vec3 p, vec3 n, float h) {
            return dot(p, n) + h;
        }

        float sdf_scene(vec3 p) {
            float d1 = sdf_sphere(p - vec3(0.0, 0.0, sin(time * 6.0) * 0.1), 0.1);
            float d2 = sdf_sphere(p - vec3(0.0, -0.3, sin(time * 6.0 - 2.0) * 0.1), 0.1);
            float d3 = sdf_sphere(p - vec3(0.0, 0.3, sin(time * 6.0 + 2.0) * 0.1), 0.1);
            float d4 = sdf_plane(p - vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), 0.0);
            return sdf_soft_min(min(min(d1, d2), d3), d4, 0.1);
        }

        vec3 sdf_normal(vec3 p) {
            vec2 e = vec2(0.0001, 0.0);
            return normalize(vec3(
                sdf_scene(p + e.xyy) - sdf_scene(p - e.xyy),
                sdf_scene(p + e.yxy) - sdf_scene(p - e.yxy),
                sdf_scene(p + e.yyx) - sdf_scene(p - e.yyx)
            ));
        }

        float soft_checkers(vec3 p) {
            vec3 q = floor(p * 12.0);
            return mod(q.x + q.y, 2.0) - 1.0;
        }

        in vec2 v_vertex;

        layout (location = 0) out vec4 out_color;

        void main() {
            mat4 inv_camera_matrix = inverse(camera_matrix);
            vec4 position_temp = inv_camera_matrix * vec4(v_vertex, -1.0, 1.0);
            vec4 target_temp = inv_camera_matrix * vec4(v_vertex, 1.0, 1.0);
            vec3 position = position_temp.xyz / position_temp.w;
            vec3 target = target_temp.xyz / target_temp.w;
            vec3 direction = normalize(target - position);

            for (int i = 0; i < 64; i++) {
                position += direction * sdf_scene(position);
            }

            vec3 normal = sdf_normal(position);
            vec3 light_dir = light_position.xyz - position;
            float light = max(dot(normal, normalize(light_dir)), 0.0);
            light /= exp(length(light_dir) * 2.0);
            light = light * 0.99 + 0.01;
            vec3 color = vec3(0.2, 0.4, 1.0) * soft_checkers(position) * 0.5 + 0.5;
            out_color = vec4(color * light, 1.0);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));

            vec4 vertex = camera_matrix * vec4(position, 1.0);
            gl_FragDepth = (vertex.z / vertex.w) * 0.5 + 0.5;
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
    vertex_count=3,
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
    uniform_buffer.write(struct.pack('64s3f4x3f4xf', camera, *eye, *light, now))

    pipeline.render()
    debug_light.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
