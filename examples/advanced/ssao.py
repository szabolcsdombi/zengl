import math
import struct
import sys

import assets
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.2, 0.2, 0.2, 1.0)

vertex_buffer = ctx.buffer(open(assets.get('door.mesh'), 'rb').read())
uniform_buffer = ctx.buffer(size=96)

temp_position = ctx.image(window_size, 'rgba32float')
temp_normal = ctx.image(window_size, 'rgba32float')
temp_depth = ctx.image(window_size, 'depth24plus')

temp_pass = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 light_pos;
            vec4 camera_pos;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec3 in_norm;

        out vec3 v_vert;
        out vec3 v_norm;

        void main() {
            v_vert = in_vert;
            v_norm = in_norm;
            gl_Position = mvp * vec4(v_vert, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 light_pos;
            vec4 camera_pos;
        };

        in vec3 v_vert;
        in vec3 v_norm;

        layout (location = 0) out vec4 out_position;
        layout (location = 1) out vec4 out_normal;

        void main() {
            out_position = vec4(v_vert, 0.0);
            out_normal = vec4(v_norm, 0.0);
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
    framebuffer=[temp_position, temp_normal, temp_depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

ssao = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        float hash13(vec3 p3) {
            p3 = fract(p3 * 0.1031);
            p3 += dot(p3, p3.zyx + 31.32);
            return fract((p3.x + p3.y) * p3.z);
        }

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 light_pos;
            vec4 camera_pos;
        };

        vec3 points[16] = vec3[](
            vec3(-0.2227, 0.7356, 0.4659),
            vec3(0.1868, 0.7353, 0.0808),
            vec3(0.0237, 0.1313, 0.3295),
            vec3(-0.0870, 0.6000, 0.6392),
            vec3(0.4394, -0.2024, 0.6497),
            vec3(0.5340, -0.5363, 0.1430),
            vec3(0.6900, 0.4900, 0.2762),
            vec3(-0.3859, 0.1028, 0.6229),
            vec3(-0.0469, -0.3474, 0.5782),
            vec3(0.6194, -0.0360, 0.4796),
            vec3(0.2145, 0.1505, 0.5911),
            vec3(-0.0271, 0.2603, 0.6902),
            vec3(0.1280, -0.5275, 0.2892),
            vec3(-0.6619, 0.1548, 0.2046),
            vec3(-0.1323, -0.7756, 0.3461),
            vec3(-0.4802, -0.5583, 0.2939)
        );

        uniform sampler2D Position;
        uniform sampler2D Normal;

        layout (location = 0) out vec4 out_color;

        void main() {
            ivec2 at = ivec2(gl_FragCoord.xy);
            vec3 position = texelFetch(Position, at, 0).rgb;
            vec3 normal = texelFetch(Normal, at, 0).rgb;
            vec3 x = cross(normal, vec3(0.0, 0.0, 1.0));
            vec3 y = cross(normal, x);
            float angle = hash13(gl_FragCoord.xyz);
            mat3 rot = mat3(
                cos(angle), -sin(angle), 0.0,
                sin(angle), cos(angle), 0.0,
                0.0, 0.0, 1.0
            );
            mat3 basis = mat3(x, y, normal) * rot;

            float lum = 0.0;

            for (int i = 0; i < 16; i++) {
                vec4 tmp = mvp * vec4(position + basis * points[i] * 0.08, 1.0);
                vec2 uv = (tmp.xy / tmp.w) * 0.5 + 0.5;
                vec3 pick = texture(Position, uv).rgb;
                if (distance(pick, camera_pos.xyz) > distance(position, camera_pos.xyz) - 1e-2) {
                    lum += 1.0 / 16.0;
                }
            }

            out_color = vec4(lum, lum, lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Position',
            'binding': 0,
        },
        {
            'name': 'Normal',
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
            'type': 'sampler',
            'binding': 0,
            'image': temp_position,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': temp_normal,
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now  = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    eye = (0.0 + math.sin(now) * 0.5, -4.0, 3.0)
    camera = zengl.camera(eye, (0.0, 0.0, 1.0), aspect=window_aspect, fov=45.0)

    uniform_buffer.write(struct.pack('=64s3f4x3f4x', camera, *eye, *eye))

    temp_position.clear()
    temp_normal.clear()
    temp_depth.clear()
    temp_pass.render()

    image.clear()
    depth.clear()
    ssao.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
