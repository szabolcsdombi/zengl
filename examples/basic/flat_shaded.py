import math
import struct
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()
zengl_extras.download('arena.zip')

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def load_texture(name):
    img = pygame.image.load(name)
    pixels = pygame.image.tobytes(img, 'RGBA', True)
    return ctx.image(img.get_size(), 'rgba8unorm', pixels)


def load_model(name):
    with open(name, 'rb') as f:
        model = f.read()
    return ctx.buffer(model)


size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

texture = load_texture('downloads/arena/arena.png')
vertex_buffer = load_model('downloads/arena/arena.bin')

uniform_buffer = ctx.buffer(size=96, uniform=True)

background = ctx.pipeline(
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
        };

        in vec2 v_vertex;

        layout (location = 0) out vec4 out_color;

        void main() {
            mat4 inv_camera_matrix = inverse(camera_matrix);
            vec4 position_temp = inv_camera_matrix * vec4(v_vertex, -1.0, 1.0);
            vec4 target_temp = inv_camera_matrix * vec4(v_vertex, 1.0, 1.0);
            vec3 position = position_temp.xyz / position_temp.w;
            vec3 target = target_temp.xyz / target_temp.w;
            vec3 direction = normalize(target - position);

            vec3 color;
            if (direction.z > 0.0) {
                vec3 color1 = vec3(1.0, 1.0, 1.0);
                vec3 color2 = vec3(0.1, 0.5, 0.9);
                color = mix(color1, color2, pow(direction.z, 0.4));
            } else {
                vec3 color1 = vec3(1.0, 1.0, 1.0);
                vec3 color2 = vec3(0.2, 0.2, 0.2);
                color = mix(color1, color2, pow(direction.z, 0.1));
            }

            out_color = vec4(color, 1.0);
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
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

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
            gl_Position = camera_matrix * vec4(in_vertex, 1.0);
            v_vertex = in_vertex;
            v_normal = in_normal;
            v_uv = in_uv;
        }
    ''',
    fragment_shader='''
        #version 330 core

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
            vec3 light_direction = normalize(light_position.xyz - v_vertex);
            float lum = clamp(dot(light_direction, normalize(v_normal)), 0.0, 1.0);
            lum = lum * 0.3 + 0.7;
            vec3 color = pow(texture(Texture, v_uv).rgb, vec3(2.2));
            out_color = vec4(color * lum, 1.0);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
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
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    ctx.new_frame()
    time = pygame.time.get_ticks() / 1000.0
    eye = (math.cos(time * 0.1) * 10.0, math.sin(time * 0.1) * 10.0, 3.0)
    light = (3.0, 4.0, 30.0)
    camera = zengl.camera(eye, (0.0, 0.0, 0.0), aspect=1.0, fov=45.0)
    uniform_buffer.write(struct.pack('64s3f4x3f4x', camera, *eye, *light))
    image.clear()
    depth.clear()
    background.render()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
