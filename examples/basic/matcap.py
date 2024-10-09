import glob
import itertools
import math
import struct
import sys

import glm
import pygame
import zengl
import zengl_extras

zengl_extras.init()

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

matcaps = itertools.cycle(glob.glob('downloads/matcap/*.png'))


def replace_texture(texture, name):
    img = pygame.image.load(name)
    pixels = pygame.image.tobytes(img, 'RGBA', True)
    texture.write(pixels)


def load_model(name):
    with open(name, 'rb') as f:
        model = f.read()
    return ctx.buffer(model)


size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

texture = ctx.image((512, 512), 'rgba8unorm')
replace_texture(texture, next(matcaps))

vertex_buffer = load_model('downloads/matcap/statue.bin')
uniform_buffer = ctx.buffer(size=160, uniform=True)

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
            mat4 modelview_matrix;
            vec4 camera_position;
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
                color = mix(color1, color2, pow(-direction.z, 0.1));
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
            mat4 modelview_matrix;
            vec4 camera_position;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;

        out vec3 v_normal;

        void main() {
            gl_Position = camera_matrix * vec4(in_vertex, 1.0);
            v_normal = (modelview_matrix * vec4(in_normal, 0.0)).xyz;
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 camera_matrix;
            mat4 modelview_matrix;
            vec4 camera_position;
        };

        uniform sampler2D Matcap;

        in vec3 v_normal;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec2 uv = vec2(0.5, 0.5) + normalize(v_normal).xy * 0.49;
            vec3 color = texture(Matcap, uv).rgb;
            out_color = vec4(color, 1.0);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
            // out_color.rgb *= 0.000000001;
            // out_color.rgb += v_normal * 0.5 + 0.5;
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Matcap',
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
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f'),
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                replace_texture(texture, next(matcaps))

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    image.clear()
    depth.clear()

    eye = (math.cos(now * 0.8) * 1.0, math.sin(now * 0.8) * 1.0, 0.5)
    target = (0.0, 0.0, 0.2)

    projection = glm.perspective(glm.radians(75.0), 1.0, 0.1, 100.0)
    lookat = glm.lookAt(eye, target, (0.0, 0.0, 1.0))
    mvp = projection * lookat

    uniform_buffer.write(struct.pack('64s64s3f4x', bytes(glm.transpose(mvp)), bytes(glm.transpose(lookat)), *eye))

    background.render()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
