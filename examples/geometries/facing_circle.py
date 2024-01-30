import struct

import pygame
import zengl

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)
uniform_buffer = ctx.buffer(size=80, uniform=True)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
        };

        vec2 vertices[16] = vec2[](
            vec2(1.0000, 0.0000),
            vec2(0.9239, 0.3827),
            vec2(0.7071, 0.7071),
            vec2(0.3827, 0.9239),
            vec2(0.0000, 1.0000),
            vec2(-0.3827, 0.9239),
            vec2(-0.7071, 0.7071),
            vec2(-0.9239, 0.3827),
            vec2(-1.0000, 0.0000),
            vec2(-0.9239, -0.3827),
            vec2(-0.7071, -0.7071),
            vec2(-0.3827, -0.9239),
            vec2(0.0000, -1.0000),
            vec2(0.3827, -0.9239),
            vec2(0.7071, -0.7071),
            vec2(0.9239, -0.3827)
        );

        vec3 position = vec3(0.0, 0.0, 0.0);
        vec3 up = vec3(0.0, 0.0, 1.0);

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            v_normal = normalize(camera_position.xyz - position);
            vec3 tangent = normalize(cross(up, v_normal));
            vec3 bitangent = cross(v_normal, tangent);
            v_vertex = position + tangent * vertices[gl_VertexID].x + bitangent * vertices[gl_VertexID].y;
            v_texcoord = vertices[gl_VertexID] + 0.5;
            gl_Position = camera_matrix * vec4(v_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_normal;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light_direction = vec3(0.48, 0.32, 0.81);
            float lum = dot(light_direction, normalize(v_normal)) * 0.7 + 0.3;
            out_color = vec4(lum, lum, lum, 1.0);
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
    topology='triangle_fan',
    cull_face='back',
    vertex_count=16,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    image.clear()
    depth.clear()
    camera_position = (4.0, 3.0, 2.0)
    camera = zengl.camera(camera_position, aspect=1.0, fov=45.0)
    uniform_buffer.write(struct.pack('64s3f4x', camera, *camera_position))
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
