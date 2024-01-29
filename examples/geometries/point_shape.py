import struct

import pygame
import zengl

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF)

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

        vec3 vertices[12] = vec3[](
            vec3(0.000000, 0.000000, -1.000000),
            vec3(0.723600, -0.525720, -0.447215),
            vec3(-0.276385, -0.850640, -0.447215),
            vec3(-0.894425, 0.000000, -0.447215),
            vec3(-0.276385, 0.850640, -0.447215),
            vec3(0.723600, 0.525720, -0.447215),
            vec3(0.276385, -0.850640, 0.447215),
            vec3(-0.723600, -0.525720, 0.447215),
            vec3(-0.723600, 0.525720, 0.447215),
            vec3(0.276385, 0.850640, 0.447215),
            vec3(0.894425, 0.000000, 0.447215),
            vec3(0.000000, 0.000000, 1.000000)
        );

        vec2 texcoords[22] = vec2[](
            vec2(0.181819, 0.000000),
            vec2(0.363637, 0.000000),
            vec2(0.909091, 0.000000),
            vec2(0.727273, 0.000000),
            vec2(0.545455, 0.000000),
            vec2(0.272728, 0.157461),
            vec2(1.000000, 0.157461),
            vec2(0.090910, 0.157461),
            vec2(0.818182, 0.157461),
            vec2(0.636364, 0.157461),
            vec2(0.454546, 0.157461),
            vec2(0.181819, 0.314921),
            vec2(0.000000, 0.314921),
            vec2(0.909091, 0.314921),
            vec2(0.727273, 0.314921),
            vec2(0.545455, 0.314921),
            vec2(0.363637, 0.314921),
            vec2(0.272728, 0.472382),
            vec2(0.090910, 0.472382),
            vec2(0.818182, 0.472382),
            vec2(0.636364, 0.472382),
            vec2(0.454546, 0.472382)
        );

        int vertex_indices[60] = int[](
            0, 1, 2, 1, 0, 5, 0, 2, 3, 0, 3, 4, 0, 4, 5, 1, 5, 10, 2, 1, 6, 3, 2, 7, 4, 3, 8, 5, 4, 9, 1, 10,
            6, 2, 6, 7, 3, 7, 8, 4, 8, 9, 5, 9, 10, 6, 10, 11, 7, 6, 11, 8, 7, 11, 9, 8, 11, 10, 9, 11
        );

        int texcoord_indices[60] = int[](
            0, 5, 7, 5, 1, 10, 2, 6, 8, 3, 8, 9, 4, 9, 10, 5, 10, 16, 7, 5, 11, 8, 6, 13, 9, 8, 14, 10, 9, 15,
            5, 16, 11, 7, 11, 12, 8, 13, 14, 9, 14, 15, 10, 15, 16, 11, 16, 17, 12, 11, 18, 14, 13, 19, 15, 14,
            20, 16, 15, 21
        );

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            v_vertex = vertices[vertex_indices[gl_VertexID]];
            v_normal = vertices[vertex_indices[gl_VertexID]];
            v_texcoord = texcoords[texcoord_indices[gl_VertexID]];
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
    topology='triangles',
    cull_face='back',
    vertex_count=60,
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
