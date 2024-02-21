import struct
import sys

import pygame
import zengl
import zengl_extras

zengl_extras.init()
zengl_extras.download('metal_plate.zip')

pygame.init()
pygame.display.set_mode((720, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

ctx = zengl.context()


def load_texture(path):
    img = pygame.image.load(path)
    pixels = pygame.image.tobytes(img, 'RGBA')
    return ctx.image(img.get_size(), 'rgba8unorm', pixels)


size = pygame.display.get_window_size()

image = ctx.image(size, 'rgba8unorm', samples=4)
depth = ctx.image(size, 'depth24plus', samples=4)

diffuse_texture = load_texture('downloads/metal_plate/diffuse_2k.jpg')
roughness_texture = load_texture('downloads/metal_plate/roughness_2k.jpg')
normal_texture = load_texture('downloads/metal_plate/normal_2k.jpg')

model = open('downloads/metal_plate/plate.bin', 'rb').read()
vertex_buffer = ctx.buffer(model)

uniform_buffer = ctx.buffer(size=144)

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

        const float pi = 3.14159265358979323;

        uniform sampler2D Texture;

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
        };

        in vec2 v_vertex;

        layout (location = 0) out vec4 out_color;

        vec3 hash33(vec3 p3) {
            p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yxz + 33.33);
            return fract((p3.xxy + p3.yxx) * p3.zyx);
        }

        vec3 desaturate(vec3 color, float factor) {
            vec3 lum = vec3(0.2126, 0.7152, 0.0722);
            vec3 gray = vec3(dot(lum, color));
            return mix(color, gray, factor);
        }

        void main() {
            mat4 inv_camera_matrix = inverse(camera_matrix);
            vec4 position_temp = inv_camera_matrix * vec4(v_vertex, -1.0, 1.0);
            vec4 target_temp = inv_camera_matrix * vec4(v_vertex, 1.0, 1.0);
            vec3 position = position_temp.xyz / position_temp.w;
            vec3 target = target_temp.xyz / target_temp.w;
            vec3 direction = normalize(target - position);

            vec3 color = vec3(0.0);
            for (int i = 0; i < 8; i++) {
                vec3 ray = normalize(direction + hash33(vec3(gl_FragCoord.xy, float(i))) * 0.1);
                vec2 uv = vec2(atan(ray.y, ray.x) / pi, ray.z) * 0.5 + 0.5;
                color += texture(Texture, uv).rgb;
            }
            color = desaturate(color / 8.0, 0.8);
            out_color = vec4(color, 1.0);
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
            'image': diffuse_texture,
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
        layout (location = 2) in vec2 in_texcoords;
        layout (location = 3) in vec3 in_tangent;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec2 v_texcoords;
        out vec3 v_tangent;

        void main() {
            gl_Position = camera_matrix * vec4(in_vertex, 1.0);
            v_vertex = in_vertex;
            v_normal = in_normal;
            v_texcoords = in_texcoords;
            v_tangent = in_tangent;
        }
    ''',
    fragment_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 camera_matrix;
            vec4 camera_position;
            vec4 light_position;
        };

        uniform sampler2D DiffuseTexture;
        uniform sampler2D RoughnessTexture;
        uniform sampler2D NormalTexture;

        in vec3 v_vertex;
        in vec3 v_normal;
        in vec2 v_texcoords;
        in vec3 v_tangent;

        layout (location = 0) out vec4 out_color;

        const float shininess = 32.0;

        void main() {
            vec3 bitangent = cross(v_tangent, v_normal);
            mat3 btn = mat3(v_tangent, bitangent, v_normal);
            vec3 texture_normal = texture(NormalTexture, v_texcoords).rgb - 0.5;
            vec3 normal = normalize(btn * texture_normal);
            vec3 light_direction = normalize(light_position.xyz - v_vertex);
            vec3 eye_direction = normalize(camera_position.xyz - v_vertex);
            vec3 halfway_direction = normalize(light_direction + eye_direction);
            vec3 surface_normal = texture(NormalTexture, v_texcoords).rgb;
            float roughness = texture(RoughnessTexture, v_texcoords).r;
            float specular = pow(max(dot(normal, halfway_direction), 0.0), shininess) * roughness;
            vec3 color = pow(texture(DiffuseTexture, v_texcoords).rgb, vec3(2.2)) + vec3(1.0, 1.0, 1.0) * specular;
            out_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'DiffuseTexture',
            'binding': 0,
        },
        {
            'name': 'RoughnessTexture',
            'binding': 1,
        },
        {
            'name': 'NormalTexture',
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
            'image': diffuse_texture,
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': roughness_texture,
        },
        {
            'type': 'sampler',
            'binding': 2,
            'image': normal_texture,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f 3f', 0, 1, 2, 3),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f 3f'),
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0
    mouse = pygame.mouse.get_pos()

    ctx.new_frame()
    image.clear()
    depth.clear()

    mx, my = mouse[0] / size[0] - 0.5, 0.5 - mouse[1] / size[1]
    camera_position = (mx * 2.0, my * 2.0, 3.0)
    light_position = (mx * 2.0, my * 2.0, 1.0)
    camera = zengl.camera(camera_position, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), fov=45.0)
    uniform_buffer.write(struct.pack('=64s3f4x3f4x', camera, *camera_position, *light_position))

    background.render()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
