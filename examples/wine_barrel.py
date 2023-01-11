import math
import struct
import zipfile

import zengl
from PIL import Image

import assets
from window import Window

pack = zipfile.ZipFile(assets.get('wine_barrel_01.zip'))
img_diff = Image.open(pack.open('wine_barrel_01_diff_2k.jpg'))
img_arm = Image.open(pack.open('wine_barrel_01_arm_2k.jpg'))
img_norm = Image.open(pack.open('wine_barrel_01_nor_gl_2k.jpg'))

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm-srgb', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

vertex_buffer = ctx.buffer(pack.read('wine_barrel_01.mesh'))  # https://polyhaven.com/a/wine_barrel_01

texture_diff = ctx.image(img_diff.size, 'rgba8unorm-srgb', img_diff.tobytes('raw', 'RGBA', 0, -1))
texture_arm = ctx.image(img_arm.size, 'rgba8unorm', img_arm.tobytes('raw', 'RGBA', 0, -1))
texture_norm = ctx.image(img_norm.size, 'rgba8unorm', img_norm.tobytes('raw', 'RGBA', 0, -1))

uniform_buffer = ctx.buffer(size=96)

crate = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
            vec3 eye_pos;
            vec3 light_pos;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_tangent;
        layout (location = 2) in vec3 in_bitangent;
        layout (location = 3) in vec3 in_normal;
        layout (location = 4) in vec2 in_texcoord;

        out vec3 v_vertex;
        out vec3 v_tangent;
        out vec3 v_bitangent;
        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            gl_Position = mvp * vec4(in_vertex, 1.0);
            v_vertex = in_vertex;
            v_tangent = in_tangent;
            v_bitangent = in_bitangent;
            v_normal = in_normal;
            v_texcoord = in_texcoord;
        }
    ''',
    fragment_shader='''
        #version 450 core

        layout (std140, binding = 0) uniform Common {
            mat4 mvp;
            vec3 eye_pos;
            vec3 light_pos;
        };

        layout (binding = 0) uniform sampler2D Texture1;
        layout (binding = 1) uniform sampler2D Texture2;
        layout (binding = 2) uniform sampler2D Texture3;

        in vec3 v_vertex;
        in vec3 v_tangent;
        in vec3 v_bitangent;
        in vec3 v_normal;
        in vec2 v_texcoord;

        layout (location = 0) out vec4 out_color;

        void main() {
            mat3 btn = mat3(v_tangent, v_bitangent, v_normal);
            vec3 texture_normal = texture(Texture3, v_texcoord).rgb - 0.5;
            vec3 normal = normalize(btn * texture_normal);
            float shininess = 32.0;
            vec3 light_dir = normalize(light_pos - v_vertex);
            vec3 eye_dir = normalize(eye_pos - v_vertex);
            vec3 halfway_dir = normalize(light_dir + eye_dir);
            vec3 surface_normal = texture(Texture3, v_texcoord).rgb;
            float rought = texture(Texture2, v_texcoord).g;
            float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess) * rought;
            vec3 color = texture(Texture1, v_texcoord).rgb + vec3(1.0, 1.0, 1.0) * spec;
            out_color = vec4(color, 1.0);
        }
    ''',
    resources=[
        {
            'type': 'uniform_buffer',
            'binding': 0,
            'buffer': uniform_buffer,
        },
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture_diff,
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': texture_arm,
        },
        {
            'type': 'sampler',
            'binding': 2,
            'image': texture_norm,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f 3f 2f', 0, 1, 2, 3, 4),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 3f 3f 2f'),
)

while window.update():
    x, y = math.sin(window.time * 0.5) * 2.0, math.cos(window.time * 0.5) * 2.0
    camera = zengl.camera((x, y, 1.2), (0.0, 0.0, 0.5), aspect=window.aspect, fov=45.0)

    uniform_buffer.write(camera)
    uniform_buffer.write(struct.pack('3f4x', x, y, 1.5), offset=64)
    uniform_buffer.write(struct.pack('3f4x', -4.0, -4.0, 4.0), offset=80)

    image.clear()
    depth.clear()
    crate.render()
    image.blit(srgb=True)
