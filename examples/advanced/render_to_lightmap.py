import sys

import assets
import numpy as np
import pygame
import zengl
import zengl_extras
from objloader import Obj
from PIL import Image
from progress.bar import Bar

samples = 64
size = 1024

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

ctx = zengl.context()

ctx.includes['samples'] = f'const int samples = {samples};'

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

model = Obj.open(assets.get('suzanne-lightmap-uv.obj')).pack('vx vy vz tx ty')
# model = Obj.open(assets.get('ao-map-target.obj')).pack('vx vy vz tx ty')
vertex_buffer = ctx.buffer(model)

temp_depth = ctx.image((size, size), 'depth24plus')
temp_texture = ctx.image((size, size), 'r32float')
texture = ctx.image((size, size), 'r32float')

uniform_buffer = ctx.buffer(size=64)

depth_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;

        out vec3 v_vertex;

        void main() {
            gl_Position = mvp * vec4(in_vertex, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        void main() {
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
    framebuffer=[temp_depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 2f', 0, -1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 2f'),
)

texture_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec2 in_texcoord;

        out vec3 v_vertex;

        void main() {
            gl_Position = vec4(in_texcoord * 2.0 - 1.0, 0.0, 1.0);
            v_vertex = in_vertex;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;
        precision highp int;

        #include "samples"

        layout (std140) uniform Common {
            mat4 mvp;
        };

        uniform sampler2DShadow DepthTexture;

        in vec3 v_vertex;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec4 vertex = mvp * vec4(v_vertex, 1.0);
            vertex.xyz = vertex.xyz / vertex.w * 0.5 + 0.5;
            float lum = texture(DepthTexture, vertex.xyz);
            out_color = vec4(vec3(lum) / float(samples), 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'DepthTexture',
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
            'image': temp_depth,
            'compare_func': 'less',
            'compare_mode': 'ref_to_texture',
        },
    ],
    blend={
        'enable': True,
        'src_color': 'one',
        'dst_color': 'one',
    },
    framebuffer=[temp_texture],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 2f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 2f'),
)

fill_pipeline = ctx.pipeline(
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

        uniform sampler2D Texture;

        layout (location = 0) out float out_color;

        ivec2 offsets[8] = ivec2[](
            ivec2(-1, -1),
            ivec2(-1, 0),
            ivec2(-1, 1),
            ivec2(0, -1),
            ivec2(0, 1),
            ivec2(1, -1),
            ivec2(1, 0),
            ivec2(1, 1)
        );

        void main() {
            ivec2 uv = ivec2(gl_FragCoord.xy);
            float color = texelFetch(Texture, uv, 0).r;
            if (color == 0.0) {
                int count = 0;
                for (int i = 0; i < 8; ++i) {
                    float temp = texelFetch(Texture, uv + offsets[i], 0).r;
                    if (temp > 0.0) {
                        ++count;
                    }
                    color += temp;
                }
                if (count > 0) {
                    color /= float(count);
                }
            }
            out_color = color;
        }
    ''',
    layout=[
        {
            'name': 'Texture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': temp_texture,
        },
    ],
    framebuffer=[texture],
    topology='triangles',
    vertex_count=3,
)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec2 in_texcoord;

        out vec3 v_vertex;
        out vec2 v_texcoord;

        void main() {
            gl_Position = mvp * vec4(in_vertex, 1.0);
            v_vertex = in_vertex;
            v_texcoord = in_texcoord;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        uniform sampler2D Texture;

        in vec3 v_vertex;
        in vec2 v_texcoord;

        layout (location = 0) out vec4 out_color;

        void main() {
            float lum = texture(Texture, v_texcoord).r;
            out_color = vec4(lum, lum, lum, 1.0);
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
            'min_filter': 'nearest',
            'mag_filter': 'nearest',
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 2f', 0, 1),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 2f'),
)

temp_texture.clear()

bar = Bar('Progress', fill='-', suffix='%(percent)d%%', max=samples)

for i in range(samples):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (i / (samples - 1.0)) * 2.0
    x = np.cos(phi * i) * np.sqrt(1.0 - y * y)
    z = np.sin(phi * i) * np.sqrt(1.0 - y * y)

    camera = zengl.camera((x * 5.0, y * 5.0, z * 5.0), (0.0, 0.0, 0.0), aspect=1.0, fov=45.0, near=1.0, far=10.0)
    uniform_buffer.write(camera)
    temp_depth.clear()
    depth_pipeline.render()
    texture_pipeline.render()
    bar.next()


fill_pipeline.render()

ao = np.frombuffer(texture.read(), 'f4').reshape(size, size)[::-1]
Image.fromarray((ao * 255.0).astype('u1'), 'L').save('generated-ao-map.png')

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now  = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    x, y = np.sin(now * 0.5 + 1.0) * 5.0, np.cos(now * 0.5 + 1.0) * 5.0
    camera = zengl.camera((x, y, 1.5), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
