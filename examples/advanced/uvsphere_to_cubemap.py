import numpy as np
import zengl
from PIL import Image

import assets

'''
    A better solution can be found in panorama_to_cubemap
'''

sphere_resolution = 32
image_size = (1024, 1024)

img = Image.open(assets.get('comfy_cafe.jpg'))  # https://polyhaven.com/a/comfy_cafe

ctx = zengl.context(zengl.loader(headless=True))

texture = ctx.image(img.size, 'rgba8unorm', img.convert('RGBA').tobytes())
texture.mipmaps()

N, M = sphere_resolution + 1, sphere_resolution + 1
u = np.tile(np.linspace(0.0, 1.0, N), M)
v = np.repeat(np.linspace(0.0, 1.0, M), N)

vertices = np.array([
    np.sin(u * np.pi * 2.0 + np.pi * 3.0 / 2) * np.sin(v * np.pi),
    np.cos(u * np.pi * 2.0 + np.pi * 3.0 / 2) * np.sin(v * np.pi),
    np.cos(v * np.pi),
    u,
    v,
]).T

indices = np.pad(np.array([
    np.arange(N * M - N),
    np.arange(N * M - N) + N,
]).T.reshape(-1, N * 2), ((0, 0), (0, 1)), constant_values=-1)

image = ctx.image(image_size, 'rgba8unorm')
vertex_buffer = ctx.buffer(vertices.astype('f4').tobytes())
index_buffer = ctx.buffer(indices.astype('i4').tobytes())
uniform_buffer = ctx.buffer(size=64)

sphere = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vert;
        layout (location = 1) in vec2 in_text;

        out vec2 v_text;

        void main() {
            gl_Position = mvp * vec4(in_vert, 1.0);
            v_text = in_text;
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        layout (std140) uniform Common {
            mat4 mvp;
        };

        uniform sampler2D Texture;

        in vec2 v_text;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = texture(Texture, v_text);
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
            'min_filter': 'linear_mipmap_linear',
            'mag_filter': 'linear',
            'lod_bias': -1.0,
        },
    ],
    framebuffer=[image],
    topology='triangle_strip',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 2f', 0, 1),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
)

faces = [
    ('face_0', zengl.camera((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), fov=90.0)),
    ('face_1', zengl.camera((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), fov=90.0)),
    ('face_2', zengl.camera((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), fov=90.0)),
    ('face_3', zengl.camera((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), fov=90.0)),
    ('face_4', zengl.camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0), fov=90.0)),
    ('face_5', zengl.camera((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), fov=90.0)),
]

print('rendering')

ctx.new_frame()
for face, camera in faces:
    uniform_buffer.write(camera)
    sphere.render()
    img = Image.frombuffer('RGBA', image.size, image.read(), 'raw', 'RGBA', 0, -1)
    img.save(f'downloads/skybox_{face}.png')
ctx.end_frame()

print('done')
