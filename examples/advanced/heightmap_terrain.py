import sys

import assets
import imageio
import numpy as np
import pygame
import zengl
import zengl_extras
from skimage.filters import gaussian

zengl_extras.init()

pygame.init()
pygame.display.set_mode((1280, 720), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

window_size = pygame.display.get_window_size()
window_aspect = window_size[0] / window_size[1]

imageio.plugins.freeimage.download()
img = imageio.imread(assets.get('Terrain002.exr'))  # https://ambientcg.com/view?id=Terrain002

normals = np.zeros((512, 512, 3))
normals[:, 1:-1, 0] = img[:, :-2, 0] - img[:, 2:, 0]
normals[1:-1, :, 1] = img[:-2, :, 0] - img[2:, :, 0]
normals[:, :, 0] = gaussian(normals[:, :, 0])
normals[:, :, 1] = gaussian(normals[:, :, 1])
normals[:, :, 2] = 0.01

normals /= np.repeat(np.sum(np.sqrt(normals * normals), axis=2), 3).reshape(512, 512, 3)
normals = normals * 0.5 + 0.5

norm_img = np.full((512, 512, 4), 255, 'u1')
norm_img[:, :, :3] = np.clip(normals * 255, 0, 255)

color_img = np.full((512, 512, 4), 255, 'u1')
gray = np.random.randint(0, 32, (512, 512))
shade = np.where(gaussian(normals[:, :, 2]) > 0.75, 200, 50).astype('u1')
color_img[:, :, 0] = gray + shade
color_img[:, :, 1] = gray + shade
color_img[:, :, 2] = gray + shade


def create_terrain(N):
    vert = np.zeros((N * N, 2), 'i4')
    idx = np.full((N - 1, N * 2 + 1), -1, 'i4')
    vert[:] = np.array([np.repeat(np.arange(N), N), np.tile(np.arange(N), N)]).T
    idx[:, :-1] = (np.repeat(np.arange(N * N - N), 2) + np.tile([0, N], N * N - N)).reshape(-1, N * 2)
    return vert, idx


ctx = zengl.context()

image = ctx.image(window_size, 'rgba8unorm', samples=4)
depth = ctx.image(window_size, 'depth24plus', samples=4)
image.clear_value = (1.0, 1.0, 1.0, 1.0)

vertices, indices = create_terrain(512)
vertex_buffer = ctx.buffer(vertices)
index_buffer = ctx.buffer(indices)

img = imageio.imread(assets.get('Terrain002.exr'))
heightmap = ctx.image((512, 512), 'r32float', img[:, :, 0].tobytes())
normalmap = ctx.image((512, 512), 'rgba8unorm', norm_img.tobytes())
colormap = ctx.image((512, 512), 'rgba8unorm', color_img.tobytes())

uniform_buffer = ctx.buffer(size=64)

ctx.includes['terrain_info'] = '''
    const vec2 TerrainSize = vec2(512.0, 512.0);
    const vec3 TerrainScale = vec3(0.1, 0.1, 10.0);
    const vec3 TerrainPosition = vec3(-25.6, -25.6, 0.0);
'''

terrain = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        #include "terrain_info"

        layout (std140) uniform Common {
            mat4 mvp;
        };

        uniform sampler2D Heightmap;
        uniform sampler2D Normalmap;

        layout (location = 0) in ivec2 in_vert;

        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            v_normal = texelFetch(Normalmap, in_vert, 0).rgb * 2.0 - 1.0;
            float z = texelFetch(Heightmap, in_vert, 0).r;
            v_texcoord = (vec2(in_vert) + 0.5) / TerrainSize;
            gl_Position = mvp * vec4(vec3(in_vert, z) * TerrainScale + TerrainPosition, 1.0);
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec2 v_texcoord;

        uniform sampler2D Colormap;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            vec3 color = texture(Colormap, v_texcoord).rgb;
            float lum = dot(normalize(light), normalize(v_normal)) * 0.7 + 0.3;
            out_color = vec4(color * lum, 1.0);
        }
    ''',
    layout=[
        {
            'name': 'Common',
            'binding': 0,
        },
        {
            'name': 'Heightmap',
            'binding': 0,
        },
        {
            'name': 'Normalmap',
            'binding': 1,
        },
        {
            'name': 'Colormap',
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
            'image': heightmap,
        },
        {
            'type': 'sampler',
            'binding': 1,
            'image': normalmap,
        },
        {
            'type': 'sampler',
            'binding': 2,
            'image': colormap,
        },
    ],
    framebuffer=[image, depth],
    topology='triangle_strip',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '2i', 0),
    index_buffer=index_buffer,
    vertex_count=index_buffer.size // 4,
)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    now = pygame.time.get_ticks() / 1000.0

    ctx.new_frame()
    x, y = np.sin(now * 0.5) * 30.0, np.cos(now * 0.5) * 30.0
    camera = zengl.camera((x, y, 25.0), (0.0, 0.0, 0.0), aspect=window_aspect, fov=45.0)
    uniform_buffer.write(camera)

    image.clear()
    depth.clear()
    terrain.render()
    image.blit()
    ctx.end_frame()

    pygame.display.flip()
