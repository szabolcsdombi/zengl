import glm
import numpy as np
import zengl
from PIL import Image

from utils import (
    Camera, Window, download, read_file, read_vertices, set_uniform_float, set_uniform_glm, set_uniform_int
)

window = Window()
camera = Camera(4.72, 0.15, 2.16)
ctx = zengl.context()

final_image = ctx.image(window.size, 'rgba8unorm')

hdr_image = ctx.image(window.size, 'rgba32float')
hdr_depth = ctx.image(window.size, 'depth24plus')
hdr_image.clear_value = (0.0, 0.0, 0.0, 1.0)

cube_vertex_buffer = ctx.buffer(read_vertices('cube.json'))
quad_vertex_buffer = ctx.buffer(read_vertices('quad.json'))
quad_index_buffer = ctx.buffer(np.array([0, 2, 1, 1, 2, 3], 'i4'))

img = Image.open(download('wood.png')).convert('RGBA')
floor_texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())
floor_texture.mipmaps()

lighting = ctx.pipeline(
    vertex_shader=read_file('6.lighting.vs'),
    fragment_shader=read_file('6.lighting.fs'),
    layout=[
        {
            'name': 'diffuseTexture',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': floor_texture,
            'max_anisotropy': 16.0,
            'min_filter': 'linear_mipmap_linear',
            'mag_filter': 'linear',
        },
    ],
    uniforms='all',
    framebuffer=[hdr_image, hdr_depth],
    topology='triangles',
    vertex_buffers=zengl.bind(cube_vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=cube_vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

hdr = ctx.pipeline(
    vertex_shader=read_file('6.hdr.vs'),
    fragment_shader=read_file('6.hdr.fs'),
    layout=[
        {
            'name': 'hdrBuffer',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': hdr_image,
        },
    ],
    uniforms='all',
    framebuffer=[final_image],
    topology='triangles',
    index_buffer=quad_index_buffer,
    vertex_buffers=zengl.bind(quad_vertex_buffer, '3f 2f', 0, 1),
    vertex_count=quad_index_buffer.size // 4,
)

light_positions = [
    glm.vec3(0.0, 0.0, 49.5),
    glm.vec3(-1.4, -1.9, 9.0),
    glm.vec3(0.0, -1.8, 4.0),
    glm.vec3(0.8, -1.7, 6.0),
]

light_colors = [
    glm.vec3(200.0, 200.0, 200.0),
    glm.vec3(0.1, 0.0, 0.0),
    glm.vec3(0.0, 0.0, 0.2),
    glm.vec3(0.0, 0.1, 0.0),
]

exposure = 1.0

while window.update():
    camera.update(window)

    if window.key_pressed('1'):
        exposure = 0.1
    if window.key_pressed('2'):
        exposure = 0.5
    if window.key_pressed('3'):
        exposure = 1.0
    if window.key_pressed('4'):
        exposure = 5.0
    if window.key_pressed('5'):
        exposure = 20.0

    set_uniform_glm(lighting, 'projection', camera.projection_matrix)
    set_uniform_glm(lighting, 'view', camera.view_matrix)

    set_uniform_glm(lighting, 'lights[0].Position', light_positions[0])
    set_uniform_glm(lighting, 'lights[0].Color', light_colors[0])
    set_uniform_glm(lighting, 'lights[1].Position', light_positions[1])
    set_uniform_glm(lighting, 'lights[1].Color', light_colors[1])
    set_uniform_glm(lighting, 'lights[2].Position', light_positions[2])
    set_uniform_glm(lighting, 'lights[2].Color', light_colors[2])
    set_uniform_glm(lighting, 'lights[3].Position', light_positions[3])
    set_uniform_glm(lighting, 'lights[3].Color', light_colors[3])

    set_uniform_int(hdr, 'hdr', not window.key_down('space'))
    set_uniform_float(hdr, 'exposure', exposure)

    final_image.clear()
    hdr_image.clear()
    hdr_depth.clear()

    model = glm.mat4(1.0)
    model = glm.translate(model, glm.vec3(0.0, 0.0, 25.0))
    model = glm.scale(model, glm.vec3(2.5, 2.5, 27.5))
    set_uniform_glm(lighting, 'model', model)
    set_uniform_int(lighting, 'inverse_normals', True)
    lighting.render()
    hdr.render()

    final_image.blit()
