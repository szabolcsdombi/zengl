import glm
import zengl
from PIL import Image

from utils import Camera, Window, download, read_file, read_vertices, set_uniform_glm, set_uniform_int

window = Window()
camera = Camera(2.3, 0.32, 8.5)
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

vertex_buffer = ctx.buffer(read_vertices('plane.json'))

img = Image.open(download('wood.png')).convert('RGBA')
floor_texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())
floor_texture.mipmaps()

pipeline = ctx.pipeline(
    vertex_shader=read_file('1.advanced_lighting.vs'),
    fragment_shader=read_file('1.advanced_lighting.fs'),
    layout=[
        {
            'name': 'floorTexture',
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
    framebuffer=[image, depth],
    topology='triangles',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 2f'),
)

light_pos = glm.vec3(0.0, 0.0, 0.0)

while window.update():
    camera.update(window)

    set_uniform_glm(pipeline, 'projection', camera.projection_matrix)
    set_uniform_glm(pipeline, 'view', camera.view_matrix)
    set_uniform_glm(pipeline, 'viewPos', camera.position)
    set_uniform_glm(pipeline, 'lightPos', light_pos)
    set_uniform_int(pipeline, 'blinn', not window.key_down('space'))

    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
