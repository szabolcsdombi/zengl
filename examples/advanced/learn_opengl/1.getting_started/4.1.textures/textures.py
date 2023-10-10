import numpy as np
import zengl
from PIL import Image

from utils import Window, download, read_file

window = Window()
ctx = zengl.context()

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.3, 0.3, 0.3, 1.0)

vertex_buffer = ctx.buffer(np.array([
    0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,    # top right
    0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,   # bottom right
    -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # bottom left
    -0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,   # top left
], 'f4'))

index_buffer = ctx.buffer(np.array([
    0, 1, 3,  # first triangle
    1, 2, 3,  # second triangle
], 'i4'))

img = Image.open(download('container2.png')).convert('RGBA')
texture = ctx.image(img.size, 'rgba8unorm', img.tobytes())

pipeline = ctx.pipeline(
    vertex_shader=read_file('4.1.texture.vs'),
    fragment_shader=read_file('4.1.texture.fs'),
    layout=[
        {
            'name': 'texture1',
            'binding': 0,
        },
    ],
    resources=[
        {
            'type': 'sampler',
            'binding': 0,
            'image': texture,
        },
    ],
    framebuffer=[image, depth],
    topology='triangles',
    index_buffer=index_buffer,

    # Bind vertex_buffer to multiple attributes.
    # The buffer format is (vec3, vec3, vec2).
    # The attribute location=1 does not exist. Unused attributes are optimized out.
    # Unused members can be bound to attribute location -1 (a special value for skipped attributes).
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 2f', 0, -1, 2),
    vertex_count=6,
)

while window.update():
    image.clear()
    depth.clear()
    pipeline.render()
    image.blit()
