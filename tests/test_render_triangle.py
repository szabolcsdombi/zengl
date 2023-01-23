import numpy as np
import zengl

from utils import glsl


def test_render_triangle(ctx: zengl.Context):
    img = ctx.image((256, 256), 'rgba8unorm')
    triangle = ctx.pipeline(
        vertex_shader=glsl('triangle.vert'),
        fragment_shader=glsl('triangle.frag'),
        framebuffer=[img],
        vertex_count=3,
    )
    img.clear()
    triangle.run()
    pixels = np.frombuffer(img.read(), 'u1').reshape(256, 256, 4)
    x = np.repeat(np.arange(4) * 50 + 50, 4)
    y = np.tile(np.arange(4) * 50 + 50, 4)
    r = [255, 0, 0, 255]
    z = [0, 0, 0, 0]
    np.testing.assert_array_equal(pixels[x, y], [
        r, r, r, r,
        z, r, r, z,
        z, r, r, z,
        z, z, z, z,
    ])
    # from matplotlib import pyplot as plt
    # plt.imshow(pixels)
    # plt.plot(x, y, 'bx')
    # plt.show()
