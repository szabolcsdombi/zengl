import numpy as np
import zengl


def test_blit_array_layer(ctx: zengl.Context):
    array = ctx.image((64, 64), 'rgba8unorm', np.full((3, 64, 64, 4), (200, 100, 0, 255), 'u1'), array=3, levels=2)
    image = ctx.image((64, 64), 'rgba8unorm')

    ctx.new_frame()
    image.clear()
    array.clear_value = (0.0, 0.0, 0.0, 1.0)
    array.face(1).clear()
    array.face(2).blit(image.face())
    ctx.end_frame()
    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [200, 100, 0, 255],
            [200, 100, 0, 255],
            [200, 100, 0, 255],
            [200, 100, 0, 255],
        ],
    )

    ctx.new_frame()
    image.clear()
    array.clear_value = (0.0, 0.0, 1.0, 1.0)
    array.face(2).clear()
    array.face(2).blit(image.face())
    ctx.end_frame()
    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
    )

    ctx.new_frame()
    array.mipmaps()
    image.clear()
    array.face(layer=2, level=1).blit(image.face(), size=(32, 32))
    ctx.end_frame()
    pixels = np.frombuffer(image.read(), 'u1').reshape(64, 64, 4)
    np.testing.assert_array_equal(
        pixels[[16, 16, 48, 48], [16, 48, 16, 48]],
        [
            [0, 0, 255, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )
