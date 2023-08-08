import numpy as np
import zengl


def test_read_depth16(ctx: zengl.Context):
    img = ctx.image((4, 4), 'depth16unorm')

    img.clear_value = 0.0
    img.clear()
    content = np.frombuffer(img.read(), 'u2').reshape(4, 4).astype(float) / 0xFFFF
    expected = np.zeros((4, 4))
    np.testing.assert_array_almost_equal(content, expected, 2)

    img.clear_value = 0.5
    img.clear()
    content = np.frombuffer(img.read(), 'u2').reshape(4, 4).astype(float) / 0xFFFF
    expected = np.full((4, 4), 0.5)
    np.testing.assert_array_almost_equal(content, expected, 2)


def test_read_depth24(ctx: zengl.Context):
    img = ctx.image((4, 4), 'depth24plus')

    img.clear_value = 0.0
    img.clear()
    packed = np.frombuffer(img.read(), 'u4').reshape(4, 4)
    content = (packed >> 8).astype(float) / 0xFFFFFF
    expected = np.zeros((4, 4))
    np.testing.assert_array_almost_equal(content, expected, 2)

    img.clear_value = 0.5
    img.clear()
    packed = np.frombuffer(img.read(), 'u4').reshape(4, 4)
    content = (packed >> 8).astype(float) / 0xFFFFFF
    expected = np.full((4, 4), 0.5)
    np.testing.assert_array_almost_equal(content, expected, 2)


def test_read_depth24_stencil8(ctx: zengl.Context):
    img = ctx.image((4, 4), 'depth24plus-stencil8')

    img.clear_value = (0.0, 0)
    img.clear()
    packed = np.frombuffer(img.read(), 'u4').reshape(4, 4)
    content = (packed >> 8).astype(float) / 0xFFFFFF
    expected = np.zeros((4, 4))
    np.testing.assert_array_almost_equal(content, expected, 2)
    content = packed & 0xFF
    expected = np.zeros((4, 4), int)
    np.testing.assert_array_equal(content, expected)

    img.clear_value = (0.5, 17)
    img.clear()
    packed = np.frombuffer(img.read(), 'u4').reshape(4, 4)
    content = (packed >> 8).astype(float) / 0xFFFFFF
    expected = np.full((4, 4), 0.5)
    np.testing.assert_array_almost_equal(content, expected, 2)
    content = packed & 0xFF
    expected = np.full((4, 4), 17)
    np.testing.assert_array_equal(content, expected)


def test_read_depth32(ctx: zengl.Context):
    img = ctx.image((4, 4), 'depth32float')

    img.clear_value = 0.0
    img.clear()
    content = np.frombuffer(img.read(), 'f4').reshape(4, 4)
    expected = np.zeros((4, 4))
    np.testing.assert_array_almost_equal(content, expected, 2)

    img.clear_value = 0.5
    img.clear()
    content = np.frombuffer(img.read(), 'f4').reshape(4, 4)
    expected = np.full((4, 4), 0.5)
    np.testing.assert_array_almost_equal(content, expected, 2)
