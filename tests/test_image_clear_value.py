import numpy as np
import pytest
import zengl


def test_float_image(ctx: zengl.Context):
    img = ctx.image((4, 4), 'r32float')
    img.clear_value = 0.33
    np.testing.assert_array_almost_equal([img.clear_value], [0.33], 2)
    img.clear_value = 0.75
    np.testing.assert_array_almost_equal([img.clear_value], [0.75], 2)


def test_int_image(ctx: zengl.Context):
    img = ctx.image((4, 4), 'r32sint')
    img.clear_value = 42
    np.testing.assert_array_almost_equal([img.clear_value], [42], 2)
    img.clear_value = 123456789
    np.testing.assert_array_almost_equal([img.clear_value], [123456789], 2)


def test_float_vec_image(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rg32float')
    img.clear_value = 0.33, 0.66
    np.testing.assert_array_almost_equal(img.clear_value, [0.33, 0.66], 2)
    img.clear_value = 0.75, 0.12
    np.testing.assert_array_almost_equal(img.clear_value, [0.75, 0.12], 2)


def test_int_vec_image(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba16sint')
    img.clear_value = 10, 20, 30, 40
    np.testing.assert_array_almost_equal(img.clear_value, [10, 20, 30, 40], 2)
    img.clear_value = 50, 60, 70, 80
    np.testing.assert_array_almost_equal(img.clear_value, [50, 60, 70, 80], 2)


def test_depth_stencil_image(ctx: zengl.Context):
    img = ctx.image((4, 4), 'depth24plus-stencil8')
    img.clear_value = 0.3, 200
    np.testing.assert_array_almost_equal(img.clear_value, [0.3, 200], 2)
    img.clear_value = 0.88, 12
    np.testing.assert_array_almost_equal(img.clear_value, [0.88, 12], 2)


def test_invalid_clear_value(ctx: zengl.Context):
    img_float = ctx.image((4, 4), 'r32float')
    img_int = ctx.image((4, 4), 'r32sint')
    img_depth = ctx.image((4, 4), 'depth24plus')
    img_depth_stencil = ctx.image((4, 4), 'depth24plus-stencil8')

    with pytest.raises(TypeError):
        img_float.clear_value = 100

    with pytest.raises(TypeError):
        img_float.clear_value = (100, 100)

    with pytest.raises(TypeError):
        img_float.clear_value = '100'

    with pytest.raises(TypeError):
        img_int.clear_value = 0.5

    with pytest.raises(TypeError):
        img_int.clear_value = (0.5, 0.5)

    with pytest.raises(TypeError):
        img_int.clear_value = b'0.5'

    with pytest.raises(TypeError):
        img_depth.clear_value = 2

    # with pytest.raises(TypeError):
    #     img_depth_stencil.clear_value = 2, 2

    with pytest.raises(TypeError):
        img_depth_stencil.clear_value = 0.5, 0.5
