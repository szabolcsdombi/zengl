import pytest
import zengl


def test_invalid_image_read_multisample(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', samples=4)
    with pytest.raises(TypeError):
        img.read()


def test_invalid_image_read_cubemap(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', cubemap=True)
    with pytest.raises(TypeError):
        img.read()


def test_invalid_image_read_array(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', array=4)
    with pytest.raises(TypeError):
        img.read()
