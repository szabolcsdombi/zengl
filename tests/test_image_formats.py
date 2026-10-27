import pytest
import zengl

import utils

IMAGE_FORMATS = [
    'r8unorm',
    'rg8unorm',
    'rgba8unorm',
    'r8snorm',
    'rg8snorm',
    'rgba8snorm',
    'r8uint',
    'rg8uint',
    'rgba8uint',
    'r16uint',
    'rg16uint',
    'rgba16uint',
    'r32uint',
    'rg32uint',
    'rgba32uint',
    'r8sint',
    'rg8sint',
    'rgba8sint',
    'r16sint',
    'rg16sint',
    'rgba16sint',
    'r32sint',
    'rg32sint',
    'rgba32sint',
    'r16float',
    'rg16float',
    'rgba16float',
    'r32float',
    'rg32float',
    'rgba32float',
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
]


def test_texture_image_formats(ctx: zengl.Context):
    utils.clear_gl_error()
    for fmt in IMAGE_FORMATS:
        img = ctx.image((4, 4), fmt, texture=True)
        utils.assert_gl_error(fmt)
        ctx.release(img)
        utils.assert_gl_error()


def test_renderbuffer_image_formats(ctx: zengl.Context):
    utils.clear_gl_error()
    for fmt in IMAGE_FORMATS:
        img = ctx.image((4, 4), fmt, texture=False)
        utils.assert_gl_error(fmt)
        ctx.release(img)
        utils.assert_gl_error()


@pytest.mark.parametrize("fmt, bytes_per_pxl", [
    ("r16float", 2),
    ("rg16float", 4),
    ("rgba16float", 8),
])
@pytest.mark.parametrize("tex_size", [
    (112, 112),
    (1112, 1112),  # Larger size that previously triggered a segfault
])
def test_half_float_texture_creation_and_readback(
    ctx: zengl.Context,
    fmt: str,
    bytes_per_pxl: int,
    tex_size: tuple[int, int]
):
    utils.clear_gl_error()

    width, height = tex_size
    data = b'\x00' * width * height * bytes_per_pxl

    img = ctx.image(tex_size, fmt, data)
    assert img is not None

    read_back = img.read()
    assert isinstance(read_back, bytes)
    assert len(read_back) > 0

    utils.assert_gl_error(fmt)
    ctx.release(img)
    utils.assert_gl_error()
