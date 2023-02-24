import zengl

import utils

IMAGE_FORMATS = [
    'r8unorm', 'rg8unorm', 'rgba8unorm', 'r8snorm', 'rg8snorm', 'rgba8snorm', 'r8uint', 'rg8uint',
    'rgba8uint', 'r16uint', 'rg16uint', 'rgba16uint', 'r32uint', 'rg32uint', 'rgba32uint', 'r8sint', 'rg8sint',
    'rgba8sint', 'r16sint', 'rg16sint', 'rgba16sint', 'r32sint', 'rg32sint', 'rgba32sint', 'r16float', 'rg16float',
    'rgba16float', 'r32float', 'rg32float', 'rgba32float', 'rgba8unorm-srgb', 'stencil8',
    'depth16unorm', 'depth24plus', 'depth24plus-stencil8', 'depth32float',
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
