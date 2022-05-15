import pytest
import zengl


def test_invalid_image_size(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.image()

    with pytest.raises(TypeError):
        ctx.image(size=1, format='rgba8unorm')

    with pytest.raises(TypeError):
        ctx.image(size=(1,), format='rgba8unorm')

    with pytest.raises(TypeError):
        ctx.image(size=(1, '2'), format='rgba8unorm')

    with pytest.raises(TypeError):
        ctx.image(size=(2.3, 7.1), format='rgba8unorm')

    with pytest.raises(TypeError):
        ctx.image(size=(1, 2, 3), format='rgba8unorm')


def test_invalid_image_format(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.image(size=(4, 4), format=None)

    with pytest.raises(ValueError):
        ctx.image(size=(4, 4), format='')


def test_invalid_image_data(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.image((1, 1), 'rgba8unorm', data='data')

    with pytest.raises(BufferError):
        ctx.image((1, 1), 'rgba8unorm', data=memoryview(b'12345678')[::2])

    with pytest.raises(TypeError):
        ctx.image((1, 1), 'rgba8unorm', data=100)

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', data=b'123')

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', data=b'12345')

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', data=b'1234', samples=2)

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', data=b'12345678', samples=2)

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', data=b'1234', texture=False)


def test_invalid_image_samples(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', samples=0)

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', samples=3)

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', samples=1024)

    with pytest.raises(TypeError):
        ctx.image((1, 1), 'rgba8unorm', samples=2, texture=True)


def test_invalid_image_array(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', array=-1)

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', array=2, data=b'1234')

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', array=2, data=b'1234567')

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', array=2, data=b'123456789')


def test_invalid_image_cubemap(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', cubemap=True, data=b'1234')

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', cubemap=True, data=b'aaaabbbbccccddddeeeefff')

    with pytest.raises(ValueError):
        ctx.image((1, 1), 'rgba8unorm', cubemap=True, data=b'aaaabbbbccccddddeeeefffff')


def test_invalid_image_type(ctx: zengl.Context):
    with pytest.raises(TypeError):
        ctx.image((1, 1), 'rgba8unorm', samples=2, array=2)

    with pytest.raises(TypeError):
        ctx.image((1, 1), 'rgba8unorm', samples=2, cubemap=True)

    with pytest.raises(TypeError):
        ctx.image((1, 1), 'rgba8unorm', array=2, cubemap=True)
