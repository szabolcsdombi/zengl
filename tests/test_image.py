import zengl


def test_image_data(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'RGBA' * 16)
    assert img.read() == b'RGBA' * 16


def test_image_write(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm')
    img.write(b'RGBA' * 16)
    assert img.read() == b'RGBA' * 16


def test_image_write(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'RGBA' * 16)
    img.write(b'00001111', (2, 1), offset=(2, 3))
    assert img.read() == b'RGBA' * 14 + b'00001111'
    assert img.read(offset=(1, 3), size=(1, 1)) == b'RGBA'
    assert img.read(offset=(2, 3), size=(1, 1)) == b'0000'
    assert img.read(offset=(3, 3), size=(1, 1)) == b'1111'


def test_image_read_into(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'RGBA' * 16)
    img.write(b'00001111', (2, 1), offset=(2, 3))

    mem = memoryview(bytearray(64))
    img.read(into=mem)
    assert bytes(mem) == b'RGBA' * 14 + b'00001111'

    mem = memoryview(bytearray(4))
    img.read(offset=(1, 3), size=(1, 1), into=mem)
    assert bytes(mem) == b'RGBA'

    mem = memoryview(bytearray(4))
    img.read(offset=(2, 3), size=(1, 1), into=mem)
    assert bytes(mem) == b'0000'

    mem = memoryview(bytearray(4))
    img.read(offset=(3, 3), size=(1, 1), into=mem)
    assert bytes(mem) == b'1111'
