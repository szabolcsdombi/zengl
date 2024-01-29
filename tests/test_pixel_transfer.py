import zengl


def test_image_from_buffer(ctx: zengl.Context):
    buf = ctx.buffer(b'AAAA' * 16)
    img = ctx.image((4, 4), 'rgba8unorm', buf)
    assert img.read() == b'AAAA' * 16


def test_image_write_from_buffer(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'AAAA' * 16)
    buf = ctx.buffer(b'BBBB' * 16)
    img.write(buf)
    assert img.read() == b'BBBB' * 16


def test_image_write_from_buffer_offset(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'AAAA' * 16)
    buf = ctx.buffer(b'UUUUUUUU' + b'BBBB' * 16 + b'VVVVVVVV')
    img.write(buf.view(64, 8))
    assert img.read() == b'BBBB' * 16


def test_image_read_to_buffer(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'AAAA' * 16)
    buf = ctx.buffer(b'BBBB' * 16)
    img.read(into=buf)
    assert buf.read() == b'AAAA' * 16


def test_image_read_to_buffer_offset(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', b'AAAA' * 16)
    buf = ctx.buffer(b'UUUUUUUU' + b'BBBB' * 16 + b'VVVVVVVV')
    img.read(into=buf.view(64, 8))
    assert buf.read() == b'UUUUUUUU' + b'AAAA' * 16 + b'VVVVVVVV'
