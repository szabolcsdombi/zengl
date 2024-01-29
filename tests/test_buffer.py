import pytest
import zengl


def test_buffer_data(ctx: zengl.Context):
    buf = ctx.buffer(b'Hello World!')
    assert buf.read() == b'Hello World!'


def test_buffer_data_non_contiguous(ctx: zengl.Context):
    buf = ctx.buffer(memoryview(b'Hello World!')[::2])
    assert buf.read() == b'HloWrd'


def test_buffer_write(ctx: zengl.Context):
    buf = ctx.buffer(size=12)
    buf.write(b'Hello World!')
    assert buf.read() == b'Hello World!'


def test_buffer_write_offset(ctx: zengl.Context):
    buf = ctx.buffer(b'---------------')
    buf.write(b'Hello', offset=2)
    buf.write(b'World', offset=9)
    assert buf.read() == b'--Hello--World-'


def test_buffer_copy(ctx: zengl.Context):
    buf1 = ctx.buffer(b'aaaaaaaaaa')
    buf2 = ctx.buffer(b'bbbbbbbbbb')
    assert buf2.read() == b'bbbbbbbbbb'
    buf2.write(buf1)
    assert buf2.read() == b'aaaaaaaaaa'


def test_buffer_copy_with_read(ctx: zengl.Context):
    buf1 = ctx.buffer(b'aaaaaaaaaa')
    buf2 = ctx.buffer(b'bbbbbbbbbb')
    assert buf2.read() == b'bbbbbbbbbb'
    buf1.read(into=buf2)
    assert buf2.read() == b'aaaaaaaaaa'


def test_buffer_copy_offset(ctx: zengl.Context):
    buf1 = ctx.buffer(b'aaaaaaaaaa')
    buf2 = ctx.buffer(b'bbbbbbbbbb')
    assert buf2.read() == b'bbbbbbbbbb'
    buf2.write(buf1.view(4), offset=4)
    assert buf2.read() == b'bbbbaaaabb'


def test_buffer_copy_offset_with_read(ctx: zengl.Context):
    buf1 = ctx.buffer(b'aaaaaaaaaa')
    buf2 = ctx.buffer(b'bbbbbbbbbb')
    assert buf2.read() == b'bbbbbbbbbb'
    buf1.read(size=4, into=buf2.view(offset=4, size=4))
    assert buf2.read() == b'bbbbaaaabb'


def test_buffer_write_error(ctx: zengl.Context):
    buf = ctx.buffer(b'bbbbbbbbbb')

    with pytest.raises(ValueError):
        buf.write(b'aaaaaaaaaa', offset=-1)

    with pytest.raises(ValueError):
        buf.write(b'aaaaaaaaaa', offset=4)


def test_buffer_copy_error(ctx: zengl.Context):
    buf1 = ctx.buffer(b'aaaaaaaaaa')
    buf2 = ctx.buffer(b'bbbbbbbbbb')

    with pytest.raises(ValueError):
        buf2.write(buf1.view(), offset=-1)

    with pytest.raises(ValueError):
        buf2.write(buf1.view(), offset=4)


def test_buffer_data_and_size(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.buffer(b'Hello World!', size=64)


def test_buffer_read_offset(ctx: zengl.Context):
    buf = ctx.buffer(b'0123456789')
    assert buf.read() == b'0123456789'
    assert buf.read(4) == b'0123'
    assert buf.read(offset=5) == b'56789'
    assert buf.read(offset=6, size=3) == b'678'


def test_buffer_read_error(ctx: zengl.Context):
    buf = ctx.buffer(b'bbbbbbbbbb')

    with pytest.raises(ValueError):
        buf.read(100)

    with pytest.raises(ValueError):
        buf.read(offset=-1)

    with pytest.raises(ValueError):
        buf.read(offset=4, size=10)

    with pytest.raises(ValueError):
        buf.read(size=-1)


def test_buffer_read_into(ctx: zengl.Context):
    buf = ctx.buffer(b'0123456789')

    mem = memoryview(bytearray(10))
    buf.read(into=mem)
    assert bytes(mem) == b'0123456789'

    mem = memoryview(bytearray(4))
    buf.read(4, into=mem)
    assert bytes(mem) == b'0123'

    mem = memoryview(bytearray(5))
    buf.read(offset=5, into=mem)
    assert bytes(mem) == b'56789'

    mem = memoryview(bytearray(3))
    buf.read(offset=6, size=3, into=mem)
    assert bytes(mem) == b'678'


def test_invalid_access(ctx):
    with pytest.raises(ValueError):
        ctx.buffer(size=64, access='bad')
