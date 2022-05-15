import pytest
import zengl


def test_invalid_buffer_parameters(ctx: zengl.Context):
    with pytest.raises(ValueError):
        ctx.buffer()

    with pytest.raises(TypeError):
        ctx.buffer('test')

    with pytest.raises(BufferError):
        ctx.buffer(memoryview(b'test')[::2])

    with pytest.raises(TypeError):
        ctx.buffer(100)

    with pytest.raises(ValueError):
        ctx.buffer(b'')

    with pytest.raises(ValueError):
        ctx.buffer(None)

    with pytest.raises(TypeError):
        ctx.buffer(size='999')

    with pytest.raises(ValueError):
        ctx.buffer(size=0)

    with pytest.raises(ValueError):
        ctx.buffer(size=-1)


def test_invalid_buffer_write_parameters(ctx: zengl.Context):
    buf = ctx.buffer(size=32)

    with pytest.raises(TypeError):
        buf.write('test')

    with pytest.raises(BufferError):
        buf.write(memoryview(b'test')[::2])

    with pytest.raises(TypeError):
        buf.write(10)

    with pytest.raises(TypeError):
        buf.write(None)

    with pytest.raises(ValueError):
        buf.write(b'123', 32 - 2)

    with pytest.raises(ValueError):
        buf.write(b'1', -1)

    with pytest.raises(ValueError):
        buf.write(b'1' * 33)

    with pytest.raises(TypeError):
        buf.write(b'1', offset=0.1)


def test_buffer_map_parameters(ctx: zengl.Context):
    buf = ctx.buffer(size=32)

    with pytest.raises(TypeError):
        buf.map(size=2.5)

    with pytest.raises(TypeError):
        buf.map(size='25')

    with pytest.raises(ValueError):
        buf.map(size=33)

    with pytest.raises(ValueError):
        buf.map(size=-1)

    with pytest.raises(ValueError):
        buf.map(size=0)

    with pytest.raises(TypeError):
        buf.map(size=1, offset=1.5)

    with pytest.raises(TypeError):
        buf.map(size=1, offset='15')

    with pytest.raises(ValueError):
        buf.map(offset=-1, size=1)

    with pytest.raises(ValueError):
        buf.map(offset=30, size=3)

    with pytest.raises(ValueError):
        buf.map(offset=64, size=1)

    buf.map(size=16, offset=8)
    with pytest.raises(RuntimeError):
        buf.map(size=16, offset=8)
