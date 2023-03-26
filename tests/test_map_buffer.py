import pytest
import zengl


def test_map_buffer(ctx: zengl.Context):
    buffer = ctx.buffer(b'\x55\xAA' * 32)
    mem = buffer.map()
    data = mem.tobytes()
    buffer.unmap()

    assert data == b'\x55\xAA' * 32


def test_map_buffer_twice(ctx: zengl.Context):
    buffer = ctx.buffer(b'\x55\xAA' * 32)
    buffer.map()

    with pytest.raises(RuntimeError):
        buffer.write(b'hello')

    with pytest.raises(RuntimeError):
        buffer.map()

    buffer.unmap()


def test_map_buffer_many_times(ctx: zengl.Context):
    buffer = ctx.buffer(size=32)

    buffer.map()
    buffer.unmap()
    buffer.map()
    buffer.unmap()
    buffer.map()
    buffer.unmap()
