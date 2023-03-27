import zengl


def test_read_multisampled_image(ctx: zengl.Context):
    img = ctx.image((4, 4), 'rgba8unorm', samples=4)
    img.clear_value = (0.0, 1.0, 0.0, 1.0)
    img.clear()
    assert img.read() == b'\x00\xff\x00\xff' * 16
