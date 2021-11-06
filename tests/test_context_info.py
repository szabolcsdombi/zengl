import zengl


def test_context_info(ctx: zengl.Context):
    assert isinstance(ctx.info, tuple)
    assert len(ctx.info) == 3
