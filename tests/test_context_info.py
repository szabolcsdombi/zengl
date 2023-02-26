import zengl


def test_context_info(ctx: zengl.Context):
    assert isinstance(ctx.info, dict)
    assert len(ctx.info) == 4
