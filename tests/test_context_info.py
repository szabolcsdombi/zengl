import zengl


def test_context_info(ctx: zengl.Context):
    assert isinstance(ctx.info, dict)

    assert 'vendor' in ctx.info
    assert 'renderer' in ctx.info
    assert 'version' in ctx.info
    assert 'glsl' in ctx.info

    assert isinstance(ctx.info['vendor'], str)
    assert isinstance(ctx.info['renderer'], str)
    assert isinstance(ctx.info['version'], str)
    assert isinstance(ctx.info['glsl'], str)
