import pytest
import zengl
from glcontext import egl


@pytest.fixture
def ctx():
    ctx = zengl.context(egl.create_context(glversion=330, mode='standalone'))
    yield ctx
    ctx.release('all')
    ctx.release('shader_cache')
    assert len(ctx.gc()) == 0
