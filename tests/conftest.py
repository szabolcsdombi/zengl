import pytest
import zengl
from glcontext import egl


class state:
    initialized = False


@pytest.fixture
def ctx():
    if not state.initialized:
        zengl.init(egl.create_context(glversion=330, mode='standalone'))
        state.initialized = True
    ctx = zengl.context()
    ctx.new_frame(reset=True, clear=False)
    yield ctx
    ctx.end_frame()
    ctx.release('all')
    ctx.release('shader_cache')
    assert len(ctx.gc()) == 0
