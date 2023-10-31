import pytest
import zengl
from glcontext import egl


@pytest.fixture
def ctx():
    zengl.init(egl.create_context(glversion=330, mode="standalone"))
    ctx = zengl.context()
    yield ctx
    ctx.release("all")
    ctx.release("shader_cache")
    assert len(ctx.gc()) == 0
