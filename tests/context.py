import pytest
import zengl
from glcontext import egl


class MissingLoaderMethod:
    pass


class BrokenLoaderMethod:
    def load_opengl_function(self):
        return 0


class BrokenLoaderResult:
    def load_opengl_function(self, name):
        return lambda: None


class BrokenLoader:
    def load_opengl_function(self, name):
        return 0


def test_context():
    with pytest.raises(RuntimeError):
        zengl.context()

    with pytest.raises(RuntimeError):
        zengl.init()

    with pytest.raises(ValueError):
        zengl.init(MissingLoaderMethod())

    with pytest.raises(TypeError):
        zengl.init(BrokenLoaderMethod())

    with pytest.raises(TypeError):
        zengl.init(BrokenLoaderResult())

    with pytest.raises(RuntimeError):
        zengl.init(BrokenLoader())

    loader = egl.create_context(glversion=330, mode="standalone")
    zengl.init(loader)

    ctx1 = zengl.context()
    ctx2 = zengl.context()
    assert ctx1 is ctx2
