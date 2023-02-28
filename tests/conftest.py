import pytest
import zengl
from glcontext import headless


@pytest.fixture
def ctx():
    devices = headless.devices()
    headless.init(device=next(x['device'] for x in devices if 'EGL_MESA_device_software' in x['extensions']))
    return zengl.context(headless)
