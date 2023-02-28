import sys

import pytest
import zengl


@pytest.fixture
def ctx():
    if sys.platform.startswith('linux'):
        from glcontext import headless
        devices = headless.devices()
        headless.init(device=next(x['device'] for x in devices if 'EGL_MESA_device_software' in x['extensions']))
        return zengl.context(headless)
    else:
        return zengl.context(zengl.loader(headless=True))
