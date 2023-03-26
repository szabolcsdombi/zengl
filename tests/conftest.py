import sys

import pytest
import zengl


@pytest.fixture
def ctx():
    if sys.platform.startswith('linux'):
        from glcontext import headless
        devices = headless.devices()
        headless.init(device=next(x['device'] for x in devices if 'EGL_MESA_device_software' in x['extensions']))
        ctx = zengl.context(headless)
    else:
        ctx = zengl.context(zengl.loader(headless=True))
    yield ctx
    ctx.release('all')
    ctx.release('shader_cache')
    assert len(ctx.gc()) == 0
