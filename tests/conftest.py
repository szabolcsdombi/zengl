import pytest
import zengl


@pytest.fixture
def ctx():
    return zengl.context(zengl.loader(headless=True))
