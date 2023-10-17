from morpheus._lib.tests.pycoro import hey
import pytest

@pytest.mark.asyncio
def test_hey():
    hey()
