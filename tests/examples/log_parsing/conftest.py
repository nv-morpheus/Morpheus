import pytest


@pytest.fixture
def config(config):
    from morpheus.config import PipelineModes
    config.mode = PipelineModes.NLP
    yield config
