import os
from morpheus.utils.env_config_value import EnvConfigValue

class MyApi:

    class BaseUri(EnvConfigValue):
        _CONFIG_NAME = "MY_API_BASE_URI"

    def __init__(self, base_uri: BaseUri):
        self._base_uri = str(base_uri)


from unittest import mock


def test_os_config_value():

    with mock.patch.dict(os.environ, clear=True, values={"MY_API_BASE_URI_DEFAULT": "default.api.com"}):
        assert str(MyApi.BaseUri(None)) == "default.api.com"

    with mock.patch.dict(os.environ, clear=True, values={"MY_API_BASE_URI_DEFAULT": "default.api.com"}):
        assert str(MyApi.BaseUri("api.com")) == "api.com"

    with mock.patch.dict(os.environ, clear=True, values={"MY_API_BASE_URI_DEFAULT": "default.api.com", "MY_API_BASE_URI_OVERRIDE": "override.api.com"}):
        assert str(MyApi.BaseUri("api.com")) == "override.api.com"

    with mock.patch.dict(os.environ, clear=True, values={"MY_API_BASE_URI_DEFAULT": "default.api.com", "MY_API_BASE_URI_OVERRIDE": "override.api.com"}):
        assert str(MyApi.BaseUri("api.com", use_env=False)) == "api.com"
