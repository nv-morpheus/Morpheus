import os
from morpheus.utils.env_config_value import EnvConfigValue
from unittest import mock
import pytest

class EnvDrivenValue(EnvConfigValue):
    _ENV_KEY          = "DEFAULT"
    _ENV_KEY_OVERRIDE = "OVERRIDE"


class EnvDriverValueNoOverride(EnvConfigValue):
    _ENV_KEY = "DEFAULT"


class EnvDrivenValueNoDefault(EnvConfigValue):
    _ENV_KEY_OVERRIDE = "OVERRIDE"


def test_env_driven_value():
    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com"}):
        assert str(EnvDrivenValue(None)) == "default.api.com"
        assert str(EnvDrivenValue("api.com")) == "api.com"

    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com", "OVERRIDE": "override.api.com"}):
        assert str(EnvDrivenValue("api.com")) == "override.api.com"
        assert str(EnvDrivenValue("api.com", use_env=False)) == "api.com"

        with pytest.raises(ValueError):
            EnvDrivenValue(None, use_env=False)


def test_env_driven_value_no_override():
    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com"}):
        assert str(EnvDriverValueNoOverride(None)) == "default.api.com"
        assert str(EnvDriverValueNoOverride("api.com")) == "api.com"

    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com", "OVERRIDE": "override.api.com"}):
        assert str(EnvDriverValueNoOverride("api.com")) == "api.com"
        assert str(EnvDriverValueNoOverride("api.com", use_env=False)) == "api.com"


def test_env_driven_value_no_default():
    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com"}):
        with pytest.raises(ValueError):
            EnvDrivenValueNoDefault(None)

        assert str(EnvDrivenValueNoDefault("api.com")) == "api.com"

    with mock.patch.dict(os.environ, clear=True, values={"DEFAULT": "default.api.com", "OVERRIDE": "override.api.com"}):
        assert str(EnvDrivenValueNoDefault("api.com")) == "override.api.com"
        assert str(EnvDrivenValueNoDefault("api.com", use_env=False)) == "api.com"

        with pytest.raises(ValueError):
            EnvDrivenValueNoDefault(None, use_env=False)
