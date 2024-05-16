from abc import ABC
import os

class EnvConfigValue(ABC):

    _ENV_KEY: str | None = None
    _ENV_KEY_OVERRIDE: str | None = None

    def __init__(self, value: str, use_env: bool = True):

        if use_env:

            if value is None and self.__class__._ENV_KEY is not None:
                    value = os.environ.get(f"{self.__class__._ENV_KEY}", None)

            if self.__class__._ENV_KEY_OVERRIDE is not None:
                value = os.environ.get(f"{self.__class__._ENV_KEY_OVERRIDE}", value)

            if value is None:
                raise ValueError("value must not be None, but provided value was None and no environment-based default or override was found")

        else:
            if value is None:
                raise ValueError("value must not be none")

        self._value = value

    def __str__(self):
        return self._value


# PB, DOCA, Asynchronious Programming, Strong C++ Programmer