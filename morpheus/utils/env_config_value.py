from abc import ABC
import os

class EnvConfigValue(ABC):

    _CONFIG_NAME: str | None = None

    def __init__(self, value: str, use_env: bool = True):

        if use_env:
            assert self.__class__._CONFIG_NAME is not None

            if value is None:
                value = os.environ.get(f"{self.__class__._CONFIG_NAME}_DEFAULT", None)

            value = os.environ.get(f"{self.__class__._CONFIG_NAME}_OVERRIDE", value)

            if value is None:
                raise ValueError("value is None and no environment-based default or override was found")

        self._value = value

    def __str__(self):
        return self._value
