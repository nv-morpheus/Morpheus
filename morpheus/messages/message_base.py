# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import dataclasses
import os
import typing

from morpheus.config import CppConfig

# If set, this disables all CPP class creation
NO_CPP = os.getenv("MORPHEUS_NO_CPP", 'False').lower() in ('true', '1', 't')


class MessageImpl(abc.ABCMeta):

    _cpp_class: typing.Union[type, typing.Callable] = None
    """
    Metaclass to switch between Python & C++ message implementations at construction time.
    Note: some classes don't have a C++ implementation, but do inherit from a class that
    does (ex UserMessageMeta & InferenceMemoryAE) these classes also need this metaclass
    to prevent creating instances of their parent's C++ impl.
    """

    def __new__(cls, classname, bases, classdict, cpp_class=None):
        result = super().__new__(cls, classname, bases, classdict)

        # Set the C++ class type into the object to use for creation later if desired
        result._cpp_class = None if NO_CPP else cpp_class

        # Register the C++ class as an instances of this metaclass to support isinstance(cpp_instance, PythonClass)
        if (cpp_class is not None):
            result.register(cpp_class)

        return result


class MessageBase(metaclass=MessageImpl):

    def __new__(cls, *args, **kwargs):

        # If _cpp_class is set, and use_cpp is enabled, create the C++ instance
        if (getattr(cls, "_cpp_class", None) is not None and CppConfig.get_should_use_cpp()):
            return cls._cpp_class(*args, **kwargs)

        # Otherwise, do the default init
        return super().__new__(cls)


@dataclasses.dataclass
class MessageData(MessageBase):

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
