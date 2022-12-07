# Copyright (c) 2022, NVIDIA CORPORATION.
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

import functools
import logging

import srf

from morpheus.utils.version_utils import get_srf_version_as_list

logger = logging.getLogger(f"morpheus.{__name__}")

registry = srf.ModuleRegistry


def is_module_registered(func):
    """
    Module availability in the module registry is verified by this function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        module_id = kwargs["module_id"]
        namespace = kwargs["namespace"]

        if module_id is None or namespace is None:
            raise TypeError("TypeError: a string-like object is required for module_id and namespace, not 'NoneType'")

        if not registry.contains(module_id, namespace):
            raise Exception("Module {} doesn't exist in the namespace {}".format(module_id, namespace))

        return func(*args, **kwargs)

    return wrapper


def register_module(**kwargs):

    def wrapper(func):

        module_id = kwargs["module_id"]
        namespace = kwargs["namespace"]

        # Register a module if not exists in the registry.
        if not registry.contains(module_id, namespace):
            registry.register_module(module_id, namespace, get_srf_version_as_list(), func)
            logger.info("Module {} was successfully registered with {} namespace.".format(module_id, namespace))
        else:
            logger.info("Module: {} already exists in the given namespace: {}".format(module_id, namespace))

    return wrapper
