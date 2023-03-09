# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import typing

import mrc

logger = logging.getLogger(__name__)

registry = mrc.ModuleRegistry
mrc_version = [int(i) for i in mrc.__version__.split('.')]


def verify_module_registration(func):
    """
    Module availability in the module registry is verified by this function.

    Parameters
    ----------
    func : Function that requires wrapping.

    Returns
    -------
    inner_func
        Encapsulated function.
    """

    @functools.wraps(func)
    def inner_func(config, **kwargs):

        verify_module_meta_fields(config)

        module_id = config.get("module_id")
        namespace = config.get("namespace")

        if module_id is None or namespace is None:
            raise TypeError("TypeError: a string-like object is required for module_id and namespace, not 'NoneType'")

        if not registry.contains(module_id, namespace):
            raise Exception("Module '{}' doesn't exist in the namespace '{}'".format(module_id, namespace))

        return func(config, **kwargs)

    return inner_func


def register_module(module_id, namespace):
    """
    Registers a module if not exists in the module registry.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.

    Returns
    -------
    inner_func
        Encapsulated function.
    """

    def inner_func(func):
        # Register a module if not exists in the registry.
        if not registry.contains(module_id, namespace):
            registry.register_module(module_id, namespace, mrc_version, func)
            logger.debug("Module '{}' was successfully registered with '{}' namespace.".format(module_id, namespace))
        else:
            logger.debug("Module: '{}' already exists in the given namespace '{}'".format(module_id, namespace))

        return func

    return inner_func


@verify_module_registration
def load_module(config: typing.Dict, builder: mrc.Builder):
    """
    Loads a module that exists in the module registry.

    Parameters
    ----------
    config : typing.Dict
        Module configuration.
    builder : mrc.Builder
        MRC Builder object.

    Returns
    -------
    module
        Module object.
    """

    module_id = config.get("module_id")
    namespace = config.get("namespace")
    module_name = config.get("module_name")

    module = builder.load_module(module_id, namespace, module_name, config)

    logger.debug("Module '{}' with namespace '{}' is successfully loaded.".format(module_id, namespace))

    return module


def verify_module_meta_fields(config: typing.Dict):
    """
    This function make sure the module configuration contains meta fields.

    Parameters
    ----------
    config : typing.Dict
        Module configuration.
    """

    if "module_id" not in config:
        raise KeyError("Required attribute 'module_id' is missing in the module configuration.")
    if "namespace" not in config:
        raise KeyError("Required attribute 'namespace' is missing in the module configuration.")
    if "module_name" not in config:
        raise KeyError("Required attribute 'module_name' is missing in the module configuration.")


def merge_dictionaries(primary_dict, secondary_dict):
    """Recursively merge two dictionaries, using primary_dict as a tie-breaker.

    Lists are treated as a special case, and all unique elements from both dictionaries are included in the final list.

    Args:
        primary_dict (dict): The primary dictionary.
        secondary_dict (dict): The secondary dictionary.

    Returns:
        dict: The merged dictionary.
    """
    result_dict = primary_dict.copy()

    for key, value in secondary_dict.items():
        if key in result_dict:
            if isinstance(value, list) and isinstance(result_dict[key], list):
                # Combine the two lists and remove duplicates while preserving order
                # This isn't perfect, its possible we could still end up with duplicates in some scenarios
                combined_list = result_dict[key] + value
                unique_list = []
                for item in combined_list:
                    if item not in unique_list:
                        unique_list.append(item)
                result_dict[key] = unique_list
            elif isinstance(value, dict) and isinstance(result_dict[key], dict):
                # Recursively merge the two dictionaries
                result_dict[key] = merge_dictionaries(result_dict[key], value)
        else:
            result_dict[key] = value

    return result_dict


def get_config_with_overrides(config, module_id, module_name=None, module_namespace="morpheus"):
    sub_config = config.get(module_id, None)

    try:
        if module_name is None:
            module_name = sub_config.get("module_name")
    except Exception:
        raise KeyError(f"'module_name' is not set in the '{module_id}' module configuration")

    sub_config.setdefault("module_id", module_id)
    sub_config.setdefault("module_name", module_name)
    sub_config.setdefault("namespace", module_namespace)

    return sub_config


def get_module_config(module_id: str, builder: mrc.Builder):
    """
    Returns the module configuration for the specified module id.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    builder : mrc.Builder
        MRC Builder object.

    Returns
    -------
    config : typing.Dict
        Module configuration.

    """

    config = builder.get_current_module_config()
    if module_id in config:
        config = config[module_id]

    verify_module_meta_fields(config)

    return config


def make_nested_module(module_id: str, namespace: str, ordered_modules_meta: typing.List[typing.Dict[str, str]]):
    """
    This function creates a nested module and registers it in the module registry.
    This module unifies a chain of two or more modules into a single module.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    ordered_modules_meta : typing.List[typing.Dict[str, str]]
        The sequence in which the edges between the nodes are made will be determined by ordered modules meta.
    """

    @register_module(module_id, namespace)
    def module_init(builder: mrc.Builder):

        prev_module = None
        head_module = None

        # Make edges between set of modules and wrap the internally connected modules as module.
        #                        Wrapped Module
        #                _______________________________
        #
        #    input >>   | Module1 -- Module2 -- Module3 |   >> output
        #                ________________________ ______
        for config in ordered_modules_meta:

            curr_module = load_module(config, builder=builder)

            if prev_module:
                builder.make_edge(prev_module.output_port("output"), curr_module.input_port("input"))
            else:
                head_module = curr_module

            prev_module = curr_module

        # Register input and output port for a module.
        builder.register_module_input("input", head_module.input_port("input"))
        builder.register_module_output("output", prev_module.output_port("output"))
