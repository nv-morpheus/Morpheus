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

import logging
import typing

import srf

from morpheus.utils.decorators import register_module
from morpheus.utils.decorators import verify_module_registration

logger = logging.getLogger(f"morpheus.{__name__}")


def make_nested_module(module_id: str, namespace: str, ordered_modules_meta: typing.List[typing.Tuple[str, str]]):
    """
    This function creates a nested module and registers it in the module registry.
    This module unifies a chain of two or more modules into a single module.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    ordered_modules_meta : typing.List[typing.Tuple[str, str]]
        The sequence in which the edges between the nodes are made will be determined by ordered modules meta.
    """

    @verify_module_registration
    def verify_module_existance(module_id, namespace):
        logger.debug("Registry contains module {} with namespace {}".format(module_id, namespace))
        return True

    @register_module(module_id, namespace)
    def module_init(builder: srf.Builder):

        config = builder.get_current_module_config()

        if module_id in config:
            config = config[module_id]

        prev_module = None
        head_module = None

        # Make edges between set of modules and wrap the internally connected modules as module.
        #                        Wrapped Module
        #                _______________________________
        #
        #    input >>   | Module1 -- Module2 -- Module3 |   >> output
        #                ________________________ ______
        for x, y in ordered_modules_meta:

            verify_module_existance(x, y)

            curr_module = builder.load_module(x, y, module_id, config)

            if prev_module:
                builder.make_edge(prev_module.output_port("output"), curr_module.input_port("input"))
            else:
                head_module = curr_module

            prev_module = curr_module

        # Register input and output port for a module.
        builder.register_module_input("input", head_module.input_port("input"))
        builder.register_module_output("output", prev_module.output_port("output"))
