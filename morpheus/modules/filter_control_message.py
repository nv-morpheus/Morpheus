# Copyright (c) 2023, NVIDIA CORPORATION.
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

import mrc
from mrc.core import operators as ops

from morpheus.messages import MessageControl
from morpheus.utils.module_ids import FILTER_CONTROL_MESSAGE
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILTER_CONTROL_MESSAGE, MODULE_NAMESPACE)
def filter_control_message(builder: mrc.Builder):
    """
    When the requirements are met, this module gently discards the control messages.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.
    """

    config = get_module_config(FILTER_CONTROL_MESSAGE, builder)

    filter = config.get("data_type", None)
    enable_task_check = config.get("enable_task_check", False)
    task_type = config.get("task_type", None)

    def on_data(control_message: MessageControl):
        data_type = control_message.get_metadata("data_type")

        if enable_task_check:
            # Verify if control message has expected task_type.
            task_exist = control_message.has_task(task_type)
            # Dispose messages if it has no expected task and it's data_type does not matches with filter.
            if (not task_exist and filter and data_type != filter):
                return None
        else:
            # Regardless of whether tasks are present, discard messages
            # if the data_type don't match the filter.
            if filter and data_type != filter:
                return None

        return control_message

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

    node = builder.make_node_full(FILTER_CONTROL_MESSAGE, node_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
