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
import logging

import mrc
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import SUPPORTED_DATA_TYPES
from morpheus.utils.module_ids import SUPPORTED_TASK_TYPES
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(TO_CONTROL_MESSAGE, MORPHEUS_MODULE_NAMESPACE)
def to_control_message(builder: mrc.Builder):
    """
    This module converts MessageMeta to a ControlMessage.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - tasks (array[dict]): Control message tasks configuration.
            - meta_data (dict): Control message metadata Configuration.
    """

    config = builder.get_current_module_config()

    meta_data = config.get("meta_data", None)
    tasks = config.get("tasks", None)

    if not meta_data:
        raise ValueError(
            "The `meta_data` required to add to the control message has not been provided in the module configuration.")

    # Validate meta_data configuration
    if "data_type" in meta_data and meta_data["data_type"] not in SUPPORTED_DATA_TYPES:
        raise ValueError(f"Unsupported data type: {meta_data['data_type']}")

    # Validate tasks configuration
    if not tasks:
        raise ValueError(
            "The `tasks` required to add to the control message has not been provided in the module configuration.")

    control_message = ControlMessage()

    # Set meta_data configuration to control message
    for key, value in meta_data.items():
        control_message.set_metadata(key, value)

    # Validate and add tasks to control message
    for task in tasks:
        task_type = task.get("type")
        task_properties = task.get("properties", {})
        if task_type in SUPPORTED_TASK_TYPES:
            control_message.add_task(task.get("type"), task_properties)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def meta_to_control_message(x: MessageMeta) -> ControlMessage:
        # Copying control message to avoid setting same metadata for every batch.
        control_message_copy = control_message.copy()

        control_message_copy.payload(x)

        return control_message_copy

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(meta_to_control_message)).subscribe(sub)

    node = builder.make_node(TO_CONTROL_MESSAGE, mrc.core.operators.build(node_fn))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
