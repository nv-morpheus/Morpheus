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

import asyncio
import typing

from morpheus.llm import InputMap
from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase


def execute_node(node: LLMNodeBase, **input_values: dict) -> typing.Any:
    """
    Executes an LLM Node with the necessary LLM context, and extracts the output values.
    """
    inputs: list[InputMap] = []
    parent_context = LLMContext()

    for input_name, input_value in input_values.items():
        inputs.append(InputMap(f"/{input_name}", input_name))
        parent_context.set_output(input_name, input_value)

    context = LLMContext(parent_context, "test", inputs)

    asyncio.run(node.execute(context))

    return context.view_outputs