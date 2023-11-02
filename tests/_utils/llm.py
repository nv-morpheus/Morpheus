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
from morpheus.llm import LLMTask
from morpheus.llm import LLMTaskHandler
from morpheus.messages import ControlMessage


def _mk_context(parent_context: LLMContext, input_values: dict) -> LLMContext:
    inputs: list[InputMap] = []

    for input_name, input_value in input_values.items():
        inputs.append(InputMap(f"/{input_name}", input_name))
        parent_context.set_output(input_name, input_value)

    return parent_context.push("test", inputs)


def _execute_node(node: LLMNodeBase, context: LLMContext) -> typing.Any:

    async def execute():
        # `asyncio.run(obj)`` will raise a `ValueError`` if `asyncio.iscoutine(obj)` is `False` for composite nodes
        # that don't directly implement `execute()` this causes a failure because while
        # `mrc.core.coro.CppToPyAwaitable` is awaitable it is not a coroutine.

        return await node.execute(context)

    context = asyncio.run(execute())

    return context.view_outputs


def execute_node(node: LLMNodeBase,
                 task_dict: dict = None,
                 input_message: ControlMessage = None,
                 **input_values: dict) -> typing.Any:
    """
    Executes an LLM Node with the necessary LLM context, and extracts the output values.

    If `task_dict` and `input_message` are provided, then the context will be created from the task and message.
    """
    if task_dict is not None:
        assert input_message is not None, "If `task_dict` is provided, then `input_message` must also be provided."
        task = LLMTask("unittests", task_dict)
        parent_context = LLMContext(task, input_message)
    else:
        assert input_message is None, "If `input_message` is provided, then `task_dict` must also be provided."
        parent_context = LLMContext()

    context = _mk_context(parent_context, input_values)

    async def execute():
        # `asyncio.run(obj)`` will raise a `ValueError`` if `asyncio.iscoutine(obj)` is `False` for composite nodes
        # that don't directly implement `execute()` this causes a failure because while
        # `mrc.core.coro.CppToPyAwaitable` is awaitable it is not a coroutine.

        return await node.execute(context)

    context = asyncio.run(execute())

    return context.view_outputs


def execute_task_handler(task_handler: LLMTaskHandler,
                         task_dict: dict,
                         input_message: ControlMessage,
                         **input_values: dict) -> ControlMessage:
    """
    Executes an LLM task handler with the necessary LLM context.
    """
    task = LLMTask("unittests", task_dict)
    parent_context = LLMContext(task, input_message)
    context = _mk_context(parent_context, input_values)

    message = asyncio.run(task_handler.try_handle(context))

    return message
