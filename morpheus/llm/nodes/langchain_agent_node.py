# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import logging
import typing

from langchain_core.exceptions import OutputParserException

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from langchain.agents import AgentExecutor


class LangChainAgentNode(LLMNodeBase):
    """
    Executes a LangChain agent in an LLMEngine

    Parameters
    ----------
    agent_executor : AgentExecutor
        The agent executor to use to execute.
    """

    def __init__(self,
                 agent_executor: "AgentExecutor",
                 replace_exceptions: bool = False,
                 replace_exceptions_value: typing.Optional[str] = None):
        super().__init__()

        self._agent_executor = agent_executor

        self._input_names = self._agent_executor.input_keys

        self._replace_exceptions = replace_exceptions
        self._replace_exceptions_value = replace_exceptions_value

        if self._replace_exceptions:
            assert self._replace_exceptions_value is not None, "When replace_exceptions is enabled, replace_exceptions_value must be provided."

    def get_input_names(self):
        return self._input_names

    async def _run_single(self, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:

        all_lists = all(isinstance(v, list) for v in kwargs.values())

        # Check if all values are a list
        if all_lists:

            # Transform from dict[str, list[Any]] to list[dict[str, Any]]
            input_list = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

            # Run multiple again
            results_async = [self._run_single(**x) for x in input_list]

            results = await asyncio.gather(*results_async, return_exceptions=True)

            # # Transform from list[dict[str, Any]] to dict[str, list[Any]]
            # results = {k: [x[k] for x in results] for k in results[0]}

            return results

        # We are not dealing with a list, so run single
        try:
            return await self._agent_executor.arun(**kwargs)
        except Exception as e:
            logger.exception("Error running agent: %s", e)
            return e

    async def execute(self, context: LLMContext) -> LLMContext:  # pylint: disable=invalid-overridden-method

        input_dict = context.get_inputs()

        results = await self._run_single(**input_dict)

        if self._replace_exceptions:
            # Processes the results to replace exceptions with a default message
            for i, answer_list in enumerate(results):
                for j, answer in enumerate(answer_list):
                    if isinstance(answer, (OutputParserException, Exception)):
                        # If the agent encounters a parsing error or a server error after retries, replace the error
                        # with a default value to prevent the pipeline from crashing
                        results[i][j] = self._replace_exceptions_value
                        logger.warning(f"Exception encountered in result[{i}][{j}]: {answer}. "
                                       f"Replacing with default message: \"{self._replace_exceptions_value}\".")

        context.set_output(results)

        return context
