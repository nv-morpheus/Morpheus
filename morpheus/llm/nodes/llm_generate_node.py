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

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase
from morpheus.llm.services.llm_service import LLMClient

logger = logging.getLogger(__name__)


class LLMGenerateNode(LLMNodeBase):
    """
    Generates responses from an LLM using the provided `llm_client` instance based on prompts provided as input from
    upstream nodes.

    Parameters
    ----------
    llm_client : LLMClient
        The client instance to use to generate responses.

    input_names : list[str], optional
        The names of the inputs to this node. Defaults to `["prompt"]`.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__()

        self._llm_client = llm_client

    def get_input_names(self) -> list[str]:
        return self._llm_client.get_input_names()

    async def execute(self, context: LLMContext) -> LLMContext:

        # Get the inputs
        inputs: dict[str, list[str]] = context.get_inputs()

        results = await self._llm_client.generate_batch_async(inputs)

        context.set_output(results)

        return context
