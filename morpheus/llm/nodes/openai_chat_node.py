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
import logging

from langchain.llms.openai import OpenAIChat
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase

logger = logging.getLogger(__name__)


class OpenAIChatNode(LLMNodeBase):

    def __init__(self, model_name: str, set_assistant=False) -> None:
        super().__init__()

        self._model_name = model_name
        self._set_assistant = set_assistant

        self._model = OpenAIChat(model_name=self._model_name, temperature=0, cache=True)

    def get_input_names(self):
        if (self._set_assistant):
            return ["assistant", "user"]

        return ["user"]

    async def _run_one(self, user: str, assistant: str = None):

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user),
        ]

        if (assistant is not None):
            messages.append(AIMessage(content=assistant))

        output2 = await self._model.apredict_messages(messages=messages)

        return {"message": output2.content}

    async def execute(self, context: LLMContext):

        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        output_coros = [self._run_one(**inp) for inp in input_list]

        outputs = await asyncio.gather(*output_coros)

        # Convert from list[dict] to dict[list]
        outputs = {k: [x[k] for x in outputs] for k in outputs[0]}

        context.set_output(outputs)
