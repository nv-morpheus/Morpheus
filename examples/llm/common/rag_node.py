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
import typing

from morpheus.llm import LLMNode
from morpheus.service.vector_db_service import VectorDBResourceService

from .llm_generate_node import LLMGenerateNode
from .llm_service import LLMClient
from .prompt_template_node import PromptTemplateNode
from .retriever_node import RetrieverNode

logger = logging.getLogger(__name__)


class RAGNode(LLMNode):

    def __init__(self,
                 *,
                 prompt: str,
                 vdb_service: VectorDBResourceService,
                 embedding: typing.Callable[[list[str]], typing.Coroutine[typing.Any, typing.Any,
                                                                          list[list[float]]]] = None,
                 llm_client: LLMClient) -> None:
        super().__init__()

        self._prompt = prompt
        self._vdb_service = vdb_service
        self._embedding = embedding
        self._llm_service = llm_client

        self.add_node("retriever", node=RetrieverNode(service=vdb_service, embedding=embedding))

        self.add_node("prompt",
                      inputs=[("/retriever", "contexts"), ("query", "query")],
                      node=PromptTemplateNode(self._prompt, template_format="jinja"))

        self.add_node("generate", inputs=["/prompt"], node=LLMGenerateNode(llm_client=llm_client), is_output=True)
