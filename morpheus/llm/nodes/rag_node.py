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
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.nodes.retriever_node import RetrieverNode
from morpheus.llm.services.llm_service import LLMClient
from morpheus.service.vdb.vector_db_service import VectorDBResourceService

logger = logging.getLogger(__name__)


class RAGNode(LLMNode):
    """
    Performs Retrieval Augmented Generation (RAG).

    Parameters
    ----------
    prompt : str
        The prompt template string to populate with input values.

    template_format : str, optional default="jinja"
        The format of the template string. Must be one of: f-string, jinja.

    vdb_service : VectorDBResourceService
        The VectorDB service to use for retrieval.

    embedding : Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]], optional
        An asynchronous function to use to determine the embeddings to search the `vdb_service` for. If `None`,
        upstream nodes must provide the embeddings.

    llm_client : LLMClient
        The LLM client to use for generation.
    """

    def __init__(self,
                 *,
                 prompt: str,
                 template_format: typing.Literal["f-string", "jinja"] = "jinja",
                 vdb_service: VectorDBResourceService,
                 embedding: typing.Callable[[list[str]], typing.Coroutine[typing.Any, typing.Any,
                                                                          list[list[float]]]] = None,
                 llm_client: LLMClient) -> None:
        super().__init__()

        self.add_node("retriever", node=RetrieverNode(service=vdb_service, embedding=embedding))

        self.add_node("prompt",
                      inputs=[("/retriever", "contexts"), ("query", "query")],
                      node=PromptTemplateNode(prompt, template_format=template_format))

        self.add_node("generate", inputs=["/prompt"], node=LLMGenerateNode(llm_client=llm_client), is_output=True)
