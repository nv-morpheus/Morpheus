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

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase
from morpheus.service.vdb.vector_db_service import VectorDBResourceService

logger = logging.getLogger(__name__)


class RetrieverNode(LLMNodeBase):

    def __init__(
        self,
        *,
        embedding: typing.Callable[[list[str]], typing.Coroutine[typing.Any, typing.Any, list[list[float]]]] | None,
        service: VectorDBResourceService,
        **similarity_search_kwargs,
    ) -> None:
        super().__init__()

        self._service = service
        self._embedding = embedding
        self._similarity_search_kwargs = similarity_search_kwargs

    def get_input_names(self) -> list[str]:
        if (self._embedding is None):
            return ["embedding"]

        return ["query"]

    async def execute(self, context: LLMContext):

        if (self._embedding is not None):
            # Get the keys from the task
            input_strings: list[str] = typing.cast(list[str], context.get_input())

            # Call the embedding function to get the vector embeddings
            embeddings = await self._embedding(input_strings)
        else:
            embeddings: list[list[float]] = typing.cast(list[list[float]], context.get_input())

        # Query the vector database
        results = await self._service.similarity_search(embeddings=embeddings, **self._similarity_search_kwargs)

        context.set_output(results)

        return context
