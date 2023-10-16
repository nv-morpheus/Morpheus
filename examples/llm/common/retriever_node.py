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

import numpy as np

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase
from morpheus.service.vector_db_service import VectorDBResourceService

logger = logging.getLogger(__name__)


class RetrieverNode(LLMNodeBase):

    def __init__(
            self,
            service: VectorDBResourceService,
            embedding: typing.Callable[[list[str]], typing.Coroutine[typing.Any, typing.Any,
                                                                     list[np.ndarray]]]) -> None:
        super().__init__()

        self._service = service
        self._embedding = embedding

    def get_input_names(self) -> list[str]:
        return ["query"]

    async def execute(self, context: LLMContext):

        # Get the keys from the task
        input_strings: list[str] = typing.cast(list[str], context.get_input())

        # Call the embedding function to get the vector embeddings
        embeddings = self._embedding(input_strings)

        # Query the vector database
        results = await self._service.similarity_search(embeddings=embeddings, k=4)

        context.set_output(results)

        return context
