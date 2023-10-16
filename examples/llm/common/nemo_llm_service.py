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
import os
import typing

from .llm_service import LLMClient
from .llm_service import LLMService

logger = logging.getLogger(__name__)

try:
    from nemollm.api import NemoLLM
except ImportError:
    logger.error("NemoLLM not found. Please install NemoLLM to use this service.")


class NeMoLLMClient(LLMClient):

    def __init__(self, parent: "NeMoLLMService", model_name: str, **model_kwargs) -> None:
        super().__init__()

        self._parent = parent
        self._model_name = model_name
        self._model_kwargs = model_kwargs

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    async def generate_async(self, prompt: str) -> str:
        return (await self.generate_batch_async([prompt]))[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:

        return typing.cast(
            list[str],
            self._parent._conn.generate_multiple(model=self._model_name,
                                                 prompts=prompts,
                                                 return_type="text",
                                                 **self._model_kwargs))

    async def generate_batch_async(self, prompts: list[str]) -> list[str]:

        futures = [
            asyncio.wrap_future(
                self._parent._conn.generate(self._model_name, p, return_type="async", **self._model_kwargs))
            for p in prompts
        ]

        results = await asyncio.gather(*futures)

        return [
            typing.cast(str, NemoLLM.post_process_generate_response(r, return_text_completion_only=True))
            for r in results
        ]


class NeMoLLMService(LLMService):

    def __init__(self, *, api_key: str = None, org_id: str = None) -> None:
        super().__init__()

        api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        org_id = org_id if org_id is not None else os.environ.get("NGC_ORG_ID", None)

        self._api_key = api_key
        self._org_id = org_id

        # Do checking on api key

        # Class variables
        self._conn: NemoLLM = NemoLLM(
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=self._api_key,

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id=self._org_id,
        )

    def get_client(self, model_name: str, **model_kwargs: dict) -> NeMoLLMClient:

        return NeMoLLMClient(self, model_name, **model_kwargs)
