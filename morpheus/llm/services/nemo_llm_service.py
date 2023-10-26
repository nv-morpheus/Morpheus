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

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_ERROR_MESSAGE = (
    "NemoLLM not found. Install it and other additional dependencies by running the following command:\n"
    "`mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`")

try:
    from nemollm.api import NemoLLM
except ImportError:
    logger.error(IMPORT_ERROR_MESSAGE)


def _verify_nemo_llm():
    """
    When NemoLLM is not installed, raise an ImportError with a helpful message, rather than an attribute error.
    """
    if 'NemoLLM' not in globals():
        raise ImportError(IMPORT_ERROR_MESSAGE)


class NeMoLLMClient(LLMClient):

    def __init__(self, parent: "NeMoLLMService", model_name: str, **model_kwargs) -> None:
        super().__init__()
        _verify_nemo_llm()

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
    """
    A service for interacting with NeMo LLM models, this class should be used to create a client for a specific model.

    Parameters
    ----------
    api_key : str, optional
        The API key for the LLM service, by default None. If `None` the API key will be read from the `NGC_API_KEY`
        environment variable. If neither are present an error will be raised.

    org_id : str, optional
        The organization ID for the LLM service, by default None. If `None` the organization ID will be read from the
        `NGC_ORG_ID` environment variable. This value is only required if the account associated with the `api_key` is
        a member of multiple NGC organizations.
    """

    def __init__(self, *, api_key: str = None, org_id: str = None) -> None:
        super().__init__()
        _verify_nemo_llm()

        api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        org_id = org_id if org_id is not None else os.environ.get("NGC_ORG_ID", None)

        # Class variables
        self._conn: NemoLLM = NemoLLM(
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=api_key,

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id=org_id,
        )

    def get_client(self, model_name: str, **model_kwargs) -> NeMoLLMClient:

        return NeMoLLMClient(self, model_name, **model_kwargs)
