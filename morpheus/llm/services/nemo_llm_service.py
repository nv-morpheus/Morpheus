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
import os
import typing

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.llm_service import LLMService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = (
    "NemoLLM not found. Install it and other additional dependencies by running the following command:\n"
    "`conda env update --solver=libmamba -n morpheus "
    "--file morpheus/conda/environments/dev_cuda-121_arch-x86_64.yaml --prune`")

try:
    import nemollm
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class NeMoLLMClient(LLMClient):
    """
    Client for interacting with a specific model in Nemo. This class should be constructed with the
    `NeMoLLMService.get_client` method.

    Parameters
    ----------
    parent : NeMoLLMService
        The parent service for this client.
    model_name : str
        The name of the model to interact with.

    model_kwargs : dict[str, typing.Any]
        Additional keyword arguments to pass to the model when generating text.
    """

    def __init__(self, parent: "NeMoLLMService", *, model_name: str, **model_kwargs) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()

        assert parent is not None, "Parent service cannot be None."

        self._parent = parent
        self._model_name = model_name
        self._model_kwargs = model_kwargs
        self._prompt_key = "prompt"

    def get_input_names(self) -> list[str]:
        return [self._prompt_key]

    def generate(self, input_dict: dict[str, str]) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return self.generate_batch({self._prompt_key: [input_dict[self._prompt_key]]})[0]

    async def generate_async(self, input_dict: dict[str, str]) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.
        """
        return (await self.generate_batch_async({self._prompt_key: [input_dict[self._prompt_key]]}))[0]

    def generate_batch(self, inputs: dict[str, list[str]]) -> list[str]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        """
        return typing.cast(
            list[str],
            self._parent._conn.generate_multiple(model=self._model_name,
                                                 prompts=inputs[self._prompt_key],
                                                 return_type="text",
                                                 **self._model_kwargs))

    async def generate_batch_async(self, inputs: dict[str, list[str]]) -> list[str]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        """
        prompts = inputs[self._prompt_key]
        futures = [
            asyncio.wrap_future(
                self._parent._conn.generate(self._model_name, p, return_type="async", **self._model_kwargs))
            for p in prompts
        ]

        results = await asyncio.gather(*futures)

        responses = []

        for result in results:
            result = nemollm.NemoLLM.post_process_generate_response(result, return_text_completion_only=False)
            if result.get('status', None) == 'fail':
                raise RuntimeError(result.get('msg', 'Unknown error'))

            responses.append(result['text'])

        return responses


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
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__()
        api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        org_id = org_id if org_id is not None else os.environ.get("NGC_ORG_ID", None)

        self._conn = nemollm.NemoLLM(
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=api_key,

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id=org_id,
        )

    def get_client(self, *, model_name: str, **model_kwargs) -> NeMoLLMClient:
        """
        Returns a client for interacting with a specific model. This method is the preferred way to create a client.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text.
        """

        return NeMoLLMClient(self, model_name=model_name, **model_kwargs)
