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
from abc import ABC
from abc import abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        prompt : str
            The prompt to generate a response for.
        """
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        prompt : str
            The prompt to generate a response for.
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        prompts : list[str]
            The prompts to generate responses for.
        """
        pass

    @abstractmethod
    async def generate_batch_async(self, prompts: list[str]) -> list[str]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        prompts : list[str]
            The prompts to generate responses for.
        """
        pass


class LLMService(ABC):

    @abstractmethod
    def get_client(self, model_name: str, **model_kwargs: dict[str, typing.Any]) -> LLMClient:
        """
        Returns a client for interacting with a specific model.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model when generating text.
        """
        pass
