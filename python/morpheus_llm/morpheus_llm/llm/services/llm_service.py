# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import importlib
import logging
import typing
from abc import ABC
from abc import abstractmethod

if typing.TYPE_CHECKING:
    from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService
    from morpheus_llm.llm.services.openai_chat_service import OpenAIChatService

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract interface for clients which are able to interact with LLM models. Concrete implementations of this class
    will have an associated implementation of `LLMService` which is able to construct instances of this class.
    """

    @abstractmethod
    def get_input_names(self) -> list[str]:
        """
        Returns the names of the inputs to the model.

        Returns
        -------
        list[str]
            List of input names.
        """
        pass

    @abstractmethod
    def generate(self, **input_dict) -> str:
        """
        Issue a request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.

        Returns
        -------
        str
            Generated response for prompt.
        """
        pass

    @abstractmethod
    async def generate_async(self, **input_dict) -> str:
        """
        Issue an asynchronous request to generate a response based on a given prompt.

        Parameters
        ----------
        input_dict : dict
            Input containing prompt data.

        Returns
        -------
        str
            Generated async response for prompt.
        """
        pass

    @typing.overload
    @abstractmethod
    def generate_batch(self, inputs: dict[str, list],
                       return_exceptions: typing.Literal[True]) -> list[str | BaseException]:
        ...

    @typing.overload
    @abstractmethod
    def generate_batch(self, inputs: dict[str, list], return_exceptions: typing.Literal[False]) -> list[str]:
        ...

    @typing.overload
    @abstractmethod
    def generate_batch(self,
                       inputs: dict[str, list],
                       return_exceptions: bool = False) -> list[str] | list[str | BaseException]:
        ...

    @abstractmethod
    def generate_batch(self, inputs: dict[str, list], return_exceptions=False) -> list[str] | list[str | BaseException]:
        """
        Issue a request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.

        Returns
        -------
        list[str] | list[str | BaseException]
            List of responses or list of responses and exceptions.
        """
        pass

    @typing.overload
    @abstractmethod
    async def generate_batch_async(self, inputs: dict[str, list],
                                   return_exceptions: typing.Literal[True]) -> list[str | BaseException]:
        ...

    @typing.overload
    @abstractmethod
    async def generate_batch_async(self, inputs: dict[str, list],
                                   return_exceptions: typing.Literal[False]) -> list[str]:
        ...

    @typing.overload
    @abstractmethod
    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions: bool = False) -> list[str] | list[str | BaseException]:
        ...

    @abstractmethod
    async def generate_batch_async(self,
                                   inputs: dict[str, list],
                                   return_exceptions=False) -> list[str] | list[str | BaseException]:
        """
        Issue an asynchronous request to generate a list of responses based on a list of prompts.

        Parameters
        ----------
        inputs : dict
            Inputs containing prompt data.
        return_exceptions : bool
            Whether to return exceptions in the output list or raise them immediately.

        Returns
        -------
        list[str] | list[str | BaseException]
            List of responses or list of responses and exceptions.
        """
        pass


class LLMService(ABC):
    """
    Abstract interface for services which are able to construct clients for interacting with LLM models.
    """

    @abstractmethod
    def get_client(self, *, model_name: str, **model_kwargs) -> LLMClient:
        """
        Returns a client for interacting with a specific model.

        Parameters
        ----------
        model_name : str
            The name of the model to create a client for.

        model_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the model.

        Returns
        -------
        LLMClient
            Client for interacting with LLM models.
        """
        pass

    @typing.overload
    @staticmethod
    def create(service_type: typing.Literal["nemo"], *service_args, **service_kwargs) -> "NeMoLLMService":
        pass

    @typing.overload
    @staticmethod
    def create(service_type: typing.Literal["openai"], *service_args, **service_kwargs) -> "OpenAIChatService":
        pass

    @typing.overload
    @staticmethod
    def create(service_type: str, *service_args, **service_kwargs) -> "LLMService":
        pass

    @staticmethod
    def create(service_type: str | typing.Literal["nemo"] | typing.Literal["openai"], *service_args, **service_kwargs):
        """
        Returns a service for interacting with LLM models.

        Parameters
        ----------
        service_type : str
            The type of the service to create
        service_args : list
            Additional arguments to pass to the service.
        service_kwargs : dict[str, typing.Any]
            Additional keyword arguments to pass to the service.
        """
        if service_type.lower() == 'openai':
            llm_or_chat = "chat"
        else:
            llm_or_chat = "llm"

        module_name = f"morpheus_llm.llm.services.{service_type.lower()}_{llm_or_chat}_service"
        module = importlib.import_module(module_name)

        # Get all of the classes in the module to find the correct service class
        mod_classes = {name: cls for name, cls in module.__dict__.items() if callable(cls)}

        class_name_lower = f"{service_type}{llm_or_chat}Service".lower()

        # Find case-insensitive match for the class name
        matching_classes = [name for name in mod_classes if name.lower() == class_name_lower]

        assert len(matching_classes) == 1, (f"Expected to find exactly one class with name {class_name_lower} "
                                            f"in module {module_name}, but found {sorted(mod_classes.keys())}")

        # Create the class
        class_ = getattr(module, matching_classes[0])

        instance = class_(*service_args, **service_kwargs)

        return instance
