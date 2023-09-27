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

import threading
import typing
from abc import ABC
from abc import abstractmethod


class VectorDBController(ABC):
    """
    Abstract class for vector document store implementation.
    """

    _mutex = threading.Lock()

    @abstractmethod
    def insert(self, name: str, data: typing.Any, **kwargs):
        """
        This abstract function is used to insert data to the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        data : typing.Any
            Data that needs to be instered in the resource.
        **kwargs : dict
            Extra optional arguments specific to the vector db implementation.
        """

        pass

    @abstractmethod
    def search(self, name: str, query: typing.Union[str, dict] = None, **kwargs) -> typing.Any:
        """
        This abstract function is used to search content in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        query : yping.Union[str, dict], default None
            Query to execute on the given resource.
        **kwargs : dict
            Extra optional arguments specific to the vector db implementation.
        """

        pass

    @abstractmethod
    def drop(self, name: str, **kwargs) -> None:
        """
        This abstract function is used to drop resources in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs : dict
            Extra optional arguments specific to the vector db implementation.
        """

        pass

    # @abstractmethod
    # def update(self, vector: list[float], vector_id: str):
    #     """
    #     """
    #     pass

    # @abstractmethod
    # def get_by_name(self, name: str) -> list[float]:
    #     """
    #     """
    #     pass

    # @abstractmethod
    # def count(self) -> int:
    #     """
    #     """
    #     pass

    @abstractmethod
    def create(self, name: str, **kwargs) -> None:
        """
        This abstract function is used to create resources in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs : dict
            Extra optional arguments specific to the vector db implementation.
        """
        pass

    @abstractmethod
    def close(self, **kwargs) -> None:
        """
        This abstract function is used to close connections to the vector database.

        Parameters
        ----------
        **kwargs : dict
            Extra optional arguments specific to the vector db implementation.
        """
        pass

    @abstractmethod
    def has_store_object(self, name: str) -> bool:
        """
        This abstract function is used to check if the resource exists in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        """

        pass

    @abstractmethod
    def list_store_objects(self) -> list[str]:
        """
        This abstract function is used to list existing resources in the vector database.
        """
