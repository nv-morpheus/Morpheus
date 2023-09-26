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
from abc import ABC
from abc import abstractmethod


class VectorDBController(ABC):
    """
    Abstract class for vector document store implementation.
    """

    _mutex = threading.Lock()

    @abstractmethod
    def insert(self, name, data, **kwargs):
        """
        """
        pass

    @abstractmethod
    def search(self, name, query=None, **kwargs):
        """
        """
        pass

    @abstractmethod
    def drop(self):
        """
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
    def create_collection(self, collection_config: dict):
        """
        Create an index on the vector document store for efficient querying
        """

    @abstractmethod
    def close(self, **kwargs):
        """
        """

    @abstractmethod
    def has_collection(self, name) -> bool:
        """
        """

    @abstractmethod
    def list_collections(self) -> list[str]:
        """
        """
