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


def with_mutex(lock_name):

    def decorator(func):

        def wrapper(*args, **kwargs):
            with getattr(args[0], lock_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class VectorDBController(ABC):

    _mutex = threading.Lock()
    """
    Abstract class for vector document store implementation.
    """

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
    #     Update an existing vector in the vector document store.

    #     Parameters:
    #     - vector (List[float]): The updated vector data.
    #     - vector_id (str): The unique identifier of the vector to update.

    #     Returns:
    #     - None

    #     Raises:
    #     - RuntimeError: If an error occurs while updating the vector.
    #     """
    #     pass

    # @abstractmethod
    # def get_by_name(self, name: str) -> list[float]:
    #     """
    #     Retrieve a vector from the vector document store by its ID.

    #     Parameters:
    #     - vector_id (str): The unique identifier of the vector to retrieve.

    #     Returns:
    #     - List[float]: The vector data.

    #     Raises:
    #     - RuntimeError: If an error occurs while retrieving the vector.
    #     """
    #     pass

    # @abstractmethod
    # def count(self) -> int:
    #     """
    #     Get the total count of vectors in the vector document store.

    #     Returns:
    #     - int: The total count of vectors.

    #     Raises:
    #     - RuntimeError: If an error occurs while counting the vectors.
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
