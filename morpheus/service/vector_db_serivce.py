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


class VectorDBService(ABC):
    """
    Class used for vectorstore specific implementation.
    """

    # Mutex lock to perform operations synchronously.
    _mutex = threading.Lock()

    @abstractmethod
    def insert(self, name: str, data: typing.Any, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Insert data into the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        data : typing.Any
            Data to be inserted into the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns insert response result.
        """

        pass

    @abstractmethod
    def search(self, name: str, query: typing.Union[str, dict] = None, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Search for content in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        query : typing.Union[str, dict], default None
            Query to execute on the given resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns search results.
        """

        pass

    @abstractmethod
    def drop(self, name: str, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop resources from the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def update(self, name: str, data: typing.Any, **kwargs: dict[str, typing.Any]) -> None:
        """
        Update data in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        data : typing.Any
            Data to be updated in the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def delete(self, name: str, expr: typing.Any, **kwargs: dict[str, typing.Any]) -> None:
        """
        Delete data in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        expr : typing.Any
            E.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def create(self, name: str, overwrite: bool = False, **kwargs: dict[str, typing.Any]) -> None:
        """
        Create resources in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        overwrite : bool, default False
            Whether to overwrite the resource if it already exists.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def close(self, **kwargs) -> None:
        """
        Close connections to the vector database.

        Parameters
        ----------
        **kwargs : dict
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def has_store_object(self, name: str) -> bool:
        """
        Check if a resource exists in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.

        Returns
        -------
        bool
            Returns True if resource exists in the vector database, otherwise False.
        """

        pass

    @abstractmethod
    def list_store_objects(self, **kwargs: dict[str, typing.Any]) -> list[str]:
        """
        List existing resources in the vector database.

        Parameters
        ----------
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[str]
            Returns available resouce names in the vector database.
        """

        pass

    def transform(self, data: typing.Any, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Transform data according to the specific vector database implementation.

        Parameters
        ----------
        data : typing.Any
            Data to be updated in the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns transformed data as per the implementation.
        """
        return data

    @abstractmethod
    def retrieve_by_keys(self, name: str, keys: typing.Any, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Retrieve the inserted vectors using keys from the resource.

        Parameters
        ----------
        name : str
            Name of the resource.
        keys : typing.Any
            Primary keys to get vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns inserted vectors of the given keys that exists in the resource.
        """
        pass

    @abstractmethod
    def delete_by_keys(self, name: str, keys: typing.Any, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the resource.

        Parameters
        ----------
        name : str
            Name of the resource.
        keys : typing.Any
            Primary keys to delete vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns vectors of the given keys that are delete from the resource.
        """
        pass

    @abstractmethod
    def count(self, name: str, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given resource.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        int
            Returns number of rows/entities in the given resource.
        """
        pass
