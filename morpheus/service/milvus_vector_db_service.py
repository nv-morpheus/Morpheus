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

import copy
import logging
import threading
import time
import typing

import pandas as pd
import pymilvus
from pymilvus.orm.mutation import MutationResult

import cudf

from morpheus.service.milvus_client import MILVUS_DATA_TYPE_MAP
from morpheus.service.milvus_client import MilvusClient
from morpheus.service.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


def with_collection_lock(func: typing.Callable) -> typing.Callable:
    """
    A decorator to synchronize access to a collection with a lock. This decorator ensures that operations on a
    specific collection within the Milvus Vector Database are synchronized by acquiring and
    releasing a collection-specific lock.

    Parameters
    ----------
    func : Callable
        The function to be wrapped with the lock.

    Returns
    -------
    Callable
        The wrapped function with the lock acquisition logic.
    """

    def wrapper(self, name, *args, **kwargs):
        collection_lock = MilvusVectorDBService.get_collection_lock(name)
        with collection_lock:
            logger.debug("Acquiring lock for collection: %s", name)
            result = func(self, name, *args, **kwargs)
            logger.debug("Releasing lock for collection: %s", name)
            return result

    return wrapper


class MilvusVectorDBService(VectorDBService):
    """
    Service class for Milvus Vector Database implementation. This class provides functions for interacting
    with a Milvus vector database.

    Parameters
    ----------
    host : str
        The hostname or IP address of the Milvus server.
    port : str
        The port number for connecting to the Milvus server.
    alias : str, optional
        Alias for the Milvus connection, by default "default".
    **kwargs : dict
        Additional keyword arguments specific to the Milvus connection configuration.
    """

    _collection_locks = {}
    _cleanup_interval = 600  # 10mins
    _last_cleanup_time = time.time()

    def __init__(self,
                 uri: str,
                 user: str = "",
                 password: str = "",
                 db_name: str = "",
                 token: str = "",
                 **kwargs: dict[str, typing.Any]):

        self._client = MilvusClient(uri=uri, user=user, password=password, db_name=db_name, token=token, **kwargs)

    def has_store_object(self, name: str) -> bool:
        """
        Check if a collection exists in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to check.

        Returns
        -------
        bool
            True if the collection exists, False otherwise.
        """
        return self._client.has_collection(collection_name=name)

    def list_store_objects(self, **kwargs: dict[str, typing.Any]) -> list[str]:
        """
        List the names of all collections in the Milvus vector database.

        Returns
        -------
        list[str]
            A list of collection names.
        """
        return self._client.list_collections(**kwargs)

    def _create_schema_field(self, field_conf: dict) -> pymilvus.FieldSchema:

        name = field_conf.pop("name")
        dtype = field_conf.pop("dtype")

        dtype = MILVUS_DATA_TYPE_MAP[dtype.lower()]

        field_schema = pymilvus.FieldSchema(name=name, dtype=dtype, **field_conf)

        return field_schema

    @with_collection_lock
    def create(self, name: str, overwrite: bool = False, **kwargs: dict[str, typing.Any]):
        """
        Create a collection in the Milvus vector database with the specified name and configuration. This method
        creates a new collection in the Milvus vector database with the provided name and configuration options.
        If the collection already exists, it can be overwritten if the `overwrite` parameter is set to True.

        Parameters
        ----------
        name : str
            Name of the collection to be created.
        overwrite : bool, optional
            If True, the collection will be overwritten if it already exists, by default False.
        **kwargs : dict
            Additional keyword arguments containing collection configuration.

        Raises
        ------
        ValueError
            If the provided schema fields configuration is empty.
        """
        logger.debug("Creating collection: %s, overwrite=%s, kwargs=%s", name, overwrite, kwargs)
        # Preserve original configuration.
        kwargs = copy.deepcopy(kwargs)

        collection_conf = kwargs.get("collection_conf")
        auto_id = collection_conf.get("auto_id", False)
        index_conf = collection_conf.get("index_conf", None)
        partition_conf = collection_conf.get("partition_conf", None)

        schema_conf = collection_conf.get("schema_conf")
        schema_fields_conf = schema_conf.pop("schema_fields")

        index_param = {}

        if not self.has_store_object(name) or overwrite:
            if overwrite and self.has_store_object(name):
                self.drop(name)

            if len(schema_fields_conf) == 0:
                raise ValueError("Cannot create collection as provided empty schema_fields configuration")

            schema_fields = [self._create_schema_field(field_conf=field_conf) for field_conf in schema_fields_conf]

            schema = pymilvus.CollectionSchema(fields=schema_fields, **schema_conf)

            if index_conf:
                field_name = index_conf.pop("field_name")
                metric_type = index_conf.pop("metric_type")
                index_param = self._client.prepare_index_params(field_name=field_name,
                                                                metric_type=metric_type,
                                                                **index_conf)

            self._client.create_collection_with_schema(collection_name=name,
                                                       schema=schema,
                                                       index_param=index_param,
                                                       auto_id=auto_id,
                                                       shards_num=collection_conf.get("shards", 2),
                                                       consistency_level=collection_conf.get(
                                                           "consistency_level", "Strong"))

            if partition_conf:
                timeout = partition_conf.get("timeout", 1.0)
                # Iterate over each partition configuration
                for part in partition_conf["partitions"]:
                    self._client.create_partition(collection_name=name, partition_name=part["name"], timeout=timeout)

    @with_collection_lock
    def insert(self, name: str, data: list[list] | list[dict], **kwargs: dict[str,
                                                                              typing.Any]) -> dict[str, typing.Any]:
        """
        Insert a collection specific data in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        data : list[list] | list[dict]
            Data to be inserted in the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the collection not exists exists.
        """

        return self._collection_insert(name, data, **kwargs)

    def _collection_insert(self, name: str, data: list[list] | list[dict],
                           **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:

        if not self.has_store_object(name):
            raise RuntimeError(f"Collection {name} doesn't exist.")

        collection = None
        try:
            collection_conf = kwargs.get("collection_conf", {})
            partition_name = collection_conf.get("partition_name", "_default")

            collection = self._client.get_collection(collection_name=name, **collection_conf)
            result = collection.insert(data, partition_name=partition_name)
            collection.flush()
        finally:
            collection.release()

        result_dict = {
            "primary_keys": result.primary_keys,
            "insert_count": result.insert_count,
            "delete_count": result.delete_count,
            "upsert_count": result.upsert_count,
            "timestamp": result.timestamp,
            "succ_count": result.succ_count,
            "err_count": result.err_count,
            "succ_index": result.succ_index,
            "err_index": result.err_index
        }

        return result_dict

    @with_collection_lock
    def insert_dataframe(self, name: str, df: cudf.DataFrame | pd.DataFrame,
                         **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Converts dataframe to rows and insert to a collection in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        df : cudf.DataFrame | pd.DataFrame
            Dataframe to be inserted in the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the collection not exists exists.
        """
        if not self.has_store_object(name):
            raise RuntimeError(f"Collection {name} doesn't exist.")

        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()

        dict_of_rows = df.to_dict(orient='records')

        return self._collection_insert(name, dict_of_rows, **kwargs)

    @with_collection_lock
    def search(self, name: str, query: str = None, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Search for data in a collection in the Milvus vector database.

        This method performs a search operation in the specified collection/partition in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to search within.
        query : str, optional
            The search query, which can be a filter expression, by default None.
        **kwargs : dict
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result, which can vary depending on the query and options.

        Raises
        ------
        RuntimeError
            If an error occurs during the search operation.
            If query argument is `None` and `data` keyword argument doesn't exist.
            If `data` keyword arguement is `None`.
        """

        logger.debug("Searching in collection: %s, query=%s, kwargs=%s", name, query, kwargs)

        try:
            self._client.load_collection(collection_name=name)
            if query is not None:
                result = self._client.query(collection_name=name, filter=query, **kwargs)
            else:
                if "data" not in kwargs:
                    raise RuntimeError("The search operation requires that search vectors be " +
                                       "provided as a keyword argument 'data'")
                if kwargs["data"] is None:
                    raise RuntimeError("Argument 'data' cannot be None")

                data = kwargs.pop("data")

                result = self._client.search(collection_name=name, data=data, **kwargs)
            return result

        except pymilvus.exceptions.MilvusException as exec_info:
            raise RuntimeError(f"Unable to perform serach: {exec_info}") from exec_info

        finally:
            self._client.release_collection(collection_name=name)

    @with_collection_lock
    def update(self, name: str, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        data : list[typing.Any]
            Data to be updated in the collection.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to upsert operation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        if not isinstance(data, list):
            raise RuntimeError("Data is not of type list.")

        result = self._client.upsert(collection_name=name, entities=data, **kwargs)

        return self._convert_mutation_result_to_dict(result=result)

    @with_collection_lock
    def delete_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the resource.

        Parameters
        ----------
        name : str
            Name of the resource.
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns result of the given keys that are delete from the collection.
        """

        result = self._client.delete(collection_name=name, pks=keys, **kwargs)

        return result

    @with_collection_lock
    def delete(self, name: str, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete vectors from the resource using expressions.

        Parameters
        ----------
        name : str
            Name of the resource.
        expr : str
            Delete expression.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are delete from the collection.
        """

        result = self._client.delete_by_expr(collection_name=name, expression=expr, **kwargs)

        return self._convert_mutation_result_to_dict(result=result)

    @with_collection_lock
    def retrieve_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using their primary keys from the Collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        keys : int | str | list
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the retrieval operation.

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the collection.
        """

        result = None

        try:
            self._client.load_collection(collection_name=name)
            result = self._client.get(collection_name=name, ids=keys, **kwargs)
        except pymilvus.exceptions.MilvusException as exec_info:
            raise RuntimeError(f"Unable to perform serach: {exec_info}") from exec_info

        finally:
            self._client.release_collection(collection_name=name)

        return result

    def count(self, name: str, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the count operation.

        Returns
        -------
        int
            Returns number of entities in the collection.
        """

        return self._client.num_entities(collection_name=name, **kwargs)

    def drop(self, name: str, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop a collection, index, or partition in the Milvus vector database.

        This method allows you to drop a collection, an index within a collection,
        or a specific partition within a collection in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection, index, or partition to be dropped.
        **kwargs : dict
            Additional keyword arguments for specifying the type and partition name (if applicable).

        Notes on Expected Keyword Arguments:
        ------------------------------------
        - 'resource' (str, optional):
        Specifies the type of resource to drop. Possible values: 'collection' (default), 'index', 'partition'.

        - 'partition_name' (str, optional):
        Required when dropping a specific partition within a collection. Specifies the partition name to be dropped.

        - 'field_name' (str, optional):
        Required when dropping an index within a collection. Specifies the field name for which the index is created.

        - 'index_name' (str, optional):
        Required when dropping an index within a collection. Specifies the name of the index to be dropped.

        Raises
        ------
        ValueError
            If mandatory arguments are missing or if the provided 'resource' value is invalid.
        """

        logger.debug("Dropping collection: %s, kwargs=%s", name, kwargs)

        if self.has_store_object(name):
            resource = kwargs.get("resource", "collection")
            if resource == "collection":
                self._client.drop_collection(collection_name=name)
            elif resource == "partition":
                if "partition_name" not in kwargs:
                    raise ValueError("Mandatory argument 'partition_name' is required when resource='partition'")
                partition_name = kwargs["partition_name"]
                if self._client.has_partition(collection_name=name, partition_name=partition_name):
                    self._client.drop_partition(collection_name=name, partition_name=partition_name)
            elif resource == "index":
                if "field_name" in kwargs and "index_name" in kwargs:
                    self._client.drop_index(collection_name=name,
                                            field_name=kwargs["field_name"],
                                            index_name=kwargs["index_name"])
                else:
                    raise ValueError(
                        "Mandatory arguments 'field_name' and 'index_name' are required when resource='index'")

    def describe(self, name: str, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Describe the collection in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments specific to the Milvus vector database.

        Returns
        -------
        dict
            Returns collection information.
        """

        return self._client.describe_collection(collection_name=name, **kwargs)

    def close(self) -> None:
        """
        Close the connection to the Milvus vector database.

        This method disconnects from the Milvus vector database by removing the connection.

        """
        self._client.close()

    def _convert_mutation_result_to_dict(self, result: MutationResult) -> dict[str, typing.Any]:
        result_dict = {
            "insert_count": result.insert_count,
            "delete_count": result.delete_count,
            "upsert_count": result.upsert_count,
            "timestamp": result.timestamp,
            "succ_count": result.succ_count,
            "err_count": result.err_count
        }
        return result_dict

    @classmethod
    def get_collection_lock(cls, name: str) -> threading.Lock:
        """
        Get a lock for a given collection name.

        Parameters
        ----------
        name : str
            Name of the collection for which to acquire the lock.

        Returns
        -------
        threading.Lock
            A thread lock specific to the given collection name.
        """

        current_time = time.time()

        if name not in cls._collection_locks:
            cls._collection_locks[name] = {"lock": threading.Lock(), "last_used": current_time}
        else:
            cls._collection_locks[name]["last_used"] = current_time

        if (current_time - cls._last_cleanup_time) >= cls._cleanup_interval:
            for lock_name, lock_info in cls._collection_locks.copy().items():
                last_used = lock_info["last_used"]
                if current_time - last_used >= cls._cleanup_interval:
                    logger.debug("Cleaning up lock for collection: %s", lock_name)
                    del cls._collection_locks[lock_name]
            cls._last_cleanup_time = current_time

        return cls._collection_locks[name]["lock"]
