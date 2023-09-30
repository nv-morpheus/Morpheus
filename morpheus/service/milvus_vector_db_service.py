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

import pandas as pd
import pymilvus
from pymilvus import Collection

import cudf

from morpheus.service.vector_db_serivce import VectorDBService
from morpheus.utils.vector_db_service_utils import MILVUS_DATA_TYPE_MAP
from morpheus.utils.vector_db_service_utils import with_mutex

logger = logging.getLogger(__name__)


class MilvusVectorDBService(VectorDBService):
    """
    Service class for Milvus Vector Database implementation.

    This class provides methods for interacting with a Milvus vector database.

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

    def __init__(self,
                 uri: str,
                 user: str = "",
                 password: str = "",
                 db_name: str = "",
                 token: str = "",
                 **kwargs: dict[str, typing.Any]):
        self._client = pymilvus.MilvusClient(uri=uri,
                                             user=user,
                                             password=password,
                                             db_name=db_name,
                                             token=token,
                                             **kwargs)
        self._handler = pymilvus.connections._fetch_handler(self._get_using())

    def _get_using(self):
        return self._client._using

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
        return self._handler.has_collection(name)

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
        dtype = MILVUS_DATA_TYPE_MAP[field_conf["dtype"].lower()]
        dim = field_conf.get("dim", None)

        if (dtype == pymilvus.DataType.BINARY_VECTOR or dtype == pymilvus.DataType.FLOAT_VECTOR):
            if not dim:
                raise ValueError(f"Dimensions for {dtype} should not be None")
            if not isinstance(dim, int):
                raise ValueError(f"Dimensions for {dtype} should be an integer")

        field_schema = pymilvus.FieldSchema(name=field_conf["name"],
                                            dtype=dtype,
                                            description=field_conf.get("description", ""),
                                            is_primary=field_conf["is_primary"],
                                            dim=dim)
        return field_schema

    @with_mutex("_mutex")
    def create(self, name: str, overwrite: bool = False, **kwargs: dict[str, typing.Any]):
        collection_conf = kwargs.get("collection_conf")
        auto_id = collection_conf.get("auto_id", False)
        index_conf = collection_conf.get("index_conf", None)
        partition_conf = collection_conf.get("partition_conf", None)

        schema_conf = collection_conf.get("schema_conf")
        schema_fields_conf = schema_conf.get("schema_fields")

        index_param = None
        if overwrite:
            if self.has_store_object(name):
                self.drop(name)

        if not self.has_store_object(name):

            if len(schema_fields_conf) == 0:
                raise ValueError("Cannot create collection as provided empty schema_fields configuration")

            schema_fields = [self._create_schema_field(field_conf=field_conf) for field_conf in schema_fields_conf]

            schema = pymilvus.CollectionSchema(fields=schema_fields,
                                               auto_id=auto_id,
                                               description=schema_conf.get("description", ""))

            if index_conf:
                index_param = self._client.prepare_index_params(field_name=index_conf["field_name"],
                                                                index_type=index_conf.get("index_type", None),
                                                                metric_type=index_conf.get("metric_type", None),
                                                                index_name=index_conf.get("index_name", ""),
                                                                timeout=index_conf.get("timeout", 1.0),
                                                                params=index_conf.get("params", 1.0))

            self._client.create_collection_with_schema(collection_name=name,
                                                       schema=schema,
                                                       index_param=index_param,
                                                       auto_id=auto_id,
                                                       shards_num=collection_conf.get("shards", 2),
                                                       consistency_level=collection_conf.get("consistency_level",
                                                                                                "Strong"))

            if partition_conf:
                timeout = partition_conf.get("timeout", 1.0)
                # Iterate over each partition configuration
                for part in partition_conf["partitions"]:
                    self._handler.create_partition(collection_name=name,
                                                   partition_name=part["name"],
                                                   timeout=timeout)

    @with_mutex("_mutex")
    def insert(self, name: str, data: typing.Union[list[list], list[dict], dict], **kwargs: dict[str, typing.Any]):
        """
        Insert a collection specific data in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        data : typing.Any
            Data to be inserted in the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

        Raises
        ------
        RuntimeError
            If the collection not exists exists.
        """
        return self._collection_insert(name, data, **kwargs)

    def _collection_insert(self, name: str,
                           data: typing.Union[list[list], list[dict], dict],
                           **kwargs: dict[str, typing.Any]) -> None:
        try:
            if not self.has_store_object(name):
                raise RuntimeError(f"Collection {name} doesn't exist.")
            collection_conf = kwargs.get("collection_conf", {})
            partition_name = collection_conf.get("partition_name", "_default")

            collection = Collection(name=name, using=self._get_using(), data=data, **collection_conf)
            result = collection.insert(data, partition_name=partition_name)
            collection.flush()
        finally:
            collection.release()

        return result

    @with_mutex("_mutex")
    def insert_dataframe(self, name: str,
                         df: typing.Union[cudf.DataFrame, pd.DataFrame],
                         **kwargs: dict[str, typing.Any]):
        """
        Converts dataframe to rows and insert to a collection in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        df : typing.Union[cudf.DataFrame, pd.DataFrame]
            Dataframe to be inserted in the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

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

    @with_mutex("_mutex")
    def search(self, name: str, query: typing.Union[str, dict] = None, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Search for data in a collection in the Milvus vector database.

        This method performs a search operation in the specified collection/partition in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to search within.
        query : Union[str, dict], optional
            The search query, which can be a JSON-like string or a dictionary,
            by default None.
        **kwargs : dict
            Additional keyword arguments for the search operation.

        Returns
        -------
        Any
            The search result, which can vary depending on the query and options.

        Raises
        ------
        RuntimeError
            If an error occurs during the search operation.
        """

        use_partitions = kwargs.get("use_partitions", False)
        collection = pymilvus.Collection(name=name)

        try:
            if use_partitions:
                partitions = kwargs.get("partitions")
                collection.load(partitions)
            else:
                collection.load()

            if query:
                result = collection.query(expr=query, **kwargs)
            else:
                result = collection.search(**kwargs)

            return result

        except Exception as exec_info:
            raise RuntimeError(f"Error while performing search: {exec_info}") from exec_info
        finally:
            collection.release()

    @with_mutex("_mutex")
    def update(self, name: str, data: typing.Any, **kwargs: dict[str, typing.Any]) -> None:
        pass

    @with_mutex("_mutex")
    def delete_by_keys(self, name: str,
                       keys: typing.Union[int, str, list],
                       **kwargs: dict[str, typing.Any]) -> list[typing.Union[str, int]]:
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

        response = self._client.delete(collection_name=name, pks=keys, **kwargs)

        return response

    def delete(self, name: str, expr: str, **kwargs: dict[str, typing.Any]) -> typing.Any:
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
        typing.Any
            Returns vectors of the given keys that are delete from the resource.
        """

        return self._handler.delete(collection_name=name, expression=expr, **kwargs)

    def retrieve_by_keys(self, name: str,
                         keys: typing.Union[int, str, list],
                         **kwargs: dict[str, typing.Any]) -> list[dict]:
        """
        Retrieve the inserted vectors using their primary keys from the Collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        keys : typing.Union[int, str, list]
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the retrieval operation.
        """
        return self._client.get(collection_name=name, ids=keys, **kwargs)

    def count(self, name: str, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the count operation.
        """

        # Updated counts is not updating when we delete the vectors from the collection.
        return self._client.num_entities(collection_name=name, **kwargs)

    @with_mutex("_mutex")
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
        """

        if self.has_store_object(name):
            type = kwargs.get("type", "collection")
            if type == "collection":
                self._client.drop_collection(collection_name=name)
            elif type == "partition" and "partition_name" in kwargs:
                partition_name = kwargs["partition_name"]
                if self._handler.has_partition(collection_name=name, partition_name=partition_name):
                    self._handler.drop_partition(collection_name=name, partition_name=partition_name)
            elif type == "index":
                if "field_name" in kwargs and "index_name" in kwargs:
                    self._handler.drop_index(collection_name=name,
                                          field_name=kwargs["field_name"],
                                          index_name=kwargs["index_name"])
                else:
                    raise ValueError("Mandatory fields missing")

    @with_mutex("_mutex")
    def close(self) -> None:
        """
        Close the connection to the Milvus vector database.

        This method disconnects from the Milvus vector database by removing the connection.

        """
        self._client.close()
