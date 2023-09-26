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
import threading
import time

import pandas as pd
from pymilvus import BulkInsertState
from pymilvus import Collection
from pymilvus import CollectionSchema
from pymilvus import DataType
from pymilvus import FieldSchema
from pymilvus import connections
from pymilvus import utility

from morpheus.controllers.vector_db_controller import VectorDBController
from morpheus.controllers.vector_db_controller import with_mutex

logger = logging.getLogger(__name__)


class MilvusVectorDBController(VectorDBController):
    """
    """

    def __init__(self, host: str, port: str, alias: str = "default", pool_size: int = 1, **kwargs):
        self._data_type_dict = {
            "int": DataType.INT64,
            "bool": DataType.BOOL,
            "float": DataType.FLOAT,
            "double": DataType.DOUBLE,
            "binary_vector": DataType.BINARY_VECTOR,
            "float_vector": DataType.FLOAT_VECTOR
        }
        self._alias = alias
        connections.connect(host=host, port=port, alias=self._alias, pool_size=pool_size, **kwargs)

    def has_collection(self, name) -> bool:
        return utility.has_collection(name)

    def list_collections(self) -> list[str]:
        return utility.list_collections()

    def _create_index(self, collection, field_name, index_params) -> None:
        collection.create_index(field_name=field_name, index_params=index_params)

    def _create_schema_field(self, field_conf: dict):
        dtype = self._data_type_dict[field_conf["dtype"].lower()]
        dim = field_conf.get("dim", None)

        if (dtype == DataType.BINARY_VECTOR or dtype == DataType.FLOAT_VECTOR):
            if not dim:
                raise ValueError(f"Dimensions for {dtype} should not be None")
            if not isinstance(dim, int):
                raise ValueError(f"Dimensions for {dtype} should be an integer")

        field_schema = FieldSchema(name=field_conf["name"],
                                   dtype=dtype,
                                   description=field_conf.get("description", ""),
                                   is_primary=field_conf["is_primary"],
                                   dim=dim)
        return field_schema

    @with_mutex("_mutex")
    def create_collection(self, collection_config):
        collection_conf = collection_config.get("collection_conf")
        collection_name = collection_conf.get("name")
        index_conf = collection_conf.get("index_conf", None)
        partition_conf = collection_conf.get("partition_conf", None)

        schema_conf = collection_conf.get("schema_conf")
        schema_fields_conf = schema_conf.get("schema_fields")

        if not self.has_collection(collection_name):

            if len(schema_fields_conf) == 0:
                raise ValueError("Cannot create collection as provided empty schema_fields configuration")

            schema_fields = [self._create_schema_field(field_conf=field_conf) for field_conf in schema_fields_conf]

            schema = CollectionSchema(fields=schema_fields,
                                      auto_id=schema_conf.get("auto_id", False),
                                      description=schema_conf.get("description", ""))
            collection = Collection(name=collection_name,
                                    schema=schema,
                                    using=self._alias,
                                    shards_num=collection_conf.get("shards", 2),
                                    consistency_level=collection_conf.get("consistency_level", "Strong"))

            if partition_conf:
                # Iterate over each partition configuration
                for part in partition_conf:
                    collection.create_partition(part["name"], description=part.get("description", ""))
            if index_conf:
                self._create_index(collection=collection,
                                   field_name=index_conf["field_name"],
                                   index_params=index_conf["index_params"])

    @with_mutex("_mutex")
    def insert(self, name, data, **kwargs):

        partition_name = kwargs.get("partition_name", "_default")

        if isinstance(data, list):
            if not self.has_collection(name):
                raise ValueError(f"Collection {name} doesn't exist.")
            collection = Collection(name=name)

            collection.insert(data, partition_name=partition_name)
            collection.flush()

        # TODO (Bhargav): Load input data from a file
        # if isinstance(data, str):
        #     task_id = utility.do_bulk_insert(collection_name=name,
        #                                      partition_name=kwargs.get("partition_name", None),
        #                                      files=[data])
        #
        #     while True:
        #         time.sleep(2)
        #         state = utility.get_bulk_insert_state(task_id=task_id)
        #         if state.state == BulkInsertState.ImportFailed or state.state == BulkInsertState.ImportFailedAndCleaned:
        #             raise Exception(f"The task {state.task_id} failed, reason: {state.failed_reason}")

        #         if state.state >= BulkInsertState.ImportCompleted:
        #             break

        elif isinstance(data, pd.DataFrame):
            collection_conf = kwargs.get("collection_conf")
            index_conf = collection_conf.get("index_conf", None)
            params = collection_conf.get("params", {})

            collection, _ = Collection.construct_from_dataframe(
                collection_conf["name"],
                data,
                primary_field=collection_conf["primary_field"],
                auto_id=collection_conf.get("auto_id", False),
                description=collection_conf.get("description", None),
                partition_name=partition_name,
                **params
            )

            if index_conf:
                self._create_index(collection=collection,
                                   field_name=index_conf["field_name"],
                                   index_params=index_conf["index_params"])

            collection.flush()
        else:
            raise ValueError("Unsupported data type for insertion.")

    @with_mutex("_mutex")
    def search(self, name, query=None, **kwargs):
        is_partition_load = kwargs.get("is_partition_load", False)

        collection = Collection(name=name)

        try:
            if is_partition_load:
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
    def drop(self, name, **kwargs):

        type = kwargs.get("type", "collection")

        collection = Collection(name=name)
        if type == "index":
            collection.drop_index()
        elif type == "partition":
            partition_name = kwargs["partition_name"]
            collection.drop_partition(partition_name)
        else:
            collection.drop()

    @with_mutex("_mutex")
    def close(self):
        connections.remove_connection(alias=self._alias)
