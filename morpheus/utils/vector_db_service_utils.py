# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from pymilvus import DataType

MILVUS_DATA_TYPE_MAP = {
    "int8": DataType.INT8,
    "int16": DataType.INT16,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "bool": DataType.BOOL,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "binary_vector": DataType.BINARY_VECTOR,
    "float_vector": DataType.FLOAT_VECTOR,
    "string": DataType.STRING,
    "varchar": DataType.VARCHAR,
    "json": DataType.JSON,
}


def with_mutex(lock_name):
    """
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            with getattr(args[0], lock_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator

class CollectionLockManager:
    def __init__(self, collection, mutex):
        self.collection = collection
        self.mutex = mutex

    def __enter__(self):
        self.mutex.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.collection.release()
        self.mutex.release()
