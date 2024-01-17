# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import pickle
import time
from dataclasses import dataclass

import mrc
from mrc.core import operators as ops
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import validator

import cudf

from morpheus.messages import ControlMessage
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.service.vdb.utils import VectorDBServiceFactory
from morpheus.service.vdb.vector_db_service import VectorDBService
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_VECTOR_DB
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@dataclass
class AccumulationStats:
    msg_count: int
    last_insert_time: float
    data: list[cudf.DataFrame]


class WriteToVDBParamContract(BaseModel):
    embedding_column_name: str = "embedding"
    recreate: bool = False
    service: str = Field(default_factory=None)
    is_service_serialized: bool = False
    resource_name: str = Field(default_factory=None)
    resource_kwargs: dict = Field(default_factory=dict)
    service_kwargs: dict = Field(default_factory=dict)
    batch_size: int = 1024
    write_time_interval: float = 3.0

    @validator('service', pre=True)
    def validate_service(cls, v):
        if not v:
            raise ValueError("Service must be a service name or a serialized instance of VectorDBService")
        return v

    @validator('resource_name', pre=True)
    def validate_resource_name(cls, v):
        if not v:
            raise ValueError("Resource name must not be None or Empty.")
        return v


@register_module(WRITE_TO_VECTOR_DB, MORPHEUS_MODULE_NAMESPACE)
def _write_to_vector_db(builder: mrc.Builder):
    """
    Deserializes incoming messages into either MultiMessage or ControlMessage format.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'embedding_column_name': str, the name of the column containing embeddings (default is "embedding").
    - 'recreate': bool, whether to recreate the resource if it already exists (default is False).
    - 'service': str, the name of the service or a serialized instance of VectorDBService.
    - 'is_service_serialized': bool, whether the provided service is serialized (default is False).
    - 'resource_name': str, the name of the collection resource (must not be None or empty).
    - 'resource_kwargs': dict, additional keyword arguments for resource creation.
    - 'service_kwargs': dict, additional keyword arguments for VectorDBService creation.
    - 'batch_size': int, accumulates messages until reaching the specified batch size for writing to VDB.
    - 'write_time_interval': float, specifies the time interval (in seconds) for writing messages, or writing messages
    when the accumulated batch size is reached.

    Raises
    ------
    ValueError
        If 'resource_name' is None or empty.
        If 'service' is not provided or is not a valid service name or a serialized instance of VectorDBService.
    """

    module_config = builder.get_current_module_config()

    try:
        write_to_vdb_config = WriteToVDBParamContract(**module_config)
    except ValidationError as e:
        # Format the error message for better readability
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid configuration for write_to_vector_db: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    embedding_column_name = write_to_vdb_config.embedding_column_name
    recreate = write_to_vdb_config.recreate
    service = write_to_vdb_config.service
    is_service_serialized = write_to_vdb_config.is_service_serialized
    resource_name = write_to_vdb_config.resource_name
    resource_kwargs = write_to_vdb_config.resource_kwargs
    service_kwargs = write_to_vdb_config.service_kwargs
    batch_size = write_to_vdb_config.batch_size
    write_time_interval = write_to_vdb_config.write_time_interval

    # Check if service is serialized and convert if needed
    service: VectorDBService = (pickle.loads(bytes(service, "latin1")) if is_service_serialized else
                                VectorDBServiceFactory.create_instance(service_name=service, **service_kwargs))

    has_object = service.has_store_object(name=resource_name)

    if (recreate and has_object):
        # Delete the existing resource
        service.drop(name=resource_name)
        has_object = False

    # Ensure that the resource exists
    if (not has_object):
        service.create(name=resource_name, **resource_kwargs)

    accumulator_dict = {resource_name: AccumulationStats(msg_count=0, last_insert_time=-1, data=[])}

    def on_completed():
        final_df_references = []

        # Pushing remaining messages
        for key, accum_stats in accumulator_dict.items():
            if accum_stats.data:
                merged_df = cudf.concat(accum_stats.data)
                service.insert_dataframe(name=key, df=merged_df)
                final_df_references.append(accum_stats.data)
        # Close vector database service connection
        service.close()

        return final_df_references

    def extract_df(msg):
        df = None
        resrc_name = None

        if isinstance(msg, ControlMessage):
            df = msg.payload().df
            resrc_name = msg.get_metadata("resource_name")
        elif isinstance(msg, MultiResponseMessage):
            df = msg.get_meta()
            if df is not None and not df.empty:
                embeddings = msg.get_probs_tensor()
                df[embedding_column_name] = embeddings.tolist()
        elif isinstance(msg, MultiMessage):
            df = msg.get_meta()
        else:
            raise RuntimeError(f"Unexpected message type '{type(msg)}' was encountered.")

        return df, resrc_name

    def on_data(msg):
        try:
            df, resrc_name = extract_df(msg)

            if df is not None and not df.empty:
                if (not isinstance(df, cudf.DataFrame)):
                    df = cudf.DataFrame(df)
                final_df_references = []
                df_size = len(df)
                current_time = time.time()

                # Use default resource name
                if not resrc_name:
                    resrc_name = resource_name
                    if not service.has_store_object(resrc_name):
                        logger.error("Resource not exists in the vector database: %s", resource_name)
                        return final_df_references

                if resrc_name in accumulator_dict:
                    accumlator: AccumulationStats = accumulator_dict[resrc_name]
                    accumlator.msg_count += df_size
                    accumlator.data.append(df)
                else:
                    accumulator_dict[resrc_name] = AccumulationStats(msg_count=df_size, last_insert_time=-1, data=[df])

                for key, accum_stats in accumulator_dict.items():
                    if accum_stats.msg_count >= batch_size or (accum_stats.last_insert_time != -1 and
                                                               (current_time - accum_stats.last_insert_time)
                                                               >= write_time_interval):
                        if accum_stats.data:
                            merged_df = cudf.concat(accum_stats.data)
                            service.insert_dataframe(name=key, df=merged_df, **resource_kwargs)
                            final_df_references.append(merged_df)
                            # Reset accumlator stats
                            accum_stats.data.clear()
                            accum_stats.last_insert_time = current_time
                            accum_stats.msg_count = 0

                return final_df_references

        except Exception as exc:
            logger.error("Unable to insert into collection: %s due to %s", resrc_name, exc)

    node = builder.make_node(WRITE_TO_VECTOR_DB, ops.map(on_data), ops.on_completed(on_completed))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)


WriteToVectorDB = ModuleInterface(WRITE_TO_VECTOR_DB, MORPHEUS_MODULE_NAMESPACE)
