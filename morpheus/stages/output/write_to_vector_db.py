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

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.service.vector_db_service import VectorDBService
from morpheus.utils.vector_db_service_utils import VectorDBServiceFactory

logger = logging.getLogger(__name__)


class WriteToVectorDBStage(SinglePortStage):
    """
    Writes messages to a Vector Database.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    resource_name : str
        The name of the resource managed by this instance.
    resource_conf : dict
        Additional resource configuration when performing vector database writes.
    service : typing.Union[str, VectorDBService]
        Either the name of the vector database service to use or an instance of VectorDBService
        for managing the resource.
    **service_kwargs : dict[str, typing.Any]
        Additional keyword arguments to pass when creating a VectorDBService instance.

    Raises
    ------
    ValueError
        If `service` is not a valid string (service name) or an instance of VectorDBService.
    """

    def __init__(self,
                 config: Config,
                 *,
                 resource_name: str,
                 embedding_column_name: str = "embedding",
                 recreate: bool = False,
                 resource_kwargs: dict = None,
                 service: typing.Union[str, VectorDBService],
                 **service_kwargs):

        super().__init__(config)

        self._resource_name = resource_name
        self._embedding_column_name = embedding_column_name
        self._recreate = recreate
        self._resource_kwargs = resource_kwargs if resource_kwargs is not None else {}

        if isinstance(service, str):
            # If service is a string, assume it's the service name
            self._service: VectorDBService = VectorDBServiceFactory.create_instance(service_name=service,
                                                                                    **service_kwargs)
        elif isinstance(service, VectorDBService):
            # If service is an instance of VectorDBService, use it directly
            self._service: VectorDBService = service
        else:
            raise ValueError("service must be a string (service name) or an instance of VectorDBService")

        has_object = self._service.has_store_object(name=self._resource_name)

        if (self._recreate and has_object):
            # Delete the existing resource
            self._service.drop(name=self._resource_name)
            has_object = False

        # Ensure that the resource exists
        if (not has_object):
            self._service.create(name=self._resource_name, collection_conf=self._resource_kwargs)

        # Get the service for just the resource we are interested in
        self._resource_service = self._service.load_resource(name=self._resource_name)

    @property
    def name(self) -> str:
        return "to-vector-db"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (ControlMessage, MultiResponseMessage)

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def on_completed(self):
        # Close vector database service connection
        self._service.close()

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        def on_data_control_message(ctrl_msg: ControlMessage) -> ControlMessage:
            # Insert entries in the dataframe to vector database.
            result = self._service.insert_dataframe(name=self._resource_name,
                                                    df=ctrl_msg.payload().df,
                                                    **self._resource_kwargs)

            ctrl_msg.set_metadata("insert_response", result)

            return ctrl_msg

        def on_data_multi_message(msg: MultiResponseMessage) -> MultiResponseMessage:
            # Probs tensor contains all of the embeddings
            embeddings = msg.get_probs_tensor()
            embeddings_list = embeddings.tolist()

            # # Figure out which columns we need
            # available_columns = set(msg.get_meta_column_names())

            # if (self._include_columns is not None):
            #     available_columns = available_columns.intersection(self._include_columns)
            # if (self._exclude_columns is not None):
            #     available_columns = available_columns.difference(self._exclude_columns)

            # Build up the metadata from the message
            metadata = msg.get_meta().to_pandas()

            # Add in the embedding results to the dataframe
            metadata[self._embedding_column_name] = embeddings_list

            # if (not self._service.has_store_object(name=self._resource_name)):
            #     # Create the vector database resource
            #     self._service.create_from_dataframe(name=self._resource_name, df=metadata, index_field="embedding")

            # Insert entries in the dataframe to vector database.
            result = self._resource_service.insert_dataframe(df=metadata, **self._resource_kwargs)

            return msg

        if (issubclass(input_stream[1], ControlMessage)):
            on_data = ops.map(on_data_control_message)
        elif (issubclass(input_stream[1], MultiResponseMessage)):
            on_data = ops.map(on_data_multi_message)
        else:
            raise RuntimeError(f"Unexpected input type {input_stream[1]}")

        to_vector_db = builder.make_node(self.unique_name, on_data, ops.on_completed(self.on_completed))

        builder.make_edge(stream, to_vector_db)
        stream = to_vector_db

        # Return input unchanged to allow passthrough
        return stream, input_stream[1]
