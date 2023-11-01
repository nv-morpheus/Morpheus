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
from morpheus.messages.multi_message import MultiMessage
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.service.vdb.utils import VectorDBServiceFactory
from morpheus.service.vdb.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


class WriteToVectorDBStage(PassThruTypeMixin, SinglePortStage):
    """
    Writes messages to a Vector Database.

    Parameters
    ----------
    config : Config
        Pipeline configuration instance.
    service : typing.Union[str, VectorDBService]
        Either the name of the vector database service to use or an instance of VectorDBService
        for managing the resource.
    resource_name : str
        The identifier of the resource on which operations are to be performed in the vector database.
    embedding_column_name : str, optional
        Name of the embedding column, by default "embedding".
    recreate : bool, optional
        Specifies whether to recreate the resource if it already exists, by default False.
    resource_kwargs : dict, optional
        Additional keyword arguments to pass when performing vector database writes on a given resource.
    **service_kwargs : dict
        Additional keyword arguments to pass when creating a VectorDBService instance.

    Raises
    ------
    ValueError
        If `service` is not a valid string (service name) or an instance of VectorDBService.
    """

    def __init__(self,
                 config: Config,
                 service: typing.Union[str, VectorDBService],
                 resource_name: str,
                 embedding_column_name: str = "embedding",
                 recreate: bool = False,
                 resource_kwargs: dict = None,
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
            self._service.create(name=self._resource_name, **self._resource_kwargs)

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
        typing.Tuple(ControlMessage, MultiResponseMessage, MultiMessage)
            Accepted input types.

        """
        return (ControlMessage, MultiResponseMessage, MultiMessage)

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def on_completed(self):
        # Close vector database service connection
        self._service.close()

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        def extract_df(msg):
            df = None

            if isinstance(msg, ControlMessage):
                df = msg.payload().df
            elif isinstance(msg, MultiResponseMessage):
                df = msg.get_meta()
                if df is not None and not df.empty:
                    embeddings = msg.get_probs_tensor()
                    df[self._embedding_column_name] = embeddings.tolist()
            elif isinstance(msg, MultiMessage):
                df = msg.get_meta()
            else:
                raise RuntimeError(f"Unexpected message type '{type(msg)}' was encountered.")

            return df

        def on_data(msg):
            try:
                df = extract_df(msg)

                if df is not None and not df.empty:
                    result = self._service.insert_dataframe(name=self._resource_name, df=df, **self._resource_kwargs)

                    if isinstance(msg, ControlMessage):
                        msg.set_metadata("insert_response", result)

                    return msg

            except Exception as exc:
                logger.error("Unable to insert into collection: %s due to %s", self._resource_name, exc)

            return None

        to_vector_db = builder.make_node(self.unique_name,
                                         ops.map(on_data),
                                         ops.filter(lambda x: x is not None),
                                         ops.on_completed(self.on_completed))

        builder.make_edge(input_node, to_vector_db)

        # Return input unchanged to allow passthrough
        return to_vector_db
