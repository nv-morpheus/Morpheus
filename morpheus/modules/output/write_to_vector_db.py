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

from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages import ControlMessage

import mrc
from mrc.core import operators as ops
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import register_module
from morpheus.utils.module_ids import WRITE_TO_VECTOR_DB
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE

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
    - 'ensure_sliceable_index': bool, whether to ensure messages have a sliceable index.
    - 'message_type': type, the type of message to output (MultiMessage or ControlMessage).
    - 'task_type': str, optional, the type of task for ControlMessages.
    - 'task_payload': dict, optional, the payload for the task in ControlMessages.
    - 'batch_size': int, the size of batches for message processing.
    - 'max_concurrency': int, optional, the maximum concurrency for processing.
    - 'should_log_timestamp': bool, optional, whether to log timestamps.
    """

    module_config = builder.get_current_module_config()
    embedding_column_name = module_config.get("embedding_column_name", "embedding")

    def on_completed():
        # Close vector database service connection
        self._service.close()

    def extract_df(msg):
            df = None

            if isinstance(msg, ControlMessage):
                df = msg.payload().df
                # For control message, check if we have a collection tag
            elif isinstance(msg, MultiResponseMessage):
                df = msg.get_meta()
                if df is not None and not df.empty:
                    embeddings = msg.get_probs_tensor()
                    df[embedding_column_name] = embeddings.tolist()
            elif isinstance(msg, MultiMessage):
                df = msg.get_meta()
            else:
                raise RuntimeError(f"Unexpected message type '{type(msg)}' was encountered.")

            return df  # Return df, collection_tag or df, None

    def on_data(msg):
        try:
            df = extract_df(msg)
            # df, collection_name = extract_df(msg)
            # Call accumulator function, progress if we have enough data or our timeout has elapsed
            # Need a different accumulator for each collection_name

            if df is not None and not df.empty:
                result = self._service.insert_dataframe(name=self._resource_name, df=df, **self._resource_kwargs)

                if isinstance(msg, ControlMessage):
                    msg.set_metadata("insert_response", result)

                return msg

        except Exception as exc:
            logger.error("Unable to insert into collection: %s due to %s", self._resource_name, exc)

        return None

    node = builder.make_node(WRITE_TO_VECTOR_DB,
                             ops.map(on_data),
                             ops.filter(lambda x: x is not None),
                             ops.on_completed(on_completed))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)


write_to_vector_db = ModuleInterface(WRITE_TO_VECTOR_DB, MORPHEUS_MODULE_NAMESPACE)
