# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import mrc
from mrc.core import operators as ops

from morpheus.messages import ControlMessage
from morpheus.utils.controllers.elasticsearch_controller import ElasticsearchController
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_ELASTICSEARCH
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(WRITE_TO_ELASTICSEARCH, MORPHEUS_MODULE_NAMESPACE)
def write_to_elasticsearch(builder: mrc.Builder):
    """
    This module reads input data stream, converts each row of data to a document format suitable for Elasticsearch,
    and writes the documents to the specified Elasticsearch index using the Elasticsearch bulk API.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.

    Notes
    -----
    Configurable Parameters:
        - index               (str): Elastic search index.
        - connection_kwargs   (dict): Elastic search connection kwrags configuration.
        - pickled_func_config (str): Pickled custom function configuration to updated connection_kwargs as needed
        to established client connection. Custom function should return connection_kwargs; default: None.
        - refresh_period_secs (int): Time in seconds to refresh the client connection; default: 2400.
        - raise_on_exception  (bool): It is used to raise or supress exceptions when writing to Elasticsearch;
        deafult: False
    """

    config = builder.get_current_module_config()

    index = config.get("index", None)
    connection_kwargs = config.get("connection_kwargs")
    raise_on_exception = config.get("raise_on_exception", False)
    pickled_func_config = config.get("pickled_func_config", None)
    refresh_period_secs = config.get("refresh_period_secs", 2400)

    controller = ElasticsearchController(index=index,
                                         connection_kwargs=connection_kwargs,
                                         raise_on_exception=raise_on_exception,
                                         refresh_period_secs=refresh_period_secs,
                                         pickled_func_config=pickled_func_config)

    def on_data(message: ControlMessage):

        controller.refresh_client()

        meta = message.payload()
        rows = meta.df.to_pandas().to_dict('records')

        actions = []
        for row in rows:
            action = {"_index": index, "_source": row}
            actions.append(action)

        controller.parallel_bulk_write(actions)  # Parallel bulk upload to Elasticsearch

        return message

    def on_complete():
        controller.close_client()  # Close client

    node = builder.make_node(WRITE_TO_ELASTICSEARCH, ops.map(on_data), ops.on_completed(on_complete))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
