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

from morpheus.messages import MessageControl
from morpheus.utils.loader_ids import FILE_TO_DF_LOADER
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILE_BATCHER, MODULE_NAMESPACE)
def file_batcher(builder: mrc.Builder):
    """
    This module loads the input files, removes files that are older than the chosen window of time,
    and then groups the remaining files by period that fall inside the window.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.
    """

    config = get_module_config(FILE_BATCHER, builder)

    period = config.get("period", None)

    def on_data(control_message: MessageControl):
        # Determine the date of the file, and apply the window filter if we have one
        # This needs to be in the payload, not a task, because batcher isn't a data loader
        # TODO(Devin)
        #control_message.pop_task("load")

        # TODO(Devin)
        data_type = "streaming"
        if (control_message.has_metadata("data_type")):
            data_type = control_message.get_metadata("data_type")

        payload = control_message.payload()
        df = payload.df.to_pandas()

        # TODO(Devin): Clean this up
        control_messages = []
        if len(df) > 0:
            # Now split by the batching settings
            df_period = df["ts"].dt.to_period(period)
            period_gb = df.groupby(df_period)
            n_groups = len(period_gb)

            logger.debug("Batching %d files => %d groups", len(df), n_groups)
            for group in period_gb.groups:
                period_df = period_gb.get_group(group)
                filenames = period_df["key"].to_list()

                load_task = {
                    "loader_id": FILE_TO_DF_LOADER,
                    "strategy": "aggregate",
                    "files": filenames,
                    "n_groups": n_groups,
                    "batcher_config": {  # TODO(Devin): remove this
                        "timestamp_column_name": config.get("timestamp_column_name"),
                        "schema": config.get("schema"),
                        "file_type": config.get("file_type"),
                        "filter_null": config.get("filter_null"),
                        "parser_kwargs": config.get("parser_kwargs"),
                        "cache_dir": config.get("cache_dir")
                    }
                }

                if (data_type == "payload"):
                    control_message.add_task("load", load_task)
                elif (data_type == "streaming"):
                    batch_control_message = control_messages.copy()
                    batch_control_message.add_task("load", load_task)
                    control_messages.append(batch_control_message)
                else:
                    raise Exception("Unknown data type")

        if (data_type == "payload"):
            control_messages.append(control_message)

        return control_messages

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.flatten()).subscribe(sub)

    node = builder.make_node_full(FILE_BATCHER, node_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
