# Copyright (c) 2021, NVIDIA CORPORATION.
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
import os
import typing

import cudf
import typing_utils

from morpheus.config import Config
from morpheus.pipeline import Pipeline
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.general_stages import SwitchStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.pipeline.preprocessing import PreprocessNLPStage
from morpheus.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class AddDataLenStage(SinglePortStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._use_dask = c.use_dask

    @property
    def name(self) -> str:
        return "add-data_len"

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, StreamFuture[cudf.DataFrame])

    @staticmethod
    def process_dataframe(x: cudf.DataFrame):

        x["data_len"] = x["data"].str.len()

        return x

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = cudf.DataFrame

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):

            stream = stream.map(AddDataLenStage.process_dataframe)
            out_type = StreamFuture[cudf.DataFrame]
        else:
            stream = stream.async_map(AddDataLenStage.process_dataframe, executor=self._pipeline.thread_pool)

        return stream, out_type


def run():

    # Find our current example folder
    this_ex_dir = os.path.dirname(__file__)
    ex_root_dir = os.path.dirname(this_ex_dir)

    c = Config.get()

    c.log_level = logging.DEBUG
    c.debug = True  # Allows timestamps to be added for certain stages

    configure_logging(c.log_level, c.log_config_file)

    pipeline = Pipeline(c)

    # Create two source stages from different files
    file_source1 = FileSourceStage(c, os.path.join(ex_root_dir, "data", "with_data_len.json"), iterative=True)
    file_source2 = FileSourceStage(c, os.path.join(ex_root_dir, "data", "without_data_len.json"), iterative=True)

    deser_stage = DeserializeStage(c)

    # pipeline.add_edge(merge_stage, deser_stage)
    pipeline.add_edge(file_source1, deser_stage)
    pipeline.add_edge(file_source2, deser_stage)

    def has_data_len(x: MultiMessage) -> int:
        return 0 if "data_len" in x.meta.df else 1

    # Create a SwitchStage to fork the stream for messages that dont have the "data_len" column
    switch_stage = SwitchStage(c, 2, has_data_len)

    pipeline.add_edge(deser_stage, switch_stage)

    preproc_stage = PreprocessNLPStage(c)

    # Attach the pre-process node to port 1 of the switch stage
    pipeline.add_edge(switch_stage.output_ports[0], preproc_stage)

    # If it doesnt have the field, serialize it back to a dataframe, then add the field manually
    ser_to_df_stage = SerializeStage(c, as_cudf_df=True)

    # Connect to port 1 of the switch
    pipeline.add_edge(switch_stage.output_ports[1], ser_to_df_stage)

    # Now use the custom stage AddDataLenStage to calculate the column
    add_data_len_stage = AddDataLenStage(c)

    pipeline.add_edge(ser_to_df_stage, add_data_len_stage)

    # # Now loop back to the deserialize stage
    pipeline.add_edge(add_data_len_stage, deser_stage)

    monitor_stage = MonitorStage(c, description="Throughput From 2 Sources")

    pipeline.add_edge(preproc_stage, monitor_stage)

    # Fork the pipeline to do two different serializations
    serial_withtime_stage = SerializeStage(c, exclude=[r'^ID$'])
    serial_notime_stage = SerializeStage(c, exclude=[r'^ID$', r'^ts_'])

    pipeline.add_edge(monitor_stage, serial_withtime_stage)
    pipeline.add_edge(monitor_stage, serial_notime_stage)

    # Write out to two files
    out_file_withtime_stage = WriteToFileStage(c, os.path.join(this_ex_dir, "mimo_out_withtime.json"), overwrite=True)
    out_file_notime_stage = WriteToFileStage(c, os.path.join(this_ex_dir, "mimo_out_notime.json"), overwrite=True)

    pipeline.add_edge(serial_withtime_stage, out_file_withtime_stage)
    pipeline.add_edge(serial_notime_stage, out_file_notime_stage)

    # Build the pipeline. This will determine all stage types for visualization
    pipeline.build()

    vis_file = os.path.join(this_ex_dir, "mimo_pipeline.png")

    # Generate the visualization to see the organization of the pipeline
    pipeline.visualize(filename=vis_file, rankdir="LR")

    logger.info("Pipeline organization vizualization generated at: %s", vis_file)

    # Finally kick off the run
    pipeline.run()


if __name__ == "__main__":
    run()
