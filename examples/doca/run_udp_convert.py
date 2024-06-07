# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import click

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.messages import RawPacketMessage
from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--nic_addr",
    help="NIC PCI Address",
    required=True,
)
@click.option(
    "--gpu_addr",
    help="GPU PCI Address",
    required=True,
)
def run_pipeline(nic_addr, gpu_addr):
    # Enable the default logger
    configure_logging(log_level=logging.WARNING)

    CppConfig.set_should_use_cpp(True)

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = os.cpu_count()
    print(f"using threads = {config.num_threads}")
    config.edge_buffer_size = 1024 * 16

    pipeline = LinearPipeline(config)

    # add doca source stage
    pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, 'udp'))

    def count_raw_packets(message: RawPacketMessage):
        return message.num

    pipeline.add_stage(
        MonitorStage(config,
                     description="DOCA GPUNetIO Raw rate",
                     unit='pkts',
                     determine_count_fn=count_raw_packets,
                     delayed_start=True))

    pipeline.add_stage(DocaConvertStage(config))

    # Uncomment the following lines to display the number of rows per MesssageMeta
    # @stage
    # def stage_counter(msg: MessageMeta) -> MessageMeta:
    #     with msg.mutable_dataframe() as df:
    #         print(f"\nlen(df) = {len(df)}\n")

    # pipeline.add_stage(stage_counter(config))

    pipeline.add_stage(MonitorStage(config, description="Convert rate", unit='pkts', delayed_start=True))
    pipeline.add_stage(WriteToFileStage(config, filename=".tmp/out.csv", overwrite=True))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
