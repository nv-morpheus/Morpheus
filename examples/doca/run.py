# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=32,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=70,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
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
@click.option(
    "--source_ip_filter",
    default="",
    help="Source IP Address to filter packets by",
)
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    model_fea_length,
    nic_addr,
    gpu_addr,
    source_ip_filter,
):
    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    CppConfig.set_should_use_cpp(True)

    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length

    config.class_labels = ["probs"]
    config.edge_buffer_size = 4

    pipeline = LinearPipeline(config)

    pipeline.set_source(DocaSourceStage(
        config,
        nic_addr,
        gpu_addr,
        source_ip_filter
    ))

    pipeline.add_stage(MonitorStage(config, description="Serialize rate"))

    # pipeline.add_stage(WriteToFileStage(config, filename=".tmp/doca_test.csv", overwrite=True))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
