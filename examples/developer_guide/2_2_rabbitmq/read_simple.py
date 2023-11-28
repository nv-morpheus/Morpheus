#!/usr/bin/env python3
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
import os

import click
from rabbitmq_source_stage import RabbitMQSourceStage
from rabbitmq_source_stage_deco import rabbitmq_source

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option("--use_source_function",
              is_flag=True,
              default=False,
              help="Use the function based version of the RabbitMQ source stage instead of the class")
def run_pipeline(use_source_function: bool):
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    config = Config()
    config.num_threads = os.cpu_count()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    if use_source_function:
        pipeline.set_source(rabbitmq_source(config, host='localhost', exchange='logs'))
    else:
        pipeline.set_source(RabbitMQSourceStage(config, host='localhost', exchange='logs'))

    # Add monitor to record the performance of our new stages
    pipeline.add_stage(MonitorStage(config))

    # Write the to the output file
    pipeline.add_stage(WriteToFileStage(config, filename='/tmp/results.json', file_type=FileTypes.JSON, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
