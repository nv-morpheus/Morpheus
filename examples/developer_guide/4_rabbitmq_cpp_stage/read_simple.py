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

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option('--use_cpp', default=True)
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
def run_pipeline(use_cpp, num_threads):
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    CppConfig.set_should_use_cpp(use_cpp)

    config = Config()
    config.num_threads = num_threads

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(RabbitMQSourceStage(config, host='localhost', exchange='logs'))

    # Add monitor to record the performance of our new stages
    pipeline.add_stage(MonitorStage(config))

    # Write the to the output file
    pipeline.add_stage(WriteToFileStage(config, filename='/tmp/results.json', file_type=FileTypes.JSON, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
