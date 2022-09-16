#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from write_to_rabbitmq_stage import WriteToRabbitMQStage

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.utils.logger import configure_logging


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']
    input_file = os.path.join(root_dir, 'examples/data/email.jsonlines')

    config = Config()
    config.num_threads = os.cpu_count()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Set source stage
    pipeline.add_stage(WriteToRabbitMQStage(config, host='localhost', exchange='logs'))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
