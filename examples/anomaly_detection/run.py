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

import click
import psutil

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.pipeline import LinearPipeline
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.utils.logging import configure_logging
from user_prof_preprocessing import UserProfPreprocessingStage
from user_prof_serialize import UserProfSerializeStage


@click.command()
@click.option(
    "--num_threads",
    default=psutil.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=50000,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=20000,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default="pcap.jsonlines",
    required=True,
    help="Input filepath",
)
@click.option(
    "--output_file",
    default="pcap_out.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--model_fea_length",
    default=13,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--model_name",
    default="anomaly_detection_fil_model",
    help="The name of the model that is deployed on Tritonserver",
)
@click.option("--server_url", required=True, help="Tritonserver url")
def run_pipeline(num_threads,
                 pipeline_batch_size,
                 model_max_batch_size,
                 model_fea_length,
                 input_file,
                 output_file,
                 model_name,
                 server_url):

    # Find our current example folder
    this_ex_dir = os.path.dirname(__file__)

    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    # Its necessary to get the global config object and configure it for FIL mode
    config = Config.get()
    config.mode = PipelineModes.FIL

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.fil.model_fea_length = model_fea_length

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Add a source stage from the file created by `pcap_data_producer.py`
    pipeline.set_source(FileSourceStage(config, filename=input_file))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Add the custom preprocessing stage
    pipeline.add_stage(UserProfPreprocessingStage(config))

    # Add a inference stage.
    pipeline.add_stage(TritonInferenceStage(config, model_name=model_name, server_url=server_url))

    # Convert the probabilities to serialized JSON strings using the custom serialization stage
    pipeline.add_stage(UserProfSerializeStage(config))

    # Write the file to the output
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Save the pipeline vizualization
    pipeline.visualize(os.path.join(this_ex_dir, "pipeline.png"), rankdir="LR")

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
