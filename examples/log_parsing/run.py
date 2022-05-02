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

import os

import click
from inference import LogParsingInferenceStage
from postprocessing import LogParsingPostProcessingStage
from preprocessing import PreprocessLogParsingStage

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import BufferStage
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.preprocessing import DeserializeStage


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
    "--input_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Input filepath",
)
@click.option(
    "--output_file",
    default="log-parsing-output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
@click.option('--model_vocab_hash_file',
              required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Model vocab hash file to use for pre-processing")
@click.option('--model_vocab_file',
              required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Model vocab file to use for post-processing")
@click.option("--model_seq_length",
              default=256,
              type=click.IntRange(min=1),
              help="Sequence length to use for the model")
@click.option(
    "--model_name",
    required=True,
    help="The name of the model that is deployed on Triton server",
)
@click.option("--model_config_file", required=True, help="Model config file")
@click.option("--server_url", required=True, help="Tritonserver url")
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    input_file,
    output_file,
    model_vocab_hash_file,
    model_vocab_file,
    model_seq_length,
    model_name,
    model_config_file,
    server_url,
):
    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_seq_length

    # Create a pipeline object
    pipeline = LinearPipeline(config)

    # Add a source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False, repeat=1))

    # Add a buffer stage
    pipeline.add_stage(BufferStage(config))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))
    # Add a preprocessing stage
    pipeline.add_stage(
        PreprocessLogParsingStage(config,
                                  vocab_hash_file=model_vocab_hash_file,
                                  truncation=False,
                                  do_lower_case=False,
                                  stride=64,
                                  add_special_tokens=False))
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage
    pipeline.add_stage(
        LogParsingInferenceStage(config, model_name=model_name, server_url=server_url, force_convert_inputs=True))
    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    pipeline.add_stage(
        LogParsingPostProcessingStage(config, vocab_path=model_vocab_file, model_config_path=model_config_file))

    # Add a write file stage
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.add_stage(MonitorStage(config, description="Postprocessing rate"))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
