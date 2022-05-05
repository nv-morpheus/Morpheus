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
from abp_pcap_preprocessing import AbpPcapPreprocessingStage

from morpheus.cli import FILE_TYPE_NAMES
from morpheus.cli import str_to_file_type
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.stages.general_stages import AddClassificationsStage
from morpheus.stages.general_stages import MonitorStage
from morpheus.stages.inference.inference_triton import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.pipeline.pipeline import LinearPipeline
from morpheus.stages.preprocess.preprocessing import DeserializeStage
from morpheus.utils.logging import configure_logging


@click.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
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
    default=40000,
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
    default="./pcap_out.jsonlines",
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
    default="abp-pcap-xgb",
    help="The name of the model that is deployed on Tritonserver",
)
@click.option(
    "--iterative",
    is_flag=True,
    default=False,
    help=("Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. "
          "Iterative mode is good for interleaving source stages."),
)
@click.option("--server_url", required=True, help="Tritonserver url")
@click.option(
    "--file_type",
    type=click.Choice(FILE_TYPE_NAMES, case_sensitive=False),
    default="auto",
    help=("Indicates what type of file to read. "
          "Specifying 'auto' will determine the file type from the extension."),
)
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    model_fea_length,
    input_file,
    output_file,
    model_name,
    iterative,
    server_url,
    file_type,
):

    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    CppConfig.set_should_use_cpp(False)

    # Its necessary to get the global config object and configure it for FIL mode
    config = Config()
    config.mode = PipelineModes.FIL

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.class_labels = ["probs"]

    kwargs = {}

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(
        FileSourceStage(
            config,
            filename=input_file,
            iterative=iterative,
            file_type=str_to_file_type(file_type.lower()),
            filter_null=False,
        ))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Add the custom preprocessing stage
    pipeline.add_stage(AbpPcapPreprocessingStage(config))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Preprocessing rate"))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
        ))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # Add a add classification stage
    pipeline.add_stage(AddClassificationsStage(config, labels=["probs"]))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Add classification rate", unit="add-class"))

    # Convert the probabilities to serialized JSON strings using the custom serialization stage
    pipeline.add_stage(SerializeStage(config, **kwargs))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Serialize rate", unit="ser"))

    # Write the file to the output
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Write to file rate", unit="to-file"))
    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
