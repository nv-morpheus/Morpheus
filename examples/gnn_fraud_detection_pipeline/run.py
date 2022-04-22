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

import click
import psutil
from stages.classification_stage import ClassificationStage
from stages.graph_construction_stage import FraudGraphConstructionStage
from stages.graph_sage_stage import GraphSAGEStage

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.pipeline import LinearPipeline
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.utils.logging import configure_logging


@click.command()
@click.option(
    "--num_threads",
    default=psutil.cpu_count(),
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
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default="validation.csv",
    required=True,
    help="Input data filepath",
)
@click.option(
    "--training_file",
    type=click.Path(exists=True, readable=True),
    default="training.csv",
    required=True,
    help="Training data filepath",
)
@click.option(
    "--model-hinsage-file",
    type=click.Path(exists=True, readable=True),
    default="model/hinsage-model.pt",
    required=True,
    help="Trained hinsage model filepath",
)
@click.option(
    "--model-xgb-file",
    type=click.Path(exists=True, readable=True),
    default="model/xgb-model.pt",
    required=True,
    help="Trained xgb model filepath",
)
@click.option(
    "--output_file",
    default="output.csv",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
    model_fea_length,
    input_file,
    training_file,
    model_hinsage_file,
    model_xgb_file,
    output_file,
):
    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    CppConfig.set_should_use_cpp(False)

    # Its necessary to get the global config object and configure it for FIL mode
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length

    config.class_labels = ["probs"]
    config.edge_buffer_size = 4

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # # Set source stage
    pipeline.set_source(FileSourceStage(
        config,
        filename=input_file,
        filter_null=False,
    ))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Add the graph construction stage
    pipeline.add_stage(FraudGraphConstructionStage(config, training_file))
    pipeline.add_stage(MonitorStage(config, description="Graph construction rate"))

    # add sage inference stage
    pipeline.add_stage(GraphSAGEStage(config, model_hinsage_file))
    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    # Add classification stage
    pipeline.add_stage(ClassificationStage(config, model_xgb_file))
    pipeline.add_stage(MonitorStage(config, description="Add classification rate"))

    # Convert the probabilities to serialized JSON strings using the custom serialization stage
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Serialize rate"))

    # # Write the file to the output
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
