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

from morpheus.config import AEFeatureScalar
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.auto_encoder_inference_stage import AutoEncoderInferenceStage
from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.preprocess_ae_stage import PreprocessAEStage
from morpheus.stages.preprocess.train_ae_stage import TrainAEStage
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
    default=1024,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--columns_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Feature columns file",
)
@click.option(
    "--input_glob",
    type=str,
    required=True,
    help="Inference input glob",
)
@click.option(
    "--train_data_glob",
    type=str,
    required=False,
    help="Train data glob",
)
@click.option(
    "--pretrained_filename",
    type=click.Path(exists=True, readable=True),
    required=False,
    help="File with pre-trained user models",
)
@click.option(
    "--models_output_filename",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--output_file",
    default="./cloudtrail-detections.csv",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(num_threads,
                 pipeline_batch_size,
                 model_max_batch_size,
                 columns_file,
                 input_glob,
                 train_data_glob,
                 pretrained_filename,
                 models_output_filename,
                 output_file):

    configure_logging(log_level=logging.DEBUG)

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.AE
    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.feature_scaler = AEFeatureScalar.STANDARD

    with open(columns_file, "r") as lf:
        config.ae.feature_columns = [x.strip() for x in lf.readlines()]

    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size

    # Create a pipeline object
    pipeline = LinearPipeline(config)

    # Add a source stage
    pipeline.set_source(CloudTrailSourceStage(config, input_glob=input_glob))

    # Add a training stage
    pipeline.add_stage(
        TrainAEStage(config,
                     pretrained_filename=pretrained_filename,
                     train_data_glob=train_data_glob,
                     source_stage_class="morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage",
                     models_output_filename=models_output_filename,
                     seed=42,
                     sort_glob=True))

    # Add a inference stage
    pipeline.add_stage(AutoEncoderInferenceStage(config))

    # Add serialize stage
    pipeline.add_stage(SerializeStage(config))

    # Add a write file stage
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.add_stage(MonitorStage(config, description="Postprocessing rate"))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
