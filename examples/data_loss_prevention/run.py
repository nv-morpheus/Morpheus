# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib

import click
from dlp_stages.datasets_source import DatasetsSourceStage
from dlp_stages.dlp_input_processor import DLPInputProcessor
from dlp_stages.gliner_processor import GliNERProcessor
from dlp_stages.regex_processor import RegexProcessor
from dlp_stages.risk_scorer import RiskScorer

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging

logger = logging.getLogger(f"morpheus.{__name__}")
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MORPHEUS_ROOT = os.environ.get('MORPHEUS_ROOT', os.path.abspath(os.path.join(CUR_DIR, "..", "..")))


@click.command()
@click.option("--log_level",
              default="INFO",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              show_default=True,
              help="Specify the logging level to use.")
@click.option("--num_threads",
              default=16,
              type=click.IntRange(min=1),
              help="Number of internal pipeline threads to use.")
@click.option("--regex_file",
              help="JSON file containing regex patterns",
              default=os.path.join(CUR_DIR, "data/regex_patterns.json"),
              show_default=True,
              type=click.Path(exists=True, readable=True))
@click.option('--dataset',
              type=str,
              default=["gretel"],
              show_default=True,
              multiple=True,
              help=("Specify the datasets to use, can be set multiple times, valid datasets are: "
                    f"{', '.join(sorted(DatasetsSourceStage.AVAILABLE_DATASETS.keys()))}."))
@click.option('--input_file',
              type=click.Path(dir_okay=False, exists=True, readable=True),
              default=None,
              show_default=True,
              help=("Input file to use, if specified, overrides the dataset option."))
@click.option('--include_privacy_masks',
              is_flag=True,
              default=False,
              show_default=True,
              help=("Include privacy masks in the output DataFrame, ignored if --input_file is set. "
                    "This is useful for evaluation."))
@click.option('--num_samples',
              type=int,
              default=2000,
              show_default=True,
              help=("Number of samples to use from each dataset, ignored if --input_file is set, "
                    "set to -1 for all samples."))
@click.option('--repeat',
              type=int,
              default=1,
              show_default=True,
              help=("Repeat the input dataset, useful for testing. A value of 1 means no repeat."))
@click.option('--regex_only',
              is_flag=True,
              default=False,
              show_default=True,
              help=("Only perform regex matching and skip the GliNER processor. "
                    "Cannot be combined with --model_only."))
@click.option('--model_only',
              is_flag=True,
              default=False,
              show_default=True,
              help=("Only perform the GliNER processor and skip the regex matching. "
                    "Cannot be combined with --regex_only."))
@click.option("--server_url", required=True, help="Tritonserver url.", default="localhost:8001", show_default=True)
@click.option('--model_max_batch_size',
              type=int,
              default=16,
              show_default=True,
              help=("Maximum batch size for model inference, used by the GliNER processor. "
                    "Larger values may improve performance but require more GPU memory."))
@click.option('--pipeline_batch_size', type=int, default=1024 * 32, show_default=True, help=("Pipeline batch size."))
@click.option('--model_source_dir',
              help="Directory containing the GliNER model files",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True, resolve_path=True),
              default=os.path.join(MORPHEUS_ROOT, "models/dlp_models/gliner_bi_encoder"),
              show_default=True)
@click.option("--out_file",
              help="Output file",
              type=click.Path(dir_okay=False),
              default=os.path.join(MORPHEUS_ROOT, ".tmp/output/data_loss_prevention.jsonlines"),
              show_default=True,
              required=True)
def main(log_level: int,
         num_threads: int,
         regex_file: pathlib.Path,
         dataset: list[str],
         input_file: pathlib.Path | None,
         include_privacy_masks: bool,
         num_samples: int,
         repeat: int,
         regex_only: bool,
         model_only: bool,
         server_url: str,
         model_max_batch_size: int,
         pipeline_batch_size: int,
         model_source_dir: pathlib.Path,
         out_file: pathlib.Path):
    configure_logging(log_level=log_level)

    if regex_only and model_only:
        raise ValueError("Cannot use --regex_only and --model_only together.")

    if num_samples < 0:
        num_samples = None

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = num_threads
    config.model_max_batch_size = model_max_batch_size
    config.pipeline_batch_size = pipeline_batch_size

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    if input_file is not None:
        pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=repeat, filter_null=False))
    else:
        pipeline.set_source(
            DatasetsSourceStage(config,
                                dataset_names=dataset,
                                num_samples=num_samples,
                                include_privacy_masks=include_privacy_masks,
                                repeat=repeat))

    pipeline.add_stage(MonitorStage(config, description="Datasets Source"))

    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Deserialize Stage"))

    pipeline.add_stage(DLPInputProcessor(config))

    pipeline.add_stage(MonitorStage(config, description="Input Processor"))

    if not model_only:
        pipeline.add_stage(RegexProcessor(config, patterns_file=regex_file, include_pattern_names=regex_only))

        pipeline.add_stage(MonitorStage(config, description="Regex Processor"))

    output_columns = [
        "original_source_index",
        'risk_level',
        'risk_score',
        'highest_confidence',
        'num_minimal',
        'num_low',
        'num_medium',
        'num_high',
        'num_critical',
        'data_types_found'
    ]

    if include_privacy_masks:
        output_columns.append('privacy_mask')

    if regex_only:
        risk_scorer_input = "labels"

    else:
        pipeline.add_stage(GliNERProcessor(config, server_url=server_url, model_source_dir=str(model_source_dir)))

        pipeline.add_stage(MonitorStage(config, description="GliNER Processor"))

        risk_scorer_input = "dlp_findings"

    output_columns.append(risk_scorer_input)

    pipeline.add_stage(RiskScorer(config, findings_column=risk_scorer_input))

    pipeline.add_stage(MonitorStage(config, description="Risk Scorer"))

    pipeline.add_stage(SerializeStage(config, include=output_columns))
    pipeline.add_stage(WriteToFileStage(config, filename=str(out_file), overwrite=True))

    pipeline.add_stage(MonitorStage(config, description="DLP Output"))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    main()
