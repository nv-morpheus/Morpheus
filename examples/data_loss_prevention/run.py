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
from stages.datasets_source import DatasetsSourceStage
from stages.dlp_input_processor import DLPInputProcessor
from stages.gliner_processor import GliNERProcessor
from stages.regex_processor import RegexProcessor
from stages.risk_scorer import RiskScorer

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.logger import configure_logging

logger = logging.getLogger(f"morpheus.{__name__}")
CUR_DIR = os.path.dirname(os.path.abspath(__file__))


@click.command()
@click.option("--log_level",
              default="INFO",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              show_default=True,
              help="Specify the logging level to use.")
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
@click.option('--num_samples',
              type=int,
              default=2000,
              show_default=True,
              help="Number of samples to use from each dataset, set to -1 for all samples.")
@click.option("--out_file",
              help="Output file",
              type=click.Path(dir_okay=False),
              default=".tmp/output/data_loss_prevention.jsonlines",
              required=True)
def main(log_level: int, regex_file: pathlib.Path, dataset: list[str], num_samples: int, out_file: pathlib.Path):
    configure_logging(log_level=log_level)

    if num_samples < 0:
        num_samples = None

    config = Config()
    config.mode = PipelineModes.NLP

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(DatasetsSourceStage(config, dataset_names=dataset, num_samples=num_samples))

    pipeline.add_stage(MonitorStage(config, description="source"))

    pipeline.add_stage(DLPInputProcessor(config))

    pipeline.add_stage(MonitorStage(config, description="dpl input processor"))

    regex_processor = RegexProcessor(config, patterns_file=regex_file)
    pipeline.add_stage(regex_processor)

    pipeline.add_stage(MonitorStage(config, description="regex processor"))

    pipeline.add_stage(GliNERProcessor(config, labels=list(regex_processor.patterns.keys())))

    pipeline.add_stage(MonitorStage(config, description="gliner processor"))

    pipeline.add_stage(RiskScorer(config))

    pipeline.add_stage(MonitorStage(config, description="risk scorer"))

    pipeline.add_stage(SerializeStage(config, include=["ID", "gliner_findings", "risk_score"]))
    pipeline.add_stage(WriteToFileStage(config, filename=out_file, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    main()
