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

import json

import click
import pandas as pd
from dlp_pipeline import DLPPipeline


def inference(texts: list[str], dlp_pipeline: DLPPipeline) -> list[str]:
    """_summary_

    Parameters
    ----------
    texts : list[str]
        The text to infer.
    dlp_pipeline : DLPPipeline
        The DLP pipeline to use.

    Returns
    -------
    list[str]
        The inferred findings.
    """
    entities = []
    for entity in texts:
        entities.append(dlp_pipeline.process(entity))
    return entities


@click.command()
@click.option('--patterns-file', default='data/regex_patterns.json', help='Path to patterns JSON file')
@click.option('--dataset-file', default='data/evaluation_dataset.csv', help='Path to input dataset CSV file')
@click.option('--output-file', default='data/inference_results.csv', help='Path to output results file')
@click.option('--confidence-threshold', default=0.3, type=float, help='Confidence threshold for model predictions')
@click.option('--model-name', default='gretelai/gretel-gliner-bi-small-v1.0', help='Name of the GLiNER model to use')
@click.option('--context-window', default=300, type=int, help='Context window size for the model')
def main(*, patterns_file, dataset_file, output_file, confidence_threshold, model_name, context_window):
    """Main function to run the DLP pipeline."""

    with open(patterns_file, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    dlp_pipeline = DLPPipeline(regex_patterns=patterns,
                               confidence_threshold=confidence_threshold,
                               model_name=model_name,
                               context_window=context_window,
                               config_file="data/config.json")

    texts = pd.read_csv(dataset_file)['source_text'].tolist()
    entities = inference(texts, dlp_pipeline)

    results = pd.DataFrame(entities)
    results.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
