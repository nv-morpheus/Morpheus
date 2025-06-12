
import json

import click
import pandas as pd
from dlp_pipeline import DLPPipeline


def inference(texts: list[str], dlp_pipeline: DLPPipeline):
    """_summary_

    Parameters
    ----------
    text : str
        The text to infer.

    Returns
    -------
    list[str]
        The inferred findings.
    """
    entities = []
    for entity in texts:
        entities.append(dlp_pipeline.infer(entity))
    return entities



@click.command()
@click.option('--patterns-file', default='data/patterns.json', help='Path to patterns JSON file')
@click.option('--dataset-file', default='data/evaluation_dataset.csv', help='Path to input dataset CSV file')
@click.option('--output-file', default='data/inference_results.csv', help='Path to output results file')
@click.option('--confidence-threshold', default=0.3, type=float, help='Confidence threshold for model predictions')
@click.option('--model-name', default='gretelai/gretel-gliner-bi-small-v1.0', help='Name of the GLiNER model to use')
@click.option('--context-window', default=300, type=int, help='Context window size for the model')
def main(patterns_file, dataset_file, output_file, confidence_threshold, model_name, context_window):
    """Main function to run the DLP pipeline."""

    patterns_file = "data/patterns.json"
    dataset_file = "data/evaluation_dataset.csv"

    with open(patterns_file, 'r') as f:
        patterns = json.load(f)


    dlp_pipeline = DLPPipeline(
        regex_patterns=patterns,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        context_window=context_window,
        config_file="data/config.json"
    )

    texts = pd.read_csv(dataset_file)['text'].tolist()
    entities = inference(texts, dlp_pipeline)

    results = pd.DataFrame(entities)
    results.to_csv(output_file, index=False)
