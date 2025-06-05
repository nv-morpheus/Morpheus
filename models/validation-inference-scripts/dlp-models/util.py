
import json
import sys
import time
from ast import literal_eval

import pandas as pd
from nervaluate import Evaluator
from tqdm import tqdm


def run_benchmark_pipeline(dataset, pipeline, regex_processor,
                           gliNER_processor):
    regex_times = []
    hybrid_times = []
    gliner_times = []
    regex_results = []
    hybrid_results = []
    gliner_results = []

    for _, sample in tqdm(dataset.iterrows(), total=len(dataset)):
        # Regex processing with timing
        start_time = time.time()
        regex_findings = regex_processor.process(sample['source_text'])
        regex_time = time.time() - start_time
        regex_times.append(regex_time)

        regex_entities = [
            {
                'label': r['label'],
                'start': r['span'][0],
                'end': r['span'][1],
                'score': r['confidence']
            }
            for r in regex_findings
        ]
        regex_results.append(json.dumps(regex_entities))

        # Hybrid model with timing
        start_time = time.time()
        hybrid_findings = pipeline.inference(
            sample['source_text'],
            failback=False
        )
        hybrid_time = time.time() - start_time
        hybrid_times.append(hybrid_time)
        hybrid_results.append(json.dumps(
            gliNER_processor.filter_entities(hybrid_findings)
        ))

        # GliNER model with timing
        start_time = time.time()
        gliner_findings = gliNER_processor.gliner_predict(
            sample['source_text'])
        gliner_time = time.time() - start_time
        gliner_times.append(gliner_time)

        gliner_results.append(json.dumps(
            gliNER_processor.filter_entities(gliner_findings)
        ))

    # dataset size for throuput inference per second
    dataset_size = sum(sys.getsizeof(txt)
                       for txt in dataset['source_text'].tolist())/1024/1024  # MB
    dataset_tokens = sum(len(txt.split())
                         for txt in dataset['source_text'].tolist())

    # Log timing statistics
    timing_metrics = {
        'regex': {
            'total_time': sum(regex_times),
            'throughput': dataset_size / sum(regex_times),
            'avg_latency': sum(regex_times)/len(regex_times),
            'tokens_per_second': dataset_tokens / sum(regex_times),
        },
        'hybrid': {
            'total_time': sum(hybrid_times),
            'throughput': dataset_size / sum(hybrid_times),
            'avg_latency': sum(hybrid_times)/len(hybrid_times),
            'tokens_per_second': dataset_tokens / sum(hybrid_times),
        },
        'gliner': {
            'total_time': sum(gliner_times),
            'throughput': dataset_size / sum(gliner_times),
            'avg_latency': sum(gliner_times)/len(gliner_times),
            'tokens_per_second': dataset_tokens / sum(gliner_times),
        }
    }

    return {
        'regex_results': regex_results,
        'hybrid_results': hybrid_results,
        'gliner_results': gliner_results,
        'dataset_size': dataset_size,
        'dataset_tokens': dataset_tokens,
        'timing_metrics': timing_metrics
    }


def evaluate_results(true_samples: pd.DataFrame, results, model="gretel", dataset=None,
                     mappings_file=None):
    """Evaluate model results against ground truth."""

    all_results = {}

    true = [
        row
        for row in true_samples[true_samples['source'] == dataset]['privacy_mask'].tolist()
    ]

    # Get list of all possible labels
    labels = set()
    for row in true:
        for masks in row:
            labels.add(masks['label'])
    list_of_entities = list(labels)

    pred = [literal_eval(results[i]) for i in range(len(results))]
    true = true[:len(pred)]
    evaluator = Evaluator(true=true, pred=pred, tags=list_of_entities)
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    results_df = pd.DataFrame(results)

    missed = results_df.loc["missed", "ent_type"]
    spurious = results_df.loc["spurious", "ent_type"]
    possible = results_df.loc["possible", "ent_type"]

    all_results[model] = results_df.loc[["f1"]].T.to_dict()["f1"]
    all_results[model]["missed"] = missed / possible
    all_results[model]["spurious"] = spurious / possible

    all_results_df = pd.DataFrame(all_results).T

    return (all_results_df, results_per_tag,
            result_indices, result_indices_by_tag)


def format_findings(results: dict[str, list]) -> str:
    """Format DLP findings as a readable report"""
    output = []
    output.append("===== DLP Analysis Report =====")
    output.append(
        f"Risk Level: {results['risk_assessment']['risk_level']} ({results['risk_assessment']['risk_score']}/100)")
    output.append(f"Total Findings: {results['total_findings']}")

    output.append("\nSensitive Data Types Found:")
    for data_type in results['risk_assessment']['data_types_found']:
        output.append(f"  - {data_type}")

    output.append("\nSeverity Distribution:")
    dist = results['risk_assessment']['severity_distribution']
    output.append(
        f"  High: {dist['high']}, Medium: {dist['medium']}, Low: {dist['low']}")

    if results['findings']:
        output.append("\nDetailed Findings:")
        # Limit to first 10 findings
        for i, finding in enumerate(results['findings'][:10], 1):
            output.append(f"\n  Finding {i}:")
            output.append(f"    Type: {finding['label']}")
            output.append(f"    Confidence: {finding['score']:.2f}")
            if finding['text']:
                # Truncate and sanitize match text if needed
                match_text = finding['text']
                if len(match_text) > 40:
                    match_text = match_text[:37] + "..."
                output.append(f"    Match: {match_text}")

        if len(results['findings']) > 10:
            output.append(
                f"\n  ... and {len(results['findings']) - 10} more findings")

    return "\n".join(output)
