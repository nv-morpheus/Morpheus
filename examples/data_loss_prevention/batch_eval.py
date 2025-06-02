import argparse
import json
import logging
import os
import sys
import time
from ast import literal_eval

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from nervaluate import Evaluator
from regex_processor import GliNERProcessor, RegexProcessor
from tqdm import tqdm

# Setup logger
logger = logging.getLogger(__name__)

def setup_logger(log_level):
    """Set up logger with specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

def remap_entities(entries, dataset, mappings_file):
    """Remap entity labels based on mappings file."""
    with open(mappings_file, 'r') as fid:
        mappings_content = fid.read()
        mappings = yaml.safe_load(mappings_content)

    for i in range(len(entries)):
        old_entry, new_entry = entries[i], []
        for mask in old_entry:
            mapped_label = mappings[dataset][mask['label']]
            if mapped_label == '_DROP':
                continue
            elif mapped_label is None:
                new_entry.append(mask)
            else:
                mask['label'] = mapped_label
                new_entry.append(mask)
        entries[i] = new_entry


def run_benchmark(dataset, eval_dataset_file, output_dir, model_name, regex_file, labels):
    """Run benchmark tests on regex, gliner and hybrid models."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the evaluation dataset
    if eval_dataset_file.endswith('.pkl'):
        samples = pd.read_pickle(eval_dataset_file)
    else:
        samples = pd.read_csv(eval_dataset_file)
        
    eval_samples = samples[samples['source'] == dataset]
    
    # Load regex patterns
    with open(regex_file, 'r') as fid:
        patterns = json.load(fid)
    patterns = {k: v for k, v in patterns.items() if k in labels}
    
    # Initialize processors
    regex_processor = RegexProcessor(patterns=patterns)
    gliNER_processor = GliNERProcessor(
        labels=labels, 
        confidence_threshold=0.3,
        context_window=300, 
        model_name=model_name
    )
    
    # Pre-allocate result lists
    hybrid_results = []
    regex_results = []
    gliNER_results = []
    
    # Initialize timing lists
    regex_times = []
    hybrid_times = []
    gliner_times = []
    
    # Process each sample
    for _, sample in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
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
        hybrid_findings = gliNER_processor.process(
            sample['source_text'],
            regex_findings, 
            failback=False
        )
        
        hybrid_time = time.time() - start_time
        hybrid_times.append(hybrid_time + regex_time)
        hybrid_results.append(json.dumps(
            gliNER_processor.filter_entities(hybrid_findings)
        ))
        
        # GliNER model with timing
        start_time = time.time()
        gliner_findings = gliNER_processor.gliner_predict(
            sample['source_text'])
        gliner_time = time.time() - start_time
        gliner_times.append(gliner_time)
        
        gliNER_results.append(json.dumps(
            gliNER_processor.filter_entities(gliner_findings)
        ))
    
    # dataset size for throuput inference per second
    dataset_size = sum(sys.getsizeof(txt) for txt in eval_samples['source_text'].tolist())/1024/1024  # MB
    dataset_tokens = sum(len(txt.split()) for txt in eval_samples['source_text'].tolist())
    logger.info(f"Dataset size: {dataset_size} MB")
    
    # Log timing statistics
    timing_metrics = {
        'regex_total_time': sum(regex_times),
        'hybrid_total_time': sum(hybrid_times),
        'gliner_total_time': sum(gliner_times),
        'hybrid_plus_regex_total_time': sum(hybrid_times + regex_times),
        
        'regex_throughput': dataset_size / sum(regex_times),
        'hybrid_throughput': dataset_size / sum(hybrid_times),
        'gliner_throughput': dataset_size / sum(gliner_times),
        'hybrid_throughput_plus_regex': dataset_size / sum(hybrid_times + regex_times),
        
        'regex_avg_latency': sum(regex_times)/len(regex_times),
        'hybrid_avg_latency': sum(hybrid_times)/len(hybrid_times),
        'gliner_avg_latency': sum(gliner_times)/len(gliner_times),
        'hybrid_plus_regex_avg_latency': sum(hybrid_times + regex_times)/len(hybrid_times + regex_times),

        'tokens_per_second_plus_regex': dataset_tokens / sum(hybrid_times + regex_times),
        'tokens_per_second_regex': dataset_tokens / sum(regex_times),
        'tokens_per_second_hybrid': dataset_tokens / sum(hybrid_times),
        'tokens_per_second_gliner': dataset_tokens / sum(gliner_times),
    }
    
    logger.info(f"Regex total time: {timing_metrics['regex_total_time']:.4f}s")
    logger.info(f"Hybrid total time: {timing_metrics['hybrid_total_time']:.4f}s")
    logger.info(f"GliNER total time: {timing_metrics['gliner_total_time']:.4f}s")
    
    # throughput inference per second
    logger.info(f"Throughput: {timing_metrics['hybrid_throughput']:.4f} MB/s")
    logger.info(f"Throughput regex: {timing_metrics['regex_throughput']:.4f} MB/s")
    logger.info(f"Throughput gliner: {timing_metrics['gliner_throughput']:.4f} MB/s")
    
    # latency
    logger.info(f"Regex average latency: {timing_metrics['regex_avg_latency']:.4f}s")
    logger.info(f"Hybrid average latency: {timing_metrics['hybrid_avg_latency']:.4f}s")
    logger.info(f"GliNER average latency: {timing_metrics['gliner_avg_latency']:.4f}s")
    
    # Save results to files
    model_name_base = model_name.split('/')[-1]
    result_files = {}
    
    hybrid_file = f'{output_dir}/hybrid_results_{dataset}_{model_name_base}_5k.txt'
    with open(hybrid_file, 'w') as output_fid:
        output_fid.write('\n'.join(hybrid_results))
    result_files['hybrid'] = hybrid_file
    
    regex_file = f'{output_dir}/regex_results_{dataset}_5k.txt'
    with open(regex_file, 'w') as output_fid:
        output_fid.write('\n'.join(regex_results))
    result_files['regex'] = regex_file
    
    gliner_file = f'{output_dir}/gliNER_results_{dataset}_{model_name_base}_5k.txt'
    with open(gliner_file, 'w') as output_fid:
        output_fid.write('\n'.join(gliNER_results))
    result_files['gliner'] = gliner_file
    
    # save timing metrics to file
    with open(f'{output_dir}/timing_metrics_{dataset}_{model_name_base}_5k.json', 'w') as output_fid:
        json.dump(timing_metrics, output_fid)
    
    # Create and save performance visualization
    visualize_performance(
        timing_metrics=timing_metrics,
        output_path=f'{output_dir}/performance_{dataset}_{model_name_base}_5k.png',
        dataset_name=dataset
    )
    
    return result_files


def evaluate_results(true_samples, output_filename, model="gretel", dataset=None, 
                    mappings_file=None):
    """Evaluate model results against ground truth."""
    true_samples = pd.read_csv(true_samples)
    all_results = {}
    
    true = [
        literal_eval(row) 
        for row in true_samples[true_samples['source'] == dataset]['privacy_mask'].tolist()
    ]
    
    if dataset == 'ai4privacy' and mappings_file:
        remap_entities(entries=true, dataset=dataset, mappings_file=mappings_file)
    
    # Get list of all possible labels
    labels = set()
    for row in true:
        for masks in row:
            labels.add(masks['label'])
    list_of_entities = list(labels)

    # Load predicted masks
    with open(output_filename, 'r') as fid:
        results = fid.readlines()
    pred = [literal_eval(results[i]) for i in range(len(results))]
    true = true[:len(pred)]
    
    evaluator = Evaluator(true=true, pred=pred, tags=list_of_entities)
    results, results_per_tag, _, _ = evaluator.evaluate()
    results_df = pd.DataFrame(results)

    missed = results_df.loc["missed", "ent_type"]
    spurious = results_df.loc["spurious", "ent_type"]
    possible = results_df.loc["possible", "ent_type"]

    all_results[model] = results_df.loc[["f1"]].T.to_dict()["f1"]
    all_results[model]["missed"] = missed / possible
    all_results[model]["spurious"] = spurious / possible

    all_results_df = pd.DataFrame(all_results).T
    
    return all_results_df, results_df


def visualize_benchmark_results(results_df, output_path=None, dataset_name=None):
    """Visualize benchmark results with bar charts."""
    plt.style.use('ggplot')
    
    # Prepare data
    df = results_df.reset_index()
    df = df.rename(columns={'index': 'model'})
    
    # Set up metrics to visualize
    metrics = ['ent_type', 'partial', 'strict', 'exact']
    error_metrics = ['missed', 'spurious']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # Bar chart for main metrics
    ax1 = plt.subplot(gs[0, 0])
    df_melt = pd.melt(
        df, 
        id_vars=['model'], 
        value_vars=metrics,
        var_name='Metric', 
        value_name='Score'
    )
    
    sns.barplot(x='Metric', y='Score', hue='model', data=df_melt, ax=ax1)
    title = 'Performance Metrics by Model'
    if dataset_name:
        title += f' - {dataset_name}'
    ax1.set_title(title, fontsize=14)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel('Metric', fontsize=12)
    ax1.set_ylabel('Score (higher is better)', fontsize=12)
    ax1.legend(title='Model')
    
    # Bar chart for error metrics
    ax2 = plt.subplot(gs[0, 1])
    df_melt_err = pd.melt(
        df,
        id_vars=['model'], 
        value_vars=error_metrics,
        var_name='Error Metric', 
        value_name='Rate'
    )
    
    sns.barplot(x='Error Metric', y='Rate', hue='model', data=df_melt_err, ax=ax2)
    ax2.set_title('Error Metrics by Model', fontsize=14)
    ax2.set_ylim(0, df_melt_err['Rate'].max() * 1.2)
    ax2.set_xlabel('Error Metric', fontsize=12)
    ax2.set_ylabel('Rate (lower is better)', fontsize=12)
    ax2.legend(title='Model')
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    return fig

def visualize_performance(timing_metrics, output_path=None, dataset_name=None):
    """Visualize latency and throughput metrics."""
    plt.style.use('ggplot')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Prepare data for plotting
    models = ['regex', 'hybrid', 'gliner', 'hybrid+regex']
    
    # Plot latency
    latencies = [
        timing_metrics['regex_avg_latency'],
        timing_metrics['hybrid_avg_latency'],
        timing_metrics['gliner_avg_latency'],
        timing_metrics['hybrid_plus_regex_avg_latency']
    ]
    ax1.bar(models, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('Average Latency by Model', fontsize=14)
    ax1.set_ylabel('Latency (seconds)', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    
    # Add value labels on bars
    # for i, v in enumerate(latencies):
    #     ax1.text(i, v + 0.01, f"{v:.4f}s", ha='center', fontsize=10)
    
    # Plot throughput
    throughputs = [
        timing_metrics['regex_throughput'],
        timing_metrics['hybrid_throughput'],
        timing_metrics['gliner_throughput'],
        timing_metrics['hybrid_throughput_plus_regex']
    ]
    ax2.bar(models, throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_title('Throughput by Model', fontsize=14)
    ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(throughputs):
        ax2.text(i, v + 0.1, f"{v:.4f} MB/s", ha='center', fontsize=10)
    
    # Plot tokens per second
    tokens_per_second = [
        timing_metrics['tokens_per_second_regex'],
        timing_metrics['tokens_per_second_hybrid'],
        timing_metrics['tokens_per_second_gliner'],
        timing_metrics['tokens_per_second_plus_regex']
    ]
    ax3.bar(models, tokens_per_second, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_title('Tokens per second by Model', fontsize=14)
    ax3.set_ylabel('Tokens per second', fontsize=12)
    ax3.set_xlabel('Model', fontsize=12)
    
    # Add value labels on bars
    # for i, v in enumerate(tokens_per_second):
    #     ax3.text(i, v + 0.1, f"{v:.4f} tokens/s", ha='center', fontsize=10)
    
    
    title = 'Performance Metrics'
    if dataset_name:
        title += f' - {dataset_name}'
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance visualization saved to {output_path}")
    
    return fig

def run_full_evaluation(args):
    """Run the full benchmark and evaluation pipeline."""
    # Setup logger
    setup_logger(args.log_level)
    
    # Define standard labels
    labels = [
        "medical_record_number", "date_of_birth", "ssn", "date", "first_name",
        "email", "last_name", "customer_id", "employee_id", "name", 
        "street_address", "phone_number", "ipv4", "credit_card_number",
        "license_plate", "address", "user_name", "device_identifier",
        "bank_routing_number", "date_time", "company_name", "unique_identifier",
        "biometric_identifier", "account_number", "city", 
        "certificate_license_number", "time", "postcode", "vehicle_identifier",
        "coordinate", "country", "api_key", "ipv6", "password",
        "health_plan_beneficiary_number", "national_id", "tax_id", "url", 
        "state", "swift_bic", "cvv", "pin"
    ]
    
    # Run benchmarks for each model
    if args.model_name:
        model_names = [args.model_name]
    elif args.model_name == 'all':
        model_names = [
            'gretelai/gretel-gliner-bi-small-v1.0',
            'gretelai/gretel-gliner-bi-base-v1.0',
            'gretelai/gretel-gliner-bi-large-v1.0'
        ]
    else:
        raise ValueError("Invalid model name")
    
    if args.eval_dataset_large:
        eval_dataset_file = args.eval_dataset_large
    else:
        eval_dataset_file = args.eval_dataset

    
    results_dir = args.output_dir  # or "results"
    os.makedirs(results_dir, exist_ok=True)
    
    for model_name in model_names:
        logger.info(f"Running benchmarks for model: {model_name}")
        result_files = run_benchmark(
            dataset=args.dataset,
            eval_dataset_file=eval_dataset_file,
            output_dir=args.output_dir,
            model_name=model_name,
            regex_file=args.regex_file,
            labels=labels
        )
        
        # Evaluate results
        logger.info(f"Evaluating results for model: {model_name}")
        model_name_base = model_name.split('/')[-1]
        all_results = pd.DataFrame()
        
        for model_type, output_file in result_files.items():
            model_id = f"{model_type}_{model_name_base}" if model_type != "regex" else "regex"
            logger.info(f"Evaluating {model_id} for {args.dataset} dataset")
            
            results, gresults = evaluate_results(
                true_samples=args.eval_dataset,
                output_filename=output_file,
                model=model_id, 
                dataset=args.dataset,
                mappings_file=args.mappings_file
            )
            
            logger.info(gresults)
            logger.info(f"---------- {model_id} ----------")
            all_results = pd.concat([all_results, results])
        
        # Save results
        logger.info(all_results)
        output_csv = os.path.join(
            results_dir, f"models_benchmark_{args.dataset}_5k_{model_name_base}.csv"
        )
        all_results.reset_index().to_csv(output_csv, index=False)
        
        # Create visualization
        output_img = os.path.join(
            results_dir, 
            f"benchmark_visualization_{args.dataset}_5k_{model_name_base}.png"
        )
        visualize_benchmark_results(
            all_results, 
            output_path=output_img,
            dataset_name=args.dataset
        )
        
        logger.info(f"Completed evaluation for model: {model_name}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark and evaluation for regex and GliNER models"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gretel",
        choices=["gretel", "ai4privacy"],
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--eval_dataset", 
        type=str, 
        default="data/eval_dataset_5k.csv",
        help="Path to evaluation dataset CSV file"
    )
    
    parser.add_argument(
        "--eval_dataset_large", 
        type=str, 
        default=None,
        help="Path to evaluation dataset CSV file"
    )
    
    parser.add_argument(
        "--regex_file", 
        type=str, 
        default="patterns_v5_append.json",
        help="Path to regex patterns JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="GliNER model name to use (if not specified, all models will be run",
        default="gretelai/gretel-gliner-bi-small-v1.0"
    )
    parser.add_argument(
        "--mappings_file",
        type=str,
        default="data/mappings.yaml",
        help="Path to mappings YAML file for entity remapping"
    )
    parser.add_argument(
        "--detailed_viz", 
        action="store_true",
        help="Generate detailed visualizations for each model"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to save log file (if not specified, logs will only be printed)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.log_file:
        # Add file handler if log file is specified
        log_handler = logging.FileHandler(args.log_file)
        log_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
    
    run_full_evaluation(args)
