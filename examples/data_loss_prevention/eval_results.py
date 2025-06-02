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

# Load true data (text + masks)
import os
from ast import literal_eval

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from nervaluate import Evaluator


def remap_entities(entries, dataset):
    with open(MAPPINGS_FILENAME, 'r') as fid:
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

def evaluate_results(true_samples,  output_filename, model="gretel", dataset=None):
    
    true_samples = pd.read_csv(true_samples)
    all_results = {}
    true = [literal_eval(row) for row in true_samples[true_samples['source'] == dataset]['privacy_mask'].tolist()]
    if dataset == 'ai4privacy':
        remap_entities(entries=true, dataset=dataset)
    # Get list of all possible labels
    labels = set()
    for row in true:
        for masks in row:
            labels.add(masks['label'])
    list_of_entities = list(labels)

    # Load predicted masks
    with open(output_filename, 'r') as fid:
        results = fid.readlines()
    pred = [literal_eval(results[i]) for i in  range(len(results))] 
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
    #all_results_df.to_pickle(f"results_{model}.pickle")
    
    return all_results_df, results_df

def visualize_benchmark_results(results_df, output_path=None, dataset_name=None):
    """
    Visualize benchmark results with bar charts and radar plots.
    
    Args:
        results_df: DataFrame containing benchmark results
        output_path: Path to save visualization files
        dataset_name: Name of the dataset used for benchmarking
    """
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

def visualize_detailed_results(gresults, model_name, output_dir, dataset_name=None):
    """
    Visualize detailed evaluation results for a single model.
    
    Args:
        gresults: DataFrame with detailed evaluation results
        model_name: Name of the model being visualized
        output_dir: Directory to save output files
        dataset_name: Name of the dataset used for evaluation
    """
    plt.style.use('ggplot')
    
    # Extract metrics categories
    count_metrics = ['correct', 'incorrect', 'partial', 'missed', 'spurious', 'possible', 'actual']
    perf_metrics = ['precision', 'recall', 'f1']
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Count metrics
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    count_data = gresults.loc[count_metrics, 'ent_type'].astype(float)
    
    # Use log scale for better visibility
    ax1.bar(count_data.index, count_data.values)
    ax1.set_ylabel('Count (log scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title(f'{model_name} - Count Metrics{" - " + dataset_name if dataset_name else ""}', 
                 fontsize=14)
    
    # Add count labels on top of bars
    for i, v in enumerate(count_data.values):
        ax1.text(i, v * 1.1, f'{int(v)}', ha='center', fontsize=10)
    
    plt.tight_layout()
    count_file = os.path.join(output_dir, f'{model_name}_counts_{dataset_name or "all"}.png')
    plt.savefig(count_file, dpi=300, bbox_inches='tight')
    
    # Plot 2: Performance metrics
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    perf_data = gresults.loc[perf_metrics].T
    
    perf_data.plot(kind='bar', ax=ax2)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'{model_name} - Performance Metrics{" - " + dataset_name if dataset_name else ""}', 
                 fontsize=14)
    ax2.legend(title='Metric')
    
    # Add value labels on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    perf_file = os.path.join(output_dir, f'{model_name}_performance_{dataset_name or "all"}.png')
    plt.savefig(perf_file, dpi=300, bbox_inches='tight')
    
    # Plot 3: Entity-specific metrics if available
    if len(gresults.columns) > 1:
        # We have entity-specific results
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        
        # Get entity types (excluding the 'ent_type' column which is the average)
        entity_types = [col for col in gresults.columns if col != 'ent_type']
        
        # Get f1 scores for each entity type
        entity_f1 = gresults.loc['f1', entity_types]
        
        # Sort by F1 score for better visualization
        entity_f1 = entity_f1.sort_values(ascending=False)
        
        # Create bar chart
        ax3.bar(entity_f1.index, entity_f1.values)
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.set_title(f'{model_name} - F1 Score by Entity Type{" - " + dataset_name if dataset_name else ""}', 
                     fontsize=14)
        ax3.set_ylim(0, 1.1)
        
        # Rotate x-labels if we have many entity types
        if len(entity_types) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(entity_f1.values):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        entity_file = os.path.join(output_dir, f'{model_name}_entities_{dataset_name or "all"}.png')
        plt.savefig(entity_file, dpi=300, bbox_inches='tight')
        
        return [fig1, fig2, fig3]
    
    return [fig1, fig2]

if __name__ == "__main__":

    
    DATASETS = ['ai4privacy', 'gretel']
    MAPPINGS_FILENAME = 'data/mappings.yaml'
    ground_truth = "data/eval_dataset_5k.csv"

    dataset = DATASETS[0]
    model_list = ['gretelai/gretel-gliner-bi-small-v1.0',
              'gretelai/gretel-gliner-bi-base-v1.0',
              'gretelai/gretel-gliner-bi-large-v1.0']

    model_name = model_list[1].split('/')[-1] 
    #"gretelai/gretel-gliner-bi-small-v1.0"

    
    models_output = {
        f"gliner_{model_name}": f"output/gliNER_results_{dataset}_{model_name}_5k.txt",
        f"hybrid_{model_name}": f"output/hybrid_results_{dataset}_{model_name}_5k.txt",
        "regex": f"output/regex_results_{dataset}_5k.txt"
    }
    
    all_results = pd.DataFrame()
    for model, output_filename in models_output.items():
        print(f"Evaluating {model} for {dataset} dataset")
        results, gresults = evaluate_results(true_samples=ground_truth,
                                             output_filename=output_filename,
                                             model=model, dataset=dataset)
        # Create detailed visualizations for this model
        # model_figs = visualize_detailed_results(
        #     gresults, 
        #     model_name=model,
        #     output_dir="results/model_metrics",
        #     dataset_name=dataset
        # )
        
        print(gresults)
        print(f"-------------------------------- {model} --------------------------------")
        all_results = pd.concat([all_results, results])
        
    print(all_results)
    
    # print(all_results)
    output_csv = f"results/models_benchmark_{dataset}_v2_5k_{model_name}.csv"
    all_results.reset_index().to_csv(output_csv, index=False)
    
    # Create visualization
    output_img = f"results/benchmark_visualization_{dataset}_5k_{model_name}.png"
    fig = visualize_benchmark_results(
        all_results, 
        output_path=output_img,
        dataset_name=dataset
    )
    fig.savefig(output_img)
    
    #wandb.log({"benchmark_plot": fig})
    
    #wandb.finish()ß