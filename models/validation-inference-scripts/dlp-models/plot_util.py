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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec


def visualize_benchmark_results(results_df: pd.DataFrame,
                                output_path: str | None = None,
                                dataset_name: str | None = None):
    """Visualize benchmark results with bar charts and radar plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        The results dataframe.
    output_path : str | None, optional
        The path to save the visualization, by default None
    dataset_name : str | None, optional
        The name of the dataset used for benchmarking, by default None

    Returns
    -------
    plt.Figure
        The figure object.
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
    df_melt = pd.melt(df, id_vars=['model'], value_vars=metrics, var_name='Metric', value_name='Score')

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
    df_melt_err = pd.melt(df, id_vars=['model'], value_vars=error_metrics, var_name='Error Metric', value_name='Rate')

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


def make_metrics_per_entity(results_per_tag: list, metric: str = "partial"):
    """_summary_

    Parameters
    ----------
    results_per_tag : list
        Tag results
    metric : str, optional

    Returns
    -------
    pd.DataFrame
        data frame results
    """

    df = pd.DataFrame()
    metrics_all = []
    for entity in results_per_tag:
        recall, precision, f1 = (results_per_tag[entity][metric]['recall'],
                                 results_per_tag[entity][metric]['precision'],
                                 results_per_tag[entity][metric]['f1'])
        metrics_all.append({'entity': entity, 'recall': recall, 'precision': precision, 'f1': f1})
    df = pd.DataFrame(metrics_all)
    return df


def plot_detail_metrics(df, only_heatmap=True):
    df_sorted = df.sort_values('f1', ascending=False)

    # Horizontal bar chart
    plt.figure(figsize=(6, 8))
    metrics = ['recall', 'precision', 'f1']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    if not only_heatmap:

        for i, metric in enumerate(metrics):
            plt.barh(df_sorted['entity'],
                     df_sorted[metric],
                     left=np.sum([df_sorted[m] for m in metrics[:i]], axis=0),
                     color=colors[i],
                     alpha=0.7,
                     label=metric)

        plt.xlabel('Score Value')
        plt.ylabel('Entity')
        plt.title('Recall, Precision, and F1 Scores by Entity')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    # Heatmap
    plt.figure(figsize=(10, 16))
    heatmap_data = df_sorted[['entity', 'recall', 'precision', 'f1']].set_index('entity')
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Score'})
    plt.title('Recall, Precision, and F1 Scores by Entity')
    plt.tight_layout()
    plt.show()


def visualize_throughput_latency(timing_metrics, output_path=None, dataset_name=None):
    """Visualize latency and throughput metrics."""
    plt.style.use('ggplot')

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, figsize=(14, 12))

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

    title = 'Performance Metrics'
    if dataset_name:
        title += f' - {dataset_name}'
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_latency(latency_df: pd.DataFrame, speedup_model_factor: str = "gliner", columns: list[str] = None, plt_type="latency"):
    """Plot latency speedup plot.

    Parameters
    ----------
    latency_df : pd.DataFrame
        The latency dataframe.
    speedup_model_factor : str, optional
        The model to use for speedup, by default "gliner"
    columns : list[str], optional
        The columns to use for the plot, by default None
    """
    if columns is None:
        columns = ["total_time", "throughput"]

    # Calculate speedup factors relative to GliNER model
    hybrid_total_time = latency_df[latency_df['model'] == speedup_model_factor][columns[0]].iloc[0]
    hybrid_avg_time = latency_df[latency_df['model'] == speedup_model_factor][columns[1]].iloc[0]

    
    if plt_type == "latency":
        _, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        # Plot average latency
        ax1.bar(latency_df['model'], latency_df[columns[0]], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Average Latency by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Latency (seconds)', fontsize=12)
        ax1.set_xlabel('Model', fontsize=12)

        for i, v in enumerate(latency_df[columns[0]]):
            speedup = hybrid_total_time / v
            if latency_df.iloc[i]['model'] == speedup_model_factor:
                ax1.text(i, v + v * 0.05, f'{v:.4f}s', ha='center', va='top', fontweight='bold')
            else:
                ax1.text(i, v + v * 0.05, f'{v:.4f}s\n({speedup:.1f}x faster)', ha='center', va='top', fontweight='bold')
    elif plt_type == "throughput":    
        # Plot throughput
        _, ax2 = plt.subplots(1, 1, figsize=(15, 6))
        ax2.bar(latency_df['model'], latency_df[columns[1]], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Throughput by Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.set_xlabel('Model', fontsize=12)
        for i, v in enumerate(latency_df[columns[1]]):
            speedup = v / hybrid_avg_time
            if latency_df.iloc[i]['model'] == speedup_model_factor:
                ax2.text(i, v + v * 0.05, f'{v:.4f} MB/s', ha='center', va='top', fontweight='bold')
            else:
                ax2.text(i,
                        v + v * 0.05,
                        f'{v:.4f} MB/s\n({speedup:.1f}x faster)',
                        ha='center',
                        va='top',
                        fontweight='bold')
    else: 
        return None 

    plt.tight_layout()
    plt.show()


def visualize_risk_assessment(risk_assessment: dict[str, list],
                              type_weights: dict[str, int] = None,
                              default_weight: int = 50):
    """Visualize risk assessment data.

    Parameters
    ----------
    risk_assessment : dict[str, list]
        The risk assessment data.
    type_weights : dict[str, int], optional
        The type weights, by default None
    default_weight : int, optional
        The default weight, by default 50
    """

    if type_weights is None:
        type_weights = {}

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Risk score gauge
    risk_score = risk_assessment["risk_score"]
    risk_level = risk_assessment["risk_level"]

    # Create a gauge chart for risk score
    gauge_colors = {
        'Critical': '#FF0000', 'High': '#FF6600', 'Medium': '#FFCC00', 'Low': '#CCFF00', 'Minimal': '#00FF00'
    }

    ax1.pie([risk_score, 100 - risk_score],
            colors=[gauge_colors.get(risk_level, '#CCCCCC'), '#F8F8F8'],
            startangle=90,
            counterclock=False,
            wedgeprops={
                'width': 0.3, 'edgecolor': 'w', 'linewidth': 3
            })
    ax1.add_patch(plt.Circle((0, 0), 0.35, color='white'))
    ax1.text(0, 0, f"{risk_score}\n{risk_level}", ha='center', va='center', fontsize=14)
    ax1.set_title("Risk Score", fontsize=18)
    ax1.axis('equal')

    # 2. Severity distribution bar chart
    severity = risk_assessment["severity_distribution"]
    labels = list(severity.keys())
    values = list(severity.values())
    colors = ['#FF0000', '#FFCC00', '#00CC00']

    bars = ax2.bar(labels, values, color=colors)
    ax2.set_title("Findings by Severity", fontsize=18)
    ax2.set_ylim(0, max(values) + 1 if values else 1)

    # Add counts above bars
    for bar_plot in bars:
        height = bar_plot.get_height()
        ax2.text(bar_plot.get_x() + bar_plot.get_width() / 2., height + 0.1, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Create a horizontal bar chart for data types found
    if risk_assessment["data_types_found"]:
        plt.figure(figsize=(10, len(risk_assessment["data_types_found"]) * 0.4 + 1))
        # Get weights for each data type
        data_types = risk_assessment["data_types_found"]
        weights = [type_weights.get(dt, default_weight) for dt in data_types]
        # Sort by weight
        data_types_sorted = [x for _, x in sorted(zip(weights, data_types), reverse=True)]
        weights_sorted = sorted(weights, reverse=True)

        # Create color map based on weights
        colors = ['#FF0000' if w >= 80 else '#FFCC00' if w >= 50 else '#00CC00' for w in weights_sorted]

        plt.barh(data_types_sorted, weights_sorted, color=colors)
        plt.xlabel('Sensitivity Weight')
        plt.title('Data Types Found by Sensitivity')
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.show()
