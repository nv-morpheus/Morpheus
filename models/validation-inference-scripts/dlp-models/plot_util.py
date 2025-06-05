import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

    sns.barplot(x='Error Metric', y='Rate',
                hue='model', data=df_melt_err, ax=ax2)
    ax2.set_title('Error Metrics by Model', fontsize=14)
    ax2.set_ylim(0, df_melt_err['Rate'].max() * 1.2)
    ax2.set_xlabel('Error Metric', fontsize=12)
    ax2.set_ylabel('Rate (lower is better)', fontsize=12)
    ax2.legend(title='Model')

    # # Radar chart
    # ax3 = plt.subplot(gs[1, :], polar=True)

    # # Number of metrics
    # N = len(metrics)

    # # What will be the angle of each axis in the plot
    # angles = [n / float(N) * 2 * np.pi for n in range(N)]
    # angles += angles[:1]  # Close the loop

    # # Draw one axis per variable and add labels
    # plt.xticks(angles[:-1], metrics, fontsize=12)

    # # Draw ylabels
    # ax3.set_rlabel_position(0)
    # plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], fontsize=10)
    # plt.ylim(0, 1)

    # # Plot each model
    # for i, model in enumerate(df['model']):
    #     values = df.loc[i, metrics].values.flatten().tolist()
    #     values += values[:1]  # Close the loop

    #     # Plot values
    #     ax3.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    #     ax3.fill(angles, values, alpha=0.1)

    # ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # ax3.set_title('Radar Chart of Model Performance', fontsize=14)

    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")

    return fig

def make_metrics_per_entity(results_per_tag, metric="partial"):
    df = pd.DataFrame()
    metrics_all = []
    for entity in results_per_tag:
        recall, precision, f1 = (results_per_tag[entity][metric]['recall'],
                                 results_per_tag[entity][metric]['precision'],
                                 results_per_tag[entity][metric]['f1'])
        metrics_all.append(
            {'entity': entity, 'recall': recall, 'precision': precision, 'f1': f1})
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
            plt.barh(df_sorted['entity'], df_sorted[metric], left=np.sum([df_sorted[m] for m in metrics[:i]], axis=0),
                     color=colors[i], alpha=0.7, label=metric)

        plt.xlabel('Score Value')
        plt.ylabel('Entity')
        plt.title('Recall, Precision, and F1 Scores by Entity')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    # Heatmap
    plt.figure(figsize=(10, 16))
    heatmap_data = df_sorted[['entity', 'recall',
                              'precision', 'f1']].set_index('entity')
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu',
                fmt='.3f', cbar_kws={'label': 'Score'})
    plt.title('Recall, Precision, and F1 Scores by Entity')
    plt.tight_layout()
    plt.show()


def visualize_throughput_latency(timing_metrics, output_path=None, dataset_name=None):
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
    ax1.bar(models, latencies, color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
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
    ax2.bar(models, throughputs, color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
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
    ax3.bar(models, tokens_per_second, color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
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

    return fig


def plot_latency_speedup(latency_df, speedup_model_factor="gliner", columns=["total_time", "throughput"]):

    # Create DataFrame from latency data
    # latency_df = pd.DataFrame(latency)

    # Calculate speedup factors relative to GliNER model
    hybrid_total_time = latency_df[latency_df['model']
                                   == speedup_model_factor][columns[0]].iloc[0]
    hybrid_avg_time = latency_df[latency_df['model']
                                 == speedup_model_factor][columns[1]].iloc[0]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot average latency
    ax1.bar(latency_df['model'], latency_df[columns[0]],
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Average Latency by Model', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Latency (seconds)', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)

    for i, v in enumerate(latency_df[columns[0]]):
        speedup = hybrid_total_time / v
        if latency_df.iloc[i]['model'] == speedup_model_factor:
            ax1.text(i, v + v*0.05, f'{v:.4f}s',
                     ha='center', va='top', fontweight='bold')
        else:
            ax1.text(i, v + v*0.05, f'{v:.4f}s\n({speedup:.1f}x faster)',
                     ha='center', va='top', fontweight='bold')

    # Plot throughput
    ax2.bar(latency_df['model'], latency_df[columns[1]],
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Throughput by Model', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    for i, v in enumerate(latency_df[columns[1]]):
        speedup = v / hybrid_avg_time
        if latency_df.iloc[i]['model'] == speedup_model_factor:
            ax2.text(i, v + v*0.05, f'{v:.4f} MB/s',
                     ha='center', va='top', fontweight='bold')
        else:
            ax2.text(i, v + v*0.05, f'{v:.4f} MB/s\n({speedup:.1f}x faster)',
                     ha='center', va='top', fontweight='bold')

    plt.tight_layout()
    plt.show()
