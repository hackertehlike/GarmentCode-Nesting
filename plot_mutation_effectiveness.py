#!/usr/bin/env python3
"""
Plot mutation effectiveness by type using improvement rate metric.
improvement_rate = (mutations_improving_fitness / total_mutations) × 100
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

# Attempt to import config for dynamic path resolution
try:
    from nesting import config as nesting_config
except Exception:  # pragma: no cover
    nesting_config = None


def _resolve_raw_mutation_csv():
    """Return best existing master_mutation_raw_data.csv path (new layout preferred).
    Search order:
      1. config.AGGREGATE_DIR/master_mutation_raw_data.csv
      2. nesting/experiments/aggregate/master_mutation_raw_data.csv (explicit new path)
      3. nesting/aggregate_stats/master_mutation_raw_data.csv (legacy path)
    Returns first path that exists & has > header bytes; else returns preferred target even if empty.
    """
    candidates = []
    if nesting_config and hasattr(nesting_config, 'AGGREGATE_DIR'):
        candidates.append(Path(nesting_config.AGGREGATE_DIR) / 'master_mutation_raw_data.csv')
    candidates.append(Path('nesting/experiments/aggregate/master_mutation_raw_data.csv'))
    candidates.append(Path('nesting/aggregate_stats/master_mutation_raw_data.csv'))
    for p in candidates:
        if p.exists() and p.stat().st_size > 50:  # > header
            return p
    return candidates[0]


def calculate_improvement_rates(df):
    """Calculate improvement rate for each mutation type."""
    if df.empty:
        return pd.DataFrame(columns=[
            'mutation_type','total_mutations','improving_mutations',
            'improvement_rate','mean_fitness_gain','std_fitness_gain'
        ])
    grouped = df.groupby('mutation_type')
    results = []
    for mutation_type, group in grouped:
        total_mutations = len(group)
        improving_mutations = (group['fitness_gain'] > 0).sum()
        improvement_rate = (improving_mutations / total_mutations) * 100 if total_mutations else 0.0
        results.append({
            'mutation_type': mutation_type,
            'total_mutations': total_mutations,
            'improving_mutations': improving_mutations,
            'improvement_rate': improvement_rate,
            'mean_fitness_gain': group['fitness_gain'].mean(),
            'std_fitness_gain': group['fitness_gain'].std()
        })
    return pd.DataFrame(results)


def plot_improvement_rates(stats_df, output_file='mutation_improvement_rates.png'):
    """Create visualizations for mutation effectiveness."""
    if stats_df.empty:
        print('No mutation data to plot.')
        return None
    plt.style.use('default')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mutation Effectiveness Analysis', fontsize=16, fontweight='bold')
    stats_df_sorted = stats_df.sort_values('improvement_rate', ascending=True)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(stats_df_sorted['mutation_type'], stats_df_sorted['improvement_rate'],
                     color=plt.cm.viridis(np.linspace(0, 1, len(stats_df_sorted))))
    ax1.set_xlabel('Improvement Rate (%)')
    ax1.set_title('Improvement Rate by Mutation Type')
    ax1.grid(axis='x', alpha=0.3)
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    ax2 = axes[0, 1]
    bars2 = ax2.barh(stats_df_sorted['mutation_type'], stats_df_sorted['mean_fitness_gain'],
                     color=plt.cm.RdYlBu(np.linspace(0, 1, len(stats_df_sorted))))
    ax2.set_xlabel('Mean Fitness Gain')
    ax2.set_title('Average Fitness Impact by Mutation Type')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend()
    for bar in bars2:
        width = bar.get_width()
        label_x = width + 0.001 if width >= 0 else width - 0.001
        ha = 'left' if width >= 0 else 'right'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha=ha, va='center', fontweight='bold')
    ax3 = axes[1, 0]
    bars3 = ax3.barh(stats_df_sorted['mutation_type'], stats_df_sorted['total_mutations'],
                     color=plt.cm.plasma(np.linspace(0, 1, len(stats_df_sorted))))
    ax3.set_xlabel('Total Mutations')
    ax3.set_title('Sample Size by Mutation Type')
    ax3.grid(axis='x', alpha=0.3)
    max_total = stats_df_sorted['total_mutations'].max() if not stats_df_sorted.empty else 0
    for bar in bars3:
        width = bar.get_width()
        ax3.text(width + (max_total * 0.01 if max_total else 0.1),
                 bar.get_y() + bar.get_height()/2,
                 f'{int(width)}', ha='left', va='center', fontweight='bold')
    ax4 = axes[1, 1]
    scatter = ax4.scatter(stats_df['improvement_rate'], stats_df['mean_fitness_gain'],
                          s=np.maximum(stats_df['total_mutations'], 1)/20,
                          c=range(len(stats_df)), cmap='tab10', alpha=0.7, edgecolors='black')
    for _, row in stats_df.iterrows():
        ax4.annotate(row['mutation_type'], (row['improvement_rate'], row['mean_fitness_gain']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Improvement Rate (%)')
    ax4.set_ylabel('Mean Fitness Gain')
    ax4.set_title('Improvement Rate vs Average Impact\n(Bubble size = sample size)')
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    return fig


def print_summary_stats(stats_df, raw_df):
    """Print summary statistics."""
    if raw_df.empty:
        print("No mutation records found – nothing to summarize.")
        return
    print("\n" + "="*60)
    print("MUTATION EFFECTIVENESS SUMMARY")
    print("="*60)
    total = len(raw_df)
    improving_total = (raw_df['fitness_gain'] > 0).sum()
    overall_rate = (improving_total / total * 100) if total else 0.0
    print(f"\nTotal mutations analyzed: {total:,}")
    print(f"Overall improvement rate: {overall_rate:.1f}%")
    print(f"Overall mean fitness gain: {raw_df['fitness_gain'].mean():.4f}")
    print("\nBy Mutation Type:")
    print("-" * 80)
    print(f"{'Type':<15} {'Count':<8} {'Improve%':<9} {'Mean Gain':<11} {'Std Gain':<10}")
    print("-" * 80)
    for _, row in stats_df.sort_values('improvement_rate', ascending=False).iterrows():
        print(f"{row['mutation_type']:<15} {row['total_mutations']:<8} "
              f"{row['improvement_rate']:<8.1f}% {row['mean_fitness_gain']:<10.4f} "
              f"{row['std_fitness_gain']:<10.4f}")
    if not stats_df.empty:
        best_type = stats_df.loc[stats_df['improvement_rate'].idxmax()]
        worst_type = stats_df.loc[stats_df['improvement_rate'].idxmin()]
        print("\nKey Insights:")
        print(f"• Best improvement rate: {best_type['mutation_type']} ({best_type['improvement_rate']:.1f}%)")
        print(f"• Worst improvement rate: {worst_type['mutation_type']} ({worst_type['improvement_rate']:.1f}%)")
        positive_gain = stats_df[stats_df['mean_fitness_gain'] > 0]
        if len(positive_gain) > 0:
            print(f"• Mutation types with positive average gain: {', '.join(positive_gain['mutation_type'].tolist())}")
        else:
            print("• No mutation types have positive average fitness gain")


if __name__ == "__main__":
    print("Resolving mutation data path...")
    csv_path = _resolve_raw_mutation_csv()
    print(f"Using mutation data file: {csv_path}")
    if not csv_path.exists() or csv_path.stat().st_size <= 50:
        print("Mutation data file missing or empty (only header). Run evolution to generate data.")
        sys.exit(0)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read mutation data: {e}")
        sys.exit(1)
    if df.empty:
        print("Mutation data frame is empty. Nothing to analyze.")
        sys.exit(0)
    print("Calculating improvement rates...")
    stats_df = calculate_improvement_rates(df)
    print_summary_stats(stats_df, df)
    print("\nGenerating plots...")
    plot_improvement_rates(stats_df)
    print("\nAnalysis complete!")
