#!/usr/bin/env python3
"""
Plot mutation effectiveness by type and generation phase
Shows how improvement rates change over time in the GA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

try:
    from nesting import config as nesting_config
except Exception:  # pragma: no cover
    nesting_config = None


def _resolve_raw_mutation_csv():
    candidates = []
    if nesting_config and hasattr(nesting_config, 'AGGREGATE_DIR'):
        candidates.append(Path(nesting_config.AGGREGATE_DIR) / 'master_mutation_raw_data.csv')
    candidates.append(Path('nesting/experiments/aggregate/master_mutation_raw_data.csv'))
    candidates.append(Path('nesting/aggregate_stats/master_mutation_raw_data.csv'))
    for p in candidates:
        if p.exists() and p.stat().st_size > 50:
            return p
    return candidates[0]


def load_and_process_data(csv_file):
    """Load mutation data and calculate improvement rates by type and generation phase"""
    df = pd.read_csv(csv_file)
    if df.empty:
        return df, pd.DataFrame(columns=['mutation_type','generation_phase','improvements','total_mutations','mean_fitness_gain','std_fitness_gain','improvement_rate'])

    def get_generation_phase(gen):
        if gen <= 5:
            return "Early (1-5)"
        elif gen <= 10:
            return "Middle (6-10)"
        else:
            return "Late (10+)"

    df['generation_phase'] = df['generation'].apply(get_generation_phase)
    df['improved'] = df['fitness_gain'] > 0
    improvement_stats = df.groupby(['mutation_type', 'generation_phase']).agg({
        'improved': ['sum', 'count'],
        'fitness_gain': ['mean', 'std']
    }).round(4)
    improvement_stats.columns = ['improvements', 'total_mutations', 'mean_fitness_gain', 'std_fitness_gain']
    improvement_stats['improvement_rate'] = (improvement_stats['improvements'] / improvement_stats['total_mutations'] * 100).round(2)
    return df, improvement_stats


def plot_temporal_improvement_rates(improvement_stats, save_path=None):
    """Create temporal visualization of improvement rates"""
    if improvement_stats.empty:
        print('No temporal mutation data to plot.')
        return None, None, None
    plot_data = improvement_stats.reset_index()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    pivot_improvement = plot_data.pivot(index='mutation_type', columns='generation_phase', values='improvement_rate')
    phase_order = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]
    pivot_improvement = pivot_improvement.reindex(columns=phase_order)
    pivot_improvement.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Mutation Improvement Rate by Type and Generation Phase', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mutation Type')
    ax1.set_ylabel('Improvement Rate (%)')
    ax1.legend(title='Generation Phase')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    sns.heatmap(pivot_improvement, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax2, cbar_kws={'label': 'Improvement Rate (%)'})
    ax2.set_title('Improvement Rate Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Generation Phase')
    ax2.set_ylabel('Mutation Type')
    pivot_fitness = plot_data.pivot(index='mutation_type', columns='generation_phase', values='mean_fitness_gain').reindex(columns=phase_order)
    pivot_fitness.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Mean Fitness Gain by Type and Generation Phase', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mutation Type')
    ax3.set_ylabel('Mean Fitness Gain')
    ax3.legend(title='Generation Phase')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    pivot_count = plot_data.pivot(index='mutation_type', columns='generation_phase', values='total_mutations').reindex(columns=phase_order)
    pivot_count.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Number of Mutations by Type and Generation Phase', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mutation Type')
    ax4.set_ylabel('Number of Mutations')
    ax4.legend(title='Generation Phase')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    return pivot_improvement, pivot_fitness, pivot_count


def print_summary_stats(df, improvement_stats):
    if df.empty:
        print('No mutation records found – nothing to summarize.')
        return
    print("=== TEMPORAL MUTATION EFFECTIVENESS ANALYSIS ===\n")
    phase_stats = df.groupby('generation_phase').agg({'improved': ['sum', 'count'], 'fitness_gain': ['mean', 'std']}).round(4)
    phase_stats.columns = ['improvements', 'total_mutations', 'mean_fitness_gain', 'std_fitness_gain']
    phase_stats['improvement_rate'] = (phase_stats['improvements'] / phase_stats['total_mutations'] * 100).round(2)
    print('Overall stats by generation phase:')
    print(phase_stats)
    print()
    print('Improvement rates by mutation type and generation phase:')
    pivot_table = improvement_stats.reset_index().pivot(index='mutation_type', columns='generation_phase', values='improvement_rate') if not improvement_stats.empty else pd.DataFrame()
    phase_order = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]
    if not pivot_table.empty:
        pivot_table = pivot_table.reindex(columns=phase_order)
        print(pivot_table)
    else:
        print('No per-type temporal data.')
    print('\n=== KEY INSIGHTS ===')
    if not pivot_table.empty:
        for mutation_type in pivot_table.index:
            rates = pivot_table.loc[mutation_type].values
            if len(rates) >= 2 and not pd.isna(rates[0]) and not pd.isna(rates[-1]):
                change = rates[-1] - rates[0]
                if abs(change) > 1:
                    trend = 'improving' if change > 0 else 'declining'
                    print(f"• {mutation_type}: {trend} over time ({rates[0]:.1f}% → {rates[-1]:.1f}%)")
        overall_rates = phase_stats['improvement_rate'].values
        if len(overall_rates) >= 2:
            if overall_rates[0] > overall_rates[-1]:
                print(f"• Overall mutation effectiveness DECLINES over generations ({overall_rates[0]:.1f}% → {overall_rates[-1]:.1f}%)")
            else:
                print(f"• Overall mutation effectiveness IMPROVES over generations ({overall_rates[0]:.1f}% → {overall_rates[-1]:.1f}%)")


def main():
    csv_path = _resolve_raw_mutation_csv()
    print(f'Using mutation data file: {csv_path}')
    if not csv_path.exists() or csv_path.stat().st_size <= 50:
        print('Mutation data file missing or empty (only header). Run evolution to generate data.')
        sys.exit(0)
    try:
        df, improvement_stats = load_and_process_data(csv_path)
    except Exception as e:
        print(f'Failed to read/process mutation data: {e}')
        sys.exit(1)
    if df.empty:
        print('Mutation data frame is empty. Nothing to analyze.')
        sys.exit(0)
    print_summary_stats(df, improvement_stats)
    print('\nCreating temporal plots...')
    piv_impr, piv_fit, piv_cnt = plot_temporal_improvement_rates(improvement_stats, save_path='mutation_effectiveness_temporal.png')
    if improvement_stats is not None and not improvement_stats.empty:
        improvement_stats.to_csv('mutation_effectiveness_temporal_stats.csv')
        print('Detailed stats saved to: mutation_effectiveness_temporal_stats.csv')
    print('\nAnalysis complete!')


if __name__ == "__main__":
    main()
