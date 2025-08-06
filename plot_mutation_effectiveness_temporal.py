#!/usr/bin/env python3
"""
Plot mutation effectiveness by type and generation phase
Shows how improvement rates change over time in the GA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process_data(csv_file):
    """Load mutation data and calculate improvement rates by type and generation phase"""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Create generation phases
    def get_generation_phase(gen):
        if gen <= 5:
            return "Early (1-5)"
        elif gen <= 10:
            return "Middle (6-10)"
        else:
            return "Late (10+)"
    
    df['generation_phase'] = df['generation'].apply(get_generation_phase)
    
    # Calculate improvements (positive fitness_gain)
    df['improved'] = df['fitness_gain'] > 0
    
    # Calculate improvement rates by mutation type and generation phase
    improvement_stats = df.groupby(['mutation_type', 'generation_phase']).agg({
        'improved': ['sum', 'count'],
        'fitness_gain': ['mean', 'std']
    }).round(4)
    
    improvement_stats.columns = ['improvements', 'total_mutations', 'mean_fitness_gain', 'std_fitness_gain']
    improvement_stats['improvement_rate'] = (improvement_stats['improvements'] / 
                                           improvement_stats['total_mutations'] * 100).round(2)
    
    return df, improvement_stats

def plot_temporal_improvement_rates(improvement_stats, save_path=None):
    """Create temporal visualization of improvement rates"""
    
    # Reset index to get mutation_type and generation_phase as columns
    plot_data = improvement_stats.reset_index()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Improvement Rate by Type and Phase (Bar plot)
    pivot_improvement = plot_data.pivot(index='mutation_type', 
                                      columns='generation_phase', 
                                      values='improvement_rate')
    
    # Ensure consistent ordering
    phase_order = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]
    pivot_improvement = pivot_improvement.reindex(columns=phase_order)
    
    pivot_improvement.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Mutation Improvement Rate by Type and Generation Phase', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mutation Type')
    ax1.set_ylabel('Improvement Rate (%)')
    ax1.legend(title='Generation Phase')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Heatmap of improvement rates
    sns.heatmap(pivot_improvement, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=0, ax=ax2, cbar_kws={'label': 'Improvement Rate (%)'})
    ax2.set_title('Improvement Rate Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Generation Phase')
    ax2.set_ylabel('Mutation Type')
    
    # 3. Mean fitness gain by type and phase
    pivot_fitness = plot_data.pivot(index='mutation_type', 
                                  columns='generation_phase', 
                                  values='mean_fitness_gain')
    pivot_fitness = pivot_fitness.reindex(columns=phase_order)
    
    pivot_fitness.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Mean Fitness Gain by Type and Generation Phase', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mutation Type')
    ax3.set_ylabel('Mean Fitness Gain')
    ax3.legend(title='Generation Phase')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 4. Total mutations count by type and phase (to show sample sizes)
    pivot_count = plot_data.pivot(index='mutation_type', 
                                columns='generation_phase', 
                                values='total_mutations')
    pivot_count = pivot_count.reindex(columns=phase_order)
    
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
    
    plt.show()
    
    return pivot_improvement, pivot_fitness, pivot_count

def print_summary_stats(df, improvement_stats):
    """Print summary statistics"""
    
    print("=== TEMPORAL MUTATION EFFECTIVENESS ANALYSIS ===\n")
    
    # Overall stats by generation phase
    phase_stats = df.groupby('generation_phase').agg({
        'improved': ['sum', 'count'],
        'fitness_gain': ['mean', 'std']
    }).round(4)
    
    phase_stats.columns = ['improvements', 'total_mutations', 'mean_fitness_gain', 'std_fitness_gain']
    phase_stats['improvement_rate'] = (phase_stats['improvements'] / 
                                     phase_stats['total_mutations'] * 100).round(2)
    
    print("Overall stats by generation phase:")
    print(phase_stats)
    print()
    
    # Best and worst performers by phase
    print("Improvement rates by mutation type and generation phase:")
    pivot_table = improvement_stats.reset_index().pivot(index='mutation_type', 
                                                       columns='generation_phase', 
                                                       values='improvement_rate')
    
    phase_order = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]
    pivot_table = pivot_table.reindex(columns=phase_order)
    print(pivot_table)
    print()
    
    # Identify trends
    print("=== KEY INSIGHTS ===")
    
    # Which mutations get better/worse over time
    for mutation_type in pivot_table.index:
        rates = pivot_table.loc[mutation_type].values
        if len(rates) >= 2 and not pd.isna(rates[0]) and not pd.isna(rates[-1]):
            change = rates[-1] - rates[0]
            if abs(change) > 1:  # Only report significant changes
                trend = "improving" if change > 0 else "declining"
                print(f"• {mutation_type}: {trend} over time ({rates[0]:.1f}% → {rates[-1]:.1f}%)")
    
    print()
    
    # Overall effectiveness by phase
    overall_rates = phase_stats['improvement_rate'].values
    if len(overall_rates) >= 2:
        if overall_rates[0] > overall_rates[-1]:
            print(f"• Overall mutation effectiveness DECLINES over generations ({overall_rates[0]:.1f}% → {overall_rates[-1]:.1f}%)")
        else:
            print(f"• Overall mutation effectiveness IMPROVES over generations ({overall_rates[0]:.1f}% → {overall_rates[-1]:.1f}%)")

def main():
    # Load and process data
    csv_file = "nesting/aggregate_stats/master_mutation_raw_data.csv"
    
    print("Loading mutation data...")
    df, improvement_stats = load_and_process_data(csv_file)
    
    # Print summary statistics
    print_summary_stats(df, improvement_stats)
    
    # Create plots
    print("\nCreating temporal plots...")
    pivot_improvement, pivot_fitness, pivot_count = plot_temporal_improvement_rates(
        improvement_stats, 
        save_path="mutation_effectiveness_temporal.png"
    )
    
    # Save detailed results
    improvement_stats.to_csv("mutation_effectiveness_temporal_stats.csv")
    print("Detailed stats saved to: mutation_effectiveness_temporal_stats.csv")

if __name__ == "__main__":
    main()
