#!/usr/bin/env python3
"""
Plot mutation effectiveness by type using improvement rate metric.
improvement_rate = (mutations_improving_fitness / total_mutations) × 100
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_improvement_rates(df):
    """Calculate improvement rate for each mutation type."""
    # Group by mutation type
    grouped = df.groupby('mutation_type')
    
    results = []
    for mutation_type, group in grouped:
        total_mutations = len(group)
        improving_mutations = len(group[group['fitness_gain'] > 0])
        improvement_rate = (improving_mutations / total_mutations) * 100
        
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
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mutation Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Sort by improvement rate for better visualization
    stats_df_sorted = stats_df.sort_values('improvement_rate', ascending=True)
    
    # 1. Improvement Rate Bar Plot
    ax1 = axes[0, 0]
    bars1 = ax1.barh(stats_df_sorted['mutation_type'], stats_df_sorted['improvement_rate'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(stats_df_sorted))))
    ax1.set_xlabel('Improvement Rate (%)')
    ax1.set_title('Improvement Rate by Mutation Type')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 2. Mean Fitness Gain
    ax2 = axes[0, 1]
    bars2 = ax2.barh(stats_df_sorted['mutation_type'], stats_df_sorted['mean_fitness_gain'],
                     color=plt.cm.RdYlBu(np.linspace(0, 1, len(stats_df_sorted))))
    ax2.set_xlabel('Mean Fitness Gain')
    ax2.set_title('Average Fitness Impact by Mutation Type')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        label_x = width + 0.001 if width >= 0 else width - 0.001
        ha = 'left' if width >= 0 else 'right'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha=ha, va='center', fontweight='bold')
    
    # 3. Total Mutations Count
    ax3 = axes[1, 0]
    bars3 = ax3.barh(stats_df_sorted['mutation_type'], stats_df_sorted['total_mutations'],
                     color=plt.cm.plasma(np.linspace(0, 1, len(stats_df_sorted))))
    ax3.set_xlabel('Total Mutations')
    ax3.set_title('Sample Size by Mutation Type')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + max(stats_df_sorted['total_mutations'])*0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    # 4. Scatter: Improvement Rate vs Mean Fitness Gain
    ax4 = axes[1, 1]
    scatter = ax4.scatter(stats_df['improvement_rate'], stats_df['mean_fitness_gain'],
                         s=stats_df['total_mutations']/20, # Size by sample size
                         c=range(len(stats_df)), cmap='tab10', alpha=0.7, edgecolors='black')
    
    # Add labels for each point
    for i, row in stats_df.iterrows():
        ax4.annotate(row['mutation_type'], 
                    (row['improvement_rate'], row['mean_fitness_gain']),
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
    print("\n" + "="*60)
    print("MUTATION EFFECTIVENESS SUMMARY")
    print("="*60)
    
    print(f"\nTotal mutations analyzed: {len(raw_df):,}")
    print(f"Overall improvement rate: {(len(raw_df[raw_df['fitness_gain'] > 0]) / len(raw_df) * 100):.1f}%")
    print(f"Overall mean fitness gain: {raw_df['fitness_gain'].mean():.4f}")
    
    print("\nBy Mutation Type:")
    print("-" * 80)
    print(f"{'Type':<15} {'Count':<8} {'Improve%':<9} {'Mean Gain':<11} {'Std Gain':<10}")
    print("-" * 80)
    
    for _, row in stats_df.sort_values('improvement_rate', ascending=False).iterrows():
        print(f"{row['mutation_type']:<15} {row['total_mutations']:<8} "
              f"{row['improvement_rate']:<8.1f}% {row['mean_fitness_gain']:<10.4f} "
              f"{row['std_fitness_gain']:<10.4f}")
    
    print("\nKey Insights:")
    best_type = stats_df.loc[stats_df['improvement_rate'].idxmax()]
    worst_type = stats_df.loc[stats_df['improvement_rate'].idxmin()]
    
    print(f"• Best improvement rate: {best_type['mutation_type']} ({best_type['improvement_rate']:.1f}%)")
    print(f"• Worst improvement rate: {worst_type['mutation_type']} ({worst_type['improvement_rate']:.1f}%)")
    
    positive_gain = stats_df[stats_df['mean_fitness_gain'] > 0]
    if len(positive_gain) > 0:
        print(f"• Mutation types with positive average gain: {', '.join(positive_gain['mutation_type'].tolist())}")
    else:
        print("• No mutation types have positive average fitness gain")

if __name__ == "__main__":
    # Load the data
    print("Loading mutation data...")
    df = pd.read_csv('nesting/aggregate_stats/master_mutation_raw_data.csv')
    
    # Calculate improvement rates
    print("Calculating improvement rates...")
    stats_df = calculate_improvement_rates(df)
    
    # Print summary
    print_summary_stats(stats_df, df)
    
    # Create plots
    print("\nGenerating plots...")
    plot_improvement_rates(stats_df)
    
    print("\nAnalysis complete!")
