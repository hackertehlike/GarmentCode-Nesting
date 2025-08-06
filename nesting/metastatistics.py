"""
metastatistics.py - Module for tracking cross-run statistics in the GarmentCode nesting system.
This module provides tools to aggregate metrics across multiple pattern optimization runs.
"""

from __future__ import annotations
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

class GAMetaStatistics:
    def __init__(self):
        # Store data across all runs
        self.runs_data = []
        
        # Per-run summary metrics
        self.best_fitness_per_run = []
        self.generations_to_convergence = []
        self.run_durations = []
        
        # Mutation operator statistics
        self.mutation_improvements = defaultdict(list)
        self.mutation_counts = defaultdict(int)
        
        # Origin statistics (where top solutions came from)
        self.elite_origins = defaultdict(int)  # random, offspring, mutation
        
        # Convergence data
        self.convergence_curves = []
    
    def add_run_data(self, run_data):
        """Add data from a single GA run"""
        self.runs_data.append(run_data)
        
        # Extract summary metrics
        self.best_fitness_per_run.append(run_data['best_fitness'])
        self.generations_to_convergence.append(run_data['generations_to_convergence'])
        self.run_durations.append(run_data['duration'])
        
        # Process mutation operator data
        for op_type, improvements in run_data['mutation_improvements'].items():
            self.mutation_improvements[op_type].extend(improvements)
            self.mutation_counts[op_type] += run_data['mutation_counts'][op_type]
        
        # Process elite origins
        for origin, count in run_data['elite_origins'].items():
            self.elite_origins[origin] += count
        
        # Store convergence curve
        self.convergence_curves.append(run_data['convergence_curve'])
    
    def analyze(self):
        """Analyze the collected data and return a dictionary of results"""
        results = {}
        
        # Basic statistics
        results['num_runs'] = len(self.runs_data)
        results['avg_best_fitness'] = np.mean(self.best_fitness_per_run)
        results['std_best_fitness'] = np.std(self.best_fitness_per_run)
        results['avg_generations_to_convergence'] = np.mean(self.generations_to_convergence)
        results['avg_duration'] = np.mean(self.run_durations)
        
        # Mutation operator effectiveness
        results['mutation_avg_improvement'] = {
            op: np.mean(improvements) if improvements else 0
            for op, improvements in self.mutation_improvements.items()
        }
        results['mutation_std_improvement'] = {
            op: np.std(improvements) if improvements else 0
            for op, improvements in self.mutation_improvements.items()
        }
        
        # Elite origins distribution
        total_elites = sum(self.elite_origins.values())
        results['elite_origins_pct'] = {
            origin: count/total_elites*100 if total_elites > 0 else 0
            for origin, count in self.elite_origins.items()
        }
        
        # Average convergence curve
        max_gens = max(len(curve) for curve in self.convergence_curves)
        padded_curves = [
            np.pad(curve, (0, max_gens - len(curve)), 'edge')
            for curve in self.convergence_curves
        ]
        results['avg_convergence_curve'] = np.mean(padded_curves, axis=0)
        results['std_convergence_curve'] = np.std(padded_curves, axis=0)
        
        return results
    
    def visualize(self, results=None):
        """Create visualizations of the meta-statistics"""
        if results is None:
            results = self.analyze()
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 15))
        
        # Plot 1: Average convergence curve with std dev band
        ax1 = fig.add_subplot(2, 2, 1)
        x = range(len(results['avg_convergence_curve']))
        ax1.plot(x, results['avg_convergence_curve'], 'b-', label='Avg. fitness')
        ax1.fill_between(
            x, 
            results['avg_convergence_curve'] - results['std_convergence_curve'],
            results['avg_convergence_curve'] + results['std_convergence_curve'],
            alpha=0.2, color='b'
        )
        ax1.set_title('Average Convergence Across Runs')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.legend()
        
        # Plot 2: Mutation operator effectiveness
        ax2 = fig.add_subplot(2, 2, 2)
        operators = list(results['mutation_avg_improvement'].keys())
        improvements = [results['mutation_avg_improvement'][op] for op in operators]
        errors = [results['mutation_std_improvement'][op] for op in operators]
        ax2.bar(operators, improvements, yerr=errors)
        ax2.set_title('Average Improvement by Mutation Operator')
        ax2.set_xlabel('Operator')
        ax2.set_ylabel('Fitness Improvement')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: Elite origins pie chart
        ax3 = fig.add_subplot(2, 2, 3)
        origins = results['elite_origins_pct']
        ax3.pie(
            origins.values(), 
            labels=origins.keys(),
            autopct='%1.1f%%'
        )
        ax3.set_title('Elite Solutions Origin')
        
        # Plot 4: Generations to convergence histogram
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(self.generations_to_convergence, bins=10)
        ax4.set_title('Generations to Convergence')
        ax4.set_xlabel('Generations')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig


class MetaStatistics:
    """
    Static class that handles cross-run statistics and aggregation for the genetic algorithm.
    """
    
    # Store master files in a dedicated 'aggregate' directory to prevent conflicts with pattern-specific logs
    MASTER_DIR = Path(config.SAVE_LOGS_PATH).parent / "aggregate_stats"
    MASTER_CSV_PATH = MASTER_DIR / "master_statistics.csv"
    MASTER_MUTATION_CSV_PATH = MASTER_DIR / "master_mutation_stats.csv"
    # New file to store raw mutation data from all runs
    MASTER_MUTATION_RAW_DATA_PATH = MASTER_DIR / "master_mutation_raw_data.csv"
    CHECKPOINT_GENERATIONS = [5, 10, 15, 20]
    
    @classmethod
    def ensure_master_files_exist(cls):
        """
        Ensures that the master CSV files exist with appropriate headers.
        """
        # Create directories if needed
        cls.MASTER_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create master CSV if it doesn't exist
        if not cls.MASTER_CSV_PATH.exists():
            with open(cls.MASTER_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pattern_name', 'num_pieces', 'total_generations',
                    'initial_fitness', 'final_fitness', 'improvement_percent',
                    'fitness_at_gen_5', 'fitness_at_gen_10', 'fitness_at_gen_15', 'fitness_at_gen_20',
                    'improvement_at_gen_5', 'improvement_at_gen_10', 'improvement_at_gen_15', 'improvement_at_gen_20',
                    'elapsed_time', 'decoder', 'fitness_metric', 'crossover_method', 'mutation_rate',
                    'container_width', 'container_height'
                ])
        
        # Create master mutation statistics CSV if it doesn't exist
        if not cls.MASTER_MUTATION_CSV_PATH.exists():
            with open(cls.MASTER_MUTATION_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pattern_name', 'mutation_type', 
                    'avg_gain_total', 'count_total',
                    'avg_gain_early', 'count_early',  # Generations 1-5
                    'avg_gain_mid', 'count_mid',      # Generations 6-10
                    'avg_gain_late', 'count_late'     # Generations 11+
                ])
                
        # Create raw mutation data CSV if it doesn't exist
        if not cls.MASTER_MUTATION_RAW_DATA_PATH.exists():
            with open(cls.MASTER_MUTATION_RAW_DATA_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pattern_name', 'mutation_type', 'fitness_gain', 'generation', 'run_id'
                ])
    
    @classmethod
    def save_run_statistics(cls, evolution_instance, elapsed_time: float):
        """
        Save statistics from a completed evolution run to the master CSV.
        
        Args:
            evolution_instance: The Evolution instance that completed a run
            elapsed_time: Total elapsed time for the run
        """
        cls.ensure_master_files_exist()
        
        # Extract checkpoint metrics
        checkpoints = cls._extract_checkpoint_metrics(evolution_instance)
        
        # Create the row data
        initial_fitness = evolution_instance.best_fitness_history[0] if evolution_instance.best_fitness_history else 0
        final_fitness = evolution_instance.best_fitness_history[-1] if evolution_instance.best_fitness_history else 0
        
        improvement = final_fitness - initial_fitness
        improvement_percent = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
        
        row = {
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'pattern_name': evolution_instance.pattern_name,
            'num_pieces': len(evolution_instance.pieces),
            'total_generations': evolution_instance.generation,
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'improvement_percent': improvement_percent,
            'elapsed_time': elapsed_time,
            'decoder': config.SELECTED_DECODER,
            'fitness_metric': config.SELECTED_FITNESS_METRIC,
            'crossover_method': config.SELECTED_CROSSOVER,
            'mutation_rate': config.MUTATION_RATE,
            'container_width': config.CONTAINER_WIDTH_CM,
            'container_height': config.CONTAINER_HEIGHT_CM
        }
        
        # Add checkpoint metrics
        for gen in cls.CHECKPOINT_GENERATIONS:
            if gen in checkpoints:
                row[f'fitness_at_gen_{gen}'] = checkpoints[gen]['fitness']
                row[f'improvement_at_gen_{gen}'] = checkpoints[gen]['improvement_percent']
            else:
                row[f'fitness_at_gen_{gen}'] = ''
                row[f'improvement_at_gen_{gen}'] = ''
        
        # Append to master CSV
        with open(cls.MASTER_CSV_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
            
        # Also save mutation statistics
        cls._save_mutation_statistics(evolution_instance)
        
        return row
    
    @classmethod
    def _extract_checkpoint_metrics(cls, evolution_instance) -> Dict[int, Dict[str, float]]:
        """
        Extract fitness metrics at checkpoint generations.
        
        Returns:
            Dictionary mapping generation numbers to metrics
        """
        result = {}
        initial_fitness = evolution_instance.best_fitness_history[0] if evolution_instance.best_fitness_history else 0
        
        for gen in cls.CHECKPOINT_GENERATIONS:
            if gen < len(evolution_instance.best_fitness_history):
                fitness = evolution_instance.best_fitness_history[gen]
                improvement = fitness - initial_fitness
                improvement_percent = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
                
                result[gen] = {
                    'fitness': fitness,
                    'improvement': improvement,
                    'improvement_percent': improvement_percent
                }
                
        return result
    
    @classmethod
    def _save_mutation_statistics(cls, evolution_instance):
        """
        Save mutation statistics to the master mutation CSV.
        
        Args:
            evolution_instance: The Evolution instance that completed a run
        """
        # Get mutation data from the swarm DataFrame
        if evolution_instance._mutation_swarm_data.empty:
            return
            
        df = evolution_instance._mutation_swarm_data
        
        # Group mutations by type and calculate statistics
        by_type = df.groupby('mutation_type')
        
        # Also group by generation ranges
        early = df[df['generation'] <= 5]
        mid = df[(df['generation'] > 5) & (df['generation'] <= 10)]
        late = df[df['generation'] > 10]
        
        early_by_type = early.groupby('mutation_type')
        mid_by_type = mid.groupby('mutation_type')
        late_by_type = late.groupby('mutation_type')
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{evolution_instance.pattern_name}_{timestamp}"
        
        # Save aggregated statistics to master mutation CSV
        with open(cls.MASTER_MUTATION_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for mutation_type in df['mutation_type'].unique():
                # Get overall stats
                total_data = by_type.get_group(mutation_type) if mutation_type in by_type.groups else None
                total_avg = total_data['fitness_gain'].mean() if total_data is not None else 0
                total_count = len(total_data) if total_data is not None else 0
                
                # Get early-generation stats (1-5)
                early_data = early_by_type.get_group(mutation_type) if mutation_type in early_by_type.groups else None
                early_avg = early_data['fitness_gain'].mean() if early_data is not None else 0
                early_count = len(early_data) if early_data is not None else 0
                
                # Get mid-generation stats (6-10)
                mid_data = mid_by_type.get_group(mutation_type) if mutation_type in mid_by_type.groups else None
                mid_avg = mid_data['fitness_gain'].mean() if mid_data is not None else 0
                mid_count = len(mid_data) if mid_data is not None else 0
                
                # Get late-generation stats (11+)
                late_data = late_by_type.get_group(mutation_type) if mutation_type in late_by_type.groups else None
                late_avg = late_data['fitness_gain'].mean() if late_data is not None else 0
                late_count = len(late_data) if late_data is not None else 0
                
                writer.writerow([
                    timestamp,
                    evolution_instance.pattern_name,
                    mutation_type,
                    total_avg,
                    total_count,
                    early_avg,
                    early_count,
                    mid_avg,
                    mid_count,
                    late_avg,
                    late_count
                ])
        
        # Save raw mutation data for detailed analysis
        with open(cls.MASTER_MUTATION_RAW_DATA_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write each individual mutation record with run identifier
            for _, row in df.iterrows():
                writer.writerow([
                    timestamp,
                    evolution_instance.pattern_name,
                    row['mutation_type'],
                    row['fitness_gain'],
                    row['generation'],
                    run_id
                ])
    
    @classmethod
    def generate_aggregate_reports(cls):
        """
        Generate aggregate reports and visualizations from the master statistics.
        This should be called after evolution runs have been completed.
        """
        # Ensure master files exist (creates empty files with headers if they don't)
        cls.ensure_master_files_exist()
            
        # Load the data - handle cases where files don't exist or are empty
        try:
            # Check if file exists and has data beyond headers
            if cls.MASTER_CSV_PATH.exists() and cls.MASTER_CSV_PATH.stat().st_size > 100:
                master_df = pd.read_csv(cls.MASTER_CSV_PATH)
            else:
                master_df = pd.DataFrame()
                
            # Check mutation data file
            if cls.MASTER_MUTATION_CSV_PATH.exists() and cls.MASTER_MUTATION_CSV_PATH.stat().st_size > 100:
                mutation_df = pd.read_csv(cls.MASTER_MUTATION_CSV_PATH)
            else:
                mutation_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading master statistics files: {e}")
            master_df = pd.DataFrame()
            mutation_df = pd.DataFrame()
        
        if len(master_df) < 1:
            print("Not enough data to generate aggregate reports yet. Run at least one evolution first.")
            return
            
        # We'll generate what we can even with a single data point
        print(f"Generating reports with {len(master_df)} data points")
            
        # Create output directory - use the same parent directory as the master files
        reports_dir = cls.MASTER_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports - each method will check if there's enough data
        cls._generate_checkpoint_comparison(master_df, reports_dir)
        
        if not mutation_df.empty:
            cls._generate_mutation_effectiveness(mutation_df, reports_dir)
        else:
            print("No mutation data available, skipping mutation effectiveness reports")
            
        # Load raw mutation data file for detailed analysis
        try:
            if cls.MASTER_MUTATION_RAW_DATA_PATH.exists() and cls.MASTER_MUTATION_RAW_DATA_PATH.stat().st_size > 100:
                mutation_raw_df = pd.read_csv(cls.MASTER_MUTATION_RAW_DATA_PATH)
                if not mutation_raw_df.empty:
                    cls._generate_mutation_fitness_changes_by_type(mutation_raw_df, reports_dir)
                else:
                    print("No raw mutation data available, skipping detailed mutation analysis")
            else:
                print("Raw mutation data file does not exist or is empty")
        except Exception as e:
            print(f"Error loading raw mutation data: {e}")
            
        cls._generate_pattern_comparison(master_df, reports_dir)
        
        print(f"Aggregate reports generated in {reports_dir}")
    
    @classmethod
    def _generate_checkpoint_comparison(cls, df, output_dir):
        """Generate checkpoint comparison charts"""
        # First save the raw data - this is always useful
        df.to_csv(output_dir / 'all_runs_data.csv', index=False)
        
        # Prepare data for checkpoint comparison
        checkpoint_data = []
        
        for gen in cls.CHECKPOINT_GENERATIONS:
            column_name = f'fitness_at_gen_{gen}'
            if column_name not in df.columns:
                continue
                
            gen_data = df[df[column_name].notnull()]
            if not gen_data.empty:
                improvement_column = f'improvement_at_gen_{gen}'
                if improvement_column in df.columns:
                    avg_improvement = gen_data[improvement_column].mean()
                    checkpoint_data.append({
                        'generation': gen,
                        'avg_improvement_percent': avg_improvement,
                        'num_patterns': len(gen_data)
                    })
        
        if not checkpoint_data:
            print("No checkpoint data available for analysis")
            # Create an empty file so we know the analysis was attempted
            with open(output_dir / 'checkpoint_comparison_no_data.txt', 'w') as f:
                f.write("No checkpoint data available for analysis")
            return
            
        # Create the dataframe for plotting
        checkpoint_df = pd.DataFrame(checkpoint_data)
        
        try:
            # Plot improvement by generation checkpoint
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='generation', y='avg_improvement_percent', data=checkpoint_df)
            
            # Add count labels
            for i, row in enumerate(checkpoint_data):
                ax.text(i, row['avg_improvement_percent'] + 1, 
                        f"n={row['num_patterns']}", 
                        ha='center')
                
            plt.title('Average Fitness Improvement at Generation Checkpoints')
            plt.xlabel('Generation')
            plt.ylabel('Average Improvement (%)')
            plt.tight_layout()
            plt.savefig(output_dir / 'checkpoint_comparison.png')
            plt.close()
        except Exception as e:
            print(f"Error generating checkpoint plot: {e}")
            # Create a text file with the data instead
            with open(output_dir / 'checkpoint_data.txt', 'w') as f:
                f.write("Checkpoint data (plot generation failed):\n")
                for item in checkpoint_data:
                    f.write(f"Generation {item['generation']}: {item['avg_improvement_percent']:.2f}% improvement (n={item['num_patterns']})\n")
        
        # Save as CSV
        checkpoint_df.to_csv(output_dir / 'checkpoint_comparison.csv', index=False)
    
    @classmethod
    def _generate_mutation_effectiveness(cls, df, output_dir):
        """Generate mutation effectiveness charts"""
        if df.empty or 'mutation_type' not in df.columns:
            print("No mutation data available for analysis")
            # Create a simple file to indicate the analysis was attempted
            with open(output_dir / 'mutation_effectiveness_no_data.txt', 'w') as f:
                f.write("No mutation data available for analysis")
            return
            
        try:
            # Group by mutation type and calculate average gains
            mutation_summary = df.groupby('mutation_type').agg({
                'avg_gain_total': 'mean',
                'count_total': 'sum',
                'avg_gain_early': 'mean',
                'count_early': 'sum',
                'avg_gain_mid': 'mean',
                'count_mid': 'sum',
                'avg_gain_late': 'mean',
                'count_late': 'sum'
            }).reset_index()
            
            # Save CSVs first (in case plotting fails)
            mutation_summary.to_csv(output_dir / 'mutation_effectiveness.csv', index=False)
            
            # Plot overall effectiveness
            plt.figure(figsize=(12, 7))
            ax = sns.barplot(x='mutation_type', y='avg_gain_total', data=mutation_summary)
            
            # Add count labels
            for i, row in mutation_summary.iterrows():
                ax.text(i, row['avg_gain_total'] + 0.005, 
                        f"n={int(row['count_total'])}", 
                        ha='center')
                
            plt.title('Overall Mutation Effectiveness')
            plt.xlabel('Mutation Type')
            plt.ylabel('Average Fitness Gain')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'mutation_effectiveness_overall.png')
            plt.close()
            
            # Create a generational comparison (early, mid, late)
            # Reshape data for plotting
            plot_data = []
            for _, row in mutation_summary.iterrows():
                plot_data.append({
                    'mutation_type': row['mutation_type'],
                    'phase': 'early (gen 1-5)',
                    'avg_gain': row['avg_gain_early'],
                    'count': row['count_early']
                })
                plot_data.append({
                    'mutation_type': row['mutation_type'],
                    'phase': 'mid (gen 6-10)',
                    'avg_gain': row['avg_gain_mid'],
                    'count': row['count_mid']
                })
                plot_data.append({
                    'mutation_type': row['mutation_type'],
                    'phase': 'late (gen 11+)',
                    'avg_gain': row['avg_gain_late'],
                    'count': row['count_late']
                })
                
            plot_df = pd.DataFrame(plot_data)
            
            # Save this CSV as well
            plot_df.to_csv(output_dir / 'mutation_effectiveness_by_phase.csv', index=False)
            
            # Only try to plot if we actually have data
            if not plot_df.empty:
                plt.figure(figsize=(15, 8))
                # Plot
                ax = sns.barplot(x='mutation_type', y='avg_gain', hue='phase', data=plot_df)
                plt.title('Mutation Effectiveness by Evolution Phase')
                plt.xlabel('Mutation Type')
                plt.ylabel('Average Fitness Gain')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'mutation_effectiveness_by_phase.png')
                plt.close()
            else:
                print("Phase plot data is empty, skipping phase plot")
                
        except Exception as e:
            print(f"Error generating mutation effectiveness charts: {e}")
            # Save basic data in text format
            with open(output_dir / 'mutation_data.txt', 'w') as f:
                f.write("Mutation data (plot generation failed):\n")
                for mutation_type in df['mutation_type'].unique():
                    type_df = df[df['mutation_type'] == mutation_type]
                    avg_gain = type_df['avg_gain_total'].mean() if 'avg_gain_total' in df.columns else "N/A"
                    count = type_df['count_total'].sum() if 'count_total' in df.columns else len(type_df)
                    f.write(f"Mutation {mutation_type}: {avg_gain:.4f} avg gain (n={count})\n")
    
    @classmethod
    def _generate_pattern_comparison(cls, df, output_dir):
        """Generate pattern comparison charts"""
        if df.empty:
            print("No pattern data for comparison")
            return
            
        # Even with one pattern, we can generate statistics
        if len(df) < 2:
            print("Only one pattern available - generating single pattern statistics")
            
        try:
            # Summary by pattern
            pattern_summary = df.groupby('pattern_name').agg({
                'improvement_percent': 'mean',
                'elapsed_time': 'mean',
                'total_generations': 'mean',
                'timestamp': 'count'  # Count of runs
            }).rename(columns={'timestamp': 'num_runs'}).reset_index()
            
            # Save as CSV (do this first in case plotting fails)
            pattern_summary.to_csv(output_dir / 'pattern_comparison.csv', index=False)
            
            # Only attempt to plot if we have actual patterns
            if not pattern_summary.empty:
                # Plot improvement by pattern
                plt.figure(figsize=(12, 6))
                ax = sns.barplot(x='pattern_name', y='improvement_percent', data=pattern_summary)
                
                # Add run count labels
                for i, row in pattern_summary.iterrows():
                    ax.text(i, row['improvement_percent'] + 1, 
                            f"n={int(row['num_runs'])}", 
                            ha='center')
                    
                plt.title('Average Improvement by Pattern')
                plt.xlabel('Pattern')
                plt.ylabel('Average Improvement (%)')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(output_dir / 'pattern_comparison.png')
                plt.close()
            else:
                print("Pattern summary is empty, skipping plot generation")
                
        except Exception as e:
            print(f"Error generating pattern comparison: {e}")
            # Create a text file with the basic information
            with open(output_dir / 'pattern_data.txt', 'w') as f:
                f.write("Pattern data (plot generation failed):\n")
                for name in df['pattern_name'].unique():
                    pattern_df = df[df['pattern_name'] == name]
                    avg_improvement = pattern_df['improvement_percent'].mean() if 'improvement_percent' in df.columns else "N/A"
                    count = len(pattern_df)
                    f.write(f"Pattern {name}: {avg_improvement:.2f}% avg improvement (n={count})\n")


    @classmethod
    def _generate_mutation_fitness_changes_by_type(cls, df, output_dir):
        """
        Generate plots showing mutation fitness changes by type across all runs.
        
        Args:
            df: DataFrame with raw mutation data
            output_dir: Directory where to save the plots
        """
        if df.empty or 'mutation_type' not in df.columns or 'fitness_gain' not in df.columns or 'generation' not in df.columns:
            print("No detailed mutation data available for analysis")
            # Create a simple file to indicate the analysis was attempted
            with open(output_dir / 'mutation_fitness_changes_no_data.txt', 'w') as f:
                f.write("No detailed mutation data available for analysis")
            return
            
        try:
            # Ensure the data types are correct
            df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
            df['fitness_gain'] = pd.to_numeric(df['fitness_gain'], errors='coerce')
            df = df.dropna(subset=['generation', 'fitness_gain', 'mutation_type'])
            
            if df.empty:
                print("No valid data after type conversion")
                return
                
            # Save processed data as CSV
            df.to_csv(output_dir / 'mutation_fitness_changes_raw.csv', index=False)
            
            print(f"Generating mutation fitness changes plot with {len(df)} data points")
            
            # Create swarm plot for all data with improved spacing
            # Use a much wider figure to avoid point overlap
            plt.figure(figsize=(24, 12))
            
            # Get unique generations for better width calculation
            unique_generations = df['generation'].unique().shape[0]
            
            # Adjust the point size based on the amount of data
            point_size = 2.5 if len(df) > 1000 else 3.5
            point_alpha = 0.5 if len(df) > 1000 else 0.7
            
            # Create the swarmplot with improved parameters
            ax = sns.swarmplot(
                data=df,
                x="generation",
                y="fitness_gain",
                hue="mutation_type",
                size=point_size,
                alpha=point_alpha,
                dodge=True,
                # Increase dodge width to separate points better
                dodge_kws={"width": 0.8}
            )
            
            plt.title("Fitness Gain by Mutation Type Across All Runs", fontsize=16)
            plt.xlabel("Generation", fontsize=14)
            plt.ylabel("Fitness Gain", fontsize=14)
            
            # Move the legend outside the plot for better visibility
            ax.legend(title="Mutation Type", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'mutation_fitness_changes_swarm.png', dpi=150)
            plt.close()
            
            # Create box plot variation for better visibility with large datasets
            plt.figure(figsize=(20, 12))
            
            # Use a more advanced boxplot with notches to show statistical significance
            ax = sns.boxplot(
                data=df,
                x="generation",
                y="fitness_gain",
                hue="mutation_type",
                notch=True,  # Add notch for statistical comparison
                palette="viridis",  # Use a more distinct color palette
                fliersize=3  # Make outlier points smaller
            )
            
            plt.title("Fitness Gain Distribution by Mutation Type Across All Runs", fontsize=16)
            plt.xlabel("Generation", fontsize=14)
            plt.ylabel("Fitness Gain", fontsize=14)
            
            # Add more informative legend
            ax.legend(title="Mutation Type", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add a horizontal line at y=0 to highlight positive vs negative gains
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'mutation_fitness_changes_boxplot.png', dpi=150)
            plt.close()
            
            # Also create a line plot showing the average fitness change by generation for each mutation type
            plt.figure(figsize=(20, 12))
            
            # Calculate mean and standard error for error bands
            avg_by_gen = df.groupby(['mutation_type', 'generation'])['fitness_gain'].agg(['mean', 'std', 'count']).reset_index()
            avg_by_gen['se'] = avg_by_gen['std'] / np.sqrt(avg_by_gen['count'])
            
            # Create a more informative line plot with error bands
            sns.lineplot(
                data=avg_by_gen,
                x='generation',
                y='mean',
                hue='mutation_type',
                marker='o',
                markersize=8,
                linewidth=2.5,
                palette="viridis",
                # Add error bands
                err_style='band',
                err_kws={'alpha': 0.2}
            )
            
            # Add point counts to the plot
            for mutation_type in avg_by_gen['mutation_type'].unique():
                for _, row in avg_by_gen[avg_by_gen['mutation_type'] == mutation_type].iterrows():
                    plt.annotate(
                        f"n={int(row['count'])}",
                        (row['generation'], row['mean']),
                        xytext=(0, 10), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7,
                        ha='center'
                    )
            
            plt.title("Average Fitness Gain by Mutation Type and Generation", fontsize=16)
            plt.xlabel("Generation", fontsize=14)
            plt.ylabel("Average Fitness Gain", fontsize=14)
            
            # Add a horizontal line at y=0 to highlight positive vs negative gains
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Add grid for better readability
            plt.grid(linestyle='--', alpha=0.7)
            
            plt.legend(title="Mutation Type", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / 'mutation_fitness_changes_line.png', dpi=150)
            plt.close()
            
            # For large datasets, create additional split views to improve readability
            if len(df) > 500:
                print("Large dataset detected, creating split views for better visibility")
                
                # Split into early, mid, and late generations
                # Define the ranges dynamically based on max generation
                max_gen = df['generation'].max()
                early_threshold = max(5, max_gen // 3)
                mid_threshold = max(10, 2 * max_gen // 3)
                
                generation_ranges = [
                    (0, early_threshold, "early"),
                    (early_threshold, mid_threshold, "mid"),
                    (mid_threshold, max_gen + 1, "late")
                ]
                
                for start_gen, end_gen, phase in generation_ranges:
                    # Filter data for this generation range
                    range_df = df[(df['generation'] >= start_gen) & (df['generation'] < end_gen)]
                    
                    if len(range_df) > 0:
                        plt.figure(figsize=(20, 12))
                        ax = sns.swarmplot(
                            data=range_df,
                            x="generation",
                            y="fitness_gain",
                            hue="mutation_type",
                            size=4,  # Larger points for the split view
                            alpha=0.7,
                            dodge=True,
                            dodge_kws={"width": 0.8}
                        )
                        
                        plt.title(f"Fitness Gain by Mutation Type ({phase.capitalize()} Generations: {start_gen}-{end_gen-1})", fontsize=16)
                        plt.xlabel("Generation", fontsize=14)
                        plt.ylabel("Fitness Gain", fontsize=14)
                        ax.legend(title="Mutation Type", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig(output_dir / f'mutation_fitness_changes_swarm_{phase}_gens.png', dpi=150)
                        plt.close()
            
            print("Mutation fitness changes plots generated successfully")
            
        except Exception as e:
            print(f"Error generating mutation fitness changes plots: {e}")
            import traceback
            traceback.print_exc()
            # Save basic data in text format
            with open(output_dir / 'mutation_fitness_changes_error.txt', 'w') as f:
                f.write(f"Error generating mutation fitness changes plots: {e}\n")
                f.write("Raw data summary:\n")
                f.write(f"Total data points: {len(df)}\n")
                for mutation_type in df['mutation_type'].unique():
                    type_df = df[df['mutation_type'] == mutation_type]
                    avg_gain = type_df['fitness_gain'].mean() 
                    count = len(type_df)
                    f.write(f"Mutation {mutation_type}: {avg_gain:.4f} avg gain (n={count})\n")


class MetaStatisticsCLI:
    """Command-line interface for generating meta-statistics reports"""
    
    @staticmethod
    def generate_reports():
        """Generate all aggregate reports from existing data"""
        print("Generating aggregate statistics reports...")
        MetaStatistics.generate_aggregate_reports()
        print("Done!")


if __name__ == "__main__":
    MetaStatisticsCLI.generate_reports()