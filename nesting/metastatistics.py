import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

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