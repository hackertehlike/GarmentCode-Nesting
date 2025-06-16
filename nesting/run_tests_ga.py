#!/usr/bin/env python
# filepath: /Users/aysegulbarlas/codestuff/GarmentCode/nesting/run_tests_ga.py
import sys, os
# add project root to sys.path for local package imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import time
from datetime import datetime
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from shapely.errors import GEOSException as TopologyException

from nesting.path_extractor import PatternPathExtractor
from nesting.layout import Layout, Container, Piece, LayoutView
from nesting.placement_engine import DECODER_REGISTRY
from nesting.evolution import Evolution
import nesting.config as config


def flush_results(rows: list[dict], timestamp: str = None) -> None:
    """Write metrics CSV and update the utilization plots with timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a directory for results if it doesn't exist
    results_dir = Path("./ga_benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    # CSV filename with timestamp
    csv_path = results_dir / f"ga_metrics_{timestamp}.csv"
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    if df.empty:
        return

    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Scatter plot of each run: bounding-box vs concave-hull utilization
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        df['usage_bb'], 
        df['concave_hull'], 
        c=df['generation'] if 'generation' in df.columns else None,
        cmap='viridis', 
        marker='o', 
        alpha=0.7,
        s=50
    )
    
    # Add colorbar if we have generation data
    if 'generation' in df.columns:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Generation')
    
    ax.set_xlabel('Bounding-box utilization')
    ax.set_ylabel('Concave-hull utilization')
    ax.set_title('GA Utilization by Generation')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save scatter chart
    scatter_path = plots_dir / f"ga_utilization_scatter_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)

    # 2. Utilization by pattern
    if 'pattern' in df.columns:
        summary = df.groupby('pattern').agg(
            avg_bb=('usage_bb', 'mean'),
            avg_hull=('concave_hull', 'mean'),
            max_bb=('usage_bb', 'max'),
            max_hull=('concave_hull', 'max')
        ).reset_index()
        
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10))
        
        # Average utilization
        axes2[0].bar(summary['pattern'], summary['avg_bb'], color='skyblue', alpha=0.7, label='Avg Bounding-box')
        axes2[0].bar(summary['pattern'], summary['avg_hull'], color='orange', alpha=0.7, label='Avg Concave-hull')
        axes2[0].set_title('Average Utilization by Pattern')
        axes2[0].set_ylabel('Utilization')
        axes2[0].legend()
        axes2[0].tick_params(axis='x', rotation=45)
        axes2[0].grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # Maximum utilization
        axes2[1].bar(summary['pattern'], summary['max_bb'], color='blue', alpha=0.7, label='Max Bounding-box')
        axes2[1].bar(summary['pattern'], summary['max_hull'], color='red', alpha=0.7, label='Max Concave-hull')
        axes2[1].set_title('Maximum Utilization by Pattern')
        axes2[1].set_ylabel('Utilization')
        axes2[1].legend()
        axes2[1].tick_params(axis='x', rotation=45)
        axes2[1].grid(True, axis='y', linestyle='--', alpha=0.6)
        
        fig2.tight_layout()
        avg_path = plots_dir / f"ga_utilization_by_pattern_{timestamp}.png"
        fig2.savefig(avg_path, dpi=300)
        plt.close(fig2)
    
    # 3. If we have generation data, plot fitness over generations
    if 'generation' in df.columns and 'best_fitness' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        for pattern in df['pattern'].unique():
            pattern_data = df[df['pattern'] == pattern]
            pattern_data = pattern_data.sort_values('generation')
            ax3.plot(pattern_data['generation'], pattern_data['best_fitness'], 
                    marker='o', label=pattern, alpha=0.7)
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Best Fitness')
        ax3.set_title('Fitness Evolution by Generation')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
        
        fitness_path = plots_dir / f"ga_fitness_evolution_{timestamp}.png"
        fig3.tight_layout()
        fig3.savefig(fitness_path, dpi=300)
        plt.close(fig3)
        
    print(f"Results saved to {csv_path} and plots saved to {plots_dir}")


def load_pieces(json_path: Path) -> dict[str, Piece]:
    extractor = PatternPathExtractor(json_path)
    pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
    for p in pieces.values():
        p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
    return pieces


def split_pieces(pieces: dict[str, Piece]) -> dict[str, Piece]:
    """Split pieces and handle topology exceptions."""
    new_pieces: dict[str, Piece] = {}
    for piece in pieces.values():
        try:
            left, right = piece.split()
            new_pieces[left.id] = left
            new_pieces[right.id] = right
        except TopologyException as e:
            print(f"Skipping split for piece {piece.id} due to topology error: {e}")
            new_pieces[piece.id] = piece  # Keep the original piece
    return new_pieces


def run_ga_with_tracking(pieces: dict[str, Piece], container: Container, pattern_name: str, split: bool = False) -> list[dict]:
    """
    Run Genetic Algorithm on the given pieces and container, tracking metrics for each generation.
    Returns a list of dictionaries with metrics for each generation.
    """
    start_time = time.time()
    results = []
    
    try:
        # Create an Evolution instance with default config settings
        evo = Evolution(
            pieces,
            container,
            num_generations=config.NUM_GENERATIONS,
            population_size=config.POPULATION_SIZE,
            mutation_rate=config.MUTATION_RATE,
            enable_dynamic_stopping=config.ENABLE_DYNAMIC_STOPPING,
            early_stop_window=config.EARLY_STOP_WINDOW,
            early_stop_tolerance=config.EARLY_STOP_TOLERANCE,
            enable_extension=config.ENABLE_EXTENSION,
            extend_window=config.EXTEND_WINDOW,
            extend_threshold=config.EXTEND_THRESHOLD,
            max_generations=config.MAX_GENERATIONS,
            crossover_method=config.SELECTED_CROSSOVER,
        )
        
        # Initialize population
        print(f"Generating initial population for {pattern_name} (Split: {split})")
        evo.generate_population()
        
        # Record initial generation (gen 0)
        elites = evo._get_elite()
        if elites:
            best_chrom = elites[0]
            # Use decoder to calculate utilization metrics
            view = LayoutView(best_chrom.genes)
            decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
            decoder.decode()
            
            results.append({
                "pattern": pattern_name,
                "split": split,
                "generation": 0,
                "best_fitness": best_chrom.fitness,
                "avg_fitness": sum(c.fitness for c in evo.population) / len(evo.population) if evo.population else 0,
                "elites_found": len(elites),
                "usage_bb": decoder.usage_BB(),
                "concave_hull": decoder.concave_hull_utilization(),
                "rest_length": decoder.rest_length(),
                "execution_time": 0,
                "algorithm": f"GA-{config.SELECTED_DECODER}",
                "decoder": config.SELECTED_DECODER,
                "crossover": config.SELECTED_CROSSOVER,
                "population_size": config.POPULATION_SIZE,
                "elite_count": evo.n_elites,
                "mutation_rate": config.MUTATION_RATE,
            })
        
        # Run for specified number of generations
        max_gens = config.NUM_GENERATIONS
        for gen in range(1, max_gens + 1):
            gen_start = time.time()
            print(f"Running generation {gen}/{max_gens} for {pattern_name} (Split: {split})")
            
            # Evolve to next generation
            evo.next_generation()
            evo.generation += 1
            
            # Record metrics for this generation
            elites = evo._get_elite()
            if elites:
                best_chrom = elites[0]
                # Use decoder to calculate utilization metrics
                view = LayoutView(best_chrom.genes)
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
                decoder.decode()
                
                gen_time = time.time() - start_time
                gen_delta = time.time() - gen_start
                
                results.append({
                    "pattern": pattern_name,
                    "split": split,
                    "generation": gen,
                    "best_fitness": best_chrom.fitness,
                    "avg_fitness": sum(c.fitness for c in evo.population) / len(evo.population) if evo.population else 0,
                    "elites_found": len(elites),
                    "usage_bb": decoder.usage_BB(),
                    "concave_hull": decoder.concave_hull_utilization(),
                    "rest_length": decoder.rest_length(),
                    "execution_time": gen_time,
                    "generation_time": gen_delta,
                    "algorithm": f"GA-{config.SELECTED_DECODER}",
                    "decoder": config.SELECTED_DECODER,
                    "crossover": config.SELECTED_CROSSOVER, 
                    "population_size": config.POPULATION_SIZE,
                    "elite_count": evo.n_elites,
                    "mutation_rate": config.MUTATION_RATE,
                })
            
            # Check if early stopped
            if evo.early_stopped:
                print(f"Early stopping triggered at generation {gen} for {pattern_name}")
                break
        
        # Record final best solution
        print(f"GA completed for {pattern_name} (Split: {split}) in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during GA run for {pattern_name}: {e}")
    
    return results


def main():
    data_dir = Path("./nesting-assets/garmentcodedata_batch0")
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}.")
        return

    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting GA benchmark run at {timestamp}")
    
    # Create container with dimensions from config
    container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)
    all_results = []

    # Limit the number of patterns to process for testing
    max_patterns = 3  # Adjust as needed
    test_files = json_files[:max_patterns]
    
    # Record GA configuration for reference
    print(f"\nGA Configuration:")
    print(f"  Population Size: {config.POPULATION_SIZE}")
    print(f"  Number of Generations: {config.NUM_GENERATIONS}")
    print(f"  Mutation Rate: {config.MUTATION_RATE}")
    print(f"  Crossover Method: {config.SELECTED_CROSSOVER}")
    print(f"  Decoder: {config.SELECTED_DECODER}")
    print(f"  Dynamic Stopping: {config.ENABLE_DYNAMIC_STOPPING}")
    print(f"  Elite %: {config.POPULATION_WEIGHTS['elites']}")
    print()

    for json_path in test_files:
        pattern_name = json_path.stem
        print(f"\n{'='*50}")
        print(f"Processing pattern: {pattern_name}")
        print(f"{'='*50}")
        
        # Load pieces
        pieces = load_pieces(json_path)
        print(f"Loaded {len(pieces)} pieces from {pattern_name}")
        
        # Run GA without splitting
        print(f"\nRunning GA for {pattern_name} without splitting")
        results = run_ga_with_tracking(
            copy.deepcopy(pieces), 
            container, 
            pattern_name, 
            split=False
        )
        all_results.extend(results)
        
        # Save intermediate results
        flush_results(all_results, timestamp)
        
        # Run GA with splitting
        print(f"\nRunning GA for {pattern_name} with splitting")
        split_dict = split_pieces(copy.deepcopy(pieces))
        print(f"After splitting: {len(split_dict)} pieces")
        
        results = run_ga_with_tracking(
            copy.deepcopy(split_dict), 
            container, 
            pattern_name, 
            split=True
        )
        all_results.extend(results)
        
        # Save results after each pattern
        flush_results(all_results, timestamp)

    # Final save of all results
    flush_results(all_results, timestamp)
    print(f"\nGA benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total benchmarks: {len(test_files)} patterns, with and without splitting")


if __name__ == "__main__":
    main()
