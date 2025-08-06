"""
Pattern nesting optimization using genetic algorithm
–––––––––––––––––––––––––––
* Runs the nesting GA on multiple pattern files
* Uses the same approach as the GUI but in batch mode
* Generates CSV outputs and plots for each run
"""

from __future__ import annotations
import os, time, shutil
import copy
import yaml
import json
from itertools import product
from pathlib import Path
import traceback
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nesting.config as config

from nesting.path_extractor import PatternPathExtractor
from .evolution import Evolution
from .layout import Container, Piece, Layout
from assets.bodies.body_params import BodyParameters
from nesting.metastatistics import MetaStatistics

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _save_plot(fig, out_dir: str, fname: str) -> None:
    """Save a matplotlib figure to a file."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def run_ga_on_patterns(pattern_paths, output_dir="results") -> None:
    """
    Run the genetic algorithm on multiple patterns.
    
    Args:
        pattern_paths: List of paths to pattern files
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track results across all patterns
    all_results = []
    
    # Process each pattern
    for pattern_path in pattern_paths:
        pattern_path = Path(pattern_path)
        # Extract the pattern name without the "_specification" suffix
        pattern_stem = pattern_path.stem
        pattern_name = pattern_stem.replace("_specification", "") if pattern_stem.endswith("_specification") else pattern_stem
        
        print(f"\n{'='*60}")
        print(f"Processing pattern: {pattern_name}")
        print(f"{'='*60}")
        
        # Create pattern-specific output directory
        pattern_output_dir = os.path.join(output_dir, pattern_name)
        os.makedirs(pattern_output_dir, exist_ok=True)
        
        # Load pattern using PatternPathExtractor (like GUI does)
        try:
            extractor = PatternPathExtractor(pattern_path)
            panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
            
            if not panel_pieces:
                print(f"No pieces found in pattern {pattern_name}. Skipping.")
                continue
                
            pieces = {
                f"{piece.id}": copy.deepcopy(piece) for piece in panel_pieces.values()
            }
            
            # Add copies if needed
            # if config.NUM_COPIES > 0:
            #     for i in range(config.NUM_COPIES):
            #         for piece in panel_pieces.values():
            #             copy_piece = copy.deepcopy(piece)
            #             copy_piece.id = f"{piece.id}_copy{i+1}"
            #             pieces[copy_piece.id] = copy_piece

            
            #     print(f"Created {config.NUM_COPIES+1} copies, {len(pieces)} pieces in total")


            # Apply seam allowance and reset translations just like GUI
            for piece in pieces.values():
                piece.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
                piece.translation = (0, 0)
        except Exception as e:
            print(f"Error loading pattern {pattern_name}: {e}")
            traceback.print_exc()
            continue
        
        # Load design parameters
        design_params = None
        try:
            yaml_path = pattern_path.parent / f"{pattern_name}_design_params.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Design parameters file not found: {yaml_path}")
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            if "design" not in yaml_data:
                raise KeyError(f"No 'design' key in {yaml_path}")
            design_params = yaml_data["design"]
            print(f"Loaded design parameters from {yaml_path}")
            
        except Exception as e:
            print(f"Error loading design parameters: {e}")
            traceback.print_exc()
        
        # Load body parameters (like GUI does)
        body_params = None
        try:
            body_path = pattern_path.parent / f"{pattern_name}_body_measurements.yaml"
            if not body_path.exists():
                raise FileNotFoundError(f"Body parameters file not found: {body_path}")
            body_params = BodyParameters(body_path)
            print(f"Body parameters loaded from {body_path}")
        except Exception as e:
            print(f"Error loading body parameters: {e}")
            traceback.print_exc()
        
        # Create container
        container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)
        
        # Set up Evolution and run it
        start_time = time.time()
        try:
            print("Initializing Evolution...")
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
                design_params=design_params,
                body_params=body_params,
                pattern_name=pattern_name,
            )
            
            print("Running evolution...")
            best_chromosome = evo.run()
            if best_chromosome is None:
                print(f"No valid solution found for {pattern_name}")
                continue
                
            # Record elapsed time
            elapsed_time = time.time() - start_time
            print(f"Evolution completed in {elapsed_time:.2f} seconds")
            print(f"Best fitness: {best_chromosome.fitness:.4f}")
            
            # Save results
            result = {
                'pattern': pattern_name,
                'num_pieces': len(pieces),
                'best_fitness': best_chromosome.fitness,
                'generation_count': evo.generation,
                'elapsed_seconds': elapsed_time
            }
            all_results.append(result)
            
            # Save generation data
            gen_csv_path = os.path.join(pattern_output_dir, "generations.csv")
            with open(gen_csv_path, "w") as f:
                f.write("Generation,BestFitness,AvgChildFitness,DeltaBest,ImprovementFromInitial\n")
                # Handle the arrays more carefully to avoid index errors
                for g in range(1, evo.generation + 1):
                    best_fitness = evo.best_fitness_history[g] if g < len(evo.best_fitness_history) else "NA"
                    avg_fitness = evo.avg_child_fitnesses[g-1] if g-1 < len(evo.avg_child_fitnesses) else "NA"
                    delta = evo.delta_best[g] if g < len(evo.delta_best) else "NA"
                    improvement_from_initial = evo.improvement_from_initial[g] if g < len(evo.improvement_from_initial) else "NA"
                    f.write(f"{g},{best_fitness},{avg_fitness},{delta},{improvement_from_initial}\n")
            
            # Save log
            log_path = os.path.join(pattern_output_dir, "evolution_log.txt")
            with open(log_path, "w") as f:
                f.write("\n".join(evo.log_lines))
                
            # Update master statistics after each pattern
            try:
                print(f"Updating master statistics for {pattern_name}...")
                MetaStatistics.save_run_statistics(evo, elapsed_time)
            except Exception as e:
                print(f"Failed to update master statistics: {e}")
                traceback.print_exc()
            
            # Save fitness plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(evo.best_fitness_history)), 
                   [evo.best_fitness_history[g] for g in range(1, len(evo.best_fitness_history))], 
                   marker='o', label='Best Fitness')
            ax.set(xlabel='Generation', ylabel='Fitness', 
                   title=f'Fitness Evolution - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "fitness_history.png")
            
            # Save delta best plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(evo.delta_best)), evo.delta_best[1:], marker='o')
            ax.set(xlabel='Generation', ylabel='Δ-Best', 
                   title=f'Generation-to-Generation Improvement - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "delta_best.png")
            
            # Save improvement from initial plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(evo.improvement_from_initial)), evo.improvement_from_initial[1:], marker='o', color='green')
            ax.set(xlabel='Generation', ylabel='Improvement from Initial', 
                   title=f'Cumulative Improvement from Generation 0 - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "improvement_from_initial.png")
            
            print(f"Results saved to {pattern_output_dir}")
            
        except Exception as e:
            print(f"Error running evolution on {pattern_name}: {e}")
            traceback.print_exc()
            continue
    
    # Save summary of all patterns
    if all_results:
        summary_path = os.path.join(output_dir, "summary.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        
        # Create summary plot
        if len(all_results) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            patterns = [r['pattern'] for r in all_results]
            fitness = [r['best_fitness'] for r in all_results]
            ax.bar(patterns, fitness)
            ax.set(xlabel='Pattern', ylabel='Best Fitness', 
                   title='Fitness Comparison Across Patterns')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            _save_plot(fig, output_dir, "pattern_comparison.png")
    else:
        print("No successful results to save")
    # This content is now handled within run_ga_on_patterns
        
# -----------------------------------------------------------------------------
# Helper functions for running on multiple patterns
# -----------------------------------------------------------------------------
def run_ga_on_directory(directory_path, pattern_filter="*/*_specification.json", limit=None):
    """
    Run the genetic algorithm on all pattern files in a directory that match the filter.
    
    Args:
        directory_path: Path to the directory containing pattern files
        pattern_filter: Glob pattern to filter files (default: "*/*_specification.json")
        limit: Optional limit on number of patterns to process
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Directory {directory_path} does not exist or is not a directory.")
        return
    
    # Find all pattern files in the directory and subdirectories
    pattern_paths = list(directory.glob(pattern_filter))
    
    if not pattern_paths:
        print(f"No pattern files found in {directory_path} matching {pattern_filter}.")
        return
    
    # Sort for consistent order and limit if specified
    pattern_paths.sort()
    if limit and limit > 0:
        pattern_paths = pattern_paths[:limit]
        print(f"Found {len(pattern_paths)} pattern files in {directory_path} (limited to {limit})")
    else:
        print(f"Found {len(pattern_paths)} pattern files in {directory_path}")
    
    # Run GA on all pattern files
    run_ga_on_patterns(pattern_paths)

# -----------------------------------------------------------------------------
# CLI convenience
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run genetic algorithm for pattern nesting optimization")
    parser.add_argument("patterns", nargs="*", help="Paths to specific pattern files")
    parser.add_argument("--dir", help="Directory containing pattern files to process")
    parser.add_argument("--filter", default="*/*_specification.json", 
                       help="Filter pattern for files in directory (default: */*_specification.json)")
    parser.add_argument("--limit", type=int, 
                       help="Limit the number of pattern files to process")
    
    args = parser.parse_args()
    
    if args.dir:
        # Run on all patterns in directory
        run_ga_on_directory(args.dir, args.filter, args.limit)
    elif args.patterns:
        # Run on specific patterns provided
        pattern_paths = args.patterns
        if args.limit and args.limit > 0 and args.limit < len(pattern_paths):
            pattern_paths = pattern_paths[:args.limit]
            print(f"Running GA on {len(pattern_paths)} pattern(s) (limited from {len(args.patterns)})")
        else:
            print(f"Running GA on {len(pattern_paths)} pattern(s)")
        run_ga_on_patterns(pattern_paths)
    else:
        # Use the default pattern path
        print("No patterns specified, using default pattern")
        run_ga_on_patterns([config.DEFAULT_PATTERN_PATH])
    
    # Generate aggregate reports after all patterns have been processed
    try:
        print("\nGenerating aggregate statistics across all runs...")
        MetaStatistics.generate_aggregate_reports()
    except Exception as e:
        print(f"Failed to generate aggregate statistics: {e}")