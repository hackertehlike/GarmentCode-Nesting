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
        pattern_name = pattern_path.stem
        
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
        except Exception as e:
            print(f"Error loading pattern {pattern_name}: {e}")
            traceback.print_exc()
            continue
        
        # Load design parameters
        design_params = None
        try:
            # Check for YAML file with design parameters
            yaml_path = pattern_path.parent / "design_params.yaml"
            if yaml_path.exists():
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                    if "design" in yaml_data:
                        design_params = yaml_data["design"]
                        print(f"Loaded design parameters from {yaml_path}")
                    else:
                        print(f"Warning: No 'design' key in {yaml_path}")
            else:
                # Check for design params in the pattern file itself
                with pattern_path.open("r", encoding="utf-8") as f:
                    pattern_data = json.load(f)
                    if "design" in pattern_data:
                        design_params = pattern_data["design"]
                        print("Design parameters loaded from pattern JSON")
                    else:
                        # Use default design params if available
                        default_params_path = Path(config.DEFAULT_DESIGN_PARAM_PATH)
                        if default_params_path.exists():
                            with open(default_params_path, "r", encoding="utf-8") as f:
                                yaml_data = yaml.safe_load(f)
                                if "design" in yaml_data:
                                    design_params = yaml_data["design"]
                                    print(f"Using default design parameters")
                                else:
                                    print(f"Warning: No 'design' key in default params")
        except Exception as e:
            print(f"Error loading design parameters: {e}")
            traceback.print_exc()
        
        # Load body parameters (like GUI does)
        body_params = None
        try:
            # Check for body parameters
            body_path = pattern_path.parent / "body_measurements.yaml"
            if body_path.exists():
                body_params = BodyParameters(body_path)
                print("Body parameters loaded from pattern folder")
            else:
                # Use default body params if available
                default_body_path = Path(config.DEFAULT_BODY_PARAM_PATH)
                if default_body_path.exists():
                    body_params = BodyParameters(default_body_path)
                    print("Default body parameters loaded")
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
                f.write("Generation,BestFitness,AvgChildFitness,DeltaBest\n")
                for g in range(1, evo.generation + 1):
                    f.write(f"{g},{evo.best_fitness_history[g]},{evo.avg_child_fitnesses[g-1]},{evo.delta_best[g]}\n")
            
            # Save log
            log_path = os.path.join(pattern_output_dir, "evolution_log.txt")
            with open(log_path, "w") as f:
                f.write("\n".join(evo.log_lines))
            
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
                   title=f'Fitness Improvement - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "delta_best.png")
            
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
# CLI convenience
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if specific patterns were provided as arguments
    if len(sys.argv) > 1:
        # Use the provided pattern paths
        pattern_paths = sys.argv[1:]
    else:
        # Use the default pattern path
        pattern_paths = [config.DEFAULT_PATTERN_PATH]
    
    print(f"Running GA on {len(pattern_paths)} pattern(s)")
    # Run the GA on each pattern
    run_ga_on_patterns(pattern_paths)