"""
Pattern nesting optimization using genetic algorithm
–––––––––––––––––––––––––––
* Runs the nesting GA on multiple pattern files
* Uses the same approach as the GUI but in batch mode
* Generates CSV outputs and plots for each run
"""

from __future__ import annotations
import os, time, shutil, traceback, json, hashlib, uuid
import yaml
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import nesting.config as config

from nesting.path_extractor import PatternPathExtractor
from .evolution import Evolution
from .layout import Container, Piece, Layout
from assets.bodies.body_params import BodyParameters
from nesting.metastatistics import MetaStatistics
from .pattern_loader import load_pattern_bundle

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _save_plot(fig, out_dir: str, fname: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)

def _serialize_config(config_obj):
    """
    Serialize the config object, filtering out non-serializable attributes.
    """
    def default_serializer(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)  # Fallback to string representation

    return {
        k: default_serializer(v) for k, v in vars(config_obj).items()
    }

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def run_ga_on_patterns(pattern_paths, output_dir="results") -> None:
    os.makedirs(output_dir, exist_ok=True)

    run_tag = uuid.uuid4().hex
    config_hash = hashlib.md5(
        json.dumps(_serialize_config(config), sort_keys=True).encode("utf-8")
    ).hexdigest()

    all_results = []

    for pattern_path in pattern_paths:
        pattern_path = Path(pattern_path)
        try:
            pieces, design_params, body_params, pattern_name = load_pattern_bundle(pattern_path)
        except Exception as e:
            print(f"Failed loading pattern {pattern_path}: {e}")
            traceback.print_exc()
            continue

        print(f"\n{'='*60}\nProcessing pattern: {pattern_name}\n{'='*60}")
        pattern_output_dir = os.path.join(output_dir, pattern_name, run_tag)
        os.makedirs(pattern_output_dir, exist_ok=True)

        container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)

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

            elapsed_time = time.time() - start_time
            print(f"Evolution completed in {elapsed_time:.2f} seconds")
            print(f"Best fitness: {best_chromosome.fitness:.4f}")

            result = {
                'pattern': pattern_name,
                'num_pieces': len(pieces),
                'best_fitness': best_chromosome.fitness,
                'generation_count': evo.generation,
                'elapsed_seconds': elapsed_time
            }
            all_results.append(result)

            gen_csv_path = os.path.join(pattern_output_dir, "generations.csv")
            with open(gen_csv_path, "w") as f:
                f.write("Generation,BestFitness,AvgChildFitness,DeltaBest,ImprovementFromInitial\n")
                for g in range(1, evo.generation + 1):
                    best_fitness = evo.best_fitness_history[g] if g < len(evo.best_fitness_history) else "NA"
                    avg_fitness = evo.avg_child_fitnesses[g-1] if g-1 < len(evo.avg_child_fitnesses) else "NA"
                    delta = evo.delta_best[g] if g < len(evo.delta_best) else "NA"
                    improvement_from_initial = evo.improvement_from_initial[g] if g < len(evo.improvement_from_initial) else "NA"
                    f.write(f"{g},{best_fitness},{avg_fitness},{delta},{improvement_from_initial}\n")

            log_path = os.path.join(pattern_output_dir, "evolution_log.txt")
            with open(log_path, "w") as f:
                f.write("\n".join(evo.log_lines))

            try:
                print(f"Updating master statistics for {pattern_name}...")
                MetaStatistics.save_run_statistics(
                    evo, elapsed_time, run_tag=run_tag, config_hash=config_hash
                )
            except Exception as e:
                print(f"Failed to update master statistics: {e}")
                traceback.print_exc()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(evo.best_fitness_history)), [evo.best_fitness_history[g] for g in range(1, len(evo.best_fitness_history))], marker='o', label='Best Fitness')
            ax.set(xlabel='Generation', ylabel='Fitness', title=f'Fitness Evolution - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "fitness_history.png")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(evo.delta_best)), evo.delta_best[1:], marker='o')
            ax.set(xlabel='Generation', ylabel='Δ-Best', title=f'Generation-to-Generation Improvement - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "delta_best.png")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(evo.improvement_from_initial)), evo.improvement_from_initial[1:], marker='o', color='green')
            ax.set(xlabel='Generation', ylabel='Improvement from Initial', title=f'Cumulative Improvement from Generation 0 - {pattern_name}')
            ax.grid(True)
            _save_plot(fig, pattern_output_dir, "improvement_from_initial.png")

            print(f"Results saved to {pattern_output_dir}")
        except Exception as e:
            print(f"Error running evolution on {pattern_name}: {e}")
            traceback.print_exc()
            continue

    if all_results:
        summary_path = os.path.join(output_dir, "summary.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        if len(all_results) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            patterns = [r['pattern'] for r in all_results]
            fitness = [r['best_fitness'] for r in all_results]
            ax.bar(patterns, fitness)
            ax.set(xlabel='Pattern', ylabel='Best Fitness', title='Fitness Comparison Across Patterns')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            _save_plot(fig, output_dir, "pattern_comparison.png")
    else:
        print("No successful results to save")

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
        raise ValueError("No pattern entered. Please enter a pattern.")
    
    # Generate aggregate reports after all patterns have been processed
    try:
        print("\nGenerating aggregate statistics across all runs...")
        MetaStatistics.generate_aggregate_reports()
    except Exception as e:
        print(f"Failed to generate aggregate statistics: {e}")