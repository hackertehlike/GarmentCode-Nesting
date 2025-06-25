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
import yaml
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
from pygarment.garmentcode.params import DesignSampler
from assets.bodies.body_params import BodyParameters


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

    # 4. Average fitness by chromosome origin
    if {'generation', 'avg_off', 'avg_mut', 'avg_rand'}.issubset(df.columns):
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(df['generation'], df['avg_off'], label='offspring', marker='o')
        ax4.plot(df['generation'], df['avg_mut'], label='mutants', marker='o')
        ax4.plot(df['generation'], df['avg_rand'], label='random', marker='o')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Average Fitness')
        ax4.set_title('Average Fitness by Origin')
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend()
        path4 = plots_dir / f"ga_origin_fitness_{timestamp}.png"
        fig4.tight_layout()
        fig4.savefig(path4, dpi=300)
        plt.close(fig4)

    # 5. Mutation efficiency by type
    mut_cols = [c for c in df.columns if c.startswith('mut_gain_')]
    if mut_cols:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        for col in mut_cols:
            ax5.plot(df['generation'], df[col], label=col.replace('mut_gain_', ''))
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Δ Fitness vs parent')
        ax5.set_title('Mutation Efficiency by Type')
        ax5.grid(True, linestyle='--', alpha=0.6)
        ax5.legend()
        path5 = plots_dir / f"ga_mutation_eff_{timestamp}.png"
        fig5.tight_layout()
        fig5.savefig(path5, dpi=300)
        plt.close(fig5)
        
    print(f"Results saved to {csv_path} and plots saved to {plots_dir}")


def flush_meta_stats(rows: list[dict], design_changes: list[dict], timestamp: str) -> None:
    """Generate meta statistics across all runs and save to disk."""
    results_dir = Path("./ga_benchmark_results") / "meta"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    changes_df = pd.DataFrame(design_changes)

    # ---- FITNESS Δ BY MUTATION TYPE --------------------------------
    gain_cols = [c for c in df.columns if c.startswith("mut_gain_")]
    if gain_cols:
        # Extract pattern information for plotting
        gains_long = (
            df[gain_cols + ["generation", "pattern"]]
            .melt(id_vars=["generation", "pattern"],
                  var_name="mutation", value_name="gain")
            .dropna()
        )
        gains_long["mutation"] = gains_long["mutation"].str.replace("mut_gain_", "")

        # save raw table
        gains_long.to_csv(results_dir / f"mut_fitness_gains_{timestamp}.csv",
                          index=False)

        # Plot 1: Create enhanced box+swarm plot combination
        plt.figure(figsize=(12, 7))
        # Create box plot for statistical summary
        ax = sns.boxplot(
            data=gains_long, 
            x="mutation", 
            y="gain",
            color="lightgrey",
            fliersize=0  # Don't show outliers in the box plot
        )
        
        # Add swarm plot on top to show individual data points
        sns.swarmplot(
            data=gains_long, 
            x="mutation", 
            y="gain", 
            hue="pattern",  # Color by pattern instead of generation
            dodge=True,     # Separate points by hue category
            size=4,
            alpha=0.7,
            ax=ax
        )
        
        plt.axhline(0, ls="--", c="r", lw=0.8, alpha=0.5)
        plt.title("Fitness Change (Δ) by Mutation Type Across Patterns")
        plt.ylabel("Δ Fitness vs Parent")
        plt.xlabel("Mutation Type")
        plt.xticks(rotation=45)
        plt.legend(title="Pattern", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(results_dir / f"mut_fitness_gains_swarm_{timestamp}.png")
        plt.close()
        
        # Plot 2: Create enhanced violin plot
        plt.figure(figsize=(12, 7))
        sns.violinplot(
            data=gains_long, 
            x="mutation", 
            y="gain", 
            inner="stick",  # Show individual data points as sticks inside the violin
            palette="Set3",
            cut=0  # Don't extend the violin past the observed data points
        )
        plt.axhline(0, ls="--", c="r", lw=0.8, alpha=0.5)
        plt.title("Fitness Change (Δ) by Mutation Type Across Patterns")
        plt.ylabel("Δ Fitness vs Parent")
        plt.xlabel("Mutation Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / f"mut_fitness_gains_violin_{timestamp}.png")
        plt.close()

        # summary stats
        summary = (
            gains_long.groupby(["mutation", "pattern"])["gain"]
            .agg(["mean", "std", "count", "min", "max"])
            .reset_index()
        )
        summary.to_csv(results_dir / f"mut_gain_summary_{timestamp}.csv",
                       index=False)
        
        # Also create overall summary by mutation type
        overall_summary = (
            gains_long.groupby("mutation")["gain"]
            .agg(["mean", "std", "count", "min", "max"])
            .reset_index()
        )
        overall_summary.to_csv(results_dir / f"mut_gain_overall_summary_{timestamp}.csv",
                       index=False)

    if df.empty:
        return

    # improvement per run
    improvements = []
    for pattern, grp in df.groupby("pattern"):
        grp = grp.sort_values("generation")
        if grp.empty:
            continue
        first = grp.iloc[0]["best_fitness"]
        last = grp.iloc[-1]["best_fitness"]
        improvements.append({"pattern": pattern, "improvement": last - first})

    imp_df = pd.DataFrame(improvements)
    imp_df.to_csv(results_dir / f"run_improvements_{timestamp}.csv", index=False)

    # best origin counts
    if "best_origin" in df.columns:
        counts = df["best_origin"].value_counts().reset_index()
        counts.columns = ["origin", "count"]
        counts.to_csv(results_dir / f"best_origin_counts_{timestamp}.csv", index=False)

        plt.figure(figsize=(6, 4))
        sns.barplot(data=counts, x="origin", y="count")
        plt.title("Best chromosome origin frequency")
        plt.tight_layout()
        plt.savefig(results_dir / f"best_origin_counts_{timestamp}.png")
        plt.close()

    # avg best fitness across runs
    avg_gen = df.groupby("generation")["best_fitness"].mean().reset_index()
    avg_gen.to_csv(results_dir / f"avg_best_fitness_{timestamp}.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(avg_gen["generation"], avg_gen["best_fitness"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Average best fitness")
    plt.title("Average best fitness across runs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / f"avg_best_fitness_{timestamp}.png")
    plt.close()



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
        left, right = piece.split()
        new_pieces[left.id] = left
        new_pieces[right.id] = right
    return new_pieces


def load_design_params(json_path: Path) -> tuple:
    """Load design and body parameters for a pattern."""
    print(f"Attempting to load design params from: {json_path}")
    
    # Method 1: Using directory name as pattern ID
    pattern_id = json_path.parent.name
    design_yaml = json_path.parent / f"{pattern_id}_design_params.yaml"
    print(f"Looking for design params at: {design_yaml} (exists: {design_yaml.exists()})")
    
    design_params = {}
    if design_yaml.exists():
        with open(design_yaml, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            if not yaml_data:
                print(f"WARNING: Empty YAML file: {design_yaml}")
            else:
                # Some files might have design params at the top level, others under 'design'
                if "design" in yaml_data:
                    design_params = yaml_data.get("design", {})
                else:
                    design_params = yaml_data
                print(f"Successfully loaded design params with {len(design_params)} top-level keys")
    else:
        print(f"WARNING: No design params found for {json_path}")
    
    # Similar approach for body measurements
    body_yaml = json_path.parent / f"{pattern_id}_body_measurements.yaml"
    print(f"Looking for body params at: {body_yaml} (exists: {body_yaml.exists()})")
    
    if not body_yaml.exists():
        from nesting import config
        body_yaml = Path(config.DEFAULT_BODY_PARAM_PATH)
        print(f"Using default body params: {body_yaml}")
    
    # Create the design sampler with the file path, not the dictionary
    design_sampler = DesignSampler(str(design_yaml)) if design_yaml.exists() else None
    body_params = BodyParameters(body_yaml) if body_yaml.exists() else None
    
    return design_params, body_params, design_sampler

def run_ga_with_tracking(pieces: dict[str, Piece], container: Container, pattern_name: str, json_path: Path) -> tuple[list[dict], list[dict]]:
    """Run GA on a pattern and collect per generation metrics.

    Returns
    -------
    results : list[dict]
        Per-generation metrics for this GA run.
    design_changes : list[dict]
        Records of design parameter mutations that actually changed a value.
    """
    start_time = time.time()
    results: list[dict] = []
    design_changes: list[dict] = []

    design_params, body_params, design_sampler = load_design_params(json_path)

    # Debug info
    print(f"Design params loaded: {bool(design_params)}")
    print(f"Body params loaded: {body_params is not None}")
    print(f"Design sampler loaded: {design_sampler is not None}")

    
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
        design_params=design_params,
        body_params=body_params,
        design_sampler=design_sampler
    )

    print(f"Generating initial population for {pattern_name}")
    evo.generate_population()

    # metrics for generation 0
    row = evo._all_metrics[-1]
    best_chrom = max(evo.population, key=lambda c: c.fitness)
    view = LayoutView(best_chrom.genes)
    decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
    decoder.decode()

    entry = {
        "pattern": pattern_name,
        "generation": int(row.get("generation", 0)),
        "best_fitness": row.get("best_fit", best_chrom.fitness),
        "best_origin": best_chrom.origin,
        "avg_fitness": row.get("avg_child_fitness", 0.0),
        "usage_bb": decoder.usage_BB(),
        "concave_hull": decoder.concave_hull_utilization(),
        "rest_length": decoder.rest_length(),
        "avg_off": row.get("avg_off", 0.0),
        "avg_mut": row.get("avg_mut", 0.0),
        "avg_rand": row.get("avg_rand", 0.0),
        "mean_offspring_gain": row.get("mean_offspring_gain", 0.0),
        "mean_mutant_gain": row.get("mean_mutant_gain", 0.0),
    }
    for k, v in row.items():
        if k.startswith("mut_gain_"):
            entry[k] = v
    results.append(entry)

    for gen in range(1, config.NUM_GENERATIONS + 1):
        print(f"Running generation {gen}/{config.NUM_GENERATIONS} for {pattern_name}")
        evo.next_generation()

        row = evo._all_metrics[-1]
        best_chrom = max(evo.population, key=lambda c: c.fitness)
        view = LayoutView(best_chrom.genes)
        decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
        decoder.decode()

        entry = {
            "pattern": pattern_name,
            "generation": int(row.get("generation", gen)),
            "best_fitness": row.get("best_fit", best_chrom.fitness),
            "best_origin": best_chrom.origin,
            "avg_fitness": row.get("avg_child_fitness", 0.0),
            "usage_bb": decoder.usage_BB(),
            "concave_hull": decoder.concave_hull_utilization(),
            "rest_length": decoder.rest_length(),
            "avg_off": row.get("avg_off", 0.0),
            "avg_mut": row.get("avg_mut", 0.0),
            "avg_rand": row.get("avg_rand", 0.0),
            "mean_offspring_gain": row.get("mean_offspring_gain", 0.0),
            "mean_mutant_gain": row.get("mean_mutant_gain", 0.0),
        }
        for k, v in row.items():
            if k.startswith("mut_gain_"):
                entry[k] = v
        results.append(entry)

        if evo.early_stopped:
            print(f"Early stopping triggered at generation {gen} for {pattern_name}")
            break

    print(f"GA completed for {pattern_name} in {time.time() - start_time:.2f} seconds")

    if config.SAVE_LOGS:
        evo.update_plots()
        evo.output_best_result()

    # collect design parameter changes
    if hasattr(evo, "design_param_changes"):
        for dc in evo.design_param_changes:
            dc["pattern"] = pattern_name
        design_changes.extend(evo.design_param_changes)

    return results, design_changes


def main():
    data_dir = Path("./nesting-assets/pattern_files")
    json_files = sorted(data_dir.glob("*/*_specification.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}.")
        return

    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting GA benchmark run at {timestamp}")
    
    # Create container with dimensions from config
    container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)
    all_results: list[dict] = []
    all_changes: list[dict] = []

    test_files = json_files
    
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
        pattern_name = json_path.stem.replace('_specification', '')
        # print(f"\n{'='*50}")
        # print(f"Processing pattern: {pattern_name}")
        # print(f"{'='*50}")
        
        # Load pieces
        pieces = load_pieces(json_path)
        print(f"Loaded {len(pieces)} pieces from {pattern_name}")
        
        results, changes = run_ga_with_tracking(
            copy.deepcopy(pieces),
            container,
            pattern_name,
            json_path,
        )
        all_results.extend(results)
        all_changes.extend(changes)

        flush_results(all_results, timestamp)
        flush_meta_stats(all_results, all_changes, timestamp)

    # Final save of all results
    flush_results(all_results, timestamp)
    flush_meta_stats(all_results, all_changes, timestamp)
    print(f"\nGA benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total benchmarks: {len(test_files)} patterns processed")


if __name__ == "__main__":
    main()
