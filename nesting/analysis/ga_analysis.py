#!/usr/bin/env python3
"""
GA Analysis Tool - Efficient Config Filtering
=============================================

Analyzes GA experimental data from:
1. experiments/runs/*.csv - Population data per run (config_hash, run_tag, pattern, generation, mutations, fitness)  
2. experiments/aggregate/final_metrics.csv - Best results per run (run summaries)

Features:
- Efficient CSV filtering by config hash (filename-based pre-filtering)
- Mutation effectiveness analysis by type, generation phase, and pattern
- Configuration comparison across runs
- Pattern performance analysis  
- Temporal evolution analysis
- Visualizations

Usage:
    python ga_analysis.py [--runs-dir experiments/runs] [--aggregate-file experiments/aggregate/final_metrics.csv] [--output-dir analysis_output] [--config-hash HASH]
"""

from __future__ import annotations
import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

class GAAnalyzer:
    """Complete GA run analysis tool with efficient config filtering"""
    
    def __init__(self, runs_dir: str = "experiments/runs", aggregate_file: str = "experiments/aggregate/final_metrics.csv", config_hash: Optional[str] = None):
        self.runs_dir = Path(runs_dir)
        self.aggregate_file = Path(aggregate_file)
        self.config_hash = config_hash
        self.population_data: pd.DataFrame = pd.DataFrame()
        self.final_metrics: pd.DataFrame = pd.DataFrame()
        self.mutation_analysis: pd.DataFrame = pd.DataFrame()
        
    def _filter_csv_files_by_config(self) -> List[Path]:
        
        """Pre-filter CSV files by config hash from filename"""
        csv_files = list(self.runs_dir.glob("*.csv"))
        
        if not self.config_hash:
            return csv_files
            
        # Filter by filename pattern: {config_hash}_{pattern}_{timestamp}.csv
        filtered_files = []
        for csv_file in csv_files:
            filename = csv_file.stem  # Remove .csv extension
            if filename.startswith(self.config_hash):
                filtered_files.append(csv_file)
                
        print(f"Filtered {len(csv_files)} files down to {len(filtered_files)} files for config {self.config_hash}")
        return filtered_files
        
    def load_data(self) -> None:
        """Load population data and final metrics with config filtering"""
        print("Loading population data from runs...")
        self._load_population_data()
        print(f"Loaded {len(self.population_data)} population records from {self.population_data['run_tag'].nunique()} runs")
        
        print("Loading final metrics...")
        self._load_final_metrics()
        print(f"Loaded final metrics for {len(self.final_metrics)} runs")
        
        print("Computing mutation analysis...")
        self._compute_mutation_analysis()
        print(f"Computed mutation stats for {len(self.mutation_analysis)} mutation records")
        
    def _load_population_data(self) -> None:
        """Load and combine filtered run CSV files"""
        csv_files = self._filter_csv_files_by_config()
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.runs_dir}" + 
                                  (f" for config {self.config_hash}" if self.config_hash else ""))
            
        dfs = []
        for csv_file in csv_files:
            try:
                # Quick peek at first few lines to validate structure
                sample = pd.read_csv(csv_file, nrows=1)
                
                # Handle different CSV formats (some have last_mutation, pre_mutation_fitness columns)
                required_cols = ["config_hash", "run_tag", "pattern_name", "generation", "origin", "self_fitness"]
                if not all(col in sample.columns for col in required_cols):
                    print(f"Skipping {csv_file.name} - missing required columns: {set(required_cols) - set(sample.columns)}")
                    continue
                
                # If config_hash filtering, double-check the content matches filename
                if self.config_hash:
                    file_config = sample["config_hash"].iloc[0] if len(sample) > 0 else ""
                    if str(file_config) != str(self.config_hash):
                        print(f"Skipping {csv_file.name} - config hash mismatch: file has {file_config}, expected {self.config_hash}")
                        continue
                
                # Load full file
                df = pd.read_csv(csv_file)
                    
                # Standardize column types
                df["generation"] = pd.to_numeric(df["generation"], errors="coerce").fillna(0).astype(int)
                df["self_fitness"] = pd.to_numeric(df["self_fitness"], errors="coerce")
                df["origin"] = df["origin"].astype(str).str.strip().str.lower()
                
                # Handle parent fitness columns
                for col in ["parent1_fitness", "parent2_fitness"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Handle mutation columns
                if "last_mutation" in df.columns:
                    df["last_mutation"] = df["last_mutation"].astype(str).str.strip()
                    df["last_mutation"] = df["last_mutation"].replace("nan", np.nan).replace("", np.nan)
                else:
                    df["last_mutation"] = np.nan
                    
                if "pre_mutation_fitness" in df.columns:
                    df["pre_mutation_fitness"] = pd.to_numeric(df["pre_mutation_fitness"], errors="coerce")
                
                # Additional config filtering if needed
                if self.config_hash:
                    df = df[df["config_hash"].astype(str) == str(self.config_hash)]
                    if df.empty:
                        continue
                
                dfs.append(df)
                print(f"Loaded {csv_file.name}: {len(df)} records")
                
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue
                
        if not dfs:
            raise ValueError("No valid CSV files could be loaded")
            
        self.population_data = pd.concat(dfs, ignore_index=True)
        
        # Clean and standardize
        self.population_data = self.population_data.dropna(subset=["self_fitness"])
        self.population_data["config_hash"] = self.population_data["config_hash"].astype(str)
        self.population_data["run_tag"] = self.population_data["run_tag"].astype(str) 
        self.population_data["pattern_name"] = self.population_data["pattern_name"].astype(str)
        
    def _load_final_metrics(self) -> None:
        """Load final metrics filtered by config hash"""
        if not self.aggregate_file.exists():
            print(f"Warning: {self.aggregate_file} not found, skipping final metrics")
            return
            
        try:
            final_metrics = pd.read_csv(self.aggregate_file)
            
            # Clean empty rows and standardize
            final_metrics = final_metrics.dropna(subset=["config_hash", "run_tag", "pattern_name"], how="all")
            final_metrics["config_hash"] = final_metrics["config_hash"].astype(str)
            final_metrics["run_tag"] = final_metrics["run_tag"].astype(str)
            
            # Filter by config hash if specified
            if self.config_hash:
                final_metrics = final_metrics[final_metrics["config_hash"] == str(self.config_hash)]
            
            self.final_metrics = final_metrics
            
        except Exception as e:
            print(f"Error loading final metrics: {e}")
            self.final_metrics = pd.DataFrame()
            
    def _compute_mutation_analysis(self) -> None:
        """Compute mutation effectiveness from population data"""
        # Filter for mutations only (mutant/offspring with valid mutation type)
        mutation_data = self.population_data[
            (self.population_data["origin"].isin(["mutant", "offspring"])) &
            (self.population_data["last_mutation"].notna()) &
            (self.population_data["last_mutation"] != "")
        ].copy()
        
        if mutation_data.empty:
            print("No mutation data found")
            return
            
        # Compute parent fitness and fitness gain
        def compute_parent_fitness(row):
            if row["origin"] == "mutant":
                return row.get("parent1_fitness", np.nan)
            elif row["origin"] == "offspring":
                p1 = row.get("parent1_fitness", np.nan) 
                p2 = row.get("parent2_fitness", np.nan)
                if pd.notna(p1) and pd.notna(p2):
                    return (p1 + p2) / 2
                return p1 if pd.notna(p1) else p2
            return np.nan
            
        mutation_data["parent_fitness"] = mutation_data.apply(compute_parent_fitness, axis=1)
        
        # Use pre_mutation_fitness if available, otherwise parent_fitness
        if "pre_mutation_fitness" in mutation_data.columns:
            mutation_data["baseline_fitness"] = mutation_data["pre_mutation_fitness"].fillna(mutation_data["parent_fitness"])
        else:
            mutation_data["baseline_fitness"] = mutation_data["parent_fitness"]
            
        mutation_data["fitness_gain"] = mutation_data["self_fitness"] - mutation_data["baseline_fitness"]
        mutation_data["improved"] = mutation_data["fitness_gain"] > 0
        
        # Add generation phase
        def get_phase(gen):
            if gen <= 5:
                return "Early (1-5)"
            elif gen <= 10:
                return "Middle (6-10)" 
            else:
                return "Late (10+)"
                
        mutation_data["generation_phase"] = mutation_data["generation"].apply(get_phase)
        
        self.mutation_analysis = mutation_data
        
    def get_run_summary(self) -> pd.DataFrame:
        """Get summary statistics per run"""
        if self.population_data.empty:
            return pd.DataFrame()
            
        # Basic stats from population data
        pop_summary = (
            self.population_data.groupby(["config_hash", "run_tag", "pattern_name"])
            .agg({
                "generation": "max",
                "self_fitness": ["min", "max", "mean", "std"],
            })
            .round(6)
        )
        
        # Flatten columns
        pop_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pop_summary.columns]
        pop_summary = pop_summary.reset_index()
        
        # Merge with final metrics if available
        if not self.final_metrics.empty:
            summary = pop_summary.merge(
                self.final_metrics, 
                on=["config_hash", "run_tag", "pattern_name"], 
                how="left"
            )
        else:
            summary = pop_summary
            
        return summary
        
    def analyze_mutations_by_type(self) -> pd.DataFrame:
        """Analyze mutation effectiveness by type"""
        if self.mutation_analysis.empty:
            return pd.DataFrame()
            
        # Per-run mutation stats
        per_run_stats = (
            self.mutation_analysis
            .groupby(["config_hash", "run_tag", "pattern_name", "last_mutation"])
            .agg({
                "improved": ["sum", "count"],
                "fitness_gain": ["mean", "std", "median"],
            })
        )
        
        per_run_stats.columns = ["improvements", "total_mutations", "mean_gain", "std_gain", "median_gain"]
        per_run_stats["improvement_rate"] = (per_run_stats["improvements"] / per_run_stats["total_mutations"] * 100).round(2)
        per_run_stats = per_run_stats.reset_index()
        
        # Aggregate across runs
        mutation_summary = (
            per_run_stats
            .groupby("last_mutation")
            .agg({
                "improvement_rate": ["mean", "std", "count"],
                "mean_gain": ["mean", "std"],
                "total_mutations": ["mean", "std", "sum"],
            })
            .round(4)
        )
        
        # Flatten columns
        mutation_summary.columns = [f"{col[0]}_{col[1]}" for col in mutation_summary.columns]
        mutation_summary = mutation_summary.reset_index()
        mutation_summary = mutation_summary.rename(columns={"last_mutation": "mutation_type"})
        
        return mutation_summary.sort_values("improvement_rate_mean", ascending=False)
        
    def analyze_mutations_by_phase(self) -> pd.DataFrame:
        """Analyze mutation effectiveness by generation phase"""
        if self.mutation_analysis.empty:
            return pd.DataFrame()
            
        # Per-run temporal stats
        per_run_temporal = (
            self.mutation_analysis
            .groupby(["config_hash", "run_tag", "pattern_name", "last_mutation", "generation_phase"])
            .agg({
                "improved": ["sum", "count"],
                "fitness_gain": ["mean", "std"],
            })
        )
        
        per_run_temporal.columns = ["improvements", "total_mutations", "mean_gain", "std_gain"]
        per_run_temporal["improvement_rate"] = (per_run_temporal["improvements"] / per_run_temporal["total_mutations"] * 100).round(2)
        per_run_temporal = per_run_temporal.reset_index()
        
        # Aggregate across runs
        temporal_summary = (
            per_run_temporal
            .groupby(["last_mutation", "generation_phase"])
            .agg({
                "improvement_rate": ["mean", "std", "count"],
                "mean_gain": ["mean", "std"],
                "total_mutations": ["mean", "sum"],
            })
            .round(4)
        )
        
        temporal_summary.columns = [f"{col[0]}_{col[1]}" for col in temporal_summary.columns]
        temporal_summary = temporal_summary.reset_index()
        temporal_summary = temporal_summary.rename(columns={"last_mutation": "mutation_type"})
        
        return temporal_summary
        
    def analyze_mutations_by_pattern(self) -> pd.DataFrame:
        """Analyze mutation effectiveness by pattern"""
        if self.mutation_analysis.empty:
            return pd.DataFrame()
            
        # Per-run pattern stats  
        per_run_pattern = (
            self.mutation_analysis
            .groupby(["config_hash", "run_tag", "pattern_name", "last_mutation"])
            .agg({
                "improved": ["sum", "count"],
                "fitness_gain": ["mean", "std"],
            })
        )
        
        per_run_pattern.columns = ["improvements", "total_mutations", "mean_gain", "std_gain"]
        per_run_pattern["improvement_rate"] = (per_run_pattern["improvements"] / per_run_pattern["total_mutations"] * 100).round(2)
        per_run_pattern = per_run_pattern.reset_index()
        
        # Aggregate across runs
        pattern_summary = (
            per_run_pattern
            .groupby(["pattern_name", "last_mutation"])
            .agg({
                "improvement_rate": ["mean", "std", "count"],
                "mean_gain": ["mean", "std"],
                "total_mutations": ["mean", "sum"],
            })
            .round(4)
        )
        
        pattern_summary.columns = [f"{col[0]}_{col[1]}" for col in pattern_summary.columns]
        pattern_summary = pattern_summary.reset_index()
        pattern_summary = pattern_summary.rename(columns={"last_mutation": "mutation_type"})
        
        return pattern_summary
        
    def compare_configurations(self) -> pd.DataFrame:
        """Compare performance across configurations (or show single config stats)"""
        summary = self.get_run_summary()
        if summary.empty:
            return pd.DataFrame()
        
        # If single config, show run-level stats instead of config-level
        if self.config_hash:
            return summary[["run_tag", "pattern_name", "generation_max", "self_fitness_max", "self_fitness_mean"] + 
                          [col for col in ["rest_length_cm", "usage_bb", "concave_hull_utilization"] if col in summary.columns]]
            
        # Multiple configs - group by config_hash
        config_comparison = (
            summary.groupby("config_hash")
            .agg({
                "generation_max": ["mean", "std", "count"],
                "self_fitness_max": ["mean", "std"],
                "self_fitness_mean": ["mean", "std"],
                # Include final metrics if available
                **{col: ["mean", "std"] for col in ["rest_length_cm", "usage_bb", "concave_hull_utilization"] 
                   if col in summary.columns}
            })
            .round(6)
        )
        
        config_comparison.columns = [f"{col[0]}_{col[1]}" for col in config_comparison.columns]
        config_comparison = config_comparison.reset_index()
        
        return config_comparison.sort_values("self_fitness_max_mean", ascending=False)
        
    def plot_mutation_effectiveness(self, output_dir: Path) -> None:
        """Create comprehensive mutation effectiveness plots"""
        if self.mutation_analysis.empty:
            print("No mutation data to plot")
            return
            
        # 1. Overall mutation effectiveness
        mut_stats = self.analyze_mutations_by_type()
        if not mut_stats.empty:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            title = f"Mutation Effectiveness Analysis"
            if self.config_hash:
                title += f" (Config: {self.config_hash[:8]})"
            fig.suptitle(title, fontsize=16, fontweight="bold")
            
            # Improvement rate
            mut_stats_sorted = mut_stats.sort_values("improvement_rate_mean")
            bars1 = ax1.barh(mut_stats_sorted["mutation_type"], mut_stats_sorted["improvement_rate_mean"], 
                            xerr=mut_stats_sorted["improvement_rate_std"], capsize=5)
            ax1.set_xlabel("Improvement Rate (%)")
            ax1.set_title("Improvement Rate by Mutation Type")
            ax1.grid(axis="x", alpha=0.3)
            
            # Mean fitness gain
            bars2 = ax2.barh(mut_stats_sorted["mutation_type"], mut_stats_sorted["mean_gain_mean"],
                            xerr=mut_stats_sorted["mean_gain_std"], capsize=5)
            ax2.set_xlabel("Mean Fitness Gain")
            ax2.set_title("Average Fitness Gain by Mutation Type")
            ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)
            ax2.grid(axis="x", alpha=0.3)
            
            # Sample size
            ax3.barh(mut_stats_sorted["mutation_type"], mut_stats_sorted["total_mutations_sum"])
            ax3.set_xlabel("Total Mutations Across All Runs")
            ax3.set_title("Sample Size by Mutation Type")
            ax3.grid(axis="x", alpha=0.3)
            
            # Scatter plot
            ax4.scatter(mut_stats["improvement_rate_mean"], mut_stats["mean_gain_mean"], 
                       s=mut_stats["total_mutations_sum"]/20, alpha=0.6)
            for _, row in mut_stats.iterrows():
                ax4.annotate(row["mutation_type"], 
                           (row["improvement_rate_mean"], row["mean_gain_mean"]),
                           xytext=(5, 5), textcoords="offset points", fontsize=9)
            ax4.set_xlabel("Improvement Rate (%)")
            ax4.set_ylabel("Mean Fitness Gain")
            ax4.set_title("Rate vs Gain (bubble size = total mutations)")
            ax4.grid(alpha=0.3)
            ax4.axhline(y=0, color="red", linestyle="--", alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_dir / "mutation_effectiveness_overview.png", dpi=300, bbox_inches="tight")
            plt.close()
            
        # 2. Temporal analysis
        temporal_stats = self.analyze_mutations_by_phase()
        if not temporal_stats.empty:
            # Create heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            title = f"Mutation Effectiveness by Generation Phase"
            if self.config_hash:
                title += f" (Config: {self.config_hash[:8]})"
            fig.suptitle(title, fontsize=16, fontweight="bold")
            
            # Improvement rate heatmap
            rate_pivot = temporal_stats.pivot(index="mutation_type", columns="generation_phase", 
                                            values="improvement_rate_mean")
            phase_order = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]
            rate_pivot = rate_pivot.reindex(columns=phase_order)
            
            if not rate_pivot.empty:
                sns.heatmap(rate_pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax1,
                           cbar_kws={"label": "Improvement Rate (%)"})
                ax1.set_title("Improvement Rate by Phase")
                ax1.set_xlabel("Generation Phase")
                ax1.set_ylabel("Mutation Type")
            
            # Fitness gain heatmap
            gain_pivot = temporal_stats.pivot(index="mutation_type", columns="generation_phase",
                                            values="mean_gain_mean")
            gain_pivot = gain_pivot.reindex(columns=phase_order)
            
            if not gain_pivot.empty:
                sns.heatmap(gain_pivot, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax2,
                           cbar_kws={"label": "Mean Fitness Gain"})
                ax2.set_title("Mean Fitness Gain by Phase")
                ax2.set_xlabel("Generation Phase")
                ax2.set_ylabel("Mutation Type")
            
            plt.tight_layout()
            plt.savefig(output_dir / "mutation_temporal_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            
        # 3. Pattern-specific analysis
        pattern_stats = self.analyze_mutations_by_pattern()
        if not pattern_stats.empty and len(pattern_stats["pattern_name"].unique()) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            title = f"Mutation Effectiveness by Pattern"
            if self.config_hash:
                title += f" (Config: {self.config_hash[:8]})"
            fig.suptitle(title, fontsize=16, fontweight="bold")
            
            # Improvement rate by pattern
            rate_pattern_pivot = pattern_stats.pivot(index="pattern_name", columns="mutation_type",
                                                   values="improvement_rate_mean")
            
            if not rate_pattern_pivot.empty:
                sns.heatmap(rate_pattern_pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax1,
                           cbar_kws={"label": "Improvement Rate (%)"})
                ax1.set_title("Improvement Rate by Pattern")
                ax1.set_xlabel("Mutation Type")
                ax1.set_ylabel("Pattern")
            
            # Fitness gain by pattern
            gain_pattern_pivot = pattern_stats.pivot(index="pattern_name", columns="mutation_type",
                                                   values="mean_gain_mean")
            
            if not gain_pattern_pivot.empty:
                sns.heatmap(gain_pattern_pivot, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax2,
                           cbar_kws={"label": "Mean Fitness Gain"})
                ax2.set_title("Mean Fitness Gain by Pattern")
                ax2.set_xlabel("Mutation Type")
                ax2.set_ylabel("Pattern")
            
            plt.tight_layout()
            plt.savefig(output_dir / "mutation_pattern_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            
    def plot_fitness_evolution(self, output_dir: Path) -> None:
        """Plot fitness evolution over generations"""
        if self.population_data.empty:
            return
            
        # Get best fitness per generation per run
        evolution_data = (
            self.population_data
            .groupby(["config_hash", "run_tag", "pattern_name", "generation"])
            ["self_fitness"]
            .max()
            .reset_index()
        )
        
        # Plot by run/pattern
        fig, axes = plt.subplots(figsize=(15, 10))
        
        unique_runs = evolution_data[["run_tag", "pattern_name"]].drop_duplicates()
        n_runs = len(unique_runs)
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_runs)))
        
        for i, (_, run_info) in enumerate(unique_runs.iterrows()):
            run_data = evolution_data[
                (evolution_data["run_tag"] == run_info["run_tag"]) &
                (evolution_data["pattern_name"] == run_info["pattern_name"])
            ]
            
            color = colors[i % len(colors)]
            axes.plot(run_data["generation"], run_data["self_fitness"], 
                     color=color, alpha=0.7, linewidth=2,
                     label=f"{run_info['run_tag'][:8]} ({run_info['pattern_name']})")
        
        axes.set_xlabel("Generation")
        axes.set_ylabel("Best Fitness")
        title = "Fitness Evolution by Run"
        if self.config_hash:
            title += f" (Config: {self.config_hash[:8]})"
        axes.set_title(title)
        axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "fitness_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    def export_analysis(self, output_dir: Path) -> None:
        """Export all analysis results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = f"_{self.config_hash[:8]}" if self.config_hash else ""
        print(f"Exporting analysis to {output_dir}...")
        
        # Export data
        self.get_run_summary().to_csv(output_dir / f"run_summary{suffix}.csv", index=False)
        self.analyze_mutations_by_type().to_csv(output_dir / f"mutation_effectiveness{suffix}.csv", index=False)
        self.analyze_mutations_by_phase().to_csv(output_dir / f"mutation_temporal{suffix}.csv", index=False)
        self.analyze_mutations_by_pattern().to_csv(output_dir / f"mutation_by_pattern{suffix}.csv", index=False)
        self.compare_configurations().to_csv(output_dir / f"config_comparison{suffix}.csv", index=False)
        
        # Generate plots
        self.plot_mutation_effectiveness(output_dir)
        self.plot_fitness_evolution(output_dir)
        
        print("Analysis export completed!")

def main():
    parser = argparse.ArgumentParser(description="GA Analysis Tool with Config Filtering")
    parser.add_argument("--runs-dir", default="experiments/runs", 
                       help="Directory containing run CSV files")
    parser.add_argument("--aggregate-file", default="experiments/aggregate/final_metrics.csv",
                       help="Final metrics CSV file")
    parser.add_argument("--output-dir", default="analysis_output",
                       help="Output directory for analysis results")
    parser.add_argument("--config-hash", default=None,
                       help="Filter analysis to specific configuration hash")
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = GAAnalyzer(args.runs_dir, args.aggregate_file, args.config_hash)
    analyzer.load_data()
    
    # Print summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    if args.config_hash:
        print(f"Configuration: {args.config_hash}")
    print(f"Total population records: {len(analyzer.population_data):,}")
    print(f"Unique runs: {analyzer.population_data['run_tag'].nunique()}")
    print(f"Unique patterns: {analyzer.population_data['pattern_name'].nunique()}")
    print(f"Unique configurations: {analyzer.population_data['config_hash'].nunique()}")
    print(f"Mutation records: {len(analyzer.mutation_analysis):,}")
    
    # Export everything
    analyzer.export_analysis(Path(args.output_dir))

if __name__ == "__main__":
    main()