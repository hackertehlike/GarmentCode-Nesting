#!/usr/bin/env python3
"""
GA plots adapted to the style you liked (pattern×mutation heatmaps, temporal bars/heatmap, 2×2 overview),
but computed per RUN first and then aggregated across runs, per CONFIG.

INPUT (run CSVs; one row per individual):
  config_hash,run_tag,pattern_name,generation,origin,parent1_fitness,parent2_fitness,self_fitness,last_mutation

Key choices:
- Only origins {mutant, offspring} are used (randoms have no parent → no fitness_gain).
- Mutation type = last_mutation (non-empty only).
- fitness_gain = self_fitness - parent_fitness
    mutant    → parent_fitness = parent1_fitness
    offspring → parent_fitness = mean(parent1_fitness, parent2_fitness)
- Generation phase buckets (fixed):
    Early (1-5), Middle (6-10), Late (10+)

Aggregation rule (your requirement):
- Compute stats PER RUN (never mix individuals from different runs).
- Then aggregate those stats ACROSS runs of the same config (mean ± std).
- For “counts”, we show MEAN per-run counts with std bars (not pooled totals).

Outputs per config (saved under --outdir/<config_hash>/):
  CSVs:
    pattern_mutation_improvement_stats.csv              (per-pattern × mutation, aggregated across runs)
    temporal_mutation_improvement_stats.csv             (per-mutation × phase, aggregated across runs)
    overall_mutation_improvement_stats.csv              (per-mutation overall, aggregated across runs)
  Figures:
    mutation_effectiveness_by_pattern_rate.png          (heatmap: improvement rate)
    mutation_effectiveness_by_pattern_gain.png          (heatmap: mean fitness gain)
    mutation_effectiveness_temporal.png                 (2×2: phase bars/heatmaps & counts)
    mutation_improvement_rates_overview.png             (2×2: improvement rate, mean gain, mean per-run counts, scatter)

Usage:
  python ga_plots.py --input "nesting/experiments/runs/*.csv" --outdir plots_out/
"""

from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


REQUIRED_COLS = [
    "config_hash","run_tag","pattern_name","generation","origin",
    "parent1_fitness","parent2_fitness","self_fitness","last_mutation"
]
VALID_ORIGINS = {"mutant","offspring"}
PHASE_LABELS = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]


# ---------------- IO & CLEANING ----------------

def read_runs(paths: List[str]) -> pd.DataFrame:
    files: List[str] = []
    for p in paths:
        ex = glob.glob(p)
        files.extend(ex if ex else [p])
    if not files:
        raise FileNotFoundError("No input CSV files found with --input.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing required columns: {missing}")
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # dtypes
    data["generation"] = pd.to_numeric(data["generation"], errors="coerce").fillna(0).astype(int)
    for c in ["parent1_fitness","parent2_fitness","self_fitness"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["origin"] = data["origin"].astype(str).str.strip().str.lower()
    data["last_mutation"] = data["last_mutation"].astype(str).str.strip()
    data["pattern_name"] = data["pattern_name"].astype(str).str.strip()
    data["config_hash"] = data["config_hash"].astype(str)
    data["run_tag"] = data["run_tag"].astype(str)
    # normalize common textual placeholders to real NaN so later filtering works
    # use inline (?i) flag for case-insensitive matching to support older pandas versions
    data["last_mutation"] = data["last_mutation"].replace(r"(?i)^\s*(nan|none|na|null)\s*$", np.nan, regex=True)
    return data


def compute_fitness_gain(df: pd.DataFrame) -> pd.DataFrame:
    """fitness_gain = self - parent; parent depends on origin."""
    df = df.copy()
    def parent_fit(r):
        if r["origin"] == "mutant":
            return r["parent1_fitness"]
        elif r["origin"] == "offspring":
            return pd.Series([r["parent1_fitness"], r["parent2_fitness"]], dtype="float").mean()
        else:
            return np.nan
    df["parent_fitness"] = df.apply(parent_fit, axis=1)
    df["fitness_gain"] = df["self_fitness"] - df["parent_fitness"]
    df["improved"] = df["fitness_gain"] > 0
    # Filter: valid origin + non-empty mutation type
    df = df[df["origin"].isin(VALID_ORIGINS)]
    df = df[(df["last_mutation"].notna()) & (df["last_mutation"] != "")]
    return df


def generation_phase(gen: int) -> str:
    if gen <= 5:
        return PHASE_LABELS[0]
    if gen <= 10:
        return PHASE_LABELS[1]
    return PHASE_LABELS[2]


# ---------------- PER-RUN STATS ----------------

def per_run_pattern_mutation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each (run, pattern, mutation): overall mean gain, rate, counts."""
    g = (
        df.groupby(["run_tag","pattern_name","last_mutation"])
          .agg(
              improvements=("improved","sum"),
              total_mutations=("improved","count"),
              mean_fitness_gain=("fitness_gain","mean"),
              std_fitness_gain=("fitness_gain","std"),
              median_fitness_gain=("fitness_gain","median"),
              total_fitness_gain=("fitness_gain","sum"),
          )
          .reset_index()
    )
    # Improvement rate per run (not pooled!)
    g["improvement_rate"] = np.where(
        g["total_mutations"]>0,
        g["improvements"]/g["total_mutations"]*100.0,
        np.nan
    )
    return g


def per_run_temporal_mutation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each (run, mutation, phase): mean gain, rate, counts."""
    tmp = df.copy()
    tmp["generation_phase"] = tmp["generation"].apply(generation_phase)
    g = (
        tmp.groupby(["run_tag","last_mutation","generation_phase"])
           .agg(
               improvements=("improved","sum"),
               total_mutations=("improved","count"),
               mean_fitness_gain=("fitness_gain","mean"),
               std_fitness_gain=("fitness_gain","std"),
           )
           .reset_index()
    )
    g["improvement_rate"] = np.where(
        g["total_mutations"]>0,
        g["improvements"]/g["total_mutations"]*100.0,
        np.nan
    )
    return g


def per_run_overall_mutation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each (run, mutation): overall mean gain, rate, counts."""
    g = (
        df.groupby(["run_tag","last_mutation"])
          .agg(
              improvements=("improved","sum"),
              total_mutations=("improved","count"),
              mean_fitness_gain=("fitness_gain","mean"),
              std_fitness_gain=("fitness_gain","std"),
          )
          .reset_index()
    )
    g["improvement_rate"] = np.where(
        g["total_mutations"]>0,
        g["improvements"]/g["total_mutations"]*100.0,
        np.nan
    )
    return g


# ---------------- AGGREGATE ACROSS RUNS (STATS ONLY) ----------------

def agg_across_runs_pattern_mutation(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run stats across runs → per pattern × mutation stats."""
    agg = (
        per_run.groupby(["pattern_name","last_mutation"])
               .agg(
                   mean_improvement_rate=("improvement_rate","mean"),
                   std_improvement_rate=("improvement_rate","std"),
                   mean_fitness_gain=("mean_fitness_gain","mean"),
                   std_fitness_gain=("mean_fitness_gain","std"),
                   mean_total_mutations=("total_mutations","mean"),  # mean per-run count
                   std_total_mutations=("total_mutations","std"),
                   n_runs=("run_tag","nunique"),
               )
               .reset_index()
    )
    # for convenience in heatmaps
    out = agg.rename(columns={
        "last_mutation":"mutation_type",
        "mean_improvement_rate":"improvement_rate",
    })
    return out


def agg_across_runs_temporal(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run temporal stats across runs → per mutation × phase stats."""
    agg = (
        per_run.groupby(["last_mutation","generation_phase"])
               .agg(
                   mean_improvement_rate=("improvement_rate","mean"),
                   std_improvement_rate=("improvement_rate","std"),
                   mean_fitness_gain=("mean_fitness_gain","mean"),
                   std_fitness_gain=("mean_fitness_gain","std"),
                   mean_total_mutations=("total_mutations","mean"),
                   std_total_mutations=("total_mutations","std"),
                   n_runs=("run_tag","nunique"),
               )
               .reset_index()
    )
    out = agg.rename(columns={"last_mutation":"mutation_type"})
    return out


def agg_across_runs_overall(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run overall mutation stats across runs → per mutation stats."""
    agg = (
        per_run.groupby(["last_mutation"])
               .agg(
                   mean_improvement_rate=("improvement_rate","mean"),
                   std_improvement_rate=("improvement_rate","std"),
                   mean_fitness_gain=("mean_fitness_gain","mean"),
                   std_fitness_gain=("mean_fitness_gain","std"),
                   mean_total_mutations=("total_mutations","mean"),
                   std_total_mutations=("total_mutations","std"),
                   n_runs=("run_tag","nunique"),
               )
               .reset_index()
    )
    out = agg.rename(columns={"last_mutation":"mutation_type"})
    return out


# ---------------- PLOTTING (STYLE YOU LIKE) ----------------

def plot_pattern_mutation_heatmaps(result: pd.DataFrame, outdir: Path):
    """Two heatmaps: improvement rate (%) and mean fitness gain, by Pattern × Mutation."""
    if result.empty:
        return
    data = result.copy()
    heat_rate = data.pivot(index="pattern_name", columns="mutation_type", values="improvement_rate")
    heat_gain = data.pivot(index="pattern_name", columns="mutation_type", values="mean_fitness_gain")
    # Drop any patterns or mutation types that are entirely missing
    heat_rate = heat_rate.dropna(axis=0, how="all").dropna(axis=1, how="all")
    heat_gain = heat_gain.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Rate heatmap
    plt.figure(figsize=(max(8, 0.6*len(heat_rate.columns) if heat_rate.columns.size else 8),
                        max(6, 0.4*len(heat_rate.index) if heat_rate.index.size else 6)))
    # mask cells that are NaN so they are not plotted/annotated
    mask_rate = heat_rate.isna() if not heat_rate.empty else None
    sns.heatmap(heat_rate, mask=mask_rate, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={"label":"Improvement Rate (%)"})
    plt.title("Improvement Rate (%) by Pattern & Mutation Type\n(per-run stats aggregated across runs)")
    plt.xlabel("Mutation Type"); plt.ylabel("Pattern")
    plt.tight_layout()
    p = outdir / "mutation_effectiveness_by_pattern_rate.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()

    # Gain heatmap
    plt.figure(figsize=(max(8, 0.6*len(heat_gain.columns) if heat_gain.columns.size else 8),
                        max(6, 0.4*len(heat_gain.index) if heat_gain.index.size else 6)))
    mask_gain = heat_gain.isna() if not heat_gain.empty else None
    sns.heatmap(heat_gain, mask=mask_gain, annot=True, fmt=".3f", cmap="coolwarm", center=0, cbar_kws={"label":"Mean Fitness Gain"})
    plt.title("Mean Fitness Gain by Pattern & Mutation Type\n(per-run stats aggregated across runs)")
    plt.xlabel("Mutation Type"); plt.ylabel("Pattern")
    plt.tight_layout()
    p = outdir / "mutation_effectiveness_by_pattern_gain.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()


def plot_temporal_improvement_rates(agg_temporal: pd.DataFrame, outdir: Path):
    """2×2 figure (bars, heatmaps, gains, counts) by mutation×phase (aggregated across runs)."""
    if agg_temporal.empty:
        return
    df = agg_temporal.copy()
    # defensive: drop rows where mutation_type is missing or the literal 'nan'
    df = df[df["mutation_type"].notna()]
    df = df[df["mutation_type"].astype(str).str.strip().str.lower() != "nan"]
    phase_order = PHASE_LABELS
    df["generation_phase"] = pd.Categorical(df["generation_phase"], categories=phase_order, ordered=True)

    # coerce numeric columns to real NaN where parsing fails (avoid literal 'nan' strings)
    for col in ["mean_improvement_rate", "mean_fitness_gain", "mean_total_mutations"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pivot tables
    piv_rate = df.pivot(index="mutation_type", columns="generation_phase", values="mean_improvement_rate").reindex(columns=phase_order)
    piv_gain = df.pivot(index="mutation_type", columns="generation_phase", values="mean_fitness_gain").reindex(columns=phase_order)
    piv_count = df.pivot(index="mutation_type", columns="generation_phase", values="mean_total_mutations").reindex(columns=phase_order)

    # remove mutation types (rows) that are completely missing across phases
    piv_rate = piv_rate.dropna(how="all")
    piv_gain = piv_gain.dropna(how="all")
    piv_count = piv_count.dropna(how="all")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Temporal Mutation Effectiveness (per-run first, aggregated across runs)", fontsize=16, fontweight="bold")

    # (1) Bar: Improvement Rate
    piv_rate.plot(kind="bar", ax=axes[0,0], width=0.8)
    axes[0,0].set_title("Improvement Rate by Type and Generation Phase")
    axes[0,0].set_xlabel("Mutation Type"); axes[0,0].set_ylabel("Improvement Rate (%)")
    axes[0,0].legend(title="Generation Phase"); axes[0,0].tick_params(axis="x", rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    # (2) Heatmap: Improvement Rate
    mask = piv_rate.isna() if not piv_rate.empty else None
    sns.heatmap(piv_rate, mask=mask, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=axes[0,1], cbar_kws={"label":"Improvement Rate (%)"})
    axes[0,1].set_title("Improvement Rate Heatmap"); axes[0,1].set_xlabel("Generation Phase"); axes[0,1].set_ylabel("Mutation Type")

    # (3) Bar: Mean Fitness Gain
    piv_gain.plot(kind="bar", ax=axes[1,0], width=0.8)
    axes[1,0].set_title("Mean Fitness Gain by Type and Generation Phase")
    axes[1,0].set_xlabel("Mutation Type"); axes[1,0].set_ylabel("Mean Fitness Gain")
    axes[1,0].legend(title="Generation Phase"); axes[1,0].tick_params(axis="x", rotation=45)
    axes[1,0].grid(True, alpha=0.3); axes[1,0].axhline(y=0, color="red", linestyle="--", alpha=0.7)

    # (4) Bar: Mean per-run counts
    piv_count.plot(kind="bar", ax=axes[1,1], width=0.8)
    # mask NaNs in gain heatmap to avoid 'nan' annotations
    mask_gain = piv_gain.isna() if not piv_gain.empty else None
    sns.heatmap(piv_gain, mask=mask_gain, annot=False if mask_gain is None else True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[1,0])
    axes[1,1].set_title("Mean Number of Mutations per Run by Type and Phase")
    axes[1,1].set_xlabel("Mutation Type"); axes[1,1].set_ylabel("Mean Count per Run")
    axes[1,1].legend(title="Generation Phase"); axes[1,1].tick_params(axis="x", rotation=45)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = outdir / "mutation_effectiveness_temporal.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_mutation_effectiveness(agg_overall: pd.DataFrame, outdir: Path):
    """2×2 overview: improvement rate, mean gain, mean per-run counts, scatter."""
    if agg_overall.empty:
        return
    df = agg_overall.copy().sort_values("mean_improvement_rate", ascending=True)

    plt.style.use("default")
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Mutation Effectiveness (per-run stats aggregated across runs)", fontsize=16, fontweight="bold")

    # 1) Improvement rate (barh)
    ax1 = axes[0,0]
    bars = ax1.barh(df["mutation_type"], df["mean_improvement_rate"], color=plt.cm.viridis(np.linspace(0,1,len(df))))
    ax1.set_xlabel("Improvement Rate (%)"); ax1.set_title("Improvement Rate by Mutation Type")
    ax1.grid(axis="x", alpha=0.3)
    for b in bars:
        w = b.get_width()
        if np.isfinite(w):
            ax1.text(w + 0.1, b.get_y()+b.get_height()/2, f"{w:.1f}%", ha="left", va="center", fontweight="bold")

    # 2) Mean fitness gain (barh)
    ax2 = axes[0,1]
    bars2 = ax2.barh(df["mutation_type"], df["mean_fitness_gain"], color=plt.cm.RdYlBu(np.linspace(0,1,len(df))))
    ax2.set_xlabel("Mean Fitness Gain"); ax2.set_title("Average Fitness Impact by Mutation Type")
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="No Change")
    ax2.grid(axis="x", alpha=0.3); ax2.legend()
    for b in bars2:
        w = b.get_width()
        if np.isfinite(w):
            label_x = w + 0.001 if w >= 0 else w - 0.001
            ha = "left" if w >= 0 else "right"
            ax2.text(label_x, b.get_y()+b.get_height()/2, f"{w:.3f}", ha=ha, va="center", fontweight="bold")

    # 3) Mean per-run counts (barh)
    ax3 = axes[1,0]
    bars3 = ax3.barh(df["mutation_type"], df["mean_total_mutations"], color=plt.cm.plasma(np.linspace(0,1,len(df))))
    ax3.set_xlabel("Mean Mutations per Run"); ax3.set_title("Sample Size by Mutation Type (mean per run)")
    ax3.grid(axis="x", alpha=0.3)
    max_total = float(np.nanmax(df["mean_total_mutations"].to_numpy())) if not df.empty else 0.0
    for b in bars3:
        w = b.get_width()
        if np.isfinite(w):
            ax3.text(w + (max_total*0.02 if max_total else 0.1), b.get_y()+b.get_height()/2, f"{w:.1f}", ha="left", va="center", fontweight="bold")

    # 4) Scatter: rate vs mean gain (bubble size = mean per-run counts)
    ax4 = axes[1,1]
    df_scatter = df.dropna(subset=["mean_improvement_rate", "mean_fitness_gain"]).copy()
    sc = ax4.scatter(df_scatter["mean_improvement_rate"], df_scatter["mean_fitness_gain"],
                     s=np.maximum(df_scatter["mean_total_mutations"].fillna(1), 1)*3.0,
                     c=range(len(df_scatter)), cmap="tab10", alpha=0.7, edgecolors="black")
    for _, row in df_scatter.iterrows():
        ax4.annotate(row["mutation_type"], (row["mean_improvement_rate"], row["mean_fitness_gain"]),
                     xytext=(5,5), textcoords="offset points", fontsize=9)
    ax4.set_xlabel("Improvement Rate (%)"); ax4.set_ylabel("Mean Fitness Gain")
    ax4.set_title("Improvement Rate vs Average Impact (bubble = mean count per run)")
    ax4.grid(alpha=0.3); ax4.axhline(y=0, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    p = outdir / "mutation_improvement_rates_overview.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- PIPELINE PER CONFIG ----------------

def process_config(cfg_df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # PER-RUN stats
    per_run_pm = per_run_pattern_mutation_stats(cfg_df)       # (run, pattern, mutation)
    per_run_temp = per_run_temporal_mutation_stats(cfg_df)    # (run, mutation, phase)
    per_run_over = per_run_overall_mutation_stats(cfg_df)     # (run, mutation)

    # Aggregate across runs
    agg_pm = agg_across_runs_pattern_mutation(per_run_pm)
    agg_temp = agg_across_runs_temporal(per_run_temp)
    agg_over = agg_across_runs_overall(per_run_over)

    # Save CSVs (inputs for plots)
    agg_pm.to_csv(outdir / "pattern_mutation_improvement_stats.csv", index=False)
    agg_temp.to_csv(outdir / "temporal_mutation_improvement_stats.csv", index=False)
    agg_over.to_csv(outdir / "overall_mutation_improvement_stats.csv", index=False)

    # PLOTS (style matched to scripts you liked)
    plot_pattern_mutation_heatmaps(agg_pm, outdir)
    plot_temporal_improvement_rates(agg_temp, outdir)
    plot_overall_mutation_effectiveness(agg_over, outdir)


# ---------------- MAIN ----------------

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="GA mutation effectiveness plots per config (per-run stats, aggregated across runs).")
    parser.add_argument("--input", nargs="+", required=True, help='CSV paths/globs, e.g. "nesting/experiments/runs/*.csv"')
    parser.add_argument("--outdir", default="plots_out", help="Root output directory")
    args = parser.parse_args(argv)

    data = read_runs(args.input)
    data = compute_fitness_gain(data)

    # per config_hash
    for cfg, cfg_df in data.groupby("config_hash"):
        cfg_out = Path(args.outdir) / cfg
        process_config(cfg_df, cfg_out)

    print(f"Done. Plots & CSVs saved under: {Path(args.outdir).resolve()}")

if __name__ == "__main__":
    main()
