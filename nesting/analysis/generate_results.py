"""Unified result generation pipeline for garment nesting paper.

Reads from runs_fresh/ CSVs and writes all figures and tables to results/.

Paper artefacts produced
------------------------
tables/
  tab_ga_vs_ne.{csv,tex}          tab:ga-vs-ne   (6 configs: ga_sticky ... random_search)
  tab_ablation.{csv,tex}          tab:ablation   (5 NE configs)
  tab_esicup.{csv,tex}            Appendix ESICUP table

figures/
  fig_correlations.pdf            fig:correlations  — pairwise metric correlation matrix
  fig_pattern_features.pdf        Pattern feature distributions (n_pieces, area, convexity…)
  fig_mut_rates.pdf               fig:mut_rates     — improvement rate + mean impact (ne_full)
  fig_mut_gain.pdf                fig:mut_gain      — cond. gain + freq-weighted gain (ne_full)
  fig_norotfreqweighted.pdf       fig:norotfreqweighted — same 2-panel for ne_no_rotations

Metric conventions (paper → CSV column)
-----------------------------------------
  F   (cm)            solution_length_cm       occupied strip length (minimize)
  η_B (%)             usage_bb  × 100          bounding-box utilization (maximize)
  Φ                   fitness                  primary objective = η_B + (L_max−F)/L_max (maximize)
  η_H (%)             concave_hull_utilization × 100  (maximize)
  Area(B) (cm²)       bb_area                  (minimize)
  Area(H) (cm²)       concave_hull_area         (minimize)

Usage
-----
    python -m nesting.analysis.generate_results
    python -m nesting.analysis.generate_results \\
        --runs-dir nesting/experiments/runs_fresh \\
        --output-dir nesting/experiments/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config ordering / display names
# ---------------------------------------------------------------------------

_GA_VS_NE_ORDER = [
    "ga_sticky", "ga_lexicographic",
    "ne_full", "ne_no_rotations",
    "two_exchange", "random_search",
]

_ABLATION_ORDER = [
    "ne_no_splits_no_params",
    "ne_no_splits",
    "ne_no_params",
    "ne_no_rotations",
    "ne_full",
]

_CONFIG_LABELS = {
    "ga_sticky":              r"GA (sticky)",
    "ga_lexicographic":       r"GA (lexicographic)",
    "ne_full":                r"NE (full)",
    "ne_no_rotations":        r"NE (\textnormal{--}rotations)",
    "ne_no_splits":           r"NE (\textnormal{--}splits)",
    "ne_no_params":           r"NE (\textnormal{--}params)",
    "ne_no_splits_no_params": r"NE (\textnormal{--}splits, \textnormal{--}params)",
    "two_exchange":           r"2-Exchange",
    "random_search":          r"Random Search",
}

_MUTATION_ORDER = ["rotate", "swap", "inversion", "insertion", "scramble", "split", "design_params"]
_MUTATION_LABELS = {
    "rotate":        "Rotate",
    "swap":          "Swap",
    "inversion":     "Inversion",
    "insertion":     "Insertion",
    "scramble":      "Scramble",
    "split":         "Split",
    "design_params": "Design\nParams",
}

# Paper metric columns — order matches the paper tables
_TABLE_METRICS = [
    ("solution_length_cm",       r"$F$ (cm)",              ".2f"),   # minimize
    ("usage_bb_pct",             r"$\eta_{\mathcal{B}}$ (\%)", ".2f"),  # maximize
    ("fitness",                  r"$\Phi$",                ".4f"),   # maximize
    ("concave_hull_util_pct",    r"$\eta_{\mathcal{H}}$ (\%)", ".2f"),  # maximize
    ("bb_area",                  r"$\mathrm{Area}(\mathcal{B})$ (cm$^2$)", ".2f"),  # minimize
    ("concave_hull_area",        r"$\mathrm{Area}(\mathcal{H})$ (cm$^2$)", ".2f"),  # minimize
]

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 50:
        df = pd.read_csv(path)
        print(f"  loaded {path.name}: {len(df):,} rows")
        return df
    print(f"  [MISSING] {path.name}")
    return pd.DataFrame()


def _add_pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add %-scaled copies of utilization columns."""
    df = df.copy()
    if "usage_bb" in df.columns:
        df["usage_bb_pct"] = df["usage_bb"] * 100
    if "concave_hull_utilization" in df.columns:
        df["concave_hull_util_pct"] = df["concave_hull_utilization"] * 100
    return df


def _per_pattern_mean(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Average per (config_name, pattern_name) over all run_ids."""
    return df.groupby(["config_name", "pattern_name"])[metrics].mean()


def _across_patterns_mean(per_pattern: pd.DataFrame) -> pd.DataFrame:
    """Average per config_name over all patterns."""
    return per_pattern.groupby("config_name").mean()


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _format_val(val: float, fmt: str) -> str:
    return format(val, fmt)


def _build_result_df(gc_df: pd.DataFrame, config_order: list[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by config (display name) with formatted metric columns."""
    raw_cols = [c for c, _, _ in _TABLE_METRICS]
    pp = _per_pattern_mean(gc_df, raw_cols)
    means = _across_patterns_mean(pp)

    rows = {}
    for cfg in config_order:
        if cfg not in means.index:
            continue
        rows[cfg] = {col: means.loc[cfg, col] for col in raw_cols}
    return pd.DataFrame(rows).T   # shape: (n_configs, n_metrics)


def _latex_table_with_bold(df: pd.DataFrame, config_order: list[str],
                            caption: str, label: str,
                            minimize_cols: set[str] | None = None) -> str:
    """Emit a booktabs LaTeX table with best values bolded per column."""
    if minimize_cols is None:
        minimize_cols = {"solution_length_cm", "bb_area", "concave_hull_area"}

    raw_cols = [c for c, _, _ in _TABLE_METRICS]
    present = [cfg for cfg in config_order if cfg in df.index]

    # Find best per column
    best: dict[str, float] = {}
    for col in raw_cols:
        if col not in df.columns:
            continue
        vals = df.loc[present, col]
        best[col] = vals.min() if col in minimize_cols else vals.max()

    col_headers = " & ".join([r"\textbf{Method}"] +
                              [hdr for _, hdr, _ in _TABLE_METRICS]) + r" \\"
    col_fmt = "l" + "r" * len(_TABLE_METRICS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        col_headers,
        r"\midrule",
    ]

    for cfg in present:
        row_vals = [_CONFIG_LABELS.get(cfg, cfg)]
        for col, _, fmt in _TABLE_METRICS:
            if col not in df.columns:
                row_vals.append("—")
                continue
            v = df.loc[cfg, col]
            s = _format_val(v, fmt)
            if col in best and np.isclose(v, best[col], rtol=1e-6):
                s = _bold(s)
            row_vals.append(s)
        lines.append(" & ".join(row_vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _save_table(df: pd.DataFrame, config_order: list[str],
                stem: Path, caption: str, label: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)

    # CSV: plain numeric
    raw_cols = [c for c, _, _ in _TABLE_METRICS if c in df.columns]
    present = [cfg for cfg in config_order if cfg in df.index]
    csv_df = df.loc[present, raw_cols].copy()
    csv_df.index = [_CONFIG_LABELS.get(c, c) for c in csv_df.index]
    csv_df.to_csv(stem.with_suffix(".csv"))

    tex = _latex_table_with_bold(df, config_order, caption, label)
    stem.with_suffix(".tex").write_text(tex)
    print(f"  [TABLE] {stem.stem}  ({len(present)} rows)")


# ---------------------------------------------------------------------------
# Section: Tables
# ---------------------------------------------------------------------------

_DECODER_ORDER = ["BL", "NFP_BL", "NFP_max_overlap", "NFP_min_bb_area", "NFP_min_bb_length"]
_DECODER_LABELS = {
    "BL":                "BL",
    "NFP_BL":            "NFP + BL",
    "NFP_max_overlap":   "NFP + max overlap",
    "NFP_min_bb_area":   r"NFP + min Area$(\mathcal{B})$",
    "NFP_min_bb_length": r"NFP + min $F$",
}
# For decoder table: lower F and Area(B) is better; higher η_B and η_H is better
_DECODER_MINIMIZE = {"solution_length_cm", "bb_area"}

def tab_decoder_comparison(dc_df: pd.DataFrame, out: Path) -> None:
    if dc_df.empty:
        print("  [SKIP] tab_decoder_comparison: no data")
        return

    dc_df = _add_pct_cols(dc_df)
    raw_cols = [c for c, _, _ in _TABLE_METRICS if c != "concave_hull_area"]  # hull area not in paper table
    metrics_shown = [(c, h, f) for c, h, f in _TABLE_METRICS if c != "concave_hull_area"]

    grp = dc_df.groupby("config_name")[raw_cols].mean()
    present = [c for c in _DECODER_ORDER if c in grp.index]

    # Bold best per column
    best = {}
    for col in raw_cols:
        if col not in grp.columns:
            continue
        vals = grp.loc[present, col]
        best[col] = vals.min() if col in _DECODER_MINIMIZE else vals.max()

    col_fmt = "l" + "r" * len(metrics_shown)
    col_headers = " & ".join([r"\textbf{Decoder}"] + [h for _, h, _ in metrics_shown]) + r" \\"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        col_headers,
        r"\midrule",
    ]
    for cfg in present:
        row_vals = [_DECODER_LABELS.get(cfg, cfg)]
        for col, _, fmt in metrics_shown:
            v = grp.loc[cfg, col]
            s = format(v, fmt)
            if col in best and np.isclose(v, best[col], rtol=1e-6):
                s = _bold(s)
            row_vals.append(s)
        lines.append(" & ".join(row_vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}",
              r"\caption{Decoder comparison averaged over 1{,}700 GarmentCodeData patterns.}",
              r"\label{tab:decoder}", r"\end{table}"]

    stem = out / "tables" / "tab_decoder_comparison"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".tex").write_text("\n".join(lines))
    grp.loc[present].to_csv(stem.with_suffix(".csv"))
    print(f"  [TABLE] tab_decoder_comparison  ({len(present)} decoders)")


def tab_ga_vs_ne(gc_df: pd.DataFrame, out: Path) -> None:
    if gc_df.empty:
        print("  [SKIP] tab_ga_vs_ne: no data")
        return
    df = _build_result_df(gc_df, _GA_VS_NE_ORDER)
    _save_table(
        df, _GA_VS_NE_ORDER,
        out / "tables" / "tab_ga_vs_ne",
        caption=(r"Comparison of GA (sticky and lexicographic crossover), NE, NE without "
                 r"rotations, 2-Exchange, and Random Search. "
                 r"Best values per metric are highlighted in bold."),
        label="tab:ga-vs-ne",
    )


def tab_ablation(gc_df: pd.DataFrame, out: Path) -> None:
    if gc_df.empty:
        print("  [SKIP] tab_ablation: no data")
        return
    df = _build_result_df(gc_df, _ABLATION_ORDER)
    _save_table(
        df, _ABLATION_ORDER,
        out / "tables" / "tab_ablation",
        caption=r"Ablation study comparing NE configurations. Best values per metric are highlighted in bold.",
        label="tab:ablation",
    )


def tab_esicup(es_df: pd.DataFrame, out: Path) -> None:
    if es_df.empty:
        print("  [SKIP] tab_esicup: no data")
        return
    esicup_order = ["ne_full", "ne_no_splits_no_params"]
    df = _build_result_df(es_df, esicup_order)
    _save_table(
        df, esicup_order,
        out / "tables" / "tab_esicup",
        caption=r"Results on ESICUP garment instances (7 instances $\times$ 5 runs).",
        label="tab:esicup",
    )


# ---------------------------------------------------------------------------
# fig:correlations — pairwise metric correlation matrix
# ---------------------------------------------------------------------------

def fig_pattern_features(stats_df: pd.DataFrame, out: Path) -> None:
    """Distribution plots for pattern geometric features (personal use)."""
    if stats_df.empty:
        print("  [SKIP] fig_pattern_features: no pattern stats")
        return

    # Features to show and their display labels
    features = [
        ("n_pieces",            "# pieces"),
        ("total_area_cm2",      "Total area (cm²)"),
        ("mean_piece_area_cm2", "Mean piece area (cm²)"),
        ("cv_piece_area",       "CV piece area"),
        ("mean_convexity",      "Mean convexity"),
        ("min_convexity",       "Min convexity"),
        ("mean_aspect_ratio",   "Mean aspect ratio"),
    ]
    features = [(c, l) for c, l in features if c in stats_df.columns]
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, (col, label) in zip(axes_flat, features):
        data = stats_df[col].dropna()
        ax.hist(data, bins=20, color="#4878CF", edgecolor="white", alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.axvline(data.median(), color="red", linestyle="--", linewidth=1,
                   label=f"median={data.median():.2g}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide any unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Category breakdown if available
    if "category" in stats_df.columns:
        cat_counts = stats_df["category"].value_counts()
        ax_cat = axes_flat[n - 1] if n < len(axes_flat) else None
        if ax_cat is not None:
            ax_cat.set_visible(True)
            ax_cat.barh(cat_counts.index, cat_counts.values, color="#ff7f0e")
            ax_cat.set_title("Category breakdown", fontsize=10)
            ax_cat.set_xlabel("Count")
            ax_cat.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Pattern feature distributions (n={len(stats_df)})", fontsize=12)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "fig_pattern_features.pdf")


def fig_correlations(gc_df: pd.DataFrame, out: Path) -> None:
    if gc_df.empty:
        print("  [SKIP] fig_correlations: no data")
        return

    # Use ne_full; one value per pattern (mean over 5 runs)
    ne = gc_df[gc_df["config_name"] == "ne_full"]
    if ne.empty:
        print("  [SKIP] fig_correlations: no ne_full rows")
        return

    raw_cols = [c for c, _, _ in _TABLE_METRICS]
    display_names = {
        "solution_length_cm":     r"$F$",
        "usage_bb_pct":           r"$\eta_\mathcal{B}$",
        "fitness":                r"$\Phi$",
        "concave_hull_util_pct":  r"$\eta_\mathcal{H}$",
        "bb_area":                r"Area$(\mathcal{B})$",
        "concave_hull_area":      r"Area$(\mathcal{H})$",
    }

    per_pattern = ne.groupby("pattern_name")[raw_cols].mean()
    corr = per_pattern.corr()
    corr.index = [display_names.get(c, c) for c in corr.index]
    corr.columns = [display_names.get(c, c) for c in corr.columns]

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax.set_title("Pairwise metric correlations (NE full, per pattern)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "fig_correlations.pdf")


# ---------------------------------------------------------------------------
# Mutation statistics helpers
# ---------------------------------------------------------------------------

def _mutation_stats_for(mut_df: pd.DataFrame, config: str) -> pd.DataFrame:
    """Return per-mutation-type stats for one config."""
    sub = mut_df[mut_df["config_name"] == config].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["improved"] = (sub["fitness_gain"] > 0).astype(int)

    agg = sub.groupby("mutation_type").agg(
        count=("fitness_gain", "count"),
        improvement_rate=("improved", "mean"),
        mean_gain_all=("fitness_gain", "mean"),
        mean_gain_success=("fitness_gain", lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
    ).reset_index()
    return agg


def _freq_weighted_gain(mut_df: pd.DataFrame, config: str) -> pd.DataFrame:
    """Frequency-weighted gain per mutation type for a config.

    For each (pattern_name, run_id): freq_i = count_i / total.
    fw_gain_i = freq_i * mean_gain_i.
    Averaged over all (pattern, run) pairs.
    """
    sub = mut_df[mut_df["config_name"] == config].copy()
    if sub.empty:
        return pd.DataFrame(columns=["mutation_type", "freq_weighted_gain"])

    run_totals = sub.groupby(["pattern_name", "run_id"])["fitness_gain"].transform("count")
    sub["freq"] = 1.0 / run_totals

    per_run = (sub.groupby(["pattern_name", "run_id", "mutation_type"])
               .agg(freq=("freq", "sum"), mean_gain=("fitness_gain", "mean"))
               .reset_index())
    per_run["fw_gain"] = per_run["freq"] * per_run["mean_gain"]

    return (per_run.groupby("mutation_type")["fw_gain"]
            .mean().reset_index()
            .rename(columns={"fw_gain": "freq_weighted_gain"}))


def _ordered_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Reindex to _MUTATION_ORDER, drop missing."""
    stats = stats.set_index("mutation_type").reindex(_MUTATION_ORDER).dropna(how="all")
    stats.index = [_MUTATION_LABELS.get(i, i) for i in stats.index]
    return stats


def _ordered_fwg(fwg: pd.DataFrame) -> tuple[list, list]:
    fwg = fwg.set_index("mutation_type").reindex(_MUTATION_ORDER).dropna()
    labels = [_MUTATION_LABELS.get(i, i) for i in fwg.index]
    return labels, fwg["freq_weighted_gain"].tolist()


# ---------------------------------------------------------------------------
# fig:mut_rates — improvement rate + mean impact (ne_full)
# ---------------------------------------------------------------------------

def fig_mut_rates(mut_df: pd.DataFrame, out: Path) -> None:
    if mut_df.empty:
        print("  [SKIP] fig_mut_rates: no mutation data")
        return
    stats = _mutation_stats_for(mut_df, "ne_full")
    if stats.empty:
        print("  [SKIP] fig_mut_rates: no ne_full rows")
        return
    s = _ordered_stats(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: improvement rate
    ax1.bar(s.index, s["improvement_rate"], color="#4878CF", edgecolor="white")
    ax1.set_ylabel("Improvement rate")
    ax1.set_title("Improvement rate per mutation type")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_ylim(0, max(s["improvement_rate"].max() * 1.2, 0.05))
    ax1.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.grid(axis="y", alpha=0.3)

    # Right: mean gain across ALL mutants (including non-improving)
    colors = ["#d62728" if v < 0 else "#4878CF" for v in s["mean_gain_all"]]
    ax2.bar(s.index, s["mean_gain_all"], color=colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Mean fitness gain (all mutants)")
    ax2.set_title("Mean fitness impact per mutation type")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("NE (full) — mutation effectiveness", fontsize=12)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "fig_mut_rates.pdf")


# ---------------------------------------------------------------------------
# fig:mut_gain — cond. gain + freq-weighted gain (ne_full)
# ---------------------------------------------------------------------------

def fig_mut_gain(mut_df: pd.DataFrame, out: Path) -> None:
    if mut_df.empty:
        print("  [SKIP] fig_mut_gain: no mutation data")
        return
    stats = _mutation_stats_for(mut_df, "ne_full")
    fwg = _freq_weighted_gain(mut_df, "ne_full")
    if stats.empty:
        print("  [SKIP] fig_mut_gain: no ne_full rows")
        return

    s = _ordered_stats(stats)
    fwg_labels, fwg_vals = _ordered_fwg(fwg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: mean gain conditional on success
    ax1.bar(s.index, s["mean_gain_success"], color="#2ca02c", edgecolor="white")
    ax1.set_ylabel("Mean fitness gain | success")
    ax1.set_title("Conditional gain per mutation type")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.3)

    # Right: frequency-weighted gain
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in fwg_vals]
    ax2.bar(fwg_labels, fwg_vals, color=colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Frequency-weighted gain per run")
    ax2.set_title("Freq-weighted gain per mutation type")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("NE (full) — mutation gain", fontsize=12)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "fig_mut_gain.pdf")


# ---------------------------------------------------------------------------
# fig:norotfreqweighted — cond. gain + freq-weighted gain (ne_no_rotations)
# ---------------------------------------------------------------------------

def fig_norotfreqweighted(mut_df: pd.DataFrame, out: Path) -> None:
    if mut_df.empty:
        print("  [SKIP] fig_norotfreqweighted: no mutation data")
        return
    stats = _mutation_stats_for(mut_df, "ne_no_rotations")
    fwg = _freq_weighted_gain(mut_df, "ne_no_rotations")
    if stats.empty:
        print("  [SKIP] fig_norotfreqweighted: no ne_no_rotations rows")
        return

    s = _ordered_stats(stats)
    fwg_labels, fwg_vals = _ordered_fwg(fwg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: mean gain conditional on success
    ax1.bar(s.index, s["mean_gain_success"], color="#ff7f0e", edgecolor="white")
    ax1.set_ylabel("Mean fitness gain | success")
    ax1.set_title("Conditional gain per mutation type")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.3)

    # Right: frequency-weighted gain
    colors = ["#d62728" if v < 0 else "#ff7f0e" for v in fwg_vals]
    ax2.bar(fwg_labels, fwg_vals, color=colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Frequency-weighted gain per run")
    ax2.set_title("Freq-weighted gain per mutation type")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(r"NE (−rot.) — mutation gain", fontsize=12)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "fig_norotfreqweighted.pdf")


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [FIG]   {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(runs_dir: Path, output_dir: Path,
         stats_file: Path | None = None) -> None:
    print(f"\nReading from : {runs_dir}")
    print(f"Writing to   : {output_dir}\n")

    gc_df  = _add_pct_cols(_read_csv(runs_dir / "garmentcode.csv"))
    es_df  = _add_pct_cols(_read_csv(runs_dir / "esicup.csv"))
    dc_df  = _read_csv(runs_dir / "decoder_comparison.csv")
    mut_df = _read_csv(runs_dir / "mutation_data.csv")

    stats_df = pd.DataFrame()
    if stats_file and stats_file.exists():
        stats_df = pd.read_csv(stats_file)
        print(f"  loaded pattern stats: {len(stats_df):,} rows")
    elif stats_file:
        print(f"  [MISSING] {stats_file}")

    print("\n--- Tables ---")
    tab_decoder_comparison(dc_df, output_dir)
    tab_ga_vs_ne(gc_df, output_dir)
    tab_ablation(gc_df, output_dir)
    tab_esicup(es_df, output_dir)

    print("\n--- Figures ---")
    fig_pattern_features(stats_df, output_dir)
    fig_correlations(gc_df, output_dir)
    fig_mut_rates(mut_df, output_dir)
    fig_mut_gain(mut_df, output_dir)
    fig_norotfreqweighted(mut_df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all paper figures and tables from runs_fresh/ CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir", default="nesting/experiments/runs_fresh",
                        help="Directory containing garmentcode.csv, esicup.csv, mutation_data.csv")
    parser.add_argument("--stats-file", default="nesting-assets/pattern_stats_100.csv",
                        help="Pattern stats CSV for feature distribution plot (optional)")
    parser.add_argument("--output-dir", default="nesting/experiments/results",
                        help="Output directory for tables/ and figures/")
    args = parser.parse_args()
    main(
        Path(args.runs_dir),
        Path(args.output_dir),
        stats_file=Path(args.stats_file) if args.stats_file else None,
    )
