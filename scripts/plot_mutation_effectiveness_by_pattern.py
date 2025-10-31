#!/usr/bin/env python3
"""
Plot mutation effectiveness (improvement rates and fitness gains) per pattern and mutation type.
Generates:
  1. CSV: pattern_mutation_improvement_stats.csv
  2. Figure: mutation_effectiveness_by_pattern.png
Similar style to plot_mutation_effectiveness_temporal but adds pattern dimension.
"""
from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys

try:  # Optional config
    from nesting import config as nesting_config  # type: ignore
except Exception:  # pragma: no cover
    nesting_config = None

PHASE_LABELS = ["Early (1-5)", "Middle (6-10)", "Late (10+)"]


def _resolve_raw_mutation_csv() -> Path:
    candidates: list[Path] = []
    if nesting_config and hasattr(nesting_config, 'AGGREGATE_DIR'):
        candidates.append(Path(nesting_config.AGGREGATE_DIR) / 'master_mutation_raw_data.csv')
    candidates.append(Path('nesting/experiments/aggregate/master_mutation_raw_data.csv'))
    candidates.append(Path('nesting/aggregate_stats/master_mutation_raw_data.csv'))
    for p in candidates:
        if p.exists() and p.stat().st_size > 50:
            return p
    return candidates[0]


def _generation_phase(gen: int) -> str:
    if gen <= 5:
        return PHASE_LABELS[0]
    if gen <= 10:
        return PHASE_LABELS[1]
    return PHASE_LABELS[2]


def load_and_aggregate(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return df, pd.DataFrame()
    required = {'pattern_name','mutation_type','fitness_gain','generation'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw mutation csv: {missing}")
    df['generation_phase'] = df['generation'].apply(_generation_phase)
    df['improved'] = df['fitness_gain'] > 0
    # Core aggregation per pattern & mutation
    grp = df.groupby(['pattern_name','mutation_type'])
    base = grp.agg(
        improvements=('improved','sum'),
        total_mutations=('improved','count'),
        mean_fitness_gain=('fitness_gain','mean'),
        std_fitness_gain=('fitness_gain','std'),
        median_fitness_gain=('fitness_gain','median'),
        total_fitness_gain=('fitness_gain','sum')
    )
    # Positive-only mean
    pos = df[df['improved']].groupby(['pattern_name','mutation_type'])['fitness_gain'] \
        .mean().rename('mean_positive_fitness_gain')
    base = base.join(pos, how='left')
    base['mean_positive_fitness_gain'] = base['mean_positive_fitness_gain'].fillna(0.0)
    base['improvement_rate'] = (base['improvements'] / base['total_mutations'] * 100).round(2)

    # Phase-level improvement rate per pattern & mutation
    phase_grp = df.groupby(['pattern_name','mutation_type','generation_phase']).agg(
        phase_improvements=('improved','sum'),
        phase_total=('improved','count'),
        phase_mean_gain=('fitness_gain','mean')
    ).reset_index()
    # Pivot phases into columns
    piv_rate = phase_grp.pivot(index=['pattern_name','mutation_type'],
                               columns='generation_phase',
                               values='phase_improvements').fillna(0)
    piv_total = phase_grp.pivot(index=['pattern_name','mutation_type'],
                                columns='generation_phase',
                                values='phase_total').fillna(0)
    for ph in PHASE_LABELS:
        if ph not in piv_rate.columns:
            piv_rate[ph] = 0
        if ph not in piv_total.columns:
            piv_total[ph] = 0
    for ph in PHASE_LABELS:
        base[f'improvement_rate_{ph.split()[0].lower()}'] = (
            np.where(piv_total[ph]>0, piv_rate[ph]/piv_total[ph]*100, np.nan).round(2)
        )

    result = base.reset_index().sort_values(['pattern_name','mutation_type'])
    return df, result


def _select_patterns_for_plot(result: pd.DataFrame, max_patterns: int = 30):
    # Choose patterns with most total mutations
    if result.empty:
        return []
    totals = result.groupby('pattern_name')['total_mutations'].sum().sort_values(ascending=False)
    return list(totals.head(max_patterns).index)


def plot_heatmaps(result: pd.DataFrame, save_path: Path | None = None):
    if result.empty:
        print('No data to plot')
        return None
    # Filter to manageable subset
    patterns = _select_patterns_for_plot(result)
    data = result[result['pattern_name'].isin(patterns)]
    # Create pivot tables
    heat_rate = data.pivot(index='pattern_name', columns='mutation_type', values='improvement_rate')
    heat_gain = data.pivot(index='pattern_name', columns='mutation_type', values='mean_fitness_gain')
    plt.figure(figsize=(max(8, 0.6 * len(heat_rate.columns)), max(6, 0.4 * len(heat_rate.index))))
    sns.heatmap(heat_rate, annot=True, fmt='.1f', cmap='RdYlGn', center=0, cbar_kws={'label':'Improvement Rate (%)'})
    plt.title('Improvement Rate (%) by Pattern & Mutation Type')
    plt.xlabel('Mutation Type'); plt.ylabel('Pattern')
    plt.tight_layout()
    if save_path:
        rate_path = save_path.with_name(save_path.stem + '_rate.png')
        plt.savefig(rate_path, dpi=300, bbox_inches='tight')
        print(f'Saved heatmap: {rate_path}')
    plt.close()
    plt.figure(figsize=(max(8, 0.6 * len(heat_gain.columns)), max(6, 0.4 * len(heat_gain.index))))
    sns.heatmap(heat_gain, annot=True, fmt='.3f', cmap='coolwarm', center=0, cbar_kws={'label':'Mean Fitness Gain'})
    plt.title('Mean Fitness Gain by Pattern & Mutation Type')
    plt.xlabel('Mutation Type'); plt.ylabel('Pattern')
    plt.tight_layout()
    if save_path:
        gain_path = save_path.with_name(save_path.stem + '_gain.png')
        plt.savefig(gain_path, dpi=300, bbox_inches='tight')
        print(f'Saved heatmap: {gain_path}')
    plt.close()


def print_summary(result: pd.DataFrame):
    if result.empty:
        print('No mutation pattern data.')
        return
    print('=== PATTERN-MUTATION EFFECTIVENESS SUMMARY ===')
    # Top mutation types overall by improvement rate (weighted by total mutations)
    type_weighted = result.groupby('mutation_type').apply(
        lambda g: pd.Series({
            'total_mutations': g['total_mutations'].sum(),
            'weighted_improvement_rate': (g['improvements'].sum()/g['total_mutations'].sum()*100) if g['total_mutations'].sum()>0 else 0,
            'mean_gain': g['mean_fitness_gain'].mean()
        })
    ).sort_values('weighted_improvement_rate', ascending=False)
    print('\nWeighted improvement rate by mutation type:')
    print(type_weighted.round(3))
    # Best pattern per mutation type by improvement rate
    print('\nBest pattern per mutation type (improvement_rate):')
    for mtype, g in result.groupby('mutation_type'):
        top = g.sort_values('improvement_rate', ascending=False).head(1)
        row = top.iloc[0]
        print(f"  {mtype}: {row['pattern_name']} ({row['improvement_rate']:.2f}% of {int(row['total_mutations'])} mutations)" )


def main():
    csv_path = _resolve_raw_mutation_csv()
    print(f'Using mutation raw data file: {csv_path}')
    if not csv_path.exists() or csv_path.stat().st_size <= 50:
        print('Mutation raw data file missing or empty (only header).')
        sys.exit(0)
    try:
        df, result = load_and_aggregate(csv_path)
    except Exception as e:
        print(f'Failed to process data: {e}')
        sys.exit(1)
    if result.empty:
        print('No aggregated data produced.')
        sys.exit(0)
    out_csv = Path('pattern_mutation_improvement_stats.csv')
    result.to_csv(out_csv, index=False)
    print(f'Saved detailed stats: {out_csv}')
    plot_heatmaps(result, save_path=Path('mutation_effectiveness_by_pattern.png'))
    print_summary(result)
    print('\nDone.')


if __name__ == '__main__':
    main()
