"""Migration script to consolidate legacy run_logs/results into unified experiments structure.

Actions:
 1. Create nesting/experiments/{runs,aggregate} directories.
 2. Move (or copy if --copy) legacy per-run folders from:
      nesting/run_logs/*  (each timestamped run folder)
      results/*           (pattern run folders)
    into nesting/experiments/runs/ (name collisions resolved by appending _migratedN)
 3. Optionally rebuild master_statistics.csv from generations.csv files if --rebuild specified.
    (Uses MetaStatistics.save_run_statistics-like reconstruction with minimal fields.)

Usage:
  python -m scripts.migrate_legacy_experiments [--copy] [--rebuild]
"""
from __future__ import annotations
import os, shutil, csv, time, json, hashlib
from pathlib import Path
import argparse
import pandas as pd

from nesting import config
from nesting.analysis.metastatistics import MetaStatistics

LEGACY_RUN_LOGS = Path('nesting/run_logs')
LEGACY_RESULTS  = Path('results')
UNIFIED_RUNS    = Path(config.RUNS_DIR)
AGG_DIR         = Path(config.AGGREGATE_DIR)


def ensure_dirs():
    UNIFIED_RUNS.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, copy: bool):
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def migrate(copy: bool):
    ensure_dirs()
    migrated = []
    for legacy_root in [LEGACY_RUN_LOGS, LEGACY_RESULTS]:
        if not legacy_root.exists():
            continue
        for child in legacy_root.iterdir():
            if not child.is_dir():
                continue
            target = UNIFIED_RUNS / child.name
            count = 1
            while target.exists():
                target = UNIFIED_RUNS / f"{child.name}_migrated{count}"
                count += 1
            move_or_copy(child, target, copy)
            migrated.append((child, target))
    return migrated


def _compute_config_hash() -> str:
    try:
        conf = {
            'decoder': getattr(config, 'SELECTED_DECODER', None),
            'fitness_metric': getattr(config, 'SELECTED_FITNESS_METRIC', None),
            'crossover': getattr(config, 'SELECTED_CROSSOVER', None),
            'mutation_rate': getattr(config, 'MUTATION_RATE', None),
            'population_size': getattr(config, 'POPULATION_SIZE', None),
            'num_generations': getattr(config, 'NUM_GENERATIONS', None),
            'max_generations': getattr(config, 'MAX_GENERATIONS', None),
            'container_width_cm': getattr(config, 'CONTAINER_WIDTH_CM', None),
            'container_height_cm': getattr(config, 'CONTAINER_HEIGHT_CM', None),
            'enable_dynamic_stopping': getattr(config, 'ENABLE_DYNAMIC_STOPPING', None),
            'early_stop_window': getattr(config, 'EARLY_STOP_WINDOW', None),
            'early_stop_tolerance': getattr(config, 'EARLY_STOP_TOLERANCE', None),
            'enable_extension': getattr(config, 'ENABLE_EXTENSION', None),
            'extend_window': getattr(config, 'EXTEND_WINDOW', None),
            'extend_threshold': getattr(config, 'EXTEND_THRESHOLD', None),
        }
        payload = json.dumps(conf, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha1(payload).hexdigest()[:12]
    except Exception:
        return ''


def rebuild_master(run_tag: str | None = None):
    
    MetaStatistics.ensure_master_files_exist()
    master_path = MetaStatistics.MASTER_CSV_PATH
    # Read header to preserve column order
    try:
        with open(master_path, 'r') as f:
            header_line = f.readline().strip()
        fieldnames = header_line.split(',') if header_line else []
    except Exception:
        fieldnames = []
    # Load existing to avoid duplicates
    existing = set()
    if master_path.exists() and master_path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(master_path)
            if 'pattern_name' in df_existing.columns and 'timestamp' in df_existing.columns:
                existing = set(zip(df_existing['pattern_name'], df_existing['timestamp']))
        except Exception:
            pass
    appended = 0
    # Determine defaults
    resolved_run_tag = run_tag or os.environ.get('RUN_TAG', '')
    resolved_config_hash = _compute_config_hash()
    for gen_file in UNIFIED_RUNS.rglob('generations.csv'):
        pattern = gen_file.parent.name
        try:
            df = pd.read_csv(gen_file)
            if df.empty:
                continue
            # Assume last row best fitness
            final_row = df.iloc[-1]
            initial_row = df.iloc[0]
            initial_fitness = float(initial_row['BestFitness']) if 'BestFitness' in initial_row else 0.0
            final_fitness = float(final_row['BestFitness']) if 'BestFitness' in final_row else initial_fitness
            improvement = final_fitness - initial_fitness
            improvement_pct = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
            total_generations = int(final_row['Generation']) if 'Generation' in final_row else len(df)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(gen_file.stat().st_mtime))
            key = (pattern, timestamp)
            if key in existing:
                continue
            row = {
                'timestamp': timestamp,
                'run_tag': resolved_run_tag,
                'config_hash': resolved_config_hash,
                'pattern_name': pattern,
                'num_pieces': '',
                'total_generations': total_generations,
                'initial_fitness': initial_fitness,
                'final_fitness': final_fitness,
                'improvement_percent': improvement_pct,
                'fitness_at_gen_5': '', 'fitness_at_gen_10': '', 'fitness_at_gen_15': '', 'fitness_at_gen_20': '',
                'improvement_at_gen_5': '', 'improvement_at_gen_10': '', 'improvement_at_gen_15': '', 'improvement_at_gen_20': '',
                'elapsed_time': '', 'decoder': '', 'fitness_metric': '', 'crossover_method': '', 'mutation_rate': '',
                'container_width': '', 'container_height': ''
            }
            with open(master_path, 'a', newline='') as f:
                # Use master header order if available
                if fieldnames:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
                else:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
            appended += 1
        except Exception:
            continue
    return appended


def main():
    parser = argparse.ArgumentParser(description='Migrate legacy run logs/results to unified experiments directory.')
    parser.add_argument('--copy', action='store_true', help='Copy instead of move (retain legacy)')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild master_statistics.csv from generations.csv')
    parser.add_argument('--run-tag', type=str, default=None, help='Run tag to apply to rebuilt entries')
    args = parser.parse_args()
    #migrated = migrate(args.copy)
    #print(f"Migrated {len(migrated)} legacy run folders")
    if args.rebuild:
        added = rebuild_master(args.run_tag)
        print(f"Rebuilt master statistics entries added: {added}")
    print('Done.')

if __name__ == '__main__':
    main()
