"""Master experiment runner.

Experiment groups
-----------------
decoder_comparison   Section 5.1 — 5 decoder configs × N patterns, single pass each
garmentcode          Sections 5.3-5.5 — 6 configs × 99 patterns × 5 runs
ablation             Section 5.5 — 3 new configs × 99 patterns × 5 runs
                     (ne_full and ne_no_rotations are shared with garmentcode, skipped)
esicup               Section 5.6 — 2 configs × 7 instances × 5 runs

Usage
-----
    # Decoder comparison (1700 patterns, 1 run each)
    python -m nesting.run_all_experiments decoder_comparison \\
        --patterns-dir nesting-assets/pattern_files --n-patterns 1700

    # Main benchmark (list of 99 pattern spec paths, 5 runs each)
    python -m nesting.run_all_experiments garmentcode \\
        --patterns-file patterns_99.txt --n-runs 5

    # Ablation only (reuses same patterns as garmentcode)
    python -m nesting.run_all_experiments ablation \\
        --patterns-file patterns_99.txt --n-runs 5

    # ESICUP (7 instance spec paths, 5 runs each)
    python -m nesting.run_all_experiments esicup \\
        --patterns-dir nesting-assets/esicup --n-runs 5

    # Run specific configs only
    python -m nesting.run_all_experiments garmentcode \\
        --patterns-file patterns_99.txt --n-runs 5 --configs ne_full ne_no_rotations

Output
------
All results land in --output-dir (default: nesting/experiments/runs_fresh/).
Each group gets its own CSV:
    <output_dir>/decoder_comparison.csv
    <output_dir>/garmentcode.csv
    <output_dir>/ablation.csv
    <output_dir>/esicup.csv

Runs that already appear in the CSV are skipped (idempotent).
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import time
import traceback
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import nesting.config as config
from nesting.layout import Container, Piece
from nesting.operations import run_decoder
from nesting.path_extractor import PatternPathExtractor
from nesting.pipeliner import temporary_config
from assets.bodies.body_params import BodyParameters

# ---------------------------------------------------------------------------
# Named experiment configurations
# ---------------------------------------------------------------------------

_GA_MUTATIONS   = {"rotate": 1, "swap": 1, "inversion": 1, "insertion": 1,
                   "scramble": 1, "split": 0, "design_params": 0}
_NE_MUTATIONS   = {"rotate": 1, "swap": 1, "inversion": 1, "insertion": 1,
                   "scramble": 1, "split": 1, "design_params": 1}

# Configs that use Evolution (GA / NE variants).
# Keys map to config overrides applied via temporary_config().
SEARCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ga_sticky": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "sticky",
        "enable_rotations": True,
        "mutation_weights": _GA_MUTATIONS,
    },
    "ga_lexicographic": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": True,
        "mutation_weights": _GA_MUTATIONS,
    },
    "ne_full": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": True,
        "mutation_weights": _NE_MUTATIONS,
    },
    "ne_no_rotations": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": False,
        "mutation_weights": {**_NE_MUTATIONS, "rotate": 0},
    },
    "ne_no_splits": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": True,
        "mutation_weights": {**_NE_MUTATIONS, "split": 0},
    },
    "ne_no_params": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": True,
        "mutation_weights": {**_NE_MUTATIONS, "design_params": 0},
    },
    "ne_no_splits_no_params": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": True,
        "mutation_weights": _GA_MUTATIONS,
    },
    # two_exchange and random_search are handled separately below
}

# Decoder-only configs for Section 5.1.
# decoder_kwargs are forwarded to the decoder constructor.
DECODER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "BL":               {"decoder": "BL",  "decoder_kwargs": {}},
    "NFP_BL":           {"decoder": "NFP", "decoder_kwargs": {"placement_mode": "bottom_left"}},
    "NFP_max_overlap":  {"decoder": "NFP", "decoder_kwargs": {"placement_mode": "max_overlap"}},
    "NFP_min_bb_area":  {"decoder": "NFP", "decoder_kwargs": {"placement_mode": "min_bbox_area"}},
    "NFP_min_bb_length":{"decoder": "NFP", "decoder_kwargs": {"placement_mode": "min_bbox_length"}},
}

# Experiment groups and which configs they include
GROUP_CONFIGS = {
    "garmentcode": ["ga_sticky", "ga_lexicographic", "ne_full",
                    "ne_no_rotations", "two_exchange", "random_search"],
    "ablation":    ["ne_no_splits", "ne_no_params", "ne_no_splits_no_params"],
    # ne_full and ne_no_rotations are reused from garmentcode — skipped here
    "esicup":      ["ne_full", "ne_no_splits_no_params"],
}

# ---------------------------------------------------------------------------
# Pattern loading
# ---------------------------------------------------------------------------

def load_pattern(spec_path: Path) -> Tuple[str, Dict[str, Piece], Optional[dict], Optional[Any]]:
    """Load pieces, design params, and body params from a pattern spec file.

    Returns (pattern_name, pieces_dict, design_params, body_params).
    design_params / body_params may be None if files are missing.
    """
    stem = spec_path.stem
    pattern_name = stem.replace("_specification", "") if stem.endswith("_specification") else stem

    extractor = PatternPathExtractor(spec_path)
    panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
    if not panel_pieces:
        raise ValueError(f"No pieces extracted from {spec_path}")

    pieces: Dict[str, Piece] = {}
    for pid, piece in panel_pieces.items():
        p = copy.deepcopy(piece)
        p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
        p.translation = (0, 0)
        pieces[str(pid)] = p

    design_params = None
    dp_path = spec_path.parent / f"{pattern_name}_design_params.yaml"
    if dp_path.exists():
        with open(dp_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        design_params = data.get("design")

    body_params = None
    bp_path = spec_path.parent / f"{pattern_name}_body_measurements.yaml"
    if bp_path.exists():
        body_params = BodyParameters(bp_path)

    return pattern_name, pieces, design_params, body_params


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_metrics(pieces: List[Piece], container: Container) -> Dict[str, float]:
    """Run the configured decoder once and return all evaluation metrics."""
    dec = run_decoder(list(pieces), config.SELECTED_DECODER, container)
    usage_bb    = dec.usage_BB()
    rest_length = dec.rest_length()
    ch_util     = dec.concave_hull_utilization()
    ch_area     = dec.concave_hull_area()
    bb_area     = dec.bbox_area()
    fitness     = (usage_bb + rest_length / config.CONTAINER_WIDTH_CM) if usage_bb > 0 else 0.0
    return {
        "fitness":                  fitness,
        "usage_bb":                 usage_bb,
        "rest_length_cm":           rest_length,
        "concave_hull_utilization": ch_util,
        "concave_hull_area":        ch_area,
        "bb_area":                  bb_area,
    }


def collect_metrics_from_decoder_instance(dec) -> Dict[str, float]:
    """Same as collect_metrics but from an already-run decoder instance."""
    usage_bb    = dec.usage_BB()
    rest_length = dec.rest_length()
    ch_util     = dec.concave_hull_utilization()
    ch_area     = dec.concave_hull_area()
    bb_area     = dec.bbox_area()
    fitness     = (usage_bb + rest_length / config.CONTAINER_WIDTH_CM) if usage_bb > 0 else 0.0
    return {
        "fitness":                  fitness,
        "usage_bb":                 usage_bb,
        "rest_length_cm":           rest_length,
        "concave_hull_utilization": ch_util,
        "concave_hull_area":        ch_area,
        "bb_area":                  bb_area,
    }


# ---------------------------------------------------------------------------
# CSV helpers (append-safe, idempotent)
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = [
    "group", "config_name", "pattern_name", "run_id",
    "fitness", "usage_bb", "rest_length_cm",
    "concave_hull_utilization", "concave_hull_area", "bb_area",
    "num_pieces", "num_generations", "runtime_sec",
]


def _ensure_csv(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES).writeheader()


def _already_run(path: Path, config_name: str, pattern_name: str, run_id: int) -> bool:
    if not path.exists():
        return False
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("config_name") == config_name
                    and row.get("pattern_name") == pattern_name
                    and row.get("run_id") == str(run_id)):
                return True
    return False


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        w.writerow(row)


# ---------------------------------------------------------------------------
# Single-run executors
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_evolution_config(
    config_name: str,
    pieces: Dict[str, Piece],
    container: Container,
    design_params: Optional[dict],
    body_params: Optional[Any],
    pattern_name: str,
    run_id: int,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one Evolution (GA/NE) experiment and return a result row."""
    from nesting.evolution import Evolution
    from nesting.metastatistics import MetaStatistics

    seed = run_id * 1000 + hash(pattern_name) % 1000
    _set_seed(seed)

    with temporary_config(overrides):
        config._update_backward_compatibility_vars()
        run_tag = MetaStatistics._get_default_run_tag()
        config_hash = config.get_stable_hash()[:8]

        evo = Evolution(
            copy.deepcopy(pieces),
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
            run_tag=run_tag,
            config_hash=config_hash,
        )

        t0 = time.time()
        best = evo.run()
        runtime = time.time() - t0

        if best is None:
            raise RuntimeError("Evolution returned no solution")

        metrics = collect_metrics(list(best.genes), container)
        num_gens = getattr(evo, "generation", config.NUM_GENERATIONS)

    return {**metrics, "num_pieces": len(pieces),
            "num_generations": num_gens, "runtime_sec": round(runtime, 2)}


def run_two_exchange(
    pieces: Dict[str, Piece],
    container: Container,
    pattern_name: str,
    run_id: int,
) -> Dict[str, Any]:
    from nesting.two_exchange_search import TwoExchangeSearch

    seed = run_id * 1000 + hash(pattern_name) % 1000
    _set_seed(seed)

    piece_list = list(copy.deepcopy(pieces).values())
    t0 = time.time()
    search = TwoExchangeSearch(piece_list, container, verbose=False)
    best_pieces, _ = search.run()
    runtime = time.time() - t0

    metrics = collect_metrics(best_pieces, container)
    return {**metrics, "num_pieces": len(pieces),
            "num_generations": getattr(search, "iterations", 0),
            "runtime_sec": round(runtime, 2)}


def run_random_search(
    pieces: Dict[str, Piece],
    container: Container,
    pattern_name: str,
    run_id: int,
) -> Dict[str, Any]:
    from nesting.random_search import RandomSearch

    seed = run_id * 1000 + hash(pattern_name) % 1000

    piece_list = list(copy.deepcopy(pieces).values())
    t0 = time.time()
    search = RandomSearch(piece_list, container, seed=seed, verbose=False)
    best_pieces, _ = search.run()
    runtime = time.time() - t0

    metrics = collect_metrics(best_pieces, container)
    return {**metrics, "num_pieces": len(pieces),
            "num_generations": getattr(search, "num_samples", 0),
            "runtime_sec": round(runtime, 2)}


def run_decoder_only(
    decoder_name_key: str,
    decoder_cfg: Dict[str, Any],
    pieces: Dict[str, Piece],
    container: Container,
) -> Dict[str, Any]:
    """Single decode pass for the decoder-comparison sweep."""
    from nesting.placement_engine import DECODER_REGISTRY

    piece_list = list(copy.deepcopy(pieces).values())
    decoder_cls = DECODER_REGISTRY[decoder_cfg["decoder"]]
    t0 = time.time()
    dec = decoder_cls(piece_list, container, **decoder_cfg.get("decoder_kwargs", {}))
    dec.decode()
    runtime = time.time() - t0

    metrics = collect_metrics_from_decoder_instance(dec)
    return {**metrics, "num_pieces": len(pieces),
            "num_generations": 0, "runtime_sec": round(runtime, 4)}


# ---------------------------------------------------------------------------
# Per-group runners
# ---------------------------------------------------------------------------

def _collect_spec_paths(patterns_dir: Optional[str], patterns_file: Optional[str],
                        n_patterns: Optional[int]) -> List[Path]:
    paths: List[Path] = []
    if patterns_file:
        with open(patterns_file) as f:
            paths = [Path(line.strip()) for line in f if line.strip()]
    elif patterns_dir:
        paths = sorted(Path(patterns_dir).glob("*/*_specification.json"))
    else:
        raise ValueError("Provide --patterns-dir or --patterns-file")
    if n_patterns:
        paths = paths[:n_patterns]
    return paths


def run_group_decoder_comparison(
    spec_paths: List[Path],
    output_csv: Path,
    configs_filter: Optional[List[str]] = None,
) -> None:
    _ensure_csv(output_csv)
    active = {k: v for k, v in DECODER_CONFIGS.items()
              if configs_filter is None or k in configs_filter}

    for spec_path in spec_paths:
        try:
            pattern_name, pieces, _, _ = load_pattern(spec_path)
        except Exception as e:
            print(f"[SKIP] {spec_path.name}: {e}")
            continue

        container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)

        for cfg_name, dec_cfg in active.items():
            if _already_run(output_csv, cfg_name, pattern_name, run_id=0):
                print(f"[SKIP] {cfg_name} / {pattern_name} already done")
                continue
            try:
                result = run_decoder_only(cfg_name, dec_cfg, pieces, container)
                row = {"group": "decoder_comparison", "config_name": cfg_name,
                       "pattern_name": pattern_name, "run_id": 0, **result}
                _append_row(output_csv, row)
                print(f"[OK] {cfg_name} / {pattern_name}  "
                      f"F={result['fitness']:.4f}  RT={result['runtime_sec']:.3f}s")
            except Exception as e:
                print(f"[ERR] {cfg_name} / {pattern_name}: {e}")
                traceback.print_exc()


def run_group_search(
    group_name: str,
    spec_paths: List[Path],
    output_csv: Path,
    n_runs: int,
    configs_filter: Optional[List[str]] = None,
) -> None:
    _ensure_csv(output_csv)
    cfg_names = GROUP_CONFIGS[group_name]
    if configs_filter:
        cfg_names = [c for c in cfg_names if c in configs_filter]

    for spec_path in spec_paths:
        try:
            pattern_name, pieces, design_params, body_params = load_pattern(spec_path)
        except Exception as e:
            print(f"[SKIP] {spec_path.name}: {e}")
            continue

        container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)

        for cfg_name in cfg_names:
            for run_id in range(1, n_runs + 1):
                if _already_run(output_csv, cfg_name, pattern_name, run_id):
                    print(f"[SKIP] {cfg_name} / {pattern_name} / run {run_id}")
                    continue
                try:
                    if cfg_name == "two_exchange":
                        result = run_two_exchange(pieces, container, pattern_name, run_id)
                    elif cfg_name == "random_search":
                        result = run_random_search(pieces, container, pattern_name, run_id)
                    else:
                        overrides = SEARCH_CONFIGS[cfg_name]
                        result = run_evolution_config(
                            cfg_name, pieces, container,
                            design_params, body_params,
                            pattern_name, run_id, overrides,
                        )
                    row = {"group": group_name, "config_name": cfg_name,
                           "pattern_name": pattern_name, "run_id": run_id, **result}
                    _append_row(output_csv, row)
                    print(f"[OK] {cfg_name} / {pattern_name} / run {run_id}  "
                          f"F={result['fitness']:.4f}  RT={result['runtime_sec']:.1f}s")
                except Exception as e:
                    print(f"[ERR] {cfg_name} / {pattern_name} / run {run_id}: {e}")
                    traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master experiment runner for garment nesting paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("group", choices=["decoder_comparison", "garmentcode", "ablation", "esicup"],
                        help="Experiment group to run")
    parser.add_argument("--patterns-dir",  help="Root directory containing pattern folders")
    parser.add_argument("--patterns-file", help="Text file with one spec path per line")
    parser.add_argument("--n-patterns", type=int, default=None,
                        help="Cap on number of patterns (decoder_comparison)")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of independent runs per config×pattern")
    parser.add_argument("--output-dir", default="nesting/experiments/runs_fresh",
                        help="Directory to write CSVs (default: nesting/experiments/runs_fresh)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Run only these named configs (optional filter)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_paths = _collect_spec_paths(args.patterns_dir, args.patterns_file, args.n_patterns)
    print(f"Found {len(spec_paths)} pattern(s). Group: {args.group}")

    if args.group == "decoder_comparison":
        run_group_decoder_comparison(
            spec_paths,
            output_dir / "decoder_comparison.csv",
            configs_filter=args.configs,
        )
    elif args.group in ("garmentcode", "ablation", "esicup"):
        run_group_search(
            args.group,
            spec_paths,
            output_dir / f"{args.group}.csv",
            n_runs=args.n_runs,
            configs_filter=args.configs,
        )


if __name__ == "__main__":
    main()
