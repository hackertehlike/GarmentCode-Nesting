"""Master experiment runner.

Experiment groups
-----------------
decoder_comparison   Section 5.1 — 5 decoder configs × N patterns, single pass each
garmentcode          Sections 5.3-5.5 — 9 configs × 100 patterns × 5 runs
                     Configs: ga_sticky, ga_lexicographic, ne_full, ne_no_rotations,
                              ne_no_splits, ne_no_params, ne_no_splits_no_params,
                              two_exchange, random_search
esicup               Section 5.6 — 2 configs × 7 instances × 5 runs

Mutation data
-------------
For ne_full and ne_no_rotations, per-mutation per-generation swarm data is saved
to <output_dir>/mutation_data.csv alongside the main results CSV.
Columns: pattern_name, config_name, run_id, generation, mutation_type, fitness_gain

Usage
-----
    # Decoder comparison (1700 patterns, 1 run each)
    python -m nesting.runners.run_all_experiments decoder_comparison \\
        --patterns-dir nesting-assets/pattern_files --n-patterns 1700

    # Main benchmark (100 pattern spec paths, 5 runs each)
    python -m nesting.runners.run_all_experiments garmentcode \\
        --patterns-file nesting-assets/patterns_100.txt --n-runs 5

    # ESICUP (instances hardcoded in ESICUP_INSTANCES, 5 runs each)
    # Spec files must exist in nesting-assets/ (albano.json, dagli.json, ...)
    # Each instance uses its own fabric width; seam allowance = 0.001 native units.
    python -m nesting.runners.run_all_experiments esicup --n-runs 5

    # Run specific configs only
    python -m nesting.runners.run_all_experiments garmentcode \\
        --patterns-file nesting-assets/patterns_100.txt --n-runs 5 --configs ne_full ne_no_rotations

Output
------
All results land in --output-dir (default: nesting/experiments/runs_fresh/).
Each group gets its own CSV:
    <output_dir>/decoder_comparison.csv
    <output_dir>/garmentcode.csv
    <output_dir>/esicup.csv
    <output_dir>/mutation_data.csv  (ne_full + ne_no_rotations only)

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
from nesting.core.layout import Container, Layout, LayoutView, Piece
from nesting.search.operations import run_decoder
from nesting.data.path_extractor import PatternPathExtractor
from nesting.runners.pipeliner import temporary_config
from assets.bodies.body_params import BodyParameters

# ---------------------------------------------------------------------------
# Named experiment configurations
# ---------------------------------------------------------------------------

_ALL_MUTATIONS  = {"rotate": 1, "swap": 1, "inversion": 1, "insertion": 1,
                   "scramble": 1, "split": 1, "design_params": 1}
_NE_MUTATIONS   = _ALL_MUTATIONS  # alias kept for clarity

# Configs that use Evolution (GA / NE variants).
# Keys map to config overrides applied via temporary_config().
SEARCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ga_sticky": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "sticky",
        "enable_rotations": True,
        "mutation_weights": _ALL_MUTATIONS,
    },
    "ga_lexicographic": {
        "selected_crossover": "cross_stitch_oxk",
        "cross_stitch_mode": "lexicographic",
        "enable_rotations": True,
        "mutation_weights": _ALL_MUTATIONS,
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
        "mutation_weights": {**_ALL_MUTATIONS, "split": 0, "design_params": 0},
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

# Experiment groups and which configs they include.
# "garmentcode" contains all 9 configs for the main benchmark table.
GROUP_CONFIGS = {
    "garmentcode": [
        "ga_sticky", "ga_lexicographic",
        "ne_full", "ne_no_rotations",
        "ne_no_splits", "ne_no_params", "ne_no_splits_no_params",
        "two_exchange", "random_search",
    ],
    "esicup": ["ne_full", "ne_no_splits_no_params"],
}

# Configs for which per-mutation per-generation data is captured
_MUTATION_LOGGING_CONFIGS = {"ne_full", "ne_no_rotations"}

# ---------------------------------------------------------------------------
# ESICUP instance metadata
# ---------------------------------------------------------------------------
# Each ESICUP instance has its own fabric width (container height) in its
# original coordinate units.  No coordinate scaling is applied.
# Seam allowance is set to 0.001 (in native units) rather than 0 because
# zero seam allowance causes geometry issues in the NFP pipeline; 0.001 is
# effectively negligible.
# Container width (open strip dimension) is 400 in the same native units.
#
# NOTE: Albano is listed in the paper but the spec file is not yet present in
# nesting-assets/; it will be skipped at runtime until the file is added.
_ASSETS = Path("nesting-assets")
ESICUP_INSTANCES: Dict[str, Dict[str, Any]] = {
    "albano":   {"spec": _ASSETS / "albano.json",   "container_width": 400, "container_height":  96},
    "dagli":    {"spec": _ASSETS / "dagli.json",    "container_width": 400, "container_height":  60},
    "mao":      {"spec": _ASSETS / "mao.json",      "container_width": 400, "container_height":  51},
    "marques":  {"spec": _ASSETS / "marques.json",  "container_width": 400, "container_height": 104},
    "shirts":   {"spec": _ASSETS / "shirts.json",   "container_width": 400, "container_height":  40},
    "swim":     {"spec": _ASSETS / "swim.json",     "container_width": 400, "container_height":  57},
    "trousers": {"spec": _ASSETS / "trousers.json", "container_width": 400, "container_height":  79},
}

# ---------------------------------------------------------------------------
# Pattern loading
# ---------------------------------------------------------------------------

def load_pattern(spec_path: Path,
                 seam_allowance: Optional[float] = None,
                 ) -> Tuple[str, Dict[str, Piece], Optional[dict], Optional[Any]]:
    """Load pieces, design params, and body params from a pattern spec file.

    Parameters
    ----------
    spec_path : Path
        Path to the pattern JSON specification.
    seam_allowance : float, optional
        Override seam allowance in coordinate units.  Defaults to
        ``config.SEAM_ALLOWANCE_CM``.  Pass 0.0 for ESICUP instances.

    Returns
    -------
    (pattern_name, pieces_dict, design_params, body_params)
    design_params / body_params may be None if files are missing.
    """
    if seam_allowance is None:
        seam_allowance = config.SEAM_ALLOWANCE_CM

    stem = spec_path.stem
    pattern_name = stem.replace("_specification", "") if stem.endswith("_specification") else stem

    extractor = PatternPathExtractor(spec_path)
    panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
    if not panel_pieces:
        raise ValueError(f"No pieces extracted from {spec_path}")

    pieces: Dict[str, Piece] = {}
    for pid, piece in panel_pieces.items():
        p = copy.deepcopy(piece)
        p.add_seam_allowance(seam_allowance)
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

def _metrics_from_dec(dec, container: Container) -> Dict[str, float]:
    usage_bb       = dec.usage_BB()
    rest_length    = dec.rest_length()
    solution_length = container.width - rest_length   # F in paper: occupied strip length
    ch_util        = dec.concave_hull_utilization()
    ch_area        = dec.concave_hull_area()
    bb_area        = dec.bbox_area()
    fitness        = (usage_bb + rest_length / container.width) if usage_bb > 0 else 0.0
    return {
        "fitness":                  fitness,
        "usage_bb":                 usage_bb,
        "solution_length_cm":       solution_length,  # F = container.width - rest_length
        "concave_hull_utilization": ch_util,
        "concave_hull_area":        ch_area,
        "bb_area":                  bb_area,
    }


def collect_metrics(pieces: List[Piece], container: Container) -> Dict[str, float]:
    """Run the configured decoder once and return all evaluation metrics."""
    dec = run_decoder(list(pieces), config.SELECTED_DECODER, container)
    return _metrics_from_dec(dec, container)


def collect_metrics_from_decoder_instance(dec, container: Container) -> Dict[str, float]:
    """Same as collect_metrics but from an already-run decoder instance."""
    return _metrics_from_dec(dec, container)


# ---------------------------------------------------------------------------
# CSV helpers (append-safe, idempotent)
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = [
    "group", "config_name", "pattern_name", "run_id",
    "fitness", "usage_bb", "solution_length_cm",
    "concave_hull_utilization", "concave_hull_area", "bb_area",
    "num_pieces", "num_generations", "runtime_sec",
]

_MUTATION_CSV_FIELDNAMES = [
    "config_name", "pattern_name", "run_id",
    "generation", "mutation_type", "fitness_gain",
]


def _ensure_csv(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES).writeheader()


def _ensure_mutation_csv(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_MUTATION_CSV_FIELDNAMES).writeheader()


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


def _append_mutation_rows(path: Path, config_name: str, pattern_name: str,
                          run_id: int, mutation_df) -> None:
    """Append per-mutation swarm rows to the mutation data CSV."""
    if mutation_df is None or mutation_df.empty:
        return
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MUTATION_CSV_FIELDNAMES, extrasaction="ignore")
        for _, r in mutation_df.iterrows():
            w.writerow({
                "config_name":   config_name,
                "pattern_name":  pattern_name,
                "run_id":        run_id,
                "generation":    r.get("generation"),
                "mutation_type": r.get("mutation_type"),
                "fitness_gain":  r.get("fitness_gain"),
            })


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
    from nesting.search.evolution import Evolution
    from nesting.analysis.metastatistics import MetaStatistics

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
        mutation_df = getattr(evo, "_mutation_swarm_data", None)

    return (
        {**metrics, "num_pieces": len(pieces),
         "num_generations": num_gens, "runtime_sec": round(runtime, 2)},
        mutation_df,
    )


def run_two_exchange(
    pieces: Dict[str, Piece],
    container: Container,
    pattern_name: str,
    run_id: int,
) -> Dict[str, Any]:
    from nesting.search.two_exchange_search import TwoExchangeSearch

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
    from nesting.search.random_search import RandomSearch

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
    from nesting.placement.placement_engine import DECODER_REGISTRY

    piece_list = list(copy.deepcopy(pieces).values())
    layout = LayoutView(piece_list)
    decoder_cls = DECODER_REGISTRY[decoder_cfg["decoder"]]
    t0 = time.time()
    dec = decoder_cls(layout, container, **decoder_cfg.get("decoder_kwargs", {}))
    dec.decode()
    runtime = time.time() - t0

    metrics = collect_metrics_from_decoder_instance(dec, container)
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

    n_total = len(spec_paths)
    n_done = 0
    t_group_start = time.time()
    print(f"[decoder_comparison] {n_total} patterns × {len(active)} decoders  →  {output_csv}")

    for i, spec_path in enumerate(spec_paths, 1):
        try:
            pattern_name, pieces, _, _ = load_pattern(spec_path)
        except Exception as e:
            print(f"  [SKIP] {spec_path.name}: {e}")
            continue

        container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)

        for cfg_name, dec_cfg in active.items():
            if _already_run(output_csv, cfg_name, pattern_name, run_id=0):
                continue
            try:
                result = run_decoder_only(cfg_name, dec_cfg, pieces, container)
                row = {"group": "decoder_comparison", "config_name": cfg_name,
                       "pattern_name": pattern_name, "run_id": 0, **result}
                _append_row(output_csv, row)
                n_done += 1
            except Exception as e:
                print(f"  [ERR] {cfg_name} / {pattern_name}: {e}")
                traceback.print_exc()

        # Progress line after each pattern
        elapsed = time.time() - t_group_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n_total - i) / rate if rate > 0 else 0
        print(f"  [{i}/{n_total}] {pattern_name}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")


def run_group_search(
    group_name: str,
    spec_paths: List[Path],
    output_csv: Path,
    n_runs: int,
    configs_filter: Optional[List[str]] = None,
) -> None:
    _ensure_csv(output_csv)
    mutation_csv = output_csv.parent / "mutation_data.csv"
    _ensure_mutation_csv(mutation_csv)

    cfg_names = GROUP_CONFIGS[group_name]
    if configs_filter:
        cfg_names = [c for c in cfg_names if c in configs_filter]

    n_patterns = len(spec_paths)
    n_total_runs = n_patterns * len(cfg_names) * n_runs
    n_done = 0
    t_group_start = time.time()
    print(f"[{group_name}] {n_patterns} patterns × {len(cfg_names)} configs × {n_runs} runs "
          f"= {n_total_runs} total runs  →  {output_csv}")
    print(f"  configs: {cfg_names}")

    for i, spec_path in enumerate(spec_paths, 1):
        try:
            pattern_name, pieces, design_params, body_params = load_pattern(spec_path)
        except Exception as e:
            print(f"  [SKIP] {spec_path.name}: {e}")
            continue

        container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)
        t_pattern_start = time.time()

        for cfg_name in cfg_names:
            for run_id in range(1, n_runs + 1):
                if _already_run(output_csv, cfg_name, pattern_name, run_id):
                    n_done += 1
                    continue
                try:
                    mutation_df = None
                    if cfg_name == "two_exchange":
                        result = run_two_exchange(pieces, container, pattern_name, run_id)
                    elif cfg_name == "random_search":
                        result = run_random_search(pieces, container, pattern_name, run_id)
                    else:
                        overrides = SEARCH_CONFIGS[cfg_name]
                        result, mutation_df = run_evolution_config(
                            cfg_name, pieces, container,
                            design_params, body_params,
                            pattern_name, run_id, overrides,
                        )
                    row = {"group": group_name, "config_name": cfg_name,
                           "pattern_name": pattern_name, "run_id": run_id, **result}
                    _append_row(output_csv, row)
                    if cfg_name in _MUTATION_LOGGING_CONFIGS and mutation_df is not None:
                        _append_mutation_rows(mutation_csv, cfg_name, pattern_name,
                                              run_id, mutation_df)
                    n_done += 1
                    print(f"    [OK] {cfg_name} run {run_id}  "
                          f"F={result['fitness']:.4f}  RT={result['runtime_sec']:.1f}s")
                except Exception as e:
                    print(f"    [ERR] {cfg_name} / run {run_id}: {e}")
                    traceback.print_exc()

        # Progress summary after each pattern
        elapsed = time.time() - t_group_start
        pattern_time = time.time() - t_pattern_start
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (n_total_runs - n_done) / rate if rate > 0 else 0
        print(f"  [{i}/{n_patterns}] {pattern_name}  "
              f"({n_done}/{n_total_runs} runs done)  "
              f"pattern_time={pattern_time:.0f}s  ETA={eta/60:.1f}min")


# ---------------------------------------------------------------------------
# ESICUP runner (per-instance container dimensions, no seam allowance)
# ---------------------------------------------------------------------------

def run_group_esicup(
    output_csv: Path,
    n_runs: int,
    configs_filter: Optional[List[str]] = None,
) -> None:
    """Run ESICUP benchmark.

    Each instance uses its own fabric width (container height) in native
    coordinate units.  Seam allowance is 0.001 native units (effectively
    negligible; 0 causes NFP geometry issues).
    """
    _ensure_csv(output_csv)
    mutation_csv = output_csv.parent / "mutation_data.csv"
    _ensure_mutation_csv(mutation_csv)

    cfg_names = GROUP_CONFIGS["esicup"]
    if configs_filter:
        cfg_names = [c for c in cfg_names if c in configs_filter]

    for instance_name, meta in ESICUP_INSTANCES.items():
        spec_path = meta["spec"]
        if not spec_path.exists():
            print(f"[SKIP] ESICUP/{instance_name}: spec file not found ({spec_path})")
            continue

        try:
            _, pieces, design_params, body_params = load_pattern(
                spec_path, seam_allowance=0.001
            )
        except Exception as e:
            print(f"[SKIP] ESICUP/{instance_name}: {e}")
            continue

        container = Container(meta["container_width"], meta["container_height"])

        for cfg_name in cfg_names:
            for run_id in range(1, n_runs + 1):
                if _already_run(output_csv, cfg_name, instance_name, run_id):
                    print(f"[SKIP] {cfg_name} / {instance_name} / run {run_id}")
                    continue
                try:
                    mutation_df = None
                    overrides = SEARCH_CONFIGS[cfg_name]
                    result, mutation_df = run_evolution_config(
                        cfg_name, pieces, container,
                        design_params, body_params,
                        instance_name, run_id, overrides,
                    )
                    row = {"group": "esicup", "config_name": cfg_name,
                           "pattern_name": instance_name, "run_id": run_id, **result}
                    _append_row(output_csv, row)
                    if cfg_name in _MUTATION_LOGGING_CONFIGS and mutation_df is not None:
                        _append_mutation_rows(mutation_csv, cfg_name, instance_name,
                                              run_id, mutation_df)
                    print(f"[OK] {cfg_name} / {instance_name} / run {run_id}  "
                          f"F={result['fitness']:.4f}  RT={result['runtime_sec']:.1f}s")
                except Exception as e:
                    print(f"[ERR] {cfg_name} / {instance_name} / run {run_id}: {e}")
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
    parser.add_argument("group", choices=["decoder_comparison", "garmentcode", "esicup"],
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

    if args.group == "esicup":
        print(f"Group: esicup  ({len(ESICUP_INSTANCES)} instances defined)")
        run_group_esicup(
            output_dir / "esicup.csv",
            n_runs=args.n_runs,
            configs_filter=args.configs,
        )
    else:
        spec_paths = _collect_spec_paths(args.patterns_dir, args.patterns_file, args.n_patterns)
        print(f"Found {len(spec_paths)} pattern(s). Group: {args.group}")

        if args.group == "decoder_comparison":
            run_group_decoder_comparison(
                spec_paths,
                output_dir / "decoder_comparison.csv",
                configs_filter=args.configs,
            )
        elif args.group == "garmentcode":
            run_group_search(
                "garmentcode",
                spec_paths,
                output_dir / "garmentcode.csv",
                n_runs=args.n_runs,
                configs_filter=args.configs,
            )


if __name__ == "__main__":
    main()
