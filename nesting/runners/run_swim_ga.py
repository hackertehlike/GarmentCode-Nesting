"""
Run GA on nesting-assets/mao.json 10 times with specific overrides.

Overrides:
- allowed_rotations: [0, 90, 180, 270]
- population_weights: { elites: 0.25, offspring: 0, mutants: 0.5, randoms: 0.25 }
- mutation_weights: { rotate: 1, swap: 1, inversion: 1, insertion: 1, scramble: 1, split: 1, design_params: 0 }
- container_width_cm: 200
- container_height_cm: 51
- seam_allowance_cm: 0.001

All other settings remain as defined in nesting.config.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for local imports when executed from any CWD
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import nesting.config as config
from nesting.runners.run_experiments import run_ga_on_patterns


def main(reps: int = 10):
    # Apply requested overrides without changing global defaults
    overrides = dict(
        allowed_rotations=[0, 180],
        population_weights={
            "elites": 0.25, "offspring": 0, "mutants": 0.5, "randoms": 0.25
        },
        mutation_weights={
            "rotate": 1, "swap": 1, "inversion": 1, "insertion": 1,
            "scramble": 1, "split": 1, "design_params": 0
        },
        container_width_cm=400,
        container_height_cm=57,
        seam_allowance_cm=0.001,
    )
    config.load_config(**overrides)

    # Resolve swim.json path relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    pattern_path = repo_root / "nesting-assets" / "swim.json"
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    # Create a top-level results folder for these runs
    base_out = repo_root / "results" / "swim_experiment"
    os.makedirs(base_out, exist_ok=True)

    print("Running GA on:", pattern_path)
    print("Repetitions:", reps)
    print("Active config overrides applied.")

    # Run multiple times with unique output directories to avoid clobbering
    for i in range(1, reps + 1):
        out_dir = base_out / f"run_{i:02d}"
        print(f"\n=== Starting repetition {i}/{reps} -> {out_dir} ===")
        run_ga_on_patterns([str(pattern_path)], output_dir=str(out_dir))


if __name__ == "__main__":
    # Default to 10 repetitions; allow override via env var if desired
    reps_str = os.getenv("SWIM_GA_REPS", "10")
    try:
        reps = int(reps_str)
    except ValueError:
        reps = 10
    main(reps)
