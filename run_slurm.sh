#!/usr/bin/env bash
#SBATCH -J nesting
#SBATCH -c 9
#SBATCH --mem-per-cpu=16G
#SBATCH --time=25:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

conda run -n garment python -m nesting.run_experiments \
  --dir nesting-assets/pattern_files \
  --filter "*/rand_*_specification.json" \
  --limit 100