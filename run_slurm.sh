#!/usr/bin/env bash
#SBATCH -J nesting
#SBATCH -c 8
#SBATCH --mem-per-cpu=2G           # Euler: per-CPU mem; 8×2G = 16G total
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --hint=nomultithread
#SBATCH --threads-per-core=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CHUNKS="${CHUNKS:-4}"
CHUNK="${CHUNK:-1}"
BASE_DIR="nesting-assets/pattern_files"
MATCH_NAME='rand_*_specification.json'

# sanity
if ! [[ "$CHUNKS" =~ ^[1-9][0-9]*$ && "$CHUNK" =~ ^[1-9][0-9]*$ && "$CHUNK" -le "$CHUNKS" ]]; then
  echo "Set CHUNKS and CHUNK correctly, e.g. CHUNKS=4 CHUNK=1"; exit 1
fi

# Option B) Cap to exactly N files from anywhere under BASE_DIR
MAX_FILES="${MAX_FILES:-100}"
mapfile -t ALL_FILES < <(cd "$BASE_DIR" && find . -type f -name "$MATCH_NAME" | sed 's|^\./||' | sort | head -n "$MAX_FILES")

TOTAL=${#ALL_FILES[@]}
(( TOTAL > 0 )) || { echo "No files matched in $BASE_DIR"; exit 1; }

# ---------- SLICE FOR THIS CHUNK ----------
PER=$(( (TOTAL + CHUNKS - 1) / CHUNKS ))   # ceil
START=$(( (CHUNK - 1) * PER ))
END=$(( START + PER )) ; (( END > TOTAL )) && END=$TOTAL
COUNT=$(( END - START ))
(( COUNT > 0 )) || { echo "Chunk $CHUNK has no files (TOTAL=$TOTAL, CHUNKS=$CHUNKS)"; exit 1; }

echo "Matched: $TOTAL files | Chunks: $CHUNKS | This chunk (#$CHUNK): $COUNT files ([$START..$((END-1))])"

# ---------- LOG FILE LIST ----------
FILES_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_files.txt"
: > "$FILES_LOG"
for ((i=START; i<END; i++)); do echo "${ALL_FILES[$i]}"; done | tee -a "$FILES_LOG" >/dev/null
echo "File list saved to $FILES_LOG"

# ---------- STAGE ONLY THIS CHUNK ----------
STAGE_DIR="$(mktemp -d -p "$PWD" .nesting_stage_XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT
while IFS= read -r rel; do
  dirpath=$(dirname "$rel")
  mkdir -p "$STAGE_DIR/$dirpath"
  ln -s "$PWD/$BASE_DIR/$rel" "$STAGE_DIR/$rel"
  # Also stage related design params and body measurements files if they exist
  base="${rel%_specification.json}"
  dp_src="$PWD/$BASE_DIR/${base}_design_params.yaml"
  if [ -f "$dp_src" ]; then
    ln -s "$dp_src" "$STAGE_DIR/${base}_design_params.yaml"
  fi
  bm_src="$PWD/$BASE_DIR/${base}_body_measurements.yaml"
  if [ -f "$bm_src" ]; then
    ln -s "$bm_src" "$STAGE_DIR/${base}_body_measurements.yaml"
  fi
done < "$FILES_LOG"

# ---------- RUNTIME ENVS ----------
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

# ---------- RUN ----------
/usr/bin/time -v -o "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_resources.txt" \
srun --cpu-bind=cores \
  conda run -n garment python -m nesting.run_experiments \
    --dir "$STAGE_DIR" \
    --filter "*/$MATCH_NAME" \
    --limit 100

echo "Done. Processed $COUNT files (chunk $CHUNK/$CHUNKS). See $FILES_LOG for the exact list."
