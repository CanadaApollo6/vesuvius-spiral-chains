#!/usr/bin/env bash
# Sim training set: randomized scrolls, N cells each. Resumable (skips scroll seeds with all cells present).
#   scripts/gen_sim.sh [out=data/sim/v1] [first_seed=101] [n_seeds=20] [cells=20]
set -euo pipefail
OUT=${1:-data/sim/v1}; FIRST=${2:-101}; NS=${3:-20}; CELLS=${4:-20}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
for ((s = FIRST; s < FIRST + NS; s++)); do
  have=$(find "$OUT" -maxdepth 1 -name "s$(printf %03d "$s")_c000_*.zarr" | wc -l)
  if [[ $have -ge $CELLS ]]; then echo "seed $s done"; continue; fi
  uv run python sim/generate.py --out "$OUT" --scroll-seed "$s" --cells "$CELLS" --randomize 2>&1 | grep --line-buffered -v Warn
done
echo ALL_DONE
