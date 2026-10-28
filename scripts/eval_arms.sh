#!/usr/bin/env bash
# Score the ablation arms on jeong200 (200 real 7.91 um patches): inference -> separation -> official metric.
#   bash scripts/eval_arms.sh [tag ...]       (default: all arms; resumable, skips finished steps)
# Outputs: data/preds/<tag>/*.tif, data/preds/sep_<tag>.{csv,json}, data/preds/score_<tag>.csv (+ .log per step).
# Holds a 6 GB lease in $HOME/.gpu-leases while inference runs (GPU shared with other sessions).
set -uo pipefail
cd "$(dirname "$0")/.."
R=data/nnunet/nnUNet_results
declare -A MODEL=(
  [real250]=$R/Dataset101_DS059real/nnUNetTrainer_real250__nnUNetPlans__3d_fullres
  [squeeze250]=$R/Dataset101_DS059real/nnUNetTrainer_squeeze250__nnUNetPlans__3d_fullres
  [simpre]=$R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres
  [simft]=$R/Dataset101_DS059real/nnUNetTrainer_simft__nnUNetPlans__3d_fullres
  [real350]=$R/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres
)
TAGS=("$@"); [[ ${#TAGS[@]} -eq 0 ]] && TAGS=(real250 squeeze250 simpre simft real350)
LEASE=$HOME/.gpu-leases/vesuvius-synth-scrolls.json
trap 'rm -f "$LEASE"' EXIT
N=$(ls data/eval/jeong200/imagesTr/*_0000.tif | wc -l)

for t in "${TAGS[@]}"; do
  P=data/preds/$t
  if [[ $(find "$P" -name "*.tif" 2>/dev/null | wc -l) -lt $N ]]; then
    echo "{\"session\": \"vesuvius-synth-scrolls\", \"max_mem_gb\": 6, \"max_procs\": 1, \"pids\": [$$], \"note\": \"nnU-Net inference $t on jeong200\", \"updated\": \"$(date -Is)\"}" > "$LEASE.tmp" && mv "$LEASE.tmp" "$LEASE"
    echo "$(date -Is) infer $t"
    uv run --with nnunetv2 python eval/infer_nnunet.py --model "${MODEL[$t]}" --fold 0 --out "$P" 2>&1 \
      | grep --line-buffered -v Warn > "data/preds/$t.log" || { echo "infer $t failed"; continue; }
    rm -f "$LEASE"
  fi
  [[ -f data/preds/sep_$t.json ]] || {
    echo "$(date -Is) separation $t"
    uv run python eval/separation.py --pred-dir "$P" --out "data/preds/sep_$t.csv" > /dev/null 2> "data/preds/sep_$t.log"
  }
  [[ -f data/preds/score_$t.csv ]] || {
    echo "$(date -Is) score $t"
    data/metric/.venv/bin/python eval/score.py --gt-dir data/eval/jeong200/labelsTr --pred-dir "$P" \
      --out "data/preds/score_$t.csv" > "data/preds/score_$t.log" 2>&1
    tail -1 "data/preds/score_$t.log"
  }
done
echo EVAL_DONE
