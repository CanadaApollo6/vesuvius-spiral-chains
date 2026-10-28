#!/usr/bin/env bash
# 9 um (ESRF stage) sim cells + winding-field training on a multi-GPU pod (campaign W1/W2). Run from the bundle root
# (~/vss), after `bash cloud/wind_pod.sh setup` (same env):
#   bash cloud/sim9_pod.sh gen [gpus=0,1,2,3]     # seeds 101-120 x 20 cells, 192^3 at 9.362 um -> data/sim/v9; GEN_DONE
#   bash cloud/sim9_pod.sh train [gpu=0] [tag=w9] [extra winding.py args...]          # -> data/winding/<tag>; TRAIN_DONE
# Needs forward/ct_params.json with stages["9p362"] (forward/fit9.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
OUT=data/sim/v9
STAGE=${STAGE:-9p362}
mkdir -p $OUT data/scratch

gen_lane() {  # $1 gpu, rest = scroll seeds
  local gpu=$1; shift
  for s in "$@"; do
    have=$(find $OUT -maxdepth 1 -name "s${s}_c001_*.zarr" | wc -l)
    [[ $have -ge 20 ]] && continue
    CUDA_VISIBLE_DEVICES=$gpu "$PY" sim/generate.py --out $OUT --scroll-seed "$s" --cell-seed 1 --cells 20 --size 192 \
      --randomize --stage "$STAGE" ${JITTER:+--stage-jitter} 2>&1 | grep --line-buffered -v Warn
  done
}

gen() {
  IFS=, read -ra G <<< "${1:-0,1,2,3}"
  local n=${#G[@]} i=0
  for g in "${G[@]}"; do
    seeds=(); for ((s = 101 + i; s <= 120; s += n)); do seeds+=("$s"); done
    gen_lane "$g" "${seeds[@]}" > "data/scratch/gen9_gpu$g.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  echo "cells: $(ls -d $OUT/*.zarr | wc -l)"
  echo GEN_DONE
}

train() {
  local gpu=${1:-0} tag=${2:-w9}; shift 2 || shift $#
  local INIT=$nnUNet_results/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train/winding.py --cells $OUT --res "$STAGE" --out "data/winding/$tag" --workers 24 \
    --init "$INIT" "$@" > "data/scratch/train_wind_$tag.log" 2>&1 && touch "data/winding/$tag/DONE"
  echo TRAIN_DONE
}

"$@"
