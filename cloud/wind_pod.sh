#!/usr/bin/env bash
# Winding-field runs on a 4-GPU pod (campaign W2). Run from the unpacked bundle root (~/vss):
#   bash cloud/wind_pod.sh setup      # env (torch 2.14.1 cu130, nnunetv2 2.8.1, eval deps), unpack cells
#   bash cloud/wind_pod.sh train      # one variant per GPU, then the jeong200 real-site test for each; prints WIND_DONE
# Bundle layout (scripts/make_wind_bundle.sh): train/ eval/ sim/ forward/ cloud/, data/winding/pack_v1.tar.zst,
# data/eval/jeong200/{imagesTr,labelsTr}, the real350 trainer folder and the Dataset101 plans.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
ENV=$HOME/env
PY=$ENV/bin/python
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
# one math-library thread per process (see cloud/pod_train.sh: oversubscription stalled the first pod)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
INIT=$nnUNet_results/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres

setup() {
  set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd >/dev/null; }
  nvidia-smi --query-gpu=name,driver_version --format=csv
  if ! "$PY" -c "import torch, nnunetv2, cc3d, skimage" 2>/dev/null; then
    uv venv --allow-existing --python 3.12 "$ENV"
    uv pip install --python "$PY" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python "$PY" nnunetv2==2.8.1 zarr tifffile imagecodecs connected-components-3d scikit-image scipy s3fs
  fi
  "$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), 'GPUs')"
  [[ -d data/winding/pack_v1 ]] || tar -I zstd -xf data/winding/pack_v1.tar.zst -C data/winding
  echo "cells: $(ls data/winding/pack_v1 | wc -l)"
  echo SETUP_DONE
}

lane() {  # $1 gpu, $2 tag, rest = winding.py args
  local gpu=$1 tag=$2; shift 2
  local out=data/winding/$tag
  if [[ ! -f $out/DONE ]]; then
    CUDA_VISIBLE_DEVICES=$gpu "$PY" train/winding.py --cells data/winding/pack_v1 --out "$out" --workers 24 "$@" \
      > "data/scratch/train_wind_$tag.log" 2>&1 && touch "$out/DONE" || { echo "$tag train failed"; return 1; }
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" eval/winding_sites.py --ckpt "$out/checkpoint.pth" --out "data/preds/wind_$tag" \
    > "data/scratch/eval_wind_$tag.log" 2>&1 || echo "$tag eval failed"
  echo "$(date -Is) $tag done"
}

train() {
  mkdir -p data/scratch data/preds
  lane 0 w1 --epochs 60 --init "$INIT" &
  lane 1 w1s --epochs 60 &                          # from scratch: does the real-data init matter?
  lane 2 w1l --epochs 150 --init "$INIT" &          # longer
  lane 3 w1p --epochs 60 --patch 160 --init "$INIT" &  # more context per patch
  wait
  for t in w1 w1s w1l w1p; do echo "== $t"; cat "data/preds/wind_$t.json" 2>/dev/null; done
  echo WIND_DONE
}

"$@"
