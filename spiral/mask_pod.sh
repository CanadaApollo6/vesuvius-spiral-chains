#!/bin/bash
# mask_pod.sh - on a multi-GPU pod: band surface masks with predict_band.py, P worker processes per GPU (nnU-Net's CPU
# resampling takes ~60 % of a block's wall time, so 3 processes keep one GPU busy). Resumable: rerun the same command;
# finished blocks are skipped (<out>/.blocks markers).
#
#   bash ~/vss/spiral/mask_pod.sh setup                       # venv with nnunetv2 + zarr 2 (torch by driver CUDA)
#   bash ~/vss/spiral/mask_pod.sh run <tag> <model_dir> [P]   # -> ~/masks/<tag>.zarr; logs ~/masks/<tag>.w*.log
#   bash ~/vss/spiral/mask_pod.sh status <tag>
# Band: Z0..Z1 = 2688..4800 (the level-0 chunk rows 14-24 that run_arm.sh fetches for tracks and SDT; level 1 covers
# rows 7-12). Volume 20250728140407 level 0, spacing 9.362 / 7.91 x plans spacing (m7 at 1.18x).
set -euo pipefail
CMD=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
CT=s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr
Z0=${Z0:-2688}; Z1=${Z1:-4800}
V=$HOME/mvenv; PY=$V/bin/python
mkdir -p ~/masks
case $CMD in
setup)
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH=$HOME/.local/bin:$PATH
  CU=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')
  if [[ ${CU%%.*} -ge 13 ]]; then IDX=https://download.pytorch.org/whl/cu130; TV=2.13.0
  else IDX=https://download.pytorch.org/whl/cu128; TV=2.11.0; fi
  [[ -x $PY ]] || uv venv -p 3.12 "$V"
  uv pip install -p "$PY" --index-url "$IDX" "torch==$TV"
  uv pip install -p "$PY" nnunetv2 "zarr==2.18.4" "numcodecs<0.16" s3fs
  "$PY" -c "import torch, nnunetv2, zarr; assert torch.cuda.is_available(); print('mvenv ok', torch.__version__, torch.cuda.device_count(), 'GPUs', zarr.__version__)"
  ;;
run)
  TAG=$1; MODEL=$2; P=${3:-3}
  OUT=$HOME/masks/$TAG.zarr
  NG=$(nvidia-smi -L | wc -l); N=$((NG * P))
  "$PY" "$HERE/predict_band.py" init --out "$OUT" --zarr "$CT"
  echo "$(date -u +%FT%TZ) $TAG: $N workers ($NG GPUs x $P)" | tee -a ~/masks/$TAG.timings
  for ((w = 0; w < N; w++)); do
    CUDA_VISIBLE_DEVICES=$((w % NG)) OMP_NUM_THREADS=2 setsid -f "$PY" "$HERE/predict_band.py" run --out "$OUT" \
      --zarr "$CT" --model "$MODEL" --fold 0 --z0 "$Z0" --z1 "$Z1" --worker $w --nworkers $N \
      > ~/masks/$TAG.w$w.log 2>&1 < /dev/null
  done
  ;;
status)
  TAG=$1; OUT=$HOME/masks/$TAG.zarr
  echo "$TAG: $(ls "$OUT/.blocks" 2>/dev/null | wc -l) blocks done; workers running: $(pgrep -fc "predict_band.py run --out $OUT" || true)"
  tail -qn1 ~/masks/$TAG.w*.log 2>/dev/null | head -4
  ;;
esac
