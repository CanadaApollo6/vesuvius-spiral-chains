#!/usr/bin/env bash
# Build the winding-field pod bundle in data/cloud/wind/ (rsync the directory to the pod as ~/vss).
#   bash scripts/make_wind_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/.."
B=data/cloud/wind
R=data/nnunet/nnUNet_results/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres
mkdir -p $B/data/winding $B/data/eval/jeong200 $B/$R/fold_0 $B/data/nnunet/nnUNet_preprocessed/Dataset101_DS059real
uv run python scripts/pack_wind.py data/sim/v1 data/winding/pack_v1
[[ -f $B/data/winding/pack_v1.tar.zst ]] || tar -C data/winding -cf - pack_v1 | zstd -T0 -3 -o $B/data/winding/pack_v1.tar.zst
rsync -a train eval sim forward cloud $B/ --exclude __pycache__
rsync -a data/eval/jeong200/imagesTr data/eval/jeong200/labelsTr $B/data/eval/jeong200/
cp $R/plans.json $R/dataset.json $B/$R/ && cp $R/fold_0/checkpoint_final.pth $B/$R/fold_0/
cp data/nnunet/nnUNet_preprocessed/Dataset101_DS059real/nnUNetPlans.json $B/data/nnunet/nnUNet_preprocessed/Dataset101_DS059real/
du -sh $B
