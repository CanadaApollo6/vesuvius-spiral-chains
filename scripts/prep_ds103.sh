#!/usr/bin/env bash
# Dataset103_Real9 (scripts/real9_trainset.py): preprocess with Dataset101's plans (same architecture/normalization,
# spacing 1.0 = native 9.362 / 8.64 um, no resampling).
set -euo pipefail
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
uv run --with nnunetv2 nnUNetv2_extract_fingerprint -d 103 -np 8
uv run --with nnunetv2 nnUNetv2_move_plans_between_datasets -s 101 -t 103 -sp nnUNetPlans -tp nnUNetPlans
cp "$nnUNet_raw/Dataset103_Real9/dataset.json" "$nnUNet_preprocessed/Dataset103_Real9/"  # move_plans does not copy it
uv run --with nnunetv2 nnUNetv2_preprocess -d 103 -plans_name nnUNetPlans -c 3d_fullres -np 8
echo PREP_DONE
