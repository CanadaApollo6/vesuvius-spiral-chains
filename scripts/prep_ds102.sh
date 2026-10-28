#!/usr/bin/env bash
# Dataset102_SimV1: raw tifs from data/sim/v1, then preprocess with Dataset101's plans (identical architecture/normalization).
set -euo pipefail
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
uv run python train/make_datasets.py sim --cells data/sim/v1
uv run --with nnunetv2 nnUNetv2_extract_fingerprint -d 102 -np 8
uv run --with nnunetv2 nnUNetv2_move_plans_between_datasets -s 101 -t 102 -sp nnUNetPlans -tp nnUNetPlans
cp "$nnUNet_raw/Dataset102_SimV1/dataset.json" "$nnUNet_preprocessed/Dataset102_SimV1/"  # move_plans does not copy it
uv run --with nnunetv2 nnUNetv2_preprocess -d 102 -plans_name nnUNetPlans -c 3d_fullres -np 8
echo PREP_DONE
