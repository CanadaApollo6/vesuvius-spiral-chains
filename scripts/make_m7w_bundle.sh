#!/usr/bin/env bash
# Bundle for the round-2 distill lanes (Dataset107 + Dataset109, real350 and m7d300 inits) -> data/scratch/m7w_bundle.tar.
# Private, pod use only (CC BY-NC derived). 109 shares inodes with 107 (hardlinks), stored once by tar.
set -euo pipefail
R=data/nnunet/nnUNet_results
P=data/nnunet/nnUNet_preprocessed/Dataset101_DS059real
tar -chf data/scratch/m7w_bundle.tar train cloud scripts/prep_ds103.sh \
  data/nnunet/nnUNet_raw/Dataset103_Real9 data/nnunet/nnUNet_raw/Dataset107_M7Distill data/nnunet/nnUNet_raw/Dataset109_M7Wide \
  $P/nnUNetPlans.json $P/dataset.json $P/dataset_fingerprint.json \
  data/nnunet/nnUNet_raw/Dataset101_DS059real/dataset.json \
  $R/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres/{plans.json,dataset.json,dataset_fingerprint.json,fold_0/checkpoint_final.pth} \
  $R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/{plans.json,dataset.json,fold_0/checkpoint_final.pth} \
  $R/Dataset107_M7Distill/nnUNetTrainer_m7d300__nnUNetPlans__3d_fullres/{plans.json,dataset.json,fold_0/checkpoint_final.pth}
ls -la data/scratch/m7w_bundle.tar
