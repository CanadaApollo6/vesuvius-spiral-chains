#!/usr/bin/env bash
# Bundle for cloud/real9_pod.sh -> data/scratch/real9_bundle.tar (stays private; pod use only).
set -euo pipefail
R=data/nnunet/nnUNet_results
P=data/nnunet/nnUNet_preprocessed/Dataset101_DS059real
tar -chf data/scratch/real9_bundle.tar train cloud scripts/prep_ds103.sh \
  data/nnunet/nnUNet_raw/Dataset103_Real9 \
  $P/nnUNetPlans.json $P/dataset.json $P/dataset_fingerprint.json \
  data/nnunet/nnUNet_raw/Dataset101_DS059real/dataset.json \
  $R/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres/{plans.json,dataset.json,dataset_fingerprint.json,fold_0/checkpoint_final.pth} \
  $R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/{plans.json,dataset.json,fold_0/checkpoint_final.pth}
ls -la data/scratch/real9_bundle.tar
