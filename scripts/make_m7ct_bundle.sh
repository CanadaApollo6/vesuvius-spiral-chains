#!/usr/bin/env bash
# Bundle for the m7 contact fine-tune pod (cloud/m7ct_pod.sh) -> data/scratch/m7ct_bundle.tar. Private, pod use only
# (CC BY-NC derived data). Dataset112/113 are hardlinks into Dataset107/111; tar -h stores the files.
set -euo pipefail
tar -chf data/scratch/m7ct_bundle.tar train cloud scripts/prep_ds112.sh scripts/make_ds112.py \
  data/nnunet/nnUNet_raw/Dataset112_M7Contact data/nnunet/nnUNet_raw/Dataset113_M7Ctrl \
  data/models/surface_m7_nnunet/plans.json data/models/surface_m7_nnunet/dataset.json \
  data/models/surface_m7_nnunet/fold_0/checkpoint_best.pth
ls -la data/scratch/m7ct_bundle.tar
