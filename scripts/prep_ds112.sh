#!/usr/bin/env bash
# Preprocess Dataset112_M7Contact and Dataset113_M7Ctrl with the public m7 plans (nnUNetResEncUNetLPlans: ResEnc-L,
# 192^3, spacing 1.0 = 7.91 um, m7's CT normalisation statistics), so pretrained m7 weights load unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
for ds in ${@:-Dataset112_M7Contact Dataset113_M7Ctrl}; do
  id=${ds:7:3}
  P=$nnUNet_preprocessed/$ds
  mkdir -p "$P"
  python3 - "$ds" "$P" <<'PY'
import json, sys
ds, P = sys.argv[1:]
p = json.load(open("data/models/surface_m7_nnunet/plans.json"))
p["dataset_name"] = ds
json.dump(p, open(f"{P}/nnUNetResEncUNetLPlans.json", "w"), indent=1)
PY
  cp "$nnUNet_raw/$ds/dataset.json" "$P/"
  uv run --with nnunetv2 nnUNetv2_extract_fingerprint -d "$id" -np 8
  uv run --with nnunetv2 nnUNetv2_preprocess -d "$id" -plans_name nnUNetResEncUNetLPlans -c 3d_fullres -np 8
done
echo PREP112_DONE
