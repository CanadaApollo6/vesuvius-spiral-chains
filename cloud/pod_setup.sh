#!/usr/bin/env bash
# Prime Intellect pod setup (image ubuntu_22_cuda_12). Run from the unpacked bundle root (~/vss).
#   bash cloud/pod_setup.sh
# 1. Python env pinned to the local versions (torch 2.14.1 cu130, nnunetv2 2.8.1); needs an NVIDIA driver >= 580.
# 2. Real training patches (Dataset059 subset listed in data/train/ds059/split.json) downloaded from dl.ash2txt.org.
# 3. nnU-Net preprocessing of Dataset101 (real) and Dataset102 (sim) with the plans shipped in the bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

ENV=$HOME/env
if ! "$ENV/bin/python" -c "import torch, nnunetv2" 2>/dev/null; then
  uv venv --allow-existing --python 3.12 "$ENV"
  uv pip install --python "$ENV/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python "$ENV/bin/python" nnunetv2==2.8.1 tifffile numpy
fi
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), 'GPUs')"

# real patches
mkdir -p data/train/ds059/imagesTr data/train/ds059/labelsTr
"$ENV/bin/python" - <<'PY' > /tmp/ds059_urls.txt
import json
d = json.load(open("data/train/ds059/split.json"))
for s in d["train"]:
    print(f"{d['source']}imagesTr/{s}_0000.tif data/train/ds059/imagesTr/{s}_0000.tif")
    print(f"{d['source']}labelsTr/{s}.tif data/train/ds059/labelsTr/{s}.tif")
PY
xargs -P 24 -n 2 sh -c '[ -s "$1" ] || curl -sfL --retry 5 -o "$1.part" "$0" && { [ -s "$1" ] || mv "$1.part" "$1"; }' < /tmp/ds059_urls.txt
n_img=$(find data/train/ds059/imagesTr -name "*.tif" -size +0 | wc -l)
n_lab=$(find data/train/ds059/labelsTr -name "*.tif" -size +0 | wc -l)
echo "real patches: $n_img images, $n_lab labels (expected $(($(wc -l < /tmp/ds059_urls.txt) / 2)))"
[[ $n_img -eq $n_lab && $n_img -eq $(($(wc -l < /tmp/ds059_urls.txt) / 2)) ]]

export nnUNet_raw=$ROOT/data/nnunet/nnUNet_raw nnUNet_preprocessed=$ROOT/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$ROOT/data/nnunet/nnUNet_results
"$ENV/bin/python" train/make_datasets.py real
"$ENV/bin/nnUNetv2_preprocess" -d 101 102 -plans_name nnUNetPlans -c 3d_fullres -np 32
for d in Dataset101_DS059real Dataset102_SimV1; do
  echo "$d: $(ls "$nnUNet_preprocessed/$d/nnUNetPlans_3d_fullres" | wc -l) preprocessed files"
done
echo SETUP_DONE
