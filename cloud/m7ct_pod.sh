#!/usr/bin/env bash
# Contact-aware fine-tune of the public m7 (ResEnc-L) on a 4-GPU pod. Run from the unpacked bundle root (~/vss):
#   bash cloud/m7ct_pod.sh
# Env as cloud/pod_setup.sh (torch 2.14.1 cu130, nnunetv2 2.8.1), preprocessing with m7's plans (scripts/prep_ds112.sh),
# then one lane per GPU, resumable. Logs: data/scratch/train_<tag>.log, data/scratch/lane<gpu>.log.
#   GPU0 m7ct150   Dataset112 (PHerc0172 contact crops x2 + 700 Dataset107), 150 epochs, init m7
#   GPU1 m7ctsq150 same + SqueezeWarp augmentation (sheets pushed together)
#   GPU2 m7ctrl150 Dataset113 (the same 700 Dataset107 crops only): control for the contact labels
#   GPU3 m7ct60    Dataset112, 60 epochs: early go/no-go checkpoint
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
ENV=$HOME/env
if ! "$ENV/bin/python" -c "import torch, nnunetv2" 2>/dev/null; then
  uv venv --allow-existing --python 3.12 "$ENV"
  # wheel CUDA must not exceed the driver's (e.g. Lambda H100 hosts ship 12.8)
  cu=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | tr -d .); tv=2.14.1 idx=cu130
  (( cu < 130 )) && tv=2.11.0 idx=cu128  # the cu128 index stops at torch 2.11
  uv pip install --python "$ENV/bin/python" torch==$tv --index-url https://download.pytorch.org/whl/$idx
  uv pip install --python "$ENV/bin/python" nnunetv2==2.8.1 tifffile numpy
fi
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), 'GPUs')"
export nnUNet_raw=$ROOT/data/nnunet/nnUNet_raw nnUNet_preprocessed=$ROOT/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$ROOT/data/nnunet/nnUNet_results
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-$(( ${NP:-$(nproc)} / $(nvidia-smi -L | wc -l) - 2 ))}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for ds in ${DATASETS:-Dataset112_M7Contact Dataset113_M7Ctrl}; do
  id=${ds:7:3}; P=$nnUNet_preprocessed/$ds
  [[ -d $P/nnUNetPlans_3d_fullres ]] && continue
  mkdir -p "$P"
  "$ENV/bin/python" - "$ds" "$P" <<'PY'
import json, sys
ds, P = sys.argv[1:]
p = json.load(open("data/models/surface_m7_nnunet/plans.json"))
p["dataset_name"] = ds
json.dump(p, open(f"{P}/nnUNetResEncUNetLPlans.json", "w"), indent=1)
PY
  cp "$nnUNet_raw/$ds/dataset.json" "$P/"
  OMP_NUM_THREADS=1 "$ENV/bin/nnUNetv2_extract_fingerprint" -d "$id" -np ${NP:-$(nproc)}
  "$ENV/bin/nnUNetv2_preprocess" -d "$id" -plans_name nnUNetResEncUNetLPlans -c 3d_fullres -np ${NP:-$(nproc)}
done
PY=$ENV/bin/python
R=$nnUNet_results
M7=$ROOT/data/models/surface_m7_nnunet/fold_0/checkpoint_best.pth
mkdir -p data/scratch
run() {  # $1 gpu, $2 dataset, $3 tag, rest = run.py args
  local gpu=${GPU_OVERRIDE:-$1} ds=$2 tag=$3; shift 3
  local out=$R/$ds/nnUNetTrainer_${tag}__nnUNetResEncUNetLPlans__3d_fullres/fold_0
  if [[ -f $out/checkpoint_final.pth ]]; then echo "skip $tag (done)"; return 0; fi
  local init=(--pretrained "$M7"); [[ -f $out/checkpoint_latest.pth ]] && init=(--continue-training)
  echo "$(date -Is) start $tag on GPU $gpu ${init[*]}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train/run.py --dataset "$ds" --tag "$tag" --plans nnUNetResEncUNetLPlans --lr 1e-3 "$@" \
    "${init[@]}" > "data/scratch/train_${tag}.log" 2>&1
  local rc=$?; echo "$(date -Is) end $tag exit $rc"; return $rc
}
lane0() { run 0 Dataset112_M7Contact m7ct150 --arm real --epochs 150; }
lane1() { run 1 Dataset112_M7Contact m7ctsq150 --arm squeeze --epochs 150; }
lane2() { run 2 Dataset113_M7Ctrl m7ctrl150 --arm real --epochs 150; }
lane3() { run 3 Dataset112_M7Contact m7ct60 --arm real --epochs 60; }
# lanes 4-7 only on 8-GPU pods
lane4() { run 4 Dataset113_M7Ctrl m7ctrl60 --arm real --epochs 60; }
lane5() { run 5 Dataset112_M7Contact m7ctsq60 --arm squeeze --epochs 60; }
lane6() { run 6 Dataset112_M7Contact m7ct150lr3 --arm real --epochs 150 --lr 3e-4; }
lane7() { run 7 Dataset112_M7Contact m7ct300 --arm real --epochs 300; }
# separator round (2026-10-20), single-GPU pods via LANES; control = m7ct150
lane8() { run 8 Dataset114_M7ContactSep m7sep01 --arm real --epochs 150 --sep-weight 0.1; }
lane9() { run 9 Dataset114_M7ContactSep m7sep03 --arm real --epochs 150 --sep-weight 0.3; }
lane10() { run 10 Dataset116_M7Contact3c m7c3 --arm real --epochs 150; }
NG=$(nvidia-smi -L | wc -l)
if [[ -n ${LANES:-} ]]; then  # single-GPU pods: LANES="2" runs lane2 on GPU 0
  for l in $LANES; do GPU_OVERRIDE=0 "lane$l" > "data/scratch/lane$l.log" 2>&1 & done
else
  for ((g = 0; g < NG && g < 8; g++)); do "lane$g" > "data/scratch/lane$g.log" 2>&1 & done
fi
wait
grep -h "end " data/scratch/lane*.log
echo TRAIN_DONE
