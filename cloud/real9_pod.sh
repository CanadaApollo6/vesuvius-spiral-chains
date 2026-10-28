#!/usr/bin/env bash
# Real 9 um surface fine-tunes on a 4-GPU pod (campaign W2, surface-first winding). Run from the bundle root (~/vss):
#   bash cloud/real9_pod.sh setup     # env + Dataset103 preprocessing (Dataset101 plans)
#   bash cloud/real9_pod.sh train     # 4 lanes in parallel, resumable
#   bash cloud/real9_pod.sh prep 104 Dataset104_Real9Thin            # preprocess another dataset (Dataset101 plans)
#   bash cloud/real9_pod.sh one GPU TAG DATASET [run.py args]         # single lane, e.g. one 0 r9thin Dataset104_Real9Thin \
#       --arm real --epochs 150 --pretrained real350
# Bundle (scripts/make_real9_bundle.sh): train/ cloud/ scripts/prep_ds103.sh, nnUNet_raw/Dataset103_Real9,
# Dataset101 plans, real350 and simpre trainer folders (checkpoint_final.pth only).
# Lanes (all Dataset103_Real9, PHerc0139 never included):
#   GPU0 r9ft     real350 init, default aug, 150 epochs
#   GPU1 r9ftsq   real350 init, squeeze/warp aug, 150 epochs
#   GPU2 r9sim    simpre (sim-only) init, default aug, 150 epochs
#   GPU3 r9ft300  real350 init, default aug, 300 epochs
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ENV=$HOME/env
PY=$ENV/bin/python
R=$nnUNet_results
mkdir -p data/scratch

setup() {
  set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  if ! "$PY" -c "import torch, nnunetv2" 2>/dev/null; then
    uv venv --allow-existing --python 3.12 "$ENV"
    uv pip install --python "$PY" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python "$PY" nnunetv2==2.8.1 tifffile numpy
  fi
  "$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), 'GPUs')"
  "$ENV/bin/nnUNetv2_extract_fingerprint" -d 103 -np 32
  "$ENV/bin/nnUNetv2_move_plans_between_datasets" -s 101 -t 103 -sp nnUNetPlans -tp nnUNetPlans
  cp "$nnUNet_raw/Dataset103_Real9/dataset.json" "$nnUNet_preprocessed/Dataset103_Real9/"
  "$ENV/bin/nnUNetv2_preprocess" -d 103 -plans_name nnUNetPlans -c 3d_fullres -np 32
  echo SETUP_DONE
}

prep() {  # $1 dataset id, $2 dataset name
  set -e
  "$ENV/bin/nnUNetv2_extract_fingerprint" -d "$1" -np 32
  "$ENV/bin/nnUNetv2_move_plans_between_datasets" -s 101 -t "$1" -sp nnUNetPlans -tp nnUNetPlans
  cp "$nnUNet_raw/$2/dataset.json" "$nnUNet_preprocessed/$2/"
  "$ENV/bin/nnUNetv2_preprocess" -d "$1" -plans_name nnUNetPlans -c 3d_fullres -np 32
  echo "PREP_DONE $2"
}

one() {  # $1 gpu, $2 tag, $3 dataset, rest = run.py args ("--pretrained real350|simpre" expands)
  local gpu=$1 tag=$2 ds=$3; shift 3
  local args=() a
  for a in "$@"; do
    case $a in
      real350) args+=("$R/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth") ;;
      simpre) args+=("$R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth") ;;
      *) args+=("$a") ;;
    esac
  done
  export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-18}
  DS=$ds run "$gpu" "$tag" "${args[@]}" > "data/scratch/lane_${tag}.log" 2>&1
}

run() {  # $1 gpu, $2 tag, rest = run.py args; dataset from $DS (default Dataset103_Real9)
  local gpu=$1 tag=$2; shift 2
  local ds=${DS:-Dataset103_Real9}
  local out=$R/$ds/nnUNetTrainer_${tag}__nnUNetPlans__3d_fullres/fold_0
  if [[ -f $out/checkpoint_final.pth ]]; then echo "skip $tag (done)"; return 0; fi
  local cont=(); [[ -f $out/checkpoint_latest.pth ]] && cont=(--continue-training)
  echo "$(date -Is) start $tag on GPU $gpu ${cont[*]:-}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train/run.py --dataset "$ds" --tag "$tag" "$@" "${cont[@]}" \
    > "data/scratch/train_${tag}.log" 2>&1
  local rc=$?
  echo "$(date -Is) end $tag exit $rc"
  return $rc
}

train() {
  export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-$(( $(nproc) / 4 - 2 ))}
  local real350=$R/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
  local simpre=$R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
  run 0 r9ft --arm real --epochs 150 --pretrained "$real350" > data/scratch/lane0.log 2>&1 &
  run 1 r9ftsq --arm squeeze --epochs 150 --pretrained "$real350" > data/scratch/lane1.log 2>&1 &
  run 2 r9sim --arm real --epochs 150 --pretrained "$simpre" > data/scratch/lane2.log 2>&1 &
  run 3 r9ft300 --arm real --epochs 300 --pretrained "$real350" > data/scratch/lane3.log 2>&1 &
  wait
  grep -h "end " data/scratch/lane*.log
  echo TRAIN_DONE
}

"$@"
