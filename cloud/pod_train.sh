#!/usr/bin/env bash
# Ablation on a 4-GPU pod: one lane per GPU, run in parallel. Resumable (skips finished runs, continues from
# checkpoint_latest.pth). Logs in data/scratch/train_<tag>.log, lane progress in data/scratch/lane<gpu>.log.
#   bash cloud/pod_train.sh
#   GPU0 real250 | GPU1 squeeze250 | GPU2 simpre (100 sim epochs) -> simft (250 real) | GPU3 real350 (epoch-matched
#   control for simpre + simft)
set -uo pipefail
cd "$(dirname "$0")/.."
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-24}  # 128 vCPUs / 4 lanes, minus main processes
# one math-library thread per process: otherwise every augmentation worker spawns threads for all host cores
# (load average ~400 on a 128-vCPU pod, no epoch finished in 12 min, 2026-10-17)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=$HOME/env/bin/python
R=$nnUNet_results
mkdir -p data/scratch

run() {  # $1 gpu, $2 dataset, $3 tag, rest = run.py args
  local gpu=$1 ds=$2 tag=$3; shift 3
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

lane0() { run 0 Dataset101_DS059real real250 --arm real --epochs 250; }
lane1() { run 1 Dataset101_DS059real squeeze250 --arm squeeze --epochs 250; }
lane2() {
  run 2 Dataset102_SimV1 simpre --arm real --epochs 100 && \
  run 2 Dataset101_DS059real simft --arm real --epochs 250 \
    --pretrained "$R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
}
lane3() { run 3 Dataset101_DS059real real350 --arm real --epochs 350; }

for g in 0 1 2 3; do
  "lane$g" > "data/scratch/lane$g.log" 2>&1 &
done
wait
grep -h "end " data/scratch/lane*.log
echo TRAIN_DONE
