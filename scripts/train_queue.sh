#!/usr/bin/env bash
# Ablation queue (one GPU run at a time). Resumable: skips runs whose checkpoint_final.pth exists.
# Keeps $HOME/.gpu-leases/vesuvius-synth-scrolls.json current (shared-GPU lease scheme).
set -euo pipefail
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
       nnUNet_results=$PWD/data/nnunet/nnUNet_results
LEASE=$HOME/.gpu-leases/vesuvius-synth-scrolls.json
R=$nnUNet_results
lease() {  # $1 = note, $2 = pid or empty
  python3 - "$1" "${2:-}" "$LEASE" <<'PY'
import json, sys, datetime
note, pid, path = sys.argv[1:]
json.dump({"session": "vesuvius-synth-scrolls", "max_mem_gb": 12.0 if pid else 0.0, "max_procs": 1,
           "pids": [int(pid)] if pid else [], "note": note,
           "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}, open(path, "w"), indent=1)
PY
}
run() {  # $1 = dataset, $2 = tag, rest = run.py args
  local ds=$1 tag=$2; shift 2
  local out=$R/$ds/nnUNetTrainer_${tag}__nnUNetPlans__3d_fullres/fold_0
  if [[ -f $out/checkpoint_final.pth ]]; then echo "skip $tag (done)"; return; fi
  local cont=(); [[ -f $out/checkpoint_latest.pth ]] && cont=(--continue-training)
  echo "$(date -Is) start $tag ${cont[*]:-}"
  uv run --with nnunetv2 python train/run.py --dataset "$ds" --tag "$tag" "$@" "${cont[@]}" > "data/scratch/train_${tag}.log" 2>&1 &
  local pid=$!
  lease "nnU-Net training $tag (1 proc, <=12 GB)" "$pid"
  wait "$pid"
  echo "$(date -Is) end $tag exit $?"
}
run Dataset102_SimV1     simpre   --arm real --epochs 100
run Dataset101_DS059real real250  --arm real --epochs 250
run Dataset101_DS059real simft    --arm real --epochs 250 \
    --pretrained "$R/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
run Dataset101_DS059real squeeze250 --arm squeeze --epochs 250
lease "idle (ablation queue finished)" ""
rm -f "$LEASE"
echo QUEUE_DONE
