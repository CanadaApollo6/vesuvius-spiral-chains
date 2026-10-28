#!/bin/bash
# snap_export.sh - keep the fit's autosave at step SNAP and, once the arm is finished, export the model AS IT WAS AT SNAP
# on the 4-vox grid (test 12, notes/2026-10-25-chain-loss-prereg.md: "does the switch excess appear before or after
# step 25000?"). Run on the pod next to pod_arms.sh (boot_and_launch.sh SNAP=1); never touches the arm's own outputs.
#
#   source ~/spiral_env.sh; setsid nohup bash ~/vss/spiral/snap_export.sh ~/vss/spiral/arms/<arm>.env 25000 \
#     > ~/work/p0139/snap_<arm>.out 2>&1 < /dev/null &
#
# 1. Polls fit/fit.log for fit_ckpt.py's "[fit_ckpt] autosave at SNAP" line and copies that checkpoint_autosave.ckpt to
#    $W/snap<SNAP>.ckpt (the autosave is written atomically and only rewritten FIT_CKPT_EVERY steps later; the copy is
#    checked to carry completed_iterations == SNAP).
# 2. Waits for the arm's own export ($W/export_step4.json), so the GPU is free, then reruns fit_ckpt.py like
#    export_fine.sh does but resumed from the snapshot with optimizer_num_training_steps = SNAP (the loop runs zero
#    steps; output_step_size is not model-shaping) -> $W/fit_snap<SNAP>/meshes, marker $W/export_snap<SNAP>.json.
# Resumable: rerun the same command; finished steps are skipped. Exit 3 if the autosave at SNAP was missed; any
# non-zero exit writes $W/snap<SNAP>.failed.
set -euo pipefail
ARMFILE=$(realpath "$1"); source "$ARMFILE"; SNAP=${2:-25000}; STEP=4
: "${ARM:?}" "${VILLA_PY:?}" "${SPIRAL_DIR:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
WROOT=${WORK:-$HOME/work}/p0139; W=$WROOT/$ARM
CK=$W/snap$SNAP.ckpt; OUT=$W/fit_snap$SNAP; DONE=$W/export_snap$SNAP.json
say(){ echo "$(date -u +%FT%TZ) [snap $ARM $SNAP] $*"; }
trap 'rc=$?; [[ $rc != 0 ]] && echo "$rc" > "$W/snap$SNAP.failed"' EXIT   # collect.sh: stop waiting for the snapshot
mkdir -p "$W"; rm -f "$W/snap$SNAP.failed"
[[ -f $DONE ]] && { say "already exported: $DONE"; exit 0; }
if [[ ! -f $CK ]]; then
  say "waiting for the autosave at $SNAP"
  while :; do
    if [[ -f $W/fit/fit.log ]] && grep -q "\[fit_ckpt\] autosave at $SNAP " "$W/fit/fit.log"; then
      SRC=$(grep "\[fit_ckpt\] autosave at $SNAP " "$W/fit/fit.log" | tail -1 | sed 's/.* -> //')
      cp "$SRC" "$CK.tmp"
      IT=$("$VILLA_PY" -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['completed_iterations']))" "$CK.tmp")
      if [[ $IT == "$SNAP" ]]; then mv "$CK.tmp" "$CK"; say "kept $SRC (completed_iterations $IT) as $CK"; break; fi
      rm -f "$CK.tmp"; say "autosave already overwritten (completed_iterations $IT); snapshot missed"; exit 3
    fi
    [[ -f $W/export_step4.json || -f $WROOT/$ARM.chain_exit ]] && { say "arm finished before the autosave at $SNAP"; exit 3; }
    sleep 30
  done
fi
say "waiting for the arm's own export (export_step4.json)"
until [[ -f $W/export_step4.json ]]; do
  if [[ -f $WROOT/$ARM.chain_exit && $(cat "$WROOT/$ARM.chain_exit") != 0 ]]; then say "arm chain failed; no snapshot export"; exit 4; fi
  sleep 60
done
OVR=$(python3 -I -c 'import json,sys; o=json.load(open(sys.argv[1])); o["output_step_size"]=int(sys.argv[2]); o["optimizer_num_training_steps"]=int(sys.argv[3]); print(json.dumps(o))' \
  "$W/fit/overrides.json" "$STEP" "$SNAP")
mkdir -p "$OUT"; echo "$OVR" > "$OUT/overrides.json"
say "export step $STEP from $CK" | tee -a "$W/logs/timings.txt"
t0=$(date +%s)
cd "$SPIRAL_DIR"
OMP_NUM_THREADS=${FIT_THREADS:-4} WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES="$OVR" FIT_SPIRAL_OUT_DIR="$OUT" \
  FIT_SPIRAL_CACHE_DIR="$OUT/cache" FIT_SPIRAL_RESUME_PATH="$CK" FIT_CKPT_EVERY=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VILLA_PY" "$HERE/fit_ckpt.py" --dataset "$W/ds" >> "$OUT/export.log" 2>&1
RUN=$(ls -td "$OUT"/*_slice-*/meshes/fitted 2>/dev/null | head -1 || true)
[[ -n $RUN ]] || { say "export wrote no meshes/fitted under $OUT"; exit 5; }
ln -sfn "$RUN" "$OUT/meshes"
python3 -I - "$OUT/meshes" "$DONE" "$STEP" "$CK" "$SNAP" "$(( $(date +%s) - t0 ))" <<'PY'
import json, os, sys
m, out, step, ck, snap, secs = sys.argv[1:]
ws = sorted(os.listdir(m))
json.dump({"meshes": os.path.realpath(m), "output_step_size": int(step), "checkpoint": ck, "snapshot_step": int(snap),
           "seconds": int(secs), "windings": ws, "spliced": [w for w in ws if w.endswith("_spliced")]}, open(out, "w"), indent=1)
print(f"{len(ws)} winding dirs at step {step} from the step-{snap} snapshot in {m} ({secs} s)")
PY
grep -m1 "resuming from" "$OUT/export.log" || true
say "done"
