#!/bin/bash
# export_fine.sh - re-export one finished arm's per-winding tifxyz on a finer grid (default 4 vox) from its
# checkpoint_fitted.ckpt, with the same pinned fitter and the same overrides. Run on the pod after run_arm.sh.
#
#   source ~/spiral_env.sh; bash ~/vss/spiral/export_fine.sh ~/vss/spiral/arms/arm0.env [STEP]
#
# The public fit writes meshes at output_step_size=20 (vertex spacing in vox, both along z and along the winding).
# A coarse grid inflates contact-zone switch rates in our metric, so this reruns fit_ckpt.py with
# FIT_SPIRAL_RESUME_PATH=<run>/checkpoint_fitted.ckpt (completed_iterations = the full horizon, so the training loop runs
# zero steps) and output_step_size=STEP. output_step_size is not a model-shaping key, so the checkpoint is accepted and
# the model is the fitted one, unchanged; only the final mesh sampling differs. Winding indices (w###) are the same
# spiral winding numbers as the 20-vox export. Output: $W/fit_step$STEP/<run>/meshes/fitted, linked as
# $W/fit_step$STEP/meshes; done marker $W/export_step$STEP.json. Resumable: rerun is a no-op once the marker exists.
set -euo pipefail
ARMFILE=$(realpath "$1"); source "$ARMFILE"; STEP=${2:-4}
: "${ARM:?}" "${VILLA_PY:?}" "${SPIRAL_DIR:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORK:-$HOME/work}/p0139/$ARM
DS=$W/ds; SRC=$W/fit; OUT=$W/fit_step$STEP; DONE=$W/export_step$STEP.json
[[ -f $DONE ]] && { echo "already exported: $DONE"; exit 0; }
CK=$(ls -t "$SRC"/*_slice-*/checkpoint_fitted.ckpt 2>/dev/null | head -1 || true)
[[ -n $CK ]] || { echo "no checkpoint_fitted.ckpt under $SRC (run run_arm.sh first)"; exit 2; }
OVR=$(python3 -I -c 'import json,sys; o=json.load(open(sys.argv[1])); o["output_step_size"]=int(sys.argv[2]); print(json.dumps(o))' \
  "$SRC/overrides.json" "$STEP")
mkdir -p "$OUT"; echo "$OVR" > "$OUT/overrides.json"
echo "$(date -u +%FT%TZ) export step $STEP from $CK" | tee -a "$W/logs/timings.txt"
t0=$(date +%s)
cd "$SPIRAL_DIR"
OMP_NUM_THREADS=${FIT_THREADS:-4} WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES="$OVR" FIT_SPIRAL_OUT_DIR="$OUT" \
  FIT_SPIRAL_CACHE_DIR="$OUT/cache" FIT_SPIRAL_RESUME_PATH="$CK" FIT_CKPT_EVERY=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VILLA_PY" "$HERE/fit_ckpt.py" --dataset "$DS" >> "$OUT/export.log" 2>&1
RUN=$(ls -td "$OUT"/*_slice-*/meshes/fitted 2>/dev/null | head -1 || true)
[[ -n $RUN ]] || { echo "export wrote no meshes/fitted under $OUT"; exit 4; }
ln -sfn "$RUN" "$OUT/meshes"
# The re-saved checkpoint is a byte-for-byte duplicate of the model in $CK; drop it to save space.
rm -f "$(dirname "$(dirname "$RUN")")/checkpoint_fitted.ckpt"
python3 -I - "$OUT/meshes" "$DONE" "$STEP" "$CK" "$(( $(date +%s) - t0 ))" <<'PY'
import json, os, sys
m, out, step, ck, secs = sys.argv[1:]
ws = sorted(os.listdir(m))
json.dump({"meshes": os.path.realpath(m), "output_step_size": int(step), "checkpoint": ck, "seconds": int(secs),
           "windings": ws, "spliced": [w for w in ws if w.endswith("_spliced")]}, open(out, "w"), indent=1)
print(f"{len(ws)} winding dirs at step {step} in {m} ({secs} s)")
PY
