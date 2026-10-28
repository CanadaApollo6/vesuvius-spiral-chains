#!/bin/bash
# pull_arm.sh - copy one arm's fit outputs (not the big inputs) from the pod to data/scratch/p0139_spiral/<arm>/.
#   bash spiral/pull_arm.sh <arm> <user@host> <port>      (key ~/.ssh/prime_ed25519)
# Copies: fitted winding surfaces (tifxyz, w###/ and w###_spliced/), checkpoint_fitted.ckpt (resume / re-render),
# fit.log, overrides, fine-grid re-exports (export_fine.sh -> meshes_step<N>/), dataset json (spiral-scroll.json, band.json, umbilicus.json), timings and the tracks extract.json.
set -euo pipefail
ARM=$1; HOST=$2; PORT=$3
cd "$(dirname "$0")/../.."
DST=data/scratch/p0139_spiral/$ARM; mkdir -p "$DST"
RS="ssh -i $HOME/.ssh/prime_ed25519 -p $PORT"
R=work/p0139/$ARM
RUN=$($RS "$HOST" "readlink -f ~/$R/fit/meshes | sed 's#/meshes/fitted\$##'")
rsync -a -e "$RS" "$HOST:$RUN/meshes/fitted/" "$DST/meshes/"
rsync -a -e "$RS" --include='checkpoint_fitted.ckpt' --include='*.json' --include='*.txt' --exclude='*/' --exclude='*' \
  "$HOST:$RUN/" "$DST/run/"
rsync -a -e "$RS" "$HOST:$R/fit/fit.log" "$HOST:$R/fit/overrides.json" "$HOST:$R/arm_done.json" "$DST/"
rsync -a -e "$RS" "$HOST:$R/ds/" "$DST/ds/"
rsync -a -e "$RS" "$HOST:$R/logs/" "$DST/logs/"
# Fine-grid re-exports from export_fine.sh (fit_step<N>/meshes -> meshes_step<N>/, export json, log and overrides -> export_step<N>/).
for F in $($RS "$HOST" "cd ~/$R && ls -d fit_step*/meshes 2>/dev/null | sed 's#/meshes##'" || true); do
  N=${F#fit_step}
  rsync -a -e "$RS" "$HOST:$R/$F/meshes/" "$DST/meshes_step$N/"
  mkdir -p "$DST/export_step$N"
  rsync -a -e "$RS" "$HOST:$R/export_step$N.json" "$HOST:$R/$F/export.log" "$HOST:$R/$F/overrides.json" "$DST/export_step$N/"
done
rsync -a -e "$RS" "$HOST:data/PHerc0139/tracks_*/*.extract.json" "$DST/" || true
du -sh "$DST"; ls "$DST/meshes" | wc -l
