#!/bin/bash
# pod_arms.sh - run several arms on one pod, one GPU each (arm i -> GPU i mod #GPUs), each as run_arm.sh then
# export_fine.sh (step 4). Arms after the first start only once the first has built the shared tracks/SDT (its fit has
# started), so the per-MASK_TAG caches under $DATA are built once. Resumable: rerun the same command after an
# interruption; finished steps are skipped by run_arm.sh / export_fine.sh.
#
#   source ~/spiral_env.sh; setsid nohup bash ~/vss/spiral/pod_arms.sh arms/arm1.env arms/arm2.env \
#     > ~/work/p0139/pod_arms.out 2>&1 < /dev/null &
# Per-arm log: ~/work/p0139/<arm>.out. Done marker per arm: ~/work/p0139/<arm>/export_step4.json.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NG=$(nvidia-smi -L | wc -l)
WROOT=${WORK:-$HOME/work}/p0139; mkdir -p "$WROOT"
i=0; prev=
for f in "$@"; do
  ENV=$(realpath "$f"); ARM=$(bash -c "source '$ENV'; echo \$ARM")
  if [[ -n $prev ]]; then   # wait until the previous arm's fit has started (shared inputs are built)
    until [[ -s $WROOT/$prev/fit/fit.log || -f $WROOT/$prev/export_step4.json ]]; do
      if [[ -f $WROOT/$prev.chain_exit ]] && [[ $(cat "$WROOT/$prev.chain_exit") != 0 ]]; then
        echo "$(date -u +%FT%TZ) $prev failed before its fit; starting $ARM anyway"; break; fi
      sleep 20; done
    sleep 60
  fi
  GPU=$(( i % NG ))
  echo "$(date -u +%FT%TZ) start $ARM on GPU $GPU"
  rm -f "$WROOT/$ARM.chain_exit"
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c "bash '$HERE/run_arm.sh' '$ENV' && bash '$HERE/export_fine.sh' '$ENV' 4; \
    echo \$? > '$WROOT/$ARM.chain_exit'" >> "$WROOT/$ARM.out" 2>&1 < /dev/null &
  prev=$ARM; i=$((i + 1))
done
echo "$(date -u +%FT%TZ) all launched"
