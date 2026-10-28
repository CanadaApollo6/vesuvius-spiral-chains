#!/bin/bash
# collect_and_score.sh - local orchestrator for the annotation arms: pull each arm once its pod reports
# export_step4.json, terminate a pod once all of its arms are pulled and verified, and score pulled arms against their
# base arm one at a time with eval/spiral_eval.py (~11 min, ~11 GB RAM each). Resumable: markers live under
# data/scratch/p0139_spiral/<arm>/ (pulled.ok) and data/scratch/spiral_eval/arms/ (the output json).
#   setsid nohup bash spiral/collect_and_score.sh spiral/arms/pods_annot.tsv > data/scratch/p0139_spiral/collect.log 2>&1 &
# pods file (tsv): arm  user@host  port  pod_id  [base]     (several arms may share a pod)
# Scoring: <base>_vs_<arm>.json, base = 5th column, else $BASE (default arm0); an arm whose base is itself is pulled
# but not scored (a band baseline). The band is read from the pulled <arm>/ds/band.json (fallback $Z0/$Z1, default
# 3000/4500). The base must be pulled (pulled.ok) or already local (<base>/meshes_step4) before an arm is scored.
set -uo pipefail
cd "$(dirname "$0")/../.."
PODS=$1
KEY=$HOME/.ssh/prime_ed25519
SC=data/scratch/p0139_spiral; EV=data/scratch/spiral_eval/arms; mkdir -p "$EV"
log(){ echo "$(date -u +%FT%TZ) $*"; }
BASE=${BASE:-arm0}; Z0=${Z0:-3000}; Z1=${Z1:-4500}
while :; do
  left=0
  while read -r ARM HOST PORT POD _; do
    [[ -z ${ARM:-} || $ARM == \#* ]] && continue
    D=$SC/$ARM
    if [[ ! -f $D/pulled.ok ]]; then
      left=1
      st=$(ssh -n -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
        "if [ -f ~/work/p0139/$ARM/export_step4.json ]; then echo DONE; elif [ -f ~/work/p0139/$ARM.chain_exit ]; then echo FAIL; else echo RUN; fi" 2>/dev/null)
      if [[ $st == DONE ]]; then
        log "pull $ARM"
        if bash spiral/pull_arm.sh "$ARM" "$HOST" "$PORT" >> "$SC/collect_pull.log" 2>&1 < /dev/null; then
          want=$(python3 -I -c "import json,sys;print(len(json.load(open(sys.argv[1]))['windings']))" "$D/export_step4/export_step4.json")
          have=$(ls "$D/meshes_step4" | wc -l)
          if [[ $want == "$have" && $have -gt 0 ]]; then touch "$D/pulled.ok"; log "$ARM pulled ($have windings)"
          else log "$ARM pull incomplete ($have/$want)"; fi
        fi
      elif [[ $st == FAIL ]]; then log "$ARM FAILED on pod (see ~/work/p0139/$ARM.out); not pulling"
      fi
    fi
  done < "$PODS"
  # terminate pods whose arms are all pulled
  for POD in $(awk '!/^#/ && NF>=4 {print $4}' "$PODS" | sort -u); do
    [[ -f $SC/terminated_$POD ]] && continue
    ok=1; for ARM in $(awk -v p="$POD" '!/^#/ && NF>=4 && $4==p {print $1}' "$PODS"); do [[ -f $SC/$ARM/pulled.ok ]] || ok=0; done
    if [[ $ok == 1 ]]; then
      log "terminate $POD"
      r=$(uvx --from prime prime --plain pods terminate "$POD" --yes < /dev/null 2>&1 | tail -1)
      date -u +%FT%TZ > "$SC/terminated_$POD"
      echo "$(date -u +%FT%TZ) terminate $POD (collect_and_score.sh): $r" >> "$SC/pods_arms.txt"
    fi
  done
  # score one pulled arm per pass
  for ARM in $(awk '!/^#/ && NF>=4 {print $1}' "$PODS"); do
    B=$(awk -v a="$ARM" -v d="$BASE" '!/^#/ && $1==a {print (NF>=5 ? $5 : d); exit}' "$PODS")
    [[ $B == "$ARM" ]] && continue
    OUT=$EV/${B}_vs_$ARM.json
    [[ -f $OUT.failed ]] && continue
    BREADY=; [[ -f $SC/$B/pulled.ok ]] && BREADY=1
    [[ -z $BREADY && -d $SC/$B/meshes_step4 ]] && ! awk -v b="$B" '!/^#/ && $1==b {f=1} END {exit !f}' "$PODS" && BREADY=1
    if [[ -f $SC/$ARM/pulled.ok && -n $BREADY && ! -f $OUT ]]; then
      left=1
      read -r AZ0 AZ1 ASC < <(python3 -I -c "import json,sys;b=json.load(open(sys.argv[1]));print(b['z0'],b['z1'],{'PHerc1667':'p1667'}.get(b.get('scroll'),'p0139'))" \
        "$SC/$ARM/ds/band.json" 2>/dev/null || echo "$Z0 $Z1 p0139")
      log "score $ARM vs $B (z $AZ0-$AZ1)"
      t0=$(date +%s)
      # spiral_eval writes --out only at the end, so a present json means a finished score
      flock /tmp/vss_eval.lock uv run python eval/spiral_eval.py --fit $SC/$B/meshes_step4 --fit-b $SC/$ARM/meshes_step4 --z0 "$AZ0" --z1 "$AZ1" \
        --scroll "$ASC" --tol 4 --out "$OUT" > "$EV/${B}_vs_$ARM.log" 2>&1
      rc=$?; [[ $rc != 0 ]] && { touch "$OUT.failed"; rm -f "$OUT"; }
      log "score $ARM exit $rc in $(( $(date +%s) - t0 )) s"
      break
    elif [[ ! -f $OUT ]]; then left=1; fi
  done
  [[ $left == 0 ]] && { log "all arms pulled and scored"; break; }
  sleep 60
done
