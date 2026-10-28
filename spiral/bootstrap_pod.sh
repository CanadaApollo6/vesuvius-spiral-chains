#!/bin/bash
# bootstrap_pod.sh - from this machine: copy spiral/, the band inputs (data/p0139/inputs) and the winding annotation
# sets to a fresh pod, and start pod_setup.sh (FIT_ONLY=1) detached. Idempotent; rerun after an interruption.
#   bash spiral/bootstrap_pod.sh <user@host> <port> [windings_set ...]
# windings sets are directory names under data/scratch/auto_windings/ (each with relative_windings.json); they land in
# ~/data/PHerc0139/windings/<set>/ on the pod. Setup log: ~/pod_setup.log; ready when it ends with the GPU name.
# Other scrolls: SCROLL=PHerc1667 LOCAL_INP=data/p1667/inputs MASK_DIR=data/p1667/masks (optional: zarrs copied to
# ~/data/$SCROLL/masks/ for MASK_LOCAL) bash spiral/bootstrap_pod.sh ...
set -euo pipefail
HOST=$1; PORT=$2; shift 2
SCROLL=${SCROLL:-PHerc0139}; LOCAL_INP=${LOCAL_INP:-data/p0139/inputs}; MASK_DIR=${MASK_DIR:-}; WIND_ROOT=${WIND_ROOT:-data/scratch/auto_windings}
cd "$(dirname "$0")/../.."
RS="ssh -i $HOME/.ssh/prime_ed25519 -p $PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
$RS "$HOST" "mkdir -p ~/vss ~/data/$SCROLL/inputs ~/data/$SCROLL/windings ~/data/$SCROLL/masks ~/work/p0139"
$RS "$HOST" '
  which rsync >/dev/null || { S=; [ "$(id -u)" -ne 0 ] && S=sudo; $S apt-get update -qq; $S apt-get install -y -qq rsync >/dev/null; }'
rsync -a --exclude __pycache__ -e "$RS" spiral "$HOST:vss/"
# VM images with python3 < 3.11 (Ubuntu 22.04: 3.10, no tomllib): put a uv-managed 3.12 first on PATH as python3
# (~/.local/bin, which pod_setup.sh and spiral_env.sh put first on PATH).
$RS "$HOST" 'python3 -c "import tomllib" 2>/dev/null && exit 0
  SUDO=; [ "$(id -u)" -ne 0 ] && SUDO=sudo
  command -v curl >/dev/null || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq curl >/dev/null; }
  export PATH=$HOME/.local/bin:$PATH
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  uv python install 3.12 >/dev/null 2>&1; ln -sfn "$(uv python find 3.12)" ~/.local/bin/python3
  ~/.local/bin/python3 -c "import tomllib, sys; print(\"python3 shim\", sys.version.split()[0])"'
if ! $RS "$HOST" 'test -f ~/pod_setup.started'; then
  $RS "$HOST" 'touch ~/pod_setup.started; cd ~/vss && PATH=$HOME/.local/bin:$PATH FIT_ONLY=1 setsid nohup bash env/pod_setup.sh > ~/pod_setup.log 2>&1 < /dev/null & echo setup started'
fi
rsync -a -e "$RS" "$LOCAL_INP/" "$HOST:data/$SCROLL/inputs/"
[[ -n $MASK_DIR ]] && rsync -a --exclude '*.log' -e "$RS" "$MASK_DIR/" "$HOST:data/$SCROLL/masks/"
for s in "$@"; do
  rsync -a -e "$RS" "$WIND_ROOT/$s/relative_windings.json" "$HOST:data/$SCROLL/windings/$s/"
  L=$(sha256sum "$WIND_ROOT/$s/relative_windings.json" | cut -c1-64)
  R=$($RS "$HOST" "sha256sum ~/data/$SCROLL/windings/$s/relative_windings.json" | cut -c1-64)
  [[ $L == "$R" ]] && echo "windings $s sha256 ok ${L:0:16}" || { echo "windings $s sha256 MISMATCH (local $L pod $R)"; exit 3; }
done
$RS "$HOST" "du -sh ~/data/$SCROLL/inputs ~/data/$SCROLL/windings/* ~/data/$SCROLL/masks; tail -2 ~/pod_setup.log"
