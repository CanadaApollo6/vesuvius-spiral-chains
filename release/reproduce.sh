#!/usr/bin/env bash
# Regenerate every number in release/SUBMISSION_DRAFT.md from the repo code alone (no data/ directory needed).
#
#   bash release/reproduce.sh [stage ...]        stages: setup bench m7 size score paired figs report   (default: all)
#   DRY_RUN=1 bash release/reproduce.sh          print the commands instead of running them
#   M7D300=<nnU-Net trainer dir> bash ...        also run the m7d300 / ensemble rows (our own checkpoint, not public)
#   REAL350=<nnU-Net trainer dir> bash ...       also run the real350 rows (our own checkpoint, not public)
#   FORCE=1                                      redo steps whose outputs already exist
#   FIG_OUT=<dir>                                where `figs` writes PNGs (default release/figures)
#
# Stages (see release/REPRODUCE.md for what each one yields):
#   setup   public m7 checkpoint from Hugging Face; the Kaggle metric env (data/metric/.venv)
#   bench   PHerc0139 winding set + hf2um set, built by streaming crops from the public buckets; bucket m7 crops
#   m7      GPU: m7 inference on both sets (+ m7d300 / real350 if given)
#   size    GPU: params, wall time and peak memory for one 256^3 block
#   score   CPU: patch graph on every mask set, GT row, hf2um official score + separation
#   paired  CPU: paired site bootstrap, certifier, GT contact undercount
#   figs    CPU: release/fig_classes.py, release/fig_compressed.py
#   report  CPU: compact table of the headline numbers (always printed at the end)
#
# Everything lives under data/ (gitignored). Nothing is committed, pushed or uploaded. HF_TOKEN is optional (public
# repo) and is never printed. Stage `bench` streams ~17.5 GB of zarr chunks (upper bound), `setup` + `bench` need disk
# of about 6 GB; see release/REPRODUCE.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL_STAGES=(setup bench m7 size score paired figs report)
M7_REPO=scrollprize/surface_m7_nnunet
M7=data/models/surface_m7_nnunet
P0139=data/eval/p0139_9um
HF=data/eval/hf2um
PREDS=data/preds
LOGDIR=data/scratch/reproduce
RES=data/reproduce
FIG_OUT="${FIG_OUT:-release/figures}"
export M7D300="${M7D300:-}"
export REAL350="${REAL350:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
WORKERS="${WORKERS:-6}"       # p0139_testset.py build workers
SEP_WORKERS="${SEP_WORKERS:-4}"
METRIC_SHA256=64d24044b7381dbb660a0e9f602ad0f8fd37d1935c09a8ee3b78088433910e68
TF_SHA256=572e69e08b37d7a92b29f78559e9de3d62c436a0fe52aee8e0cb56db9df5b161
TF_URL=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/transform.json
TF_LOCAL=data/eval/surface_labels_2026_07_sample/transform_0139_2p399_to_9p362.json

log() { printf '[reproduce] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
dry() { [ "$DRY_RUN" = 1 ]; }
force() { [ "$FORCE" = 1 ]; }

# ---------------------------------------------------------------- GPU lease (this machine's protocol)
LEASE_DIR="$HOME/.gpu-leases"
LEASE="$LEASE_DIR/vss-reproduce.json"
LEASE_HELD=0
lease_acquire() {
  [ -d "$LEASE_DIR" ] || return 0   # no lease directory: nothing to do
  dry && { log "(dry run) would write $LEASE"; return 0; }
  printf '{"owner":"vss-reproduce","max_mem_gb":16,"pid":%d,"started":"%s"}\n' "$$" "$(date -Is)" > "$LEASE"
  LEASE_HELD=1
}
lease_release() {
  if [ "$LEASE_HELD" = 1 ]; then rm -f "$LEASE"; LEASE_HELD=0; fi
}
trap lease_release EXIT
trap 'exit 130' INT TERM

gpu_check() {
  dry && return 0
  command -v nvidia-smi >/dev/null || die "no nvidia-smi: stages m7 and size need a CUDA GPU (>= 16 GB)"
  local mb
  mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -nr | head -1)"
  [ "${mb:-0}" -ge 15000 ] || die "largest GPU has ${mb:-0} MiB; m7 at 1.18x needs about 16 GB"
}

# ---------------------------------------------------------------- command helpers
mkdir -p "$LOGDIR" "$RES"

# run <cmd...>: echo, then run (or only echo under DRY_RUN)
run() {
  local c="$*"
  c="${c%%$'\n'*}"
  printf '+ %s\n' "${c:0:260}" >&2
  if ! dry; then "$@"; fi
}
# runlog <name> <cmd...>: output to $LOGDIR/<name>.log; show its tail on failure
runlog() {
  local name=$1 c; shift
  c="$*"
  c="${c%%$'\n'*}"
  printf '+ [%s] %s\n' "$name" "${c:0:240}" >&2
  if dry; then return 0; fi
  "$@" > "$LOGDIR/$name.log" 2>&1 || { log "FAILED: $name (log: $LOGDIR/$name.log)"; tail -n 15 "$LOGDIR/$name.log" >&2; return 1; }
  log "done: $name"
}
# runout <name> <cmd...>: stdout to $RES/<name>.txt (and shown), stderr to $LOGDIR/<name>.err
runout() {
  local name=$1 c; shift
  c="$*"
  c="${c%%$'\n'*}"
  printf '+ [%s] %s\n' "$name" "${c:0:240}" >&2
  if dry; then return 0; fi
  "$@" > "$RES/$name.txt" 2> "$LOGDIR/$name.err" || { log "FAILED: $name (log: $LOGDIR/$name.err)"; tail -n 15 "$LOGDIR/$name.err" >&2; return 1; }
  cat "$RES/$name.txt"
}
JOBS=()
bg() { local name=$1; shift; runlog "$name" "$@" & JOBS+=("$!:$name"); }
waitall() {
  local rc=0 j
  for j in ${JOBS[@]+"${JOBS[@]}"}; do
    wait "${j%%:*}" || { log "job failed: ${j#*:}"; rc=1; }
  done
  JOBS=()
  return "$rc"
}
need() {
  local f
  for f in "$@"; do
    if [ ! -e "$f" ]; then
      dry && { log "(dry run) $f does not exist yet"; continue; }
      die "missing $f (run the earlier stage that produces it)"
    fi
  done
}
have_all() { local f; for f in "$@"; do [ -e "$f" ] || return 1; done; ! force; }

use_m7d300() {
  [ -n "$M7D300" ] || return 1
  [ -f "$M7D300/plans.json" ] && [ -d "$M7D300/fold_0" ] || die "M7D300=$M7D300 is not an nnU-Net trainer dir (needs plans.json, dataset.json, fold_0/)"
}
use_real350() {
  [ -n "$REAL350" ] || return 1
  [ -f "$REAL350/plans.json" ] && [ -d "$REAL350/fold_0" ] || die "REAL350=$REAL350 is not an nnU-Net trainer dir"
}
skip_m7d300_msg() { use_m7d300 || log "M7D300 not set: skipping m7d300 and ensemble rows (our own, non-public checkpoint)"; }
skip_real350_msg() { use_real350 || log "REAL350 not set: skipping real350 rows (our own, non-public checkpoint)"; }

# ---------------------------------------------------------------- inline python (kept here so the repo needs nothing else)
py_meshes() { cat <<'PY'
# PHerc0139 tifxyz meshes (9.362 and 2.399 um grids): only x/y/z/meta of the 38 segments that have them, ~0.95 GB.
from pathlib import Path
import s3fs
fs = s3fs.S3FileSystem(anon=True)
B = "vesuvius-challenge-open-data/PHerc0139/segments/"
TAGS = ["20250728140407-9.362um", "20260102150214-2.399um"]
out = Path("data/meshes/p0139")
for seg in sorted(fs.ls(B)):
    name = seg.rsplit("/", 1)[1]
    try:
        meshes = fs.ls(seg + "/mesh")
    except FileNotFoundError:
        continue
    (out / name).mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        hit = [m for m in meshes if m.endswith(f"-on-{tag}.tifxyz")]
        if not hit:
            continue
        d = out / name / tag
        d.mkdir(parents=True, exist_ok=True)
        for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
            if not (d / f).exists():
                fs.get(f"{hit[0]}/{f}", str(d / (f + ".part")))
                (d / (f + ".part")).rename(d / f)
    print(name, flush=True)
PY
}

py_labels_sep() { cat <<'PY'
# hf2um labelsSep: labelsTr with the surface band dilated by 1 voxel (6-connected), never into ignore (2).
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage as ndi
src, dst = Path("data/eval/hf2um/labelsTr"), Path("data/eval/hf2um/labelsSep")
dst.mkdir(exist_ok=True)
st = ndi.generate_binary_structure(3, 1)
for f in sorted(src.glob("*.tif")):
    lab = tifffile.imread(f)
    out = lab.copy()
    out[ndi.binary_dilation(lab == 1, st) & (lab != 2)] = 1
    tifffile.imwrite(dst / f.name, out, compression="zlib")
print(len(list(dst.glob("*.tif"))), "labelsSep crops")
PY
}

py_size() { cat <<'PY'
# Params, wall time and peak GPU memory for one 256^3 block (no TTA, one warm-up), as quoted in section 5.
import os, sys, time
from pathlib import Path
import numpy as np, tifffile, torch
sys.path.insert(0, "eval")
from infer_nnunet import load_predictor
ct = sorted(Path("data/eval/hf2um/imagesTr").glob("p1667*_0000.tif"))[:2]
vol = np.concatenate([tifffile.imread(f) for f in ct], 0)[:256]
vol = np.pad(vol, ((0, 0), (0, 96), (0, 96)), mode="reflect")[:256, :256, :256].astype(np.float32)
runs = [("m7pub", "data/models/surface_m7_nnunet", 1.1836), ("m7pub_native", "data/models/surface_m7_nnunet", 1.0)]
if os.environ.get("M7D300"):
    runs.insert(0, ("m7d300", os.environ["M7D300"], 1.1836))
for name, d, sp in runs:
    pred = load_predictor(Path(d), fold="0")
    npar = sum(p.numel() for p in pred.network.parameters())
    pred.predict_single_npy_array(vol[None, :128, :128, :128], {"spacing": [sp] * 3}, None, None, False)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    pred.predict_single_npy_array(vol[None], {"spacing": [sp] * 3}, None, None, False)
    torch.cuda.synchronize()
    print(f"{name}: {npar / 1e6:.1f} M params, {time.time() - t:.1f} s per 256^3 block, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    del pred; torch.cuda.empty_cache()
PY
}

py_undercount() { cat <<'PY'
# Section 3: share of contact pairs that the graph undercounts by exactly one winding, per mask source.
import sys
import pandas as pd
col = "delta_graph_b12_r40_k30_t0"
for name in sys.argv[1:]:
    d = pd.read_csv(f"data/preds/{name}.csv")
    c = d[d.contact]
    ok = ((c[col] - c.expected).abs() < 0.25).mean()
    under = ((c.expected - c[col]).round() == 1).mean()
    print(f"{name}: contact pairs {len(c)}, exact {100 * ok:.1f} %, undercounted by one {100 * under:.1f} %")
PY
}

py_report() { cat <<'PY'
import json, re
from pathlib import Path
import pandas as pd
R, P = Path("data/reproduce"), Path("data/preds")
COL = "delta_graph_b12_r40_k30_t0"

def head(t): print(f"\n=== {t}")
def read(f):
    f = Path(f)
    return f.read_text().splitlines() if f.exists() else None

head("benchmarks (sizes)")
g = P / "graph_m7pub_s11836.csv"
if g.exists():
    d = pd.read_csv(g)
    print(f"PHerc0139 winding set: {d.stem.nunique()} sites, {len(d):,} pairs, {int(d.contact.sum()):,} contact; "
          f"sites by class {d.groupby('site_cls').stem.nunique().to_dict()}")
else:
    print("PHerc0139 winding set: graph_m7pub_s11836.csv not produced yet")
mf = Path("data/eval/hf2um/manifest.json")
if mf.exists():
    m = pd.DataFrame(json.loads(mf.read_text()))
    print(f"hf2um: {len(m)} crops, {m.vol.value_counts().to_dict()}")

head("PHerc0139 winding, exact delta % (graph | direct count), all pairs; CI = site bootstrap")
rows = [("GT labels (binary)", "graph_gt"), ("m7, 1.18x", "graph_m7pub_s11836"), ("m7d300 + m7 (mean prob)", "graph_ensB"),
        ("m7d300, 1.18x", "graph_m7d300_s11836"), ("m7, bucket predictions (dilate 0)", "graph_m7_d0"),
        ("m7, native", "graph_m7pub_s10"), ("real350", "graph_real350_s11836")]
pg, pd_ = read(R / "paired_graph.txt") or [], read(R / "paired_direct.txt") or []
def pline(lines, name):
    return next((l for l in lines if l.startswith(f"data/preds/{name}.csv:")), None)
print(f"{'masks':36s} {'graph':>6s} {'CI':>14s} {'direct':>7s} {'contact':>8s} {'open':>6s}")
for label, name in rows:
    f = P / f"{name}.csv"
    if not f.exists():
        print(f"{label:36s} not run")
        continue
    d = pd.read_csv(f)
    ok = lambda c: ((d[c] - d.expected).abs() < 0.25)
    lg = pline(pg, name)
    ci = re.search(r"CI (\[[^\]]+\])", lg).group(1) if lg else ""
    print(f"{label:36s} {100 * ok(COL).mean():6.2f} {ci:>14s} {100 * ok('delta_direct').mean():7.2f} "
          f"{100 * ok(COL)[d.contact].mean():8.1f} {100 * ok(COL)[~d.contact].mean():6.1f}")
for lines, tag in ((pg, "graph"), (pd_, "direct")):
    l = pline(lines, "graph_m7pub_s11836")
    if l:
        print(f"headline paired vs native grid ({tag}): " + re.search(r"vs ref (.*?) contact", l).group(1))

head("contact ceiling and certifier")
for l in read(R / "undercount.txt") or ["undercount.txt not produced yet"]:
    print(l)
for l in read(R / "certify.txt") or ["certify.txt not produced yet"]:
    if re.match(r"(certifier input|pairs |logistic|  coverage (70|100)%)", l):
        print(l)

head("hf2um (all 104 crops): official blend, VOI_merge, compressed separation")
print("tags: m7pubn = m7 native grid, m7pub = m7 at 1.18x (declared 7.91 um grid)")
s = read(R / "hf2um_summary.txt")
if s:
    print(s[0])
    for l in s[1:]:
        if len(l.split()) > 1 and l.split()[1] == "all":
            print(l)
else:
    print("hf2um_summary.txt not produced yet")

head("model size / speed (one 256^3 block)")
for l in read(R / "size.txt") or ["size.txt not produced yet"]:
    print(l)
PY
}

# ---------------------------------------------------------------- stages
stage_setup() {
  log "stage setup"
  command -v uv >/dev/null || die "uv is required (https://docs.astral.sh/uv/)"
  command -v curl >/dev/null || die "curl is required"
  command -v unzip >/dev/null || die "unzip is required"
  if have_all "$M7/fold_0/checkpoint_best.pth" "$M7/plans.json" "$M7/dataset.json"; then
    log "m7 checkpoint present"
  else
    mkdir -p "$M7"
    # Public repo: no login needed. An HF token in the environment or the hf cache is used silently if present.
    if command -v hf >/dev/null; then run hf download "$M7_REPO" --local-dir "$M7"
    elif command -v huggingface-cli >/dev/null; then run huggingface-cli download "$M7_REPO" --local-dir "$M7"
    else run uvx --from huggingface_hub hf download "$M7_REPO" --local-dir "$M7"; fi
  fi
  if have_all data/metric/.venv/bin/python data/metric/topological-metrics-kaggle/external/Betti-Matching-3D/build; then
    log "metric env present"
  else
    command -v c++ >/dev/null || command -v g++ >/dev/null || die "a C++ compiler is needed to build Betti-Matching-3D"
    mkdir -p data/metric
    if [ ! -f data/metric/metric.zip ] || force; then
      run curl -fL -o data/metric/metric.zip https://www.kaggle.com/api/v1/datasets/download/sohier/vesuvius-metric-resources
    fi
    dry || echo "$METRIC_SHA256  data/metric/metric.zip" | sha256sum -c - >&2
    run unzip -q -o data/metric/metric.zip -d data/metric
    # Python 3.11 env from the zip's pinned wheels (offline), as in notes/2026-10-16-eval-harness.md
    run uv venv --python 3.11 data/metric/.venv
    run uv pip install --python data/metric/.venv/bin/python --no-index --find-links data/metric/wheels \
      -r data/metric/topological-metrics-kaggle/requirements.txt
    # Betti-Matching-3D: -include functional is needed with GCC 16; the venv bin dir must be first on PATH
    # so CMake picks the venv's Python 3.11.
    dry || CXXFLAGS="-include functional" PATH="$ROOT/data/metric/.venv/bin:$PATH" \
      bash data/metric/topological-metrics-kaggle/scripts/build_betti.sh > "$LOGDIR/build_betti.log" 2>&1 \
      || { tail -n 20 "$LOGDIR/build_betti.log" >&2; die "Betti-Matching-3D build failed"; }
    if dry; then log '(dry run) would build Betti-Matching-3D with CXXFLAGS="-include functional"'; fi
  fi
}

stage_bench() {
  log "stage bench"
  # ---- PHerc0139 winding set (87 sites, 24,972 pairs)
  if have_all "$P0139/sites.json" "$P0139/scoring_pairs.csv" "$P0139/labelsTr"; then
    log "PHerc0139 set present"
  else
    if ! have_all "$TF_LOCAL"; then
      mkdir -p "$(dirname "$TF_LOCAL")"
      run curl -fsSL -o "$TF_LOCAL" "$TF_URL"
      dry || echo "$TF_SHA256  $TF_LOCAL" | sha256sum -c - >&2
    fi
    if ! have_all data/meshes/p0139; then
      runlog fetch_meshes uv run python -c "$(py_meshes)"
    fi
    have_all "$P0139/_work/overlap.json" "$P0139/_work/mine_points.npz" || \
      runlog p0139_mine uv run python scripts/p0139_testset.py mine
    have_all "$P0139/_work/candidates.json" || \
      runlog p0139_select uv run python scripts/p0139_testset.py select
    # `build` skips sites already built, so an interrupted run resumes
    runlog p0139_build uv run python scripts/p0139_testset.py build --workers "$WORKERS"
    runlog p0139_finalize uv run python scripts/p0139_testset.py finalize
    runlog p0139_pairs uv run python eval/p0139_sites.py
  fi
  # bucket m7 prediction (L0, th 0.2) cropped to the 87 sites
  if ! have_all "$PREDS/m7_seg"; then
    runlog p0139_bucket_m7 uv run --with zarr --with s3fs python scripts/p0139_m7.py --out "$PREDS/m7_seg"
  fi
  # ---- hf2um (104 crops of 160^3 from the organisers' surface-labels-2026-07)
  if have_all "$HF/manifest.json" "$HF/labelsTr"; then
    log "hf2um set present"
  else
    runlog hf2um_set uv run --with "zarr==2.18.4" --with "numcodecs<0.16" --with fsspec --with aiohttp --with requests \
      --with s3fs --with tifffile python scripts/hf2um_set.py
  fi
  have_all "$HF/labelsSep" || runlog hf2um_labels_sep uv run python -c "$(py_labels_sep)"
}

stage_m7() {
  log "stage m7 (GPU)"
  need "$M7/fold_0/checkpoint_best.pth" "$P0139/sites.json" "$HF/manifest.json"
  gpu_check
  skip_m7d300_msg; skip_real350_msg
  lease_acquire
  local sp t
  # PHerc0139: spacing 1.1836 = 9.362 / 7.91 (resample to m7's training grid), 1.0 = native grid
  for sp in 1.1836 1.0; do
    t="m7pub_s${sp/./}"
    have_all "$PREDS/xing_${t}.json" || runlog "xing_$t" uv run --with nnunetv2 --with pandas python eval/p0139_crossings.py \
      --model "$M7" --fold 0 --spacing "$sp" --out "$PREDS/xing_$t"
  done
  have_all "$PREDS/prob_m7pub_s11836" || runlog prob_m7pub uv run --with nnunetv2 python eval/p0139_probs.py \
    --model "$M7" --out "$PREDS/prob_m7pub_s11836"
  # hf2um: with the manifest, spacing = um / 7.91 (1.18x for 9.6 um crops); without it, native grid
  have_all "$PREDS/hf2um_m7pub" || runlog hf2um_m7pub uv run --with nnunetv2 python eval/infer_nnunet.py --model "$M7" \
    --fold 0 --images "$HF/imagesTr" --manifest "$HF/manifest.json" --ref-um 7.91 --out "$PREDS/hf2um_m7pub"
  have_all "$PREDS/hf2um_m7pubn" || runlog hf2um_m7pubn uv run --with nnunetv2 python eval/infer_nnunet.py --model "$M7" \
    --fold 0 --images "$HF/imagesTr" --out "$PREDS/hf2um_m7pubn"
  if use_m7d300; then
    have_all "$PREDS/xing_m7d300_s11836.json" || runlog xing_m7d300_s11836 uv run --with nnunetv2 --with pandas python \
      eval/p0139_crossings.py --model "$M7D300" --fold 0 --spacing 1.1836 --out "$PREDS/xing_m7d300_s11836"
    have_all "$PREDS/prob_m7d300_s11836" || runlog prob_m7d300 uv run --with nnunetv2 python eval/p0139_probs.py \
      --model "$M7D300" --spacing 1.1836 --out "$PREDS/prob_m7d300_s11836"
    have_all "$PREDS/hf2um_m7d300" || runlog hf2um_m7d300 uv run --with nnunetv2 python eval/infer_nnunet.py \
      --model "$M7D300" --fold 0 --images "$HF/imagesTr" --manifest "$HF/manifest.json" --ref-um 7.91 --out "$PREDS/hf2um_m7d300"
  fi
  if use_real350; then
    have_all "$PREDS/xing_real350_s11836.json" || runlog xing_real350_s11836 uv run --with nnunetv2 --with pandas python \
      eval/p0139_crossings.py --model "$REAL350" --fold 0 --spacing 1.1836 --out "$PREDS/xing_real350_s11836"
    have_all "$PREDS/hf2um_real350" || runlog hf2um_real350 uv run --with nnunetv2 python eval/infer_nnunet.py \
      --model "$REAL350" --fold 0 --images "$HF/imagesTr" --manifest "$HF/manifest.json" --ref-um 7.91 --out "$PREDS/hf2um_real350"
  fi
  lease_release
}

stage_size() {
  log "stage size (GPU)"
  need "$M7/fold_0/checkpoint_best.pth" "$HF/imagesTr"
  gpu_check
  lease_acquire
  runout size uv run --with nnunetv2 python -c "$(py_size)"
  lease_release
}

graph_job() {  # <tag> <seg dir> [extra p0139_graph.py args]
  local tag=$1 seg=$2; shift 2
  bg "graph_$tag" uv run --with pandas python eval/p0139_graph.py --seg "$seg" --out "$PREDS/graph_$tag" \
    --blocks 12 --radii 40 --ks 30 "$@"
}

stage_score() {
  log "stage score (CPU)"
  need "$P0139/sites.json" data/metric/.venv/bin/python
  skip_m7d300_msg; skip_real350_msg
  # ---- PHerc0139: frozen patch graph (block 12, radius 40, k 30) on every mask set
  need "$PREDS/xing_m7pub_s11836_seg" "$PREDS/xing_m7pub_s10_seg" "$PREDS/m7_seg"
  if use_m7d300; then
    need "$PREDS/prob_m7d300_s11836" "$PREDS/prob_m7pub_s11836"
    have_all "$PREDS/ensB_seg" || runlog ens_probs uv run python eval/p0139_probs.py \
      --ens "$PREDS/prob_m7d300_s11836,$PREDS/prob_m7pub_s11836" --thr 0.5 --out "$PREDS/ensB_seg"
  fi
  have_all "$PREDS/graph_m7pub_s11836.csv" || graph_job m7pub_s11836 "$PREDS/xing_m7pub_s11836_seg"
  have_all "$PREDS/graph_m7pub_s10.csv" || graph_job m7pub_s10 "$PREDS/xing_m7pub_s10_seg"
  # bucket m7 prediction: the draft's 80.50 row used --dilate 0 (graph_m7_d0)
  have_all "$PREDS/graph_m7_d0.csv" || graph_job m7_d0 "$PREDS/m7_seg" --dilate 0
  # GT row: the labelled surface raster is the mask (labelsTr == instancesTr > 0)
  have_all "$PREDS/graph_gt.csv" || graph_job gt "$P0139/labelsTr"
  have_all "$PREDS/xing_gt.json" || bg xing_gt uv run --with pandas python eval/p0139_crossings.py --gt --out "$PREDS/xing_gt"
  if use_m7d300; then
    have_all "$PREDS/graph_m7d300_s11836.csv" || graph_job m7d300_s11836 "$PREDS/xing_m7d300_s11836_seg"
    have_all "$PREDS/graph_ensB.csv" || graph_job ensB "$PREDS/ensB_seg"
  fi
  if use_real350; then
    have_all "$PREDS/graph_real350_s11836.csv" || graph_job real350_s11836 "$PREDS/xing_real350_s11836_seg"
  fi
  waitall
  # ---- hf2um: official Kaggle blend (ignore = 2, ~25 s per crop per run) and compressed-sheet separation
  need "$PREDS/hf2um_m7pub" "$PREDS/hf2um_m7pubn"
  local tags=(m7pub m7pubn) t
  use_m7d300 && tags+=(m7d300)
  use_real350 && tags+=(real350)
  for t in "${tags[@]}"; do
    have_all "$PREDS/hf2um_${t}_score.csv" || bg "hf2um_${t}_score" data/metric/.venv/bin/python eval/score.py \
      --gt-dir "$HF/labelsTr" --pred-dir "$PREDS/hf2um_$t" --out "$PREDS/hf2um_${t}_score.csv"
    have_all "$PREDS/hf2um_${t}_sepd.csv" || bg "hf2um_${t}_sepd" uv run python eval/separation.py \
      --gt-dir "$HF/labelsSep" --pred-dir "$PREDS/hf2um_$t" --out "$PREDS/hf2um_${t}_sepd" --workers "$SEP_WORKERS"
  done
  waitall
  # summary: the first tag is the reference for the paired patch bootstrap
  local order=()
  use_real350 && order+=(real350)
  use_m7d300 && order+=(m7d300)
  order+=(m7pub m7pubn)
  runout hf2um_summary uv run --with pandas python eval/hf2um_summary.py "${order[@]}"
}

stage_paired() {
  log "stage paired (CPU)"
  local names=(graph_m7pub_s10 graph_m7pub_s11836 graph_m7_d0 graph_gt graph_m7d300_s11836 graph_ensB graph_real350_s11836)
  local csvs=() n
  for n in "${names[@]}"; do
    if [ -f "$PREDS/$n.csv" ]; then csvs+=("$PREDS/$n.csv"); else log "no $n.csv (skipped; run stage score)"; fi
  done
  need "$PREDS/graph_m7pub_s10.csv" "$PREDS/graph_m7pub_s11836.csv"
  # first CSV is the reference: headline = graph_m7pub_s11836 vs graph_m7pub_s10 (+2.08 pp [1.26, 2.93])
  runout paired_graph uv run --with pandas python eval/p0139_paired.py "${csvs[@]}"
  runout paired_direct uv run --with pandas python eval/p0139_paired.py "${csvs[@]}" --col delta_direct
  # certifier: the draft quotes m7d300 masks; without M7D300 it runs on the public m7 masks (numbers then differ)
  # (the certifier needs the c_* columns, which p0139_graph.py writes in its current version)
  local cin="$PREDS/graph_m7d300_s11836.csv"
  if [ ! -f "$cin" ] || ! head -n 1 "$cin" | grep -q c_deg; then
    cin="$PREDS/graph_m7pub_s11836.csv"
    log "certifier on public m7 masks (no usable m7d300 graph csv); numbers differ from the draft"
  fi
  dry || echo "certifier input: $cin" > "$RES/certify.txt"
  dry || uv run --with pandas --with scikit-learn python eval/p0139_certify.py "$cin" >> "$RES/certify.txt" 2> "$LOGDIR/certify.err" \
    || { tail -n 15 "$LOGDIR/certify.err" >&2; die "certifier failed"; }
  dry || cat "$RES/certify.txt"
  local und=(graph_gt graph_m7pub_s10 graph_m7pub_s11836)
  [ -f "$PREDS/graph_m7d300_s11836.csv" ] && und+=(graph_m7d300_s11836)
  runout undercount uv run --with pandas python -c "$(py_undercount)" "${und[@]}"
}

stage_figs() {
  log "stage figs (CPU)"
  mkdir -p "$FIG_OUT"
  local c
  if have_all "$PREDS/graph_gt.csv" "$PREDS/graph_m7pub_s10.csv" "$PREDS/graph_m7pub_s11836.csv" "$PREDS/graph_m7d300_s11836.csv" || dry; then
    run uv run --with pandas --with matplotlib python release/fig_classes.py --out "$FIG_OUT/winding_by_class.png"
  else
    log "skip winding_by_class.png: needs graph_m7d300_s11836.csv (set M7D300 and run m7, score)"
  fi
  # fig_compressed.py draws a panel for m7d300 as well, so it needs that run's masks
  if have_all "$P0139/imagesTr" "$P0139/labelsTr" "$PREDS/xing_m7pub_s10_seg" "$PREDS/xing_m7pub_s11836_seg" "$PREDS/xing_m7d300_s11836_seg" || dry; then
    run uv run --with tifffile --with matplotlib python release/fig_compressed.py --stem p0139_c034_z4256_y2624_x3616 \
      --pair 97,51,150,96,25,111 --label "true winding delta A-B = 3; graph solver on m7 native grid: 2; on m7 1.18x: 3" --out "$FIG_OUT/compressed_c034.png"
    run uv run --with tifffile --with matplotlib python release/fig_compressed.py --stem p0139_c077_z4000_y3488_x4128 \
      --pair 45,89,86,45,85,43 --label "true winding delta A-B = 3; graph solver on m7 native grid: 2; on m7 1.18x: 3" --out "$FIG_OUT/compressed_c077.png"
  else
    log "skip compressed_c034/c077.png: needs xing_m7d300_s11836_seg (set M7D300 and run m7)"
  fi
}

stage_report() {
  dry && { log "(dry run) report skipped"; return 0; }
  uv run --with pandas python -c "$(py_report)"
}

# ---------------------------------------------------------------- main
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

STAGES=("$@")
[ "${#STAGES[@]}" -gt 0 ] || STAGES=(all)
RUN=()
for s in "${STAGES[@]}"; do
  case "$s" in
    -h|--help|help) usage; exit 0 ;;
    all) RUN+=("${ALL_STAGES[@]}") ;;
    setup|bench|m7|size|score|paired|figs|report) RUN+=("$s") ;;
    *) die "unknown stage '$s' (stages: ${ALL_STAGES[*]})" ;;
  esac
done
for s in "${ALL_STAGES[@]}"; do   # run in pipeline order, once each, whatever order they were given
  [ "$s" = report ] && continue
  for r in "${RUN[@]}"; do
    if [ "$r" = "$s" ]; then "stage_$s"; break; fi
  done
done
log "headline numbers"
stage_report
