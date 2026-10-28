#!/bin/bash
# run_arm.sh - one arm of the PHerc0139 spiral-fit A/B on a GPU pod: band z [Z0, Z1) of volume 20250728140407
# (9.362 um), villa 0fb38c45 fit_spiral with the 343 recipe overrides (config/recipe.toml), sense ACW (catalog:
# z top-to-bottom True, left-handed False). Every step is resumable; rerun the same command after an interruption.
#
#   source ~/spiral_env.sh; bash ~/vss/spiral/run_arm.sh ~/vss/spiral/arms/arm0.env
#
# Arm file (bash, sourced; see arms/arm0.env). What an arm may change:
#   ARM            name; outputs go to $WORK/p0139/$ARM (default WORK=~/work)
#   MASK_TAG       short name of the surface mask; tracks and SDT built from it are cached per tag under $DATA
#   MASK_S3        s3:// zarr of the mask (levels 0 and 1); only the band's chunk rows are fetched. Or:
#   MASK_LOCAL     local zarr with levels 0 and 1 (e.g. our masks copied to the pod)
#   TRACKS_DBM     prebuilt tracks DBM (packed store + crossings next to it) -> paths.tracks_dbm; skips extraction
#   SURF_SDT       prebuilt surf-SDT ome.zarr (group 1) -> paths.surf_sdt; skips make_surf_sdt
#   REL_WINDINGS   relative_windings.json (point collection) copied into the dataset root; empty = none
#   FIT_OVERRIDES  JSON merged over the recipe overrides (FIT_SPIRAL_CONFIG_OVERRIDES), e.g. '{"loss_weight_x": 2}'
#                  Fit seed: '{"optimizer_random_seed": N}' (villa default 1; seeds np/torch RNGs and the host sample
#                  pools). Replicate arms <arm>s<N>.env add only this key (notes/2026-10-22-fit-seed-variance.md).
#   STEPS          optimizer_num_training_steps (default 30000 = recipe)
#   VILLA_PATCH    optional fitter patch, a file name under spiral/patches/ (applied once to $VILLA)
# Fixed for all arms: umbilicus and lasagna normals (prep_inputs.py, uploaded to $DATA/inputs), sense, band, recipe.
# Env: FIT_THREADS (OMP threads, default 4), FIT_CKPT_EVERY (default 2500), EXTRACT_WORKERS, Z0/Z1 (3000/4500).
# Other scrolls (arm file or env): SCROLL, VOL, UM, SENSE (defaults PHerc0139 / 20250728140407 / 9.362 / ACW). Inputs are
# $DATA/inputs/${SCROLL}_{nx,ny,grad_mag}.ome.zarr + umbilicus.json (prep_inputs.py --scroll); if $DATA/inputs/ct.zarr
# exists it is used as the SDT CT (PHerc1667: the fit frame is level 2 of the 2.399 um volume, whose own level 4 does
# not exist, so prep_inputs.py ct builds it). PHerc1667: SCROLL=PHerc1667 VOL=20251217075048 UM=9.596 SENSE=ACW.
set -euo pipefail
ARMFILE=$(realpath "$1"); source "$ARMFILE"
: "${ARM:?}" "${MASK_TAG:?}" "${VILLA_PY:?}" "${SPIRAL_DIR:?}" "${VILLA:?}"
HERE=$(cd "$(dirname "$0")" && pwd); STAGES=$HERE/../stages
SCROLL=${SCROLL:-PHerc0139}; VOL=${VOL:-20250728140407}; UM=${UM:-9.362}; SENSE=${SENSE:-ACW}
Z0=${Z0:-3000}; Z1=${Z1:-4500}
TMARGIN=256          # tracks are extracted over [Z0-256, Z1+256); the fitter keeps tracks entirely inside [Z0, Z1)
DATA=${DATA_ROOT:-$HOME/data}/$SCROLL
W=${WORK:-$HOME/work}/p0139/$ARM
mkdir -p "$W/logs" "$DATA"
LOG=$W/logs/timings.txt
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
INP=$DATA/inputs
for f in umbilicus.json ${SCROLL}_nx.ome.zarr ${SCROLL}_ny.ome.zarr ${SCROLL}_grad_mag.ome.zarr; do
  [[ -e $INP/$f ]] || { echo "missing $INP/$f (upload data/p0139/inputs from prep_inputs.py)"; exit 2; }
done
eval "$(python3 -I -c "
import tomllib;r=tomllib.load(open('$HERE/../config/recipe.toml','rb'))['sdt']
print(f\"LVL={r['surface_level']} THR={r['threshold']} ZM={r['z_margin']} CTL={r['ct_mask_level']} CTD={r['ct_dilate']}\")")"
FR="bash $STAGES/fetch_zarr_rows.sh"
S3B=s3://vesuvius-challenge-open-data/$SCROLL

# --- 1. inputs: lasagna normal pool, CT level 4 (SDT occupancy mask), mask rows --------------------------------------
if [[ ! -f $INP/packed_g2.done ]]; then
  stamp "pack lasagna normals"
  "$VILLA_PY" "$SPIRAL_DIR/pack_resident_pools.py" "$INP" --what normals,grad_mag --normal-group 2 --io-threads 16 \
    --verify 500 2>&1 | tail -5 && touch "$INP/packed_g2.done"
fi
CTZ=$DATA/ct.zarr
if [[ -d $INP/ct.zarr ]]; then
  CTZ=$INP/ct.zarr; [[ -d $CTZ/$CTL ]] || { echo "$CTZ has no group $CTL"; exit 2; }
elif [[ ! -f $DATA/ct_L$CTL.done ]]; then
  CTK=$(aws s3 ls --no-sign-request "$S3B/volumes/" | awk '{print $2}' | grep "^$VOL" | head -1); CTK=${CTK%/}
  $FR "$S3B/volumes/$CTK" "$DATA/ct.zarr" "$CTL" 0 all && touch "$DATA/ct_L$CTL.done"
fi
NEED_TRACKS=; [[ -z ${TRACKS_DBM:-} ]] && NEED_TRACKS=1
NEED_SDT=; [[ -z ${SURF_SDT:-} ]] && NEED_SDT=1
if [[ -n ${MASK_LOCAL:-} ]]; then MASK=$MASK_LOCAL
elif [[ -n $NEED_TRACKS$NEED_SDT ]]; then
  : "${MASK_S3:?set MASK_S3 or MASK_LOCAL (or both TRACKS_DBM and SURF_SDT)}"
  MASK=$DATA/mask_$MASK_TAG.zarr
  C0=$(aws s3 cp --no-sign-request "$MASK_S3/0/.zarray" - | python3 -I -c "import json,sys;print(json.load(sys.stdin)['chunks'][0])")
  C1=$(aws s3 cp --no-sign-request "$MASK_S3/$LVL/.zarray" - | python3 -I -c "import json,sys;print(json.load(sys.stdin)['chunks'][0])")
  R0=$(( (Z0 - TMARGIN) / C0 )); R1=$(( (Z1 + TMARGIN - 1) / C0 + 1 ))
  S0=$(( (Z0 - ZM) / (2 ** LVL * C1) )); S1=$(( (Z1 + ZM - 1) / (2 ** LVL * C1) + 1 ))
  if [[ -n $NEED_TRACKS && ! -f $MASK.L0_$R0-$R1.done ]]; then
    stamp "fetch mask L0 rows $R0..$R1"; $FR "$MASK_S3" "$MASK" 0 $R0 $R1 32 && touch "$MASK.L0_$R0-$R1.done"; fi
  if [[ -n $NEED_SDT && ! -f $MASK.L${LVL}_$S0-$S1.done ]]; then
    stamp "fetch mask L$LVL rows $S0..$S1"; $FR "$MASK_S3" "$MASK" $LVL $S0 $S1 && touch "$MASK.L${LVL}_$S0-$S1.done"; fi
fi

# --- 2. tracks (pinned extractor, published parameters) -------------------------------------------------------------
if [[ -n $NEED_TRACKS ]]; then
  TD=$DATA/tracks_$MASK_TAG; mkdir -p "$TD"
  TRACKS_DBM=$TD/${SCROLL}_${VOL}_${MASK_TAG}_z$((Z0 - TMARGIN))-$((Z1 + TMARGIN)).dbm
  if [[ ! -f $TRACKS_DBM.done ]]; then
    stamp "extract tracks -> $TRACKS_DBM"
    "$VILLA_PY" "$HERE/extract_tracks.py" --mask "$MASK/0" --out "$TRACKS_DBM" --z-min $((Z0 - TMARGIN)) \
      --z-max $((Z1 + TMARGIN)) --workers "${EXTRACT_WORKERS:-12}" 2>&1 | tee "$W/logs/extract_tracks.log"
    touch "$TRACKS_DBM.done"; stamp "tracks done"
  fi
fi

# --- 3. surf-SDT (as stages/s03_sdt.sh: make_surf_sdt.py, level 1, threshold 128, CT L4 occupancy) --------
if [[ -n $NEED_SDT ]]; then
  SDT_TAG=$MASK_TAG; [[ $Z0 != 3000 || $Z1 != 4500 ]] && SDT_TAG=${MASK_TAG}_z$Z0-$Z1   # SDT cache keyed by band
  SURF_SDT=$DATA/sdt_$SDT_TAG/${SCROLL}_surf_sdt.ome.zarr; mkdir -p "$(dirname "$SURF_SDT")"
  if [[ ! -f $SURF_SDT.done ]]; then
    stamp "surf-SDT -> $SURF_SDT"
    RESUME=; [[ -d $SURF_SDT ]] && RESUME=--resume
    "$VILLA_PY" "$SPIRAL_DIR/make_surf_sdt.py" --surf "$MASK" --group "$LVL" --out "$SURF_SDT" \
      --threshold "$THR" --working-voxel-um "$UM" --z-range "$((Z0 - ZM)):$((Z1 + ZM))" \
      --ct "$CTZ" --ct-group "$CTL" --ct-dilate "$CTD" --no-ct-zero \
      --workers "${SDT_WORKERS:-8}" $RESUME 2>&1 | tee "$W/logs/sdt.log"
    touch "$SURF_SDT.done"; stamp "SDT done"
  fi
fi

# --- 4. dataset root -------------------------------------------------------------------------------------------------
DS=$W/ds; mkdir -p "$DS"
cp "$INP/umbilicus.json" "$DS/umbilicus.json"
rm -f "$DS/relative_windings.json"
[[ -n ${REL_WINDINGS:-} ]] && cp "$REL_WINDINGS" "$DS/relative_windings.json"
python3 -I - "$DS" "$INP" "$TRACKS_DBM" "$SURF_SDT" "$SENSE" "$UM" "$LVL" "$Z0" "$Z1" "$ARMFILE" "$SCROLL" "$VOL" <<'PY'
import json, sys, os
ds, inp, tracks, sdt, sense, um, lvl, z0, z1, armfile, scroll, vol = sys.argv[1:]
spec = {"schema_version": 1, "name": scroll, "voxel_size_um": float(um), "spiral_outward_sense": sense,
        "surf_sdt_zarr_group": lvl, "normal_zarr_group": "2", "lasagna_scale": 4,
        "paths": {"normal_x": f"{inp}/{scroll}_nx.ome.zarr", "normal_y": f"{inp}/{scroll}_ny.ome.zarr",
                  "gradient_magnitude": f"{inp}/{scroll}_grad_mag.ome.zarr",
                  "tracks_dbm": tracks, "surf_sdt": sdt}}
json.dump(spec, open(os.path.join(ds, "spiral-scroll.json"), "w"), indent=2)
json.dump({"scroll": scroll, "volume": vol, "z0": int(z0), "z1": int(z1), "sense": sense,
           "um": float(um), "arm_file": armfile, "arm": open(armfile).read()},
          open(os.path.join(ds, "band.json"), "w"), indent=2)
PY

# --- 4b. optional fitter patch (VILLA_PATCH = file name under spiral/patches/), applied once to $VILLA ---------
if [[ -n ${VILLA_PATCH:-} ]]; then
  PF=$HERE/patches/$VILLA_PATCH
  if git -C "$VILLA" apply --reverse --check "$PF" 2>/dev/null; then stamp "villa patch $VILLA_PATCH already applied ($(sha256sum "$PF" | cut -c1-16))"
  else git -C "$VILLA" apply "$PF" && stamp "applied villa patch $VILLA_PATCH ($(sha256sum "$PF" | cut -c1-16))"; fi
fi

# --- 5. fit (pinned fit_spiral via fit_ckpt.py: autosave every FIT_CKPT_EVERY steps, resumes from it) ----------------
OUT=$W/fit; mkdir -p "$OUT"
if [[ -L $OUT/meshes && -n $(ls "$OUT/meshes" 2>/dev/null | head -1) ]]; then
  stamp "fit present: $OUT/meshes"
else
  OVR=$(python3 -I - "$HERE/../config/recipe.toml" "$Z0" "$Z1" "${STEPS:-}" "${FIT_OVERRIDES:-}" <<'PY'
import json, sys, tomllib
recipe, z0, z1, steps, extra = sys.argv[1:6]
o = dict(tomllib.load(open(recipe, "rb"))["spiral_fit"]["overrides"])
o.update({"z_begin": int(z0), "z_end": int(z1)})
if steps:
    o["optimizer_num_training_steps"] = int(steps)
if extra.strip():
    o.update(json.loads(extra))
print(json.dumps(o))
PY
)
  echo "$OVR" > "$OUT/overrides.json"
  RES=$(ls -t "$OUT"/*_slice-*/checkpoint_autosave.ckpt 2>/dev/null | head -1 || true)
  stamp "fit start (resume: ${RES:-none}) overrides $OVR"
  t0=$(date +%s)
  cd "$SPIRAL_DIR"
  OMP_NUM_THREADS=${FIT_THREADS:-4} WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES="$OVR" FIT_SPIRAL_OUT_DIR="$OUT" \
    FIT_SPIRAL_CACHE_DIR="$OUT/cache" FIT_SPIRAL_RESUME_PATH="$RES" FIT_CKPT_EVERY=${FIT_CKPT_EVERY:-2500} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VILLA_PY" "$HERE/fit_ckpt.py" --dataset "$DS" >> "$OUT/fit.log" 2>&1
  RUN=$(ls -td "$OUT"/*_slice-*/meshes/fitted 2>/dev/null | head -1 || true)
  [[ -n $RUN ]] || { stamp "fit wrote no meshes/fitted under $OUT"; exit 4; }
  ln -sfn "$RUN" "$OUT/meshes"
  stamp "fit done in $(( $(date +%s) - t0 )) s: $(ls "$RUN" | grep -c _spliced) spliced windings"
fi
python3 -I - "$OUT/meshes" "$W/arm_done.json" <<'PY'
import json, os, sys
m, out = sys.argv[1:]
ws = sorted(os.listdir(m))
json.dump({"meshes": os.path.realpath(m), "windings": ws,
           "spliced": [w for w in ws if w.endswith("_spliced")]}, open(out, "w"), indent=1)
print(f"{len(ws)} winding dirs in {m}")
PY
stamp "arm $ARM done"
