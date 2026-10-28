#!/bin/bash
# render_compare.sh - flattened renders of the same few windings from several PHerc0139 spiral-fit arms and the human
# meshes, for the before/after figure (notes/2026-10-21-fit-figures.md). Runs the render stages unchanged:
# s05 snap (recipe, m7pub level 0) -> s06 lasagna flatten -> s06b montage (one winding per montage) -> s07 VC3D render
# (66 layers), then render_mid.py keeps a mean of the 7 middle layers per winding as a small uint8 tif.
#
#   source ~/spiral_env.sh; bash ~/vss/spiral/render_compare.sh ~/work/render arm0 arm19 arm2 -- human
# Layout (uploaded from the laptop): <R>/<arm>/{band.json,umbilicus.json,windings.txt,fit/meshes/w###_spliced/}
#   and <R>/human/{band.json,umbilicus.json,flat/w###.tifxyz} (human meshes already flat: only s06b + s07).
set -euo pipefail
R=$(cd "$1" && pwd); shift
FITS=(); FLATS=(); dst=FITS
for a in "$@"; do if [[ $a == -- ]]; then dst=FLATS; continue; fi; eval "$dst+=(\"\$a\")"; done
: "${PY:?source ~/spiral_env.sh}" "${VC3D:?}"
S=$(cd "$(dirname "$0")/../stages" && pwd)
D=${DATA_ROOT:-$HOME/data}/PHerc0139
M7=s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$R/timings.txt"; }
if [[ ${#FITS[@]} -gt 0 ]]; then
  C0=$(aws s3 cp --no-sign-request "$M7/0/.zarray" - | python3 -I -c "import json,sys;print(json.load(sys.stdin)['chunks'][0])")
  R0=$(( 3000 / C0 )); R1=$(( (4500 - 1) / C0 + 1 ))
  if [[ ! -f $D/m7.L0_$R0-$R1.done ]]; then
    stamp "fetch m7 L0 rows $R0..$R1"; mkdir -p "$D"
    bash "$S/fetch_zarr_rows.sh" "$M7" "$D/m7.zarr" 0 $R0 $R1 32 && touch "$D/m7.L0_$R0-$R1.done"
  fi
fi
for A in "${FITS[@]}"; do
  stamp "$A snap"; KEEP_MIN_SHEET=0 SNAP_PROCS=${SNAP_PROCS:-8} bash "$S/s05_snap.sh" "$R/$A" "$R/$A/windings.txt"
  stamp "$A flatten"; FLAT_PAR=${FLAT_PAR:-3} bash "$S/s06_flatten.sh" "$R/$A"
done
for A in "${FITS[@]}" "${FLATS[@]}"; do
  [[ -s $R/$A/montage/layout.json ]] || "$PY" -I "$S/s06b_montage.py" "$R/$A" --max-mnodes 0.01
  stamp "$A render"; bash "$S/s07_render.sh" "$R/$A"
  "$PY" -I "$(dirname "$0")/render_mid.py" "$R/$A"
done
stamp "render_compare done"
