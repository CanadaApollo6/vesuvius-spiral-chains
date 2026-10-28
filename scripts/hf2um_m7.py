"""Fetch the bucket's m7 surface prediction (run at CT level 2) for the hf2um crops: data/preds/hf2um_m7/{stem}.tif.
Only MANBp and PHerc0343P have one (no PHerc1667 prediction in the bucket).
  uv run --with "zarr==2.18.4" --with "numcodecs<0.16" --with s3fs --with tifffile python scripts/hf2um_m7.py
"""
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr

REPO = Path(__file__).resolve().parents[1]
B = "s3://vesuvius-challenge-open-data/"
M7 = {"manbp": "PHercMANBp/representations/predictions/surfaces/20251216152116-surface-20260413222639-surface-m7-L2-th0.2.zarr",
      "p0343p": "PHerc0343P/representations/predictions/surfaces/20260304131111-surface-20260413222639-surface-m7-L2-th0.2.zarr"}
out = REPO / "data/preds/hf2um_m7"
out.mkdir(parents=True, exist_ok=True)
man = json.loads((REPO / "data/eval/hf2um/manifest.json").read_text())
for vol, key in M7.items():
    g = zarr.open_group(B + key, mode="r", storage_options={"anon": True})
    a = g["0"]
    print(vol, "m7 L0 shape", a.shape, a.dtype, flush=True)
    for m in man:
        if m["vol"] != vol:
            continue
        z, y, x = m["origin_l2"]
        p = np.asarray(a[z:z + 160, y:y + 160, x:x + 160]) > 0
        tifffile.imwrite(out / f"{m['stem']}.tif", p.astype(np.uint8), compression="zlib")
print("done", len(list(out.glob("*.tif"))))
