"""Fetch the bucket's m7 surface prediction (L0, th0.2) at the 87 PHerc0139 test-site crops.

  uv run --with zarr --with s3fs python scripts/p0139_m7.py [--out data/preds/m7_seg]

Writes {stem}.tif (uint8 1 = surface, m7 value > 0) and {stem}_raw.npy (m7 uint8), for eval/p0139_graph.py --seg.
Caveat: m7 (2026-04) may have seen PHerc0139 labels, so its score is a tool check, not a held-out result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_sites import ROOT, load_sites  # noqa: E402

URL = ("s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/"
       "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr")

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=REPO / "data" / "preds" / "m7_seg")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
vol = zarr.open_group(URL, mode="r", storage_options={"anon": True})["0"]
for s in load_sites(ROOT):
    f = args.out / f"{s['stem']}.tif"
    if f.exists():
        continue
    z, y, x = s["origin_zyx"]
    d, h, w = s["shape"]
    a = np.asarray(vol[z:z + d, y:y + h, x:x + w])
    np.save(args.out / f"{s['stem']}_raw.npy", a)
    tifffile.imwrite(f, (a > 0).astype(np.uint8), compression="zlib")
    print(s["stem"], a.shape, np.unique(a)[:6], round(float((a > 0).mean()), 3), flush=True)
