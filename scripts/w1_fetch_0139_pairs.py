"""W1: same-site PHerc0139 crops at 2.399 um (CT + surface-labels-2026-07) and 9.362 um (CT), registered.

Uses the published affine 2.399 -> 9.362 (transform.json next to the 2.399 volume, xyz order), then refines
the translation by NCC between the 9.362 crop and the 2.399 crop blurred + resampled onto the 9.362 grid.

  uv run python scripts/w1_fetch_0139_pairs.py --site w047-w059:181.78.139 --site w037-w041:171.113.85

Writes data/eval/surface_labels_2026_07_sample/p0139_<zarr>_<chunk>.npz (+ .json provenance).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import urllib.request
import zarr
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "eval" / "surface_labels_2026_07_sample"
S3 = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/"
VOL24 = S3 + "20260102150214-2.399um-0.2m-78keV-masked.zarr"
VOL9 = S3 + "20250728140407-9.362um-1.2m-113keV-masked.zarr"
TF = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/transform.json"
LAB = "https://huggingface.co/buckets/scrollprize/datasets/resolve/surfaces/2um_032726/{}.zarr"
N = 512  # 2.399 crop edge (vox)
SIGMA24 = 1.6  # blur in 2.4 voxels before resampling to 9.362 (~ 9.362 um FWHM box)


def open_s3(url):
    return zarr.open_group(url, mode="r", storage_options={"anon": True})["0"]


def open_http(url):
    return zarr.open_group(url, mode="r")["0"]


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", action="append", required=True, help="<label zarr>:<z.y.x chunk index at L0 (128^3)>")
    p.add_argument("--shift", type=int, default=6, help="max integer shift (9.362 vox) in the NCC search")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tf = json.loads(urllib.request.urlopen(TF).read())
    M = np.array(tf["transformation_matrix"])  # 2.399 xyz -> 9.362 xyz
    A, t = M[:, :3], M[:, 3]
    Ainv = np.linalg.inv(A)
    ct24, ct9 = open_s3(VOL24), open_s3(VOL9)
    for site in a.site:
        zname, chunk = site.split(":")
        c = np.array(list(map(int, chunk.split("."))))
        o24 = c * 128 + 64 - N // 2  # zyx origin of the 2.399 crop
        t0 = time.time()
        sl = tuple(slice(int(o), int(o) + N) for o in o24)
        lab = np.asarray(open_http(LAB.format(zname))[sl])
        ct = np.asarray(ct24[sl])
        # 9.362 bbox of the crop
        corners = np.array([[z, y, x] for z in (0, N) for y in (0, N) for x in (0, N)], float) + o24
        c9 = corners[:, ::-1] @ A.T + t  # xyz @9.362
        lo9 = np.floor(c9.min(0))[::-1].astype(int) - a.shift - 2
        hi9 = np.ceil(c9.max(0))[::-1].astype(int) + a.shift + 2
        v9 = np.asarray(ct9[tuple(slice(int(l), int(h)) for l, h in zip(lo9, hi9))])
        # render 2.4 onto the 9.362 grid (inner cube fully inside the 2.4 crop)
        blur = ndi.gaussian_filter(ct.astype(np.float32), SIGMA24)
        zz, yy, xx = np.meshgrid(*[np.arange(s) for s in v9.shape], indexing="ij")
        p9 = np.stack([xx, yy, zz], -1).reshape(-1, 3) + lo9[::-1]  # xyz @9.362
        p24 = (p9 - t) @ Ainv.T  # xyz @2.399
        q = p24[:, ::-1] - o24  # zyx local in 2.4 crop
        inside = ((q >= 2) & (q <= N - 3)).all(1).reshape(v9.shape)
        rend = np.full(v9.shape, np.nan, np.float32)
        rend[inside] = map_coordinates(blur, q[inside.ravel()].T, order=1)
        labr = np.zeros(v9.shape, np.float32)  # label-1 fraction resampled (box-ish) onto 9.362 grid
        labr[inside] = map_coordinates(ndi.uniform_filter((lab == 1).astype(np.float32), 4), q[inside.ravel()].T, order=1)
        # integer translation search over the inner region
        idx = np.argwhere(inside)
        lo_i, hi_i = idx.min(0) + a.shift, idx.max(0) - a.shift
        R = rend[tuple(slice(l, h) for l, h in zip(lo_i, hi_i))]
        best = (-2, None)
        scores = {}
        for dz in range(-a.shift, a.shift + 1):
            for dy in range(-a.shift, a.shift + 1):
                for dx in range(-a.shift, a.shift + 1):
                    V = v9[tuple(slice(l + d, h + d) for l, h, d in zip(lo_i, hi_i, (dz, dy, dx)))].astype(np.float32)
                    s = ncc(R, V)
                    scores[(dz, dy, dx)] = s
                    if s > best[0]:
                        best = (s, (dz, dy, dx))
        d0 = np.array(best[1])
        # parabolic sub-voxel refinement per axis
        sub = []
        for ax in range(3):
            e = np.eye(3, dtype=int)[ax]
            sm, s0, sp = (scores.get(tuple(d0 + k * e), np.nan) for k in (-1, 0, 1))
            den = sm - 2 * s0 + sp
            sub.append(0.5 * (sm - sp) / den if np.isfinite(den) and den < 0 else 0.0)
        shift = d0 + np.array(sub)
        name = f"p0139_{zname}_{chunk.replace('.', '_')}"
        np.savez_compressed(OUT / f"{name}.npz", ct24=ct, lab24=lab, ct9=v9, rend9=np.nan_to_num(rend), lab9frac=labr)
        prov = {
            "label_zarr": LAB.format(zname), "ct24": VOL24, "ct9": VOL9, "transform": TF, "level": 0,
            "origin24_zyx": o24.tolist(), "shape24": [N] * 3, "origin9_zyx": lo9.tolist(), "shape9": list(v9.shape),
            "ncc_at_published_affine": scores[(0, 0, 0)], "ncc_best": best[0],
            "refine_shift9_zyx_vox": shift.tolist(), "refine_shift_um": float(np.linalg.norm(shift) * 9.362),
            "label_values": {int(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True))},
            "note": "rend9 = 2.4 CT blurred sigma 1.6 vox, resampled to the 9.362 grid with the published affine (no refinement);"
                    " lab9frac = label==1 fraction (4-vox box) resampled the same way. Apply refine_shift to ct9 indices to align.",
            "time_s": round(time.time() - t0, 1),
        }
        (OUT / f"{name}.json").write_text(json.dumps(prov, indent=1))
        print(name, {k: prov[k] for k in ("ncc_at_published_affine", "ncc_best", "refine_shift9_zyx_vox", "refine_shift_um", "label_values", "time_s")}, flush=True)


if __name__ == "__main__":
    main()
