"""hf2um: a ~9 um surface test set from the organisers' surface-labels-2026-07 (2um_032726) at pyramid level 2.

  uv run --with "zarr==2.18.4" --with "numcodecs<0.16" --with fsspec --with aiohttp --with requests --with s3fs \
      --with tifffile python scripts/hf2um_set.py [--per-vol 40]

Labels: HF bucket scrollprize/datasets surfaces/2um_032726/<vol>.zarr, level 2 (4x, nearest; rasterized meshes,
0 = labelled background, 1 = surface, 2 = ignore). Level 0/1 are only partly uploaded (404 inside labelled regions),
level 2 is complete. CT: the matching open-data volume, level 2 (same shape as the label level 2).
Voxel size: 2.399 um x 4 = 9.60 um (MANBp, 1667), 2.215 um x 4 = 8.86 um (0343P).

Scrolls: none is in Dataset103/107/109 as a 2 um scan. PHerc0343 (not 0343P) m7 pseudo-labels are in Dataset109
(round 2). PHerc0139 (w0xx volumes), 0500P2 (in Dataset107) and s1 (Kaggle real data) are left out.

Crops: 160^3 at level 2, chosen at label level 5 (20^3 L5 windows at stride 5, non-overlapping) where ignore < 30 % and
surface > 2 %, random with seed 0, then CT zero fraction <= 5 % and level-2 surface fraction >= 1 %. Writes imagesTr/<stem>_0000.tif,
labelsTr/<stem>.tif (uint8 0/1/2) and manifest.json (stem, vol, origin at L2, um).
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import numpy as np
import tifffile
import zarr

REPO = Path(__file__).resolve().parents[1]
H = "https://huggingface.co/buckets/scrollprize/datasets/resolve/surfaces/2um_032726/"
B = "s3://vesuvius-challenge-open-data/"
VOLS = {  # tag: (label zarr, CT key, um at level 2)
    "manbp": ("PHercMANBp-ct-2um_surface.zarr", "PHercMANBp/volumes/20251216152116-2.399um-0.2m-78keV-masked.zarr", 2.399 * 4),
    "p1667": ("SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked_surface.zarr",
              "PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr", 2.399 * 4),
    "p0343p": ("2.215um_0.4m_111keV_PHerc0343P_surface.zarr",
               "PHerc0343P/volumes/20260304131111-2.215um-0.4m-111keV-masked.zarr", 2.215 * 4),
}
C = 160


def retry(f, tries=8):
    """HF answers 429 under load (zarr then reports a missing array or chunk): back off and retry."""
    for k in range(tries):
        try:
            return f()
        except Exception as e:
            if k == tries - 1:
                raise
            print("retry", type(e).__name__, str(e)[:80], flush=True)
            time.sleep(15 * 2 ** min(k, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vol", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPO / "data/eval/hf2um")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    (args.out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (args.out / "labelsTr").mkdir(parents=True, exist_ok=True)
    manifest = []
    for tag, (lk, ck, um) in VOLS.items():
        tok = Path.home() / ".cache/huggingface/token"  # authenticated reads get a higher HF rate limit
        hdr = {"Authorization": f"Bearer {tok.read_text().strip()}"} if tok.exists() else {}
        lg = fsspec.get_mapper(H + lk, client_kwargs={"headers": hdr})
        lab2 = retry(lambda: zarr.open_array(lg, path="2", mode="r"))
        ct2 = zarr.open_group(B + ck, mode="r", storage_options={"anon": True})["2"]
        assert ct2.shape == lab2.shape, (tag, ct2.shape, lab2.shape)
        l5 = retry(lambda: zarr.open_array(lg, path="5", mode="r")[:])
        # 20^3 L5 windows (= 160^3 L2) at a stride of 5 L5 voxels, then greedy non-overlapping picks in random order
        def box_mean(m):
            ii = np.pad(m.astype(np.float64).cumsum(0).cumsum(1).cumsum(2), ((1, 0), (1, 0), (1, 0)))
            s_ = ii[20:, 20:, 20:] - ii[:-20, 20:, 20:] - ii[20:, :-20, 20:] - ii[20:, 20:, :-20] \
                + ii[:-20, :-20, 20:] + ii[:-20, 20:, :-20] + ii[20:, :-20, :-20] - ii[:-20, :-20, :-20]
            return s_[::5, ::5, ::5] / 8000
        surf, ign = box_mean(l5 == 1), box_mean(l5 == 2)
        cand = np.argwhere((ign < 0.3) & (surf > 0.02)) * 5  # L5 window corner
        order = cand[np.random.default_rng(0).permutation(len(cand))]
        pick = []
        for c in order:
            if all(np.abs(c - q).max() >= 20 for q in pick):
                pick.append(c)
            if len(pick) >= args.per_vol * 2:
                break
        print(tag, "candidates", len(cand), "non-overlapping", len(pick), flush=True)

        def one(c):
            return retry(lambda: one_(c))

        def one_(c):
            z, y, x = (c * 8).tolist()  # L5 -> L2
            stem = f"{tag}_z{z}_y{y}_x{x}"
            fi, fl = args.out / "imagesTr" / f"{stem}_0000.tif", args.out / "labelsTr" / f"{stem}.tif"
            if fi.exists() and fl.exists():
                return stem, [z, y, x]
            a = np.asarray(ct2[z:z + C, y:y + C, x:x + C])
            if a.shape != (C, C, C) or (a == 0).mean() > 0.05:
                return None
            lab = np.asarray(lab2[z:z + C, y:y + C, x:x + C])
            lab = np.where(lab == 255, 1, lab).astype(np.uint8)
            if (lab == 1).mean() < 0.01:
                return None
            tifffile.imwrite(fi, a, compression="zlib")
            tifffile.imwrite(fl, lab, compression="zlib")
            return stem, [z, y, x]

        with ThreadPoolExecutor(args.workers) as ex:
            got = [r for r in ex.map(one, pick) if r][: args.per_vol]
        manifest += [{"stem": s, "vol": tag, "origin_l2": o, "um": um} for s, o in got]
        print(tag, "kept", len(got), flush=True)
    keep = {m["stem"] for m in manifest}
    for d in ("imagesTr", "labelsTr"):
        for f in (args.out / d).glob("*.tif"):
            if f.name.replace("_0000.tif", "").replace(".tif", "") not in keep:
                f.unlink()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("total", len(manifest))


if __name__ == "__main__":
    main()
