#!/usr/bin/env python3
"""predict_band.py - nnU-Net surface masks for a z band of a whole OME-Zarr scroll volume, written as a 2-level mask
zarr in the bucket-mask layout (so extract_tracks.py / make_surf_sdt.py read it like the bucket m7 mask via MASK_LOCAL).

Layout copied from the bucket mask PHerc0139/.../20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr:
full-volume shape (level 0 = the CT's level 0 grid, so no shift/flip), uint8 {0, 255}, chunks 192^3, blosc zstd
clevel 1 shuffle, dimension_separator "/", fill 0; level 1 = level 0 [::2, ::2, ::2] (verified on the bucket mask: the
bucket's level 1 equals stride-2 subsampling exactly). Only the band's chunks are written; everything else is fill.

Work unit: a block of up to 384^3 level-0 voxels aligned to 384 (2x2x2 output chunks, one level-1 chunk), read with a
--margin context and predicted with predict_single_npy_array (tile step 0.5, gaussian, no TTA = eval/infer_nnunet.py's
load_predictor), surface = argmax == --surface-class. Blocks whose CT (level 2, masked volume) is all zero are skipped.
Each worker takes blocks[i::n]; one process per GPU (CUDA_VISIBLE_DEVICES). Resumable: a block is done when its marker
<out>/.blocks/<z>_<y>_<x> exists (written after both levels). Spacing: --um / --ref-um x the plans spacing
(9.362 / 7.91 = 1.1836, "m7 at 1.18x").

Bucket-recipe options (for scrolls without a bucket mask, e.g. PHerc1667): --level L predicts on CT level L (mask
level 0 = CT level L grid; occupancy test on level L+2), --threshold T writes softmax P(surface) > T instead of argmax.
The bucket m7 "-L2-th0.2" masks of the 2.4 um scans are m7 on CT level 2 at native spacing, no TTA, threshold 0.2,
so: --level 2 --um 1 --ref-um 1 --threshold 0.2.

  python predict_band.py init --out m.zarr --zarr <ct.zarr>                       # create the empty 2-level zarr
  python predict_band.py run --out m.zarr --zarr <ct.zarr> --model <dir> --z0 2688 --z1 4800 --worker 0 --nworkers 8
  python predict_band.py blocks --out m.zarr --zarr <ct.zarr> --z0 2688 --z1 4800   # list/count blocks (dry run)
  python predict_band.py run ... --only z_y_x[,z_y_x]                             # specific blocks (alignment test)
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

HERE = Path(__file__).resolve().parent
B = 384          # block edge (level 0); 2 output chunks per axis
CH = 192         # output chunk edge (both levels)


def open_ct(url):
    so = {"anon": True} if url.startswith("s3://") else None
    return zarr.open_group(url, mode="r", storage_options=so) if so else zarr.open_group(url, mode="r")


def init(out, ct_url, level=0):
    ct = open_ct(ct_url)
    s0 = ct[str(level)].shape
    s1 = tuple((n + 1) // 2 for n in s0)
    g = zarr.open_group(str(out), mode="a")
    comp = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE, blocksize=0)
    for name, shape in (("0", s0), ("1", s1)):
        if name not in g:
            g.create_dataset(name, shape=shape, chunks=(CH, CH, CH), dtype="u1", compressor=comp, fill_value=0,
                             dimension_separator="/", overwrite=False)
    g.attrs["multiscales"] = [{
        "axes": [{"name": a, "type": "space"} for a in "zyx"],
        "datasets": [{"coordinateTransformations": [{"scale": [float(2 ** i)] * 3, "type": "scale"}], "path": str(i)}
                     for i in (0, 1)],
        "name": "/", "version": "0.4"}]
    return g


def blocks(ct, z0, z1, level=0):
    """384-aligned blocks covering [z0, z1) x full yx, minus blocks whose level-(level+2) CT is all zero."""
    assert z0 % B == 0, "z0 must be a multiple of 384 (blocks map to whole level-1 chunks)"
    s0 = ct[str(level)].shape
    l2 = ct[str(level + 2)]
    zs = range(z0, z1, B)
    out = []
    for bz in zs:
        ez = min(bz + B, z1)
        occ = np.asarray(l2[bz // 4:(ez + 3) // 4, :, :]) > 0          # one slab read per z block
        for by in range(0, s0[1], B):
            for bx in range(0, s0[2], B):
                sub = occ[:, by // 4:(by + B + 3) // 4, bx // 4:(bx + B + 3) // 4]
                if sub.any():
                    out.append((bz, ez, by, min(by + B, s0[1]), bx, min(bx + B, s0[2])))
    return out


def key(b):
    return f"{b[0]}_{b[2]}_{b[4]}"


def run(args):
    sys.path.insert(0, str(HERE.parents[1] / "eval"))
    from infer_nnunet import load_predictor  # noqa: E402

    ct = open_ct(args.zarr)
    src = ct[str(args.level)]
    g = zarr.open_group(str(args.out), mode="r+")
    o0, o1 = g["0"], g["1"]
    done_dir = Path(args.out) / ".blocks"
    done_dir.mkdir(exist_ok=True)
    allb = blocks(ct, args.z0, args.z1, args.level)
    if args.only:
        want = set(args.only.split(","))
        allb = [b for b in allb if key(b) in want]
    mine = [b for b in allb[args.worker::args.nworkers] if not (done_dir / key(b)).exists()]
    print(f"worker {args.worker}/{args.nworkers}: {len(mine)} blocks to do of {len(allb)} total", flush=True)
    if not mine:
        return
    pred = load_predictor(Path(args.model), fold=args.fold, tta=False)
    spacing = [s * args.um / args.ref_um for s in pred.configuration_manager.spacing]
    M = args.margin
    S = src.shape

    def read(b):
        lo = [max(b[0] - M, 0), max(b[2] - M, 0), max(b[4] - M, 0)]
        hi = [min(b[1] + M, S[0]), min(b[3] + M, S[1]), min(b[5] + M, S[2])]
        return lo, hi, np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])

    nxt = {}

    def prefetch(b):
        nxt[key(b)] = read(b)

    th = threading.Thread(target=prefetch, args=(mine[0],)); th.start()
    t_start = time.time(); nvox = 0
    for i, b in enumerate(mine):
        th.join()
        lo, hi, ct_blk = nxt.pop(key(b))
        if i + 1 < len(mine):
            th = threading.Thread(target=prefetch, args=(mine[i + 1],)); th.start()
        t0 = time.time()
        if ct_blk.max() == 0:
            surf = np.zeros(tuple(h - l for l, h in zip(lo, hi)), bool)
        else:
            _, prob = pred.predict_single_npy_array(ct_blk.astype(np.float32)[None], {"spacing": spacing}, None, None,
                                                    True)
            if args.threshold is None:
                surf = np.argmax(prob, axis=0) == args.surface_class
            else:
                surf = prob[args.surface_class] > args.threshold
        core = surf[b[0] - lo[0]:b[1] - lo[0], b[2] - lo[1]:b[3] - lo[1], b[4] - lo[2]:b[5] - lo[2]]
        u8 = core.astype(np.uint8) * 255
        o0[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = u8
        o1[b[0] // 2:(b[1] + 1) // 2, b[2] // 2:(b[3] + 1) // 2, b[4] // 2:(b[5] + 1) // 2] = u8[::2, ::2, ::2]
        (done_dir / key(b)).write_text(json.dumps({"s": round(time.time() - t0, 1), "surf": float(core.mean())}))
        nvox += core.size
        el = time.time() - t_start
        print(f"block {key(b)} {core.shape} surf {core.mean():.3f} in {time.time() - t0:.1f}s; "
              f"{i + 1}/{len(mine)} done, {nvox / el / 1e6:.2f} Mvox/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "run", "blocks"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--model")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--um", type=float, default=9.362)
    ap.add_argument("--ref-um", type=float, default=7.91)
    ap.add_argument("--z0", type=int, default=2688)
    ap.add_argument("--z1", type=int, default=4800)
    ap.add_argument("--margin", type=int, default=32)
    ap.add_argument("--surface-class", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--level", type=int, default=0, help="CT level to predict on (mask level 0 = this grid)")
    ap.add_argument("--threshold", type=float, help="P(surface) > T instead of argmax (bucket: 0.2)")
    args = ap.parse_args()
    if args.cmd == "init":
        init(args.out, args.zarr, args.level)
    elif args.cmd == "blocks":
        bl = blocks(open_ct(args.zarr), args.z0, args.z1, args.level)
        vox = sum((b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]) for b in bl)
        print(json.dumps({"blocks": len(bl), "gvox": round(vox / 1e9, 2), "first": [key(b) for b in bl[:5]]}))
    else:
        if not Path(args.out, "1", ".zarray").exists():
            init(args.out, args.zarr, args.level)   # single worker convenience; with several workers run init first
        run(args)


if __name__ == "__main__":
    main()
