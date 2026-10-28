#!/usr/bin/env python3
"""extract_tracks.py - surface tracks for one z band from a surface-mask zarr, with the pinned villa extractor
(villa 0fb38c45 volume-cartographer/scripts/spiral/extract_surface_tracks.py) and the parameters of the published
PHerc0139 tracks (dl.ash2txt.org/datasets/spiral_datasets/PHerc0139/20250728140407/tracks/*.extract.json):
downsample 2, ribbons 2 deep every 8, vertical slabs 4 thick every 16, dust 400, max area 640000, interjoint paths of
>= 10 vertices. Why: the published DBM covers z 4500-17500 only (effective_z_bounds), so a band below 4500 has no
tracks; and the A/B arms that swap the mask need the same extractor on their own mask.

The slab computation is the pinned module's own (prepare_cc_labels + get_skeleton_tracks, unchanged); this wrapper
only (a) reads the mask in chunk-aligned blocks once and hands slabs to a process pool (the pinned script runs one
slab at a time and re-reads every chunk for each slab), and (b) pins Kimimaro's internal parallelism to 1 per worker.
Keys and values are the pinned format (h:{z}, vy:{y}, vx:{x} -> pickled list of int32 zyx arrays). Resumable: keys
already in the DBM are skipped. Then the packed store (pinned tracks.write_packed_track_store) and the crossings
sidecar (pinned build_track_crossings.py over the whole DBM) are written, as the published dataset has them.

    $VILLA_PY spiral/extract_tracks.py --mask <local m7.zarr>/0 --out <dir>/<name>.dbm --z-min 2744 --z-max 4756
Env: SPIRAL_DIR (pinned spiral scripts), PYTHONPATH with vcnative (crossings kernel).
"""
import argparse
import dbm
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import threading
import time

import numpy as np

SPIRAL_DIR = os.environ.get("SPIRAL_DIR") or sys.exit("SPIRAL_DIR not set (source ~/spiral_env.sh)")
sys.path.insert(0, SPIRAL_DIR)
import extract_surface_tracks as est  # noqa: E402  (pinned)
import kimimaro  # noqa: E402

PUBLISHED = {"downsample_factor": 2, "z_chunk_depth_h": 2, "z_chunk_stride_h": 8, "yx_chunk_thickness_v": 4,
             "yx_stride_v": 16, "dust_threshold": 400, "max_area_threshold": 640000, "path_mode": "interjoint"}
_orig_skel = kimimaro.skeletonize


def _skel_serial(*a, **k):
    k["parallel"] = 1
    return _orig_skel(*a, **k)


def configure(z_min, z_max):
    for k, v in PUBLISHED.items():
        setattr(est, k, v)
    est.z_min, est.z_max = z_min, z_max
    kimimaro.skeletonize = _skel_serial


def work(task):
    key, slab, offset = task
    pred = (slab > 0).astype(np.uint8)
    if pred.max() == 0:
        return key, pickle.dumps([])
    cc = est.prepare_cc_labels(pred)
    return key, pickle.dumps(est.get_skeleton_tracks(cc, offset_zyx=offset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="surface mask zarr array (level 0), local path")
    ap.add_argument("--out", required=True, help="tracks DBM path")
    ap.add_argument("--z-min", type=int, required=True)
    ap.add_argument("--z-max", type=int, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--crossing-workers", type=int, default=8)
    a = ap.parse_args()
    import zarr
    configure(a.z_min, a.z_max)
    arr = zarr.open_array(a.mask, mode="r")
    Z, Y, X = arr.shape
    cz, cy, cx = arr.chunks
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    zlo, zhi = min(a.z_min, Z), min(a.z_max, Z)
    t0 = time.time()
    db = dbm.open(a.out, "c")
    have = set(k.decode() for k in db.keys())
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers)
    pending = []           # AsyncResults
    n_done = [0]
    lock = threading.Lock()

    def drain(limit):
        while len(pending) > limit:
            key, val = pending.pop(0).get()
            with lock:
                db[key] = val
            n_done[0] += 1

    # horizontal ribbons, one chunk row of z at a time
    zs = [z for z in range(a.z_min, a.z_max, est.z_chunk_stride_h) if f"h:{z}" not in have]
    by_row = {}
    for z in zs:
        if min(z + est.z_chunk_depth_h, a.z_max, Z) > z:
            by_row.setdefault(z // cz, []).append(z)
        else:
            db[f"h:{z}"] = pickle.dumps([])
    for r, zl in sorted(by_row.items()):
        b0, b1 = min(zl), min(max(zl) + est.z_chunk_depth_h, a.z_max, Z)
        block = np.asarray(arr[b0:b1])
        for z in zl:
            ze = min(z + est.z_chunk_depth_h, a.z_max, Z)
            pending.append(pool.apply_async(work, ((f"h:{z}", block[z - b0:ze - b0].copy(), (z, 0, 0)),)))
        del block
        drain(2 * a.workers)
        print(f"horizontal: chunk row {r} queued, {n_done[0]} keys written [{time.time() - t0:.0f} s]", flush=True)
    drain(0)
    db.sync() if hasattr(db, "sync") else None

    # yx range of the band (pinned find_yx_range on the same array)
    min_yx, max_yx = est.find_yx_range(arr)
    print(f"yx range (full-res): {min_yx} .. {max_yx}", flush=True)

    for axis, prefix, lo, hi, n_along, cs in (("y", "vy", min_yx[0], max_yx[0], Y, cy),
                                              ("x", "vx", min_yx[1], max_yx[1], X, cx)):
        ws = [w for w in range(int(lo), int(hi), est.yx_stride_v) if f"{prefix}:{w}" not in have]
        blocks = {}
        for w in ws:
            if min(w + est.yx_chunk_thickness_v, n_along) - w < est.downsample_factor:
                db[f"{prefix}:{w}"] = pickle.dumps([])
            else:
                blocks.setdefault(w // cs, []).append(w)
        for bi, wl in sorted(blocks.items()):
            b0, b1 = min(wl), min(max(wl) + est.yx_chunk_thickness_v, n_along)
            block = np.asarray(arr[zlo:zhi, b0:b1, :] if axis == "y" else arr[zlo:zhi, :, b0:b1])
            for w in wl:
                we = min(w + est.yx_chunk_thickness_v, n_along)
                if axis == "y":
                    slab, off = block[:, w - b0:we - b0, :].copy(), (zlo, w, 0)
                else:
                    slab, off = block[:, :, w - b0:we - b0].copy(), (zlo, 0, w)
                pending.append(pool.apply_async(work, ((f"{prefix}:{w}", slab, off),)))
            del block
            drain(2 * a.workers)
            print(f"vertical {axis}: block {bi} queued, {n_done[0]} keys written [{time.time() - t0:.0f} s]", flush=True)
        drain(0)
    pool.close(); pool.join()
    db.close()
    ntr = 0
    with dbm.open(a.out, "r") as d:
        for k in d.keys():
            ntr += len(pickle.loads(d[k]))
    print(f"extraction done: {ntr} tracks in {time.time() - t0:.0f} s", flush=True)

    from tracks import _packed_store_if_current, write_packed_track_store  # pinned
    if _packed_store_if_current(a.out) is None:
        write_packed_track_store(a.out, force=True, show_progress=False)
    print(f"packed store written [{time.time() - t0:.0f} s]", flush=True)
    subprocess.run([sys.executable, os.path.join(SPIRAL_DIR, "build_track_crossings.py"), a.out,
                    "--workers", str(a.crossing_workers)], check=True)
    meta = {"made_by": "spiral/extract_tracks.py", "extractor": "villa 0fb38c45 extract_surface_tracks.py",
            "mask": os.path.abspath(a.mask), "parameters": {**PUBLISHED, "min_path_vertices": 10,
                                                             "z_min": a.z_min, "z_max": a.z_max},
            "effective_z_bounds": [zlo, zhi], "yx_bounds": [list(map(int, min_yx)), list(map(int, max_yx))],
            "num_tracks": ntr, "seconds": round(time.time() - t0)}
    json.dump(meta, open(a.out[:-4] + ".extract.json" if a.out.endswith(".dbm") else a.out + ".extract.json", "w"),
              indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    np.random.seed(0)
    main()
