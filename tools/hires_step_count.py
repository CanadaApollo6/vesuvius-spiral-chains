"""Count sheets along auto-winding chain steps in the 2.399 um PHerc0139 scan (notes/2026-10-22-hires-step-count.md).

Stage `profile --band z3000|z4500`: for every probe ray of the band (z, theta from the umbilicus, 9.362 frame) sample
  - the 2.399 um CT (volume 20260102150214, level 0) on a 7 x 7 tube of parallel lines (1 vox apart at 2.399 um, in
    the 2.399 z axis and the in-plane tangent), step 0.125 vox of 9.362 um along the ray;
  - the 9.362 um CT (volume 20250728140407, level 0) on a 3 x 3 tube (1 vox at 9.362), same radial grid.
  Points are mapped 9.362 -> 2.399 with the inverse of the published affine (p9_xyz = A p2399_xyz + t).
  Reads are byte ranges of the uncompressed zarr v2 chunks (only the z slices the tube needs), so a whole ray costs
  ~5-10 MB. Output: data/scratch/hires_step/prof_<band>/<z>_<theta>.npz (resumable).

Counting / evaluation live in tools/hires_step_eval.py.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "data" / "scratch" / "hires_step"
CACHE = REPO / "data" / "scratch" / "auto_windings"
from hires import CH, DR, PAD, R9, R24, TUBE9, TUBE24, SlabReader, ray_xyz9, to9, to24, umb  # noqa: E402,F401


def tube_points_24(z, theta, r):
    """(Nr, 7, 7, 3) 2.399 L0 zyx: ray centre line mapped to 2.399 plus offsets along the 2.399 z axis and the in-plane
    tangent (perpendicular to the mapped ray), 1 vox apart."""
    c = to24(ray_xyz9(z, theta, r))
    u = to24(ray_xyz9(z, theta, [r[0] + 100.0]))[0] - to24(ray_xyz9(z, theta, [r[0]]))[0]
    u /= np.linalg.norm(u)
    ez = np.array([1.0, 0, 0])
    w = np.cross(ez, u)
    w /= np.linalg.norm(w)
    o = np.arange(-TUBE24, TUBE24 + 1, dtype=float)
    return c[:, None, None, :] + o[None, :, None, None] * ez + o[None, None, :, None] * w


def tube_points_9(z, theta, r):
    c = ray_xyz9(z, theta, r)
    th = np.radians(theta)
    w = np.array([0.0, np.cos(th), -np.sin(th)])
    ez = np.array([1.0, 0, 0])
    o = np.arange(-TUBE9, TUBE9 + 1, dtype=float)
    return c[:, None, None, :] + o[None, :, None, None] * ez + o[None, None, :, None] * w


def band_rays(band):
    ch = json.loads((CACHE / "stepconf" / f"chains_{band}.json").read_text())
    ext = {}
    for c in ch:
        k = (int(c["z"]), float(c["theta"]))
        a, b = min(c["r"]), max(c["r"])
        e = ext.get(k)
        ext[k] = (a, b) if e is None else (min(e[0], a), max(e[1], b))
    return ext


def do_ray(job):
    (z, th), (a, b), outdir = job
    f = outdir / f"{z}_{th:g}.npz"
    if f.exists():
        return 0, 0
    r = np.arange(max(a - PAD, 4.0), b + PAD, DR)
    t0 = time.time()
    p24 = R24.sample(tube_points_24(z, th, r))
    p9 = R9.sample(tube_points_9(z, th, r), seg=192)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, r=r.astype(np.float32), p24=np.clip(np.rint(p24), 0, 255).astype(np.uint8),
             p9=np.clip(np.rint(p9), 0, 255).astype(np.uint8), z=z, theta=th, secs=time.time() - t0)
    tmp.rename(f)
    return 1, time.time() - t0


def profile(args):
    outdir = OUT / f"prof_{args.band}"
    outdir.mkdir(parents=True, exist_ok=True)
    ext = band_rays(args.band)
    keys = sorted(ext)
    if args.limit:
        rng = np.random.default_rng(0)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), args.limit, replace=False))]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (n, dt) in enumerate(ex.map(do_ray, [(k, ext[k], outdir) for k in keys])):
            done += n
            if (i + 1) % 20 == 0 or i + 1 == len(keys):
                el = time.time() - t0
                print(f"{i + 1}/{len(keys)} rays, {done} new, {el:.0f} s, 2.4um {R24.nreq} req {R24.nbytes / 1e9:.2f} GB, "
                      f"9um {R9.nreq} req {R9.nbytes / 1e9:.2f} GB", flush=True)
    (outdir / "stream_log.json").write_text(json.dumps(dict(rays=len(keys), new=done, secs=time.time() - t0,
                                                            req24=R24.nreq, gb24=R24.nbytes / 1e9, req9=R9.nreq,
                                                            gb9=R9.nbytes / 1e9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--band", choices=["z3000", "z4500"], required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"profile": profile}[a.cmd](a)


if __name__ == "__main__":
    main()
