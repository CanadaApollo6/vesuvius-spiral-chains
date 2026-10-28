"""Ray crossings of Kelly-numbered Stevens patches on the M2 ray grid, PHerc1667, all 7 whole-scroll bands.
notes/2026-10-29-kp-chains-prereg.md section 2. Label-free: reads no human label, mesh or label centre.

  uv run python -I data/scratch/kp_chains/crossings.py REPO_ROOT [--workers 8] [--limit N] [--out DIR]

Rays: exactly the M2 chain rays (tools/auto_windings.py probes(): z = zc + {-64, 0, 64} in [z0, z1), zc = 192 c + 96;
theta = 0, 15, ..., 345 deg), origin = umbilicus_p1667_L2.json control points with np.interp (clamped, as the M2 chain
rays), direction (cos theta, sin theta) in (x, y), points (cx + r cos, cy + r sin, z).
Patches: s4_good_patches.zip + s4_bad_patches.zip, read in memory (tifffile on bytes; nothing executed), mapped to L2
by the patch-numbering affine then /4. Only patches Kelly numbers in component 0 are used. Surface = valid 2x2 vertex
cells with spiral_eval.Surfaces' degenerate-quad filter (corner radius <= 4 x the patch median), each cell split into
two triangles (c0 c1 c2, c0 c2 c3). A ray crossing = intersection of the ray with a triangle's cut by the plane z.
Per (ray, patch): all hits; median radius, spread (max - min), count.
Writes data/scratch/kp_chains/crossings_z<z0>.npz per band (one row per (ray, patch) with >= 1 hit).
"""
import csv
import io
import json
import re
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(sys.argv[1])
WORKERS = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else REPO / "data/scratch/kp_chains"
ZD = REPO / "data/untrusted/stevens_s4_patches"
KELLY_CSV = REPO / "data/untrusted/kelly_scrolls/repo/community/wrap_index.csv"
M = np.array(json.loads((REPO / "data/scratch/patch_scout/affine_s4_7p91_to_2p399.json").read_text())["transformation_matrix"])
BANDS = [(400, 1500), (1500, 3000), (3000, 4500), (4500, 6000), (6000, 6500), (6500, 8000), (8000, 9000)]
THETAS = np.arange(0, 360, 15.0)
U = np.stack([np.cos(np.radians(THETAS)), np.sin(np.radians(THETAS))], 1)      # (24, 2) x, y


def ray_levels(z0, z1):
    """tools/auto_windings.py probes() with the M2 recipe (zoffs -64,0,64)."""
    out = set()
    for c in range(z0 // 192 - 1, z1 // 192 + 2):
        zc = c * 192 + 96
        out |= {zc + dz for dz in (-64, 0, 64) if z0 <= zc + dz < z1}
    return sorted(out)


def umbilicus():
    cp = json.loads((REPO / "data/eval/p1667/umbilicus_p1667_L2.json").read_text())
    cp = cp["control_points"] if isinstance(cp, dict) else cp
    Z = np.array([c["z"] for c in cp], float); X = np.array([c["x"] for c in cp], float); Y = np.array([c["y"] for c in cp], float)
    o = np.argsort(Z)
    return Z[o], X[o], Y[o]


ZU, XU, YU = umbilicus()
LEVELS = {z: bi for bi, (z0, z1) in enumerate(BANDS) for z in ray_levels(z0, z1)}
ZL = np.array(sorted(LEVELS), float)
ORIG = {z: np.array([np.interp(z, ZU, XU), np.interp(z, ZU, YU)]) for z in LEVELS}


def triangles(xyz):
    ok = (xyz >= 0).all(-1) & np.isfinite(xyz).all(-1)
    P = (xyz.astype(np.float64) @ M[:, :3].T + M[:, 3]) / 4.0            # xyz L2
    P[~ok] = np.nan
    c = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2)    # (H-1, W-1, 4, 3), as spiral_eval.Surfaces
    good = np.isfinite(c).all((-1, -2))
    c = c[good]
    if not len(c):
        return None
    cen = c.mean(1)
    rad = np.linalg.norm(c - cen[:, None], axis=-1).max(1)
    c = c[rad <= 4.0 * max(np.median(rad), 0.5)]
    return np.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]], 0)          # (n, 3, 3)


def patch_crossings(T):
    """-> list of (z, theta_index, r_median, r_spread, n_hits)."""
    out = []
    tz = T[..., 2]
    lo, hi = tz.min(1), tz.max(1)
    for z in ZL[(ZL >= lo.min()) & (ZL <= hi.max())]:
        zr = z + 1e-4
        m = (lo < zr) & (hi > zr)
        if not m.any():
            continue
        P = T[m]
        s = P[..., 2] - zr                                                   # (m, 3), never 0
        pts, msk = [], []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cr = (s[:, a] > 0) != (s[:, b] > 0)
            t = np.where(cr, s[:, a] / np.where(cr, s[:, a] - s[:, b], 1.0), 0.0)
            pts.append(P[:, a, :2] + t[:, None] * (P[:, b, :2] - P[:, a, :2]))
            msk.append(cr)
        pts, msk = np.stack(pts, 1), np.stack(msk, 1)                        # (m, 3, 2), (m, 3)
        idx = np.argsort(~msk, axis=1, kind="stable")[:, :2]
        A = np.take_along_axis(pts, idx[..., None], 1)[:, 0]
        B = np.take_along_axis(pts, idx[..., None], 1)[:, 1]
        e = B - A
        w = A - ORIG[int(z)]
        den = U[:, None, 0] * e[None, :, 1] - U[:, None, 1] * e[None, :, 0]  # cross(u, e)  (24, m)
        num_r = w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]
        num_t = w[None, :, 0] * U[:, None, 1] - w[None, :, 1] * U[:, None, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            r, t = num_r / den, num_t / den
        hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (r > 0)
        for k in np.nonzero(hit.any(1))[0]:
            rr = r[k][hit[k]]
            out.append((int(z), int(k), float(np.median(rr)), float(rr.max() - rr.min()), int(len(rr))))
    return out


def work(job):
    zname, names, elig = job
    zf = zipfile.ZipFile(ZD / zname)
    res, nerr = [], 0
    for d in names:
        pid = int(re.search(r"patch_(\d+)$", d).group(1))
        if pid not in elig:
            continue
        try:
            xyz = np.stack([tifffile.imread(io.BytesIO(zf.read(f"{d}/{c}.tif"))) for c in "xyz"], -1)
        except Exception:
            nerr += 1
            continue
        T = triangles(xyz)
        if T is None:
            continue
        for row in patch_crossings(T):
            res.append((pid,) + row)
    return zname, res, nerr


def main():
    t0 = time.time()
    KI = {}
    for r in csv.DictReader(open(KELLY_CSV)):
        if int(r["component"]) == 0:
            KI[int(r["patch"])] = (int(r["k_q3c"]), float(r["thN"]))
    elig = frozenset(KI)
    jobs, bad = [], {}
    for zname in ("s4_good_patches.zip", "s4_bad_patches.zip"):
        names = sorted({m.group(1) for x in zipfile.ZipFile(ZD / zname).namelist()
                        if (m := re.match(r"(.*patch_\d+)/x\.tif$", x))})
        if LIMIT:
            names = names[:LIMIT]
        for d in names:
            bad[int(re.search(r"patch_(\d+)$", d).group(1))] = int(zname.startswith("s4_bad"))
        jobs += [(zname, names[i:i + 250], elig) for i in range(0, len(names), 250)]
    rows, nerr = [], 0
    with Pool(WORKERS) as pool:
        for i, (zname, res, ne) in enumerate(pool.imap_unordered(work, jobs)):
            rows += res
            nerr += ne
            if i % 40 == 0:
                print(f"{i}/{len(jobs)} jobs, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    R = np.array(rows, dtype=np.float64)        # pid, z, theta_idx, r, spread, nhit
    for bi, (z0, z1) in enumerate(BANDS):
        m = (R[:, 1] >= z0) & (R[:, 1] < z1)
        S = R[m]
        o = np.lexsort((S[:, 0], S[:, 3], S[:, 2], S[:, 1]))     # z, theta, r, pid: deterministic order
        S = S[o]
        pid = S[:, 0].astype(np.int64)
        np.savez(OUT / f"crossings_z{z0}.npz", pid=pid, z=S[:, 1].astype(np.int32), ti=S[:, 2].astype(np.int16),
                 r=S[:, 3], spread=S[:, 4], nhit=S[:, 5].astype(np.int32),
                 k=np.array([KI[p][0] for p in pid], np.int64), thN=np.array([KI[p][1] for p in pid]),
                 bad=np.array([bad[p] for p in pid], np.int8),
                 levels=np.array(ray_levels(z0, z1), np.int32), thetas=THETAS,
                 origin=np.array([ORIG[z] for z in ray_levels(z0, z1)]))
        print(f"z{z0}-{z1}: {m.sum()} (ray, patch) rows, {len(np.unique(pid))} patches", flush=True)
    print(f"done: {len(R)} rows, {nerr} read errors, {len(elig)} eligible patches, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
