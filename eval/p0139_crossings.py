"""Surface-first winding on PHerc0139: delta = number of predicted sheet crossings on the straight segment A -> B.

  uv run --with nnunetv2 --with pandas python eval/p0139_crossings.py --model <nnU-Net trainer dir> --out data/preds/xing_<tag>
  uv run --with pandas python eval/p0139_crossings.py --gt --out data/preds/xing_gt

Scored on the frozen scoring pairs (data/eval/p0139_9um/scoring_pairs.csv) with the same primary metric as
eval/winding_sites.py --dataset p0139 (|delta - expected| < 0.25, beats oracle pitch only if the paired site-bootstrap
CI is above 0). Crossing count: sample the segment every 0.25 vox, nearest voxel of the mask; count entries into the
mask (off-run >= --min-gap vox before it) for t in [--skip, L + --skip]. The first --skip vox are ignored (A's own
band), and the segment runs --skip vox past B (B's band may sit just past B). --gt uses the labelled surface raster
(instancesTr > 0): an upper bound for the estimator, below 100% where unlabelled sheets lie between labelled ones.
--spacing: input spacing relative to the model's plan (9.362 / 7.91 = 1.1836 resamples to the 7.91 training grid).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))


def count_entries(mask: np.ndarray, pts: np.ndarray, arc: np.ndarray, valid: np.ndarray, skip: float, min_gap: float,
                  step: float) -> np.ndarray:
    """Entries into the mask along sampled paths: pts (n, T, 3) float, arc (n, T) arclength, valid (n, T). An entry is
    an on-sample with arc >= skip preceded by an off-run of >= min_gap (counted from the path start)."""
    p = np.rint(pts).astype(int)
    ok = valid & np.all((p >= 0) & (p < np.array(mask.shape)), -1)
    on = np.zeros(ok.shape, bool)
    on[ok] = mask[tuple(p[ok].T)]
    idx = np.broadcast_to(np.arange(on.shape[1]), on.shape)
    last = np.maximum.accumulate(np.where(on, idx, -1), axis=1)
    prev = np.concatenate([np.full((on.shape[0], 1), -1), last[:, :-1]], 1)
    gap = (idx - prev - 1) * step
    return (on & (arc >= skip) & (gap >= min_gap)).sum(1).astype(float)


def mask_normals(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Unit sheet normals (3, Z, Y, X) from the structure tensor of the smoothed mask (largest eigenvector)."""
    from scipy import ndimage as ndi

    m = ndi.gaussian_filter(mask.astype(np.float32), 1.0)
    g = np.stack(np.gradient(m))
    J = np.empty(mask.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            J[..., i, j] = J[..., j, i] = ndi.gaussian_filter(g[i] * g[j], sigma)
    _, v = np.linalg.eigh(J)
    return np.moveaxis(v[..., :, -1], -1, 0)


def count_thick(mask: np.ndarray, normals: np.ndarray, pts: np.ndarray, arc: np.ndarray, valid: np.ndarray,
                skip: float, min_gap: float, step: float, T: float) -> np.ndarray:
    """Contact-aware count: as count_entries, but each counted run (an entry plus its continuations across off-runs
    shorter than min_gap) counts max(1, round(thickness / T)) sheets, thickness = sum of step * |u . n| over its
    on-samples (u = path direction, n = local sheet normal)."""
    p = np.rint(pts).astype(int)
    ok = valid & np.all((p >= 0) & (p < np.array(mask.shape)), -1)
    on = np.zeros(ok.shape, bool)
    on[ok] = mask[tuple(p[ok].T)]
    idx = np.broadcast_to(np.arange(on.shape[1]), on.shape)
    last = np.maximum.accumulate(np.where(on, idx, -1), axis=1)
    prev = np.concatenate([np.full((on.shape[0], 1), -1), last[:, :-1]], 1)
    entry = on & (arc >= skip) & ((idx - prev - 1) * step >= min_gap)
    seg = np.cumsum(entry, 1)
    u = np.gradient(pts, axis=1)
    u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-6)
    cos = np.zeros(on.shape, np.float32)
    nn = normals[(slice(None),) + tuple(p[on].T)].T
    cos[on] = np.abs((u[on] * nn).sum(-1))
    use = on & (seg > 0)
    K = int(seg.max()) + 1
    key = np.arange(on.shape[0])[:, None] * K + seg
    thick = np.bincount(key[use], step * cos[use], on.shape[0] * K).reshape(-1, K)[:, 1:]
    n = np.where(thick > 0, np.maximum(1, np.rint(thick / T)), 0)
    has = np.bincount(key[entry], None, on.shape[0] * K).reshape(-1, K)[:, 1:] > 0
    return (n * has).sum(1).astype(float)


def orient_normals(mask: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip normals onto the crop's dominant sheet normal (top eigenvector of the summed n n^T over the mask), so
    crossing signs are consistent within a crop (sheets in a 192^3 crop are roughly parallel)."""
    n = normals[:, mask]
    w, v = np.linalg.eigh(n @ n.T)
    d = v[:, -1]
    return normals * np.sign(np.tensordot(d, normals, 1) + 1e-9)[None]


def count_signed(mask: np.ndarray, normals: np.ndarray, pts: np.ndarray, arc: np.ndarray, valid: np.ndarray,
                 skip: float, min_gap: float, step: float, graze: float, absolute: bool = True) -> np.ndarray:
    """Signed count (absolute=False keeps the sign: + along the oriented normal): entries as in count_entries, each signed by sign(u . n) (u = path direction, n = oriented
    normal), entries with |u . n| < graze dropped; delta = |sum of signs|. Exit/re-entry along a fragmented band
    grazes, and an out-and-back excursion cancels."""
    p = np.rint(pts).astype(int)
    ok = valid & np.all((p >= 0) & (p < np.array(mask.shape)), -1)
    on = np.zeros(ok.shape, bool)
    on[ok] = mask[tuple(p[ok].T)]
    idx = np.broadcast_to(np.arange(on.shape[1]), on.shape)
    last = np.maximum.accumulate(np.where(on, idx, -1), axis=1)
    prev = np.concatenate([np.full((on.shape[0], 1), -1), last[:, :-1]], 1)
    entry = on & (arc >= skip) & ((idx - prev - 1) * step >= min_gap)
    u = np.gradient(pts, axis=1)
    u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-6)
    c = np.zeros(on.shape, np.float32)
    c[entry] = (u[entry] * normals[(slice(None),) + tuple(p[entry].T)].T).sum(-1)
    sgn = np.where(entry & (np.abs(c) >= graze), np.sign(c), 0.0)
    return np.abs(sgn.sum(1)) if absolute else sgn.sum(1)


def sample_paths(A: np.ndarray, B: np.ndarray, W: np.ndarray | None, skip: float, step: float = 0.25):
    """Points every `step` along A -> B (or A -> W -> B), extended `skip` past B. A, B, W: (n, 3)."""
    legs = [(A, B)] if W is None else [(A, W), (W, B)]
    L = [np.linalg.norm(b - a, axis=1) for a, b in legs]
    tot = sum(L) + skip
    T = int(np.ceil(tot.max() / step)) + 1
    arc = np.arange(T)[None] * step + np.zeros((len(A), 1))
    pts = np.zeros((len(A), T, 3))
    s0 = np.zeros(len(A))
    for k, ((a, b), l) in enumerate(zip(legs, L)):
        u = (b - a) / np.maximum(l, 1e-6)[:, None]
        last = k == len(legs) - 1
        sel = (arc >= s0[:, None]) & ((arc <= (s0 + l + (skip if last else 0))[:, None]) if last else (arc < (s0 + l)[:, None]))
        loc = a[:, None] + (arc - s0[:, None])[..., None] * u[:, None]
        pts[sel] = loc[sel]
        s0 = s0 + l
    return pts, arc, arc <= tot[:, None]


def crossings(mask: np.ndarray, A: np.ndarray, B: np.ndarray, skip: float, min_gap: float, W: np.ndarray | None = None,
              step: float = 0.25) -> np.ndarray:
    """Crossing counts for pairs A[i] -> B[i] (optionally through W[i]); A, B: (n, 3)."""
    pts, arc, valid = sample_paths(A, B, W, skip, step)
    return count_entries(mask, pts, arc, valid, skip, min_gap, step)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=9.362 / 7.91)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--tta", action="store_true", help="nnU-Net mirroring TTA")
    ap.add_argument("--seg", type=Path, default=None, help="use cached masks <seg>/<stem>.tif (e.g. an ensemble)")
    # frozen 2026-10-18 by eval/p0139_xing_sweep.py on the even-index half of the sites (real350)
    ap.add_argument("--skip", type=float, default=3.0)
    ap.add_argument("--min-gap", type=float, default=2.5)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--paths", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--graze", type=float, default=0.2,
                    help="signed count (count_signed), tuned on the even half; < 0 = unsigned entry count")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from p0139_sites import ROOT, load_sites, read_pairs
    from winding_sites import p0139_summary, write
    from scipy import ndimage as ndi

    pred = None
    if not args.gt and args.seg is None:
        from infer_nnunet import load_predictor

        pred = load_predictor(args.model, fold=args.fold, tta=args.tta)
    sites = load_sites(ROOT)[: args.limit or None]
    keep = {s["stem"] for s in sites}
    rows = [r for r in read_pairs(ROOT) if r["kind"] == "mix" and r["stem"] in keep]
    by = {}
    for r in rows:
        by.setdefault(r["stem"], []).append(r)
    segdir = args.seg or args.out.parent / (args.out.name + "_seg")
    for i, s in enumerate(sites):
        if args.gt:
            mask = tifffile.imread(s["instances_tif"]) > 0
        else:
            cache = segdir / f"{s['stem']}.tif"
            if cache.exists():
                mask = tifffile.imread(cache) > 0
            else:
                ct = tifffile.imread(s["image"]).astype(np.float32)
                seg = pred.predict_single_npy_array(ct[None], {"spacing": [args.spacing] * 3}, None, None, False)
                mask = seg == 1  # class 1 only: a separator class (Dataset116) is not surface
                segdir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(cache, mask.astype(np.uint8), compression="zlib")
        if args.dilate:
            mask = ndi.binary_dilation(mask, iterations=args.dilate)
        rr = by.get(s["stem"], [])
        if rr:
            A = np.array([[r["za"], r["ya"], r["xa"]] for r in rr], float)
            B = np.array([[r["zb"], r["yb"], r["xb"]] for r in rr], float)
            rng = np.random.default_rng(0)
            Ws = [None] + [(A + B) / 2 + rng.normal(0, args.sigma, A.shape) for _ in range(args.paths - 1)]
            if args.graze < 0:
                c = [crossings(mask, A, B, args.skip, args.min_gap, W=W) for W in Ws]
            else:
                nrm = orient_normals(mask, mask_normals(mask))
                c = []
                for W in Ws:
                    pts, arc, valid = sample_paths(A, B, W, args.skip)
                    c.append(count_signed(mask, nrm, pts, arc, valid, args.skip, args.min_gap, 0.25, args.graze))
            for r, d in zip(rr, np.median(np.stack(c), 0)):
                r["delta_xing"] = float(d)
                r["delta_oracle_pitch"] = r["proj"] / r["pitch_site"]
        print(f"[{i + 1}/{len(sites)}] {s['stem']}: {len(by.get(s['stem'], []))} pairs, mask frac {mask.mean():.3f}",
              flush=True)
    methods = ["xing", "oracle_pitch"]
    summ = {"dataset": "p0139", "estimator": "surface crossings on segment A->B",
            "source": "gt instancesTr" if args.gt else str(args.seg or args.model), "spacing": args.spacing, "skip": args.skip,
            "min_gap": args.min_gap, "dilate": args.dilate, "paths": args.paths, "sigma": args.sigma, "graze": args.graze, "tta": args.tta, "sites": len(sites), "pairs": len(rows),
            "methods": p0139_summary(rows, methods)}
    summ["resolves_sheets"] = summ["methods"]["xing"]["vs_oracle_pitch"]["beats_oracle_pitch"]
    cols = ["stem", "site_cls", "a", "b", "expected", "dist", "proj", "contact", "pitch_site", "delta_xing",
            "delta_oracle_pitch"]
    write(args.out, [{k: r[k] for k in cols} for r in rows], summ)


if __name__ == "__main__":
    main()
