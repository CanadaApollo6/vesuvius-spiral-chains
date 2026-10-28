"""Merge-sensitive endpoint: does a prediction keep GT sheets apart at compressed sites?

  uv run python eval/separation.py --pred-dir <dir of {stem}.tif> --out results/sep_<name> [--gt-dir ...] [--limit N]

Sites come from GT bands exactly as in the contact check (eval/contact_check.py find_sites, seed 0): a band centre
point, its normal, and the next band centre along the normal at centre spacing d (7.91 um voxels).
  compressed: d < 10.5 (where models fuse sheets, Jeong villa #191); wide: 16 <= d < 24 (control).
Per site, for a binary prediction P:
  gap: some sample strictly between the two band centres (0.25 vox steps, nearest voxel) is background in P.
  cc:  both band centres are foreground in P and lie in different 26-connected components of P inside a 33^3 window
       centred on the site midpoint (local merge test; a sheet pair joined only far away still counts as separated).
  hit: both band centres have a foreground voxel of P within SNAP = 2 vox (SurfaceDice tolerance); the cc test then
       uses those nearest foreground voxels. Otherwise the site is a miss (counts as not separated in sep_cc).
Primary: separation rate = mean(cc) over compressed sites, over all sites (misses count as not separated) and over
hit sites. Writes <out>.csv (per site) and <out>.json (summary per class with a patch-bootstrap 95% CI).
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from contact_check import COMPRESSED_MAX, WIDE_MAX, WIDE_MIN, find_sites  # noqa: E402

GT_DIR = REPO / "data" / "eval" / "jeong200" / "labelsTr"
WIN = 16  # half window for the local CC test
MAX_SITES = 400
SNAP = 2.0
_off = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
OFFSETS = _off[np.linalg.norm(_off, axis=1) <= SNAP]


def snap(pred: np.ndarray, p: np.ndarray):
    """Nearest foreground voxel of pred within SNAP of point p (zyx float), or None."""
    c = np.round(p).astype(int) + OFFSETS
    c = c[((c >= 0) & (c < np.array(pred.shape))).all(1)]
    fg = c[pred[tuple(c.T)]]
    if not len(fg):
        return None
    return fg[np.argmin(np.linalg.norm(fg - p, axis=1))]


def site_scores(pred: np.ndarray, centre: np.ndarray, u: np.ndarray, d: float) -> dict:
    a, b = centre, centre + u * d
    ia, ib = np.round(a).astype(int), np.round(b).astype(int)
    n = np.array(pred.shape)
    if (ia < 0).any() or (ib < 0).any() or (ia >= n).any() or (ib >= n).any():
        return {}
    sa, sb = snap(pred, a), snap(pred, b)
    hit = sa is not None and sb is not None
    t = np.arange(0.25, d - 0.25 + 1e-9, 0.25)
    between = map_coordinates(pred.astype(np.uint8), (a[:, None] + u[:, None] * t), order=0, mode="nearest")
    gap = bool((between == 0).any())
    cc = False
    if hit:
        m = np.round((a + b) / 2).astype(int)
        lo = np.maximum(m - WIN, 0)
        hi = np.minimum(m + WIN + 1, n)
        lab = cc3d.connected_components(np.ascontiguousarray(pred[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]), connectivity=26)
        la, lb = lab[tuple(sa - lo)], lab[tuple(sb - lo)]
        cc = bool(la != lb)
    return {"hit": hit, "gap": gap, "cc": cc}


def score_patch(args) -> list[dict]:
    stem, gt_dir, pred_dir = args
    gt = tifffile.imread(Path(gt_dir) / f"{stem}.tif") == 1
    pred = tifffile.imread(Path(pred_dir) / f"{stem}.tif") == 1
    rng = np.random.default_rng(0)
    sites = find_sites(gt, rng)
    comp = [x for x in sites if x[2] < COMPRESSED_MAX]
    wide = [x for x in sites if WIDE_MIN <= x[2] < WIDE_MAX]
    comp = [comp[i] for i in rng.permutation(len(comp))[:MAX_SITES]]
    wide = [wide[i] for i in rng.permutation(len(wide))[:MAX_SITES]]
    rows = []
    for cls, items in (("compressed", comp), ("wide", wide)):
        for centre, u, d in items:
            sc = site_scores(pred, centre, u, d)
            if sc:
                rows.append({"stem": stem, "cls": cls, "d": round(d, 2), **sc})
    return rows


def summarize(rows: list[dict]) -> dict:
    stems = sorted({r["stem"] for r in rows})
    out = {"patches": len(stems)}
    boot = np.random.default_rng(1)
    for cls in ("compressed", "wide"):
        sel = [r for r in rows if r["cls"] == cls]
        if not sel:
            continue
        by = {s: [r for r in sel if r["stem"] == s] for s in stems}

        def rate(sample, key, hit_only=False):
            v = [r[key] for s in sample for r in by.get(s, []) if (r["hit"] or not hit_only)]
            return 100.0 * np.mean(v) if v else float("nan")

        bs = [rate(list(boot.choice(stems, len(stems))), "cc") for _ in range(1000)]
        out[cls] = {"n": len(sel), "sep_cc": rate(stems, "cc"), "sep_cc_ci95": [float(np.nanpercentile(bs, 2.5)),
                                                                                  float(np.nanpercentile(bs, 97.5))],
                    "sep_cc_given_hit": rate(stems, "cc", True), "gap": rate(stems, "gap"), "hit": rate(stems, "hit")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    stems = sorted(p.stem for p in Path(args.pred_dir).glob("*.tif") if (Path(args.gt_dir) / p.name).exists())
    if args.limit:
        stems = stems[: args.limit]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(score_patch, [(s, args.gt_dir, args.pred_dir) for s in stems]) for r in rs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "cls", "d", "hit", "gap", "cc"])
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
