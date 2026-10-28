"""Ridge-crossing winding on PHerc0139: signed count of probability ridges on paths A -> B.

  uv run --with pandas python eval/p0139_ridges.py --prob data/preds/prob_real350 --out data/preds/ridge_real350

Binary masks merge touching bands (a 3-4.5 vox label band cannot separate sheets < ~4 vox apart); the probability
often still dips between them. Along each path (trilinear samples every 0.25 vox) a hysteresis peak detector counts
an entry when the profile rises >= prom above its running minimum and reaches >= hi; a peak ends when the profile
falls >= prom below its running maximum. The start is in a peak if f(A) >= hi (A's own band). Entries with arc < skip
are ignored; each entry is signed by sign(u . n) (normals of the prob > 0.5 mask, oriented per crop as in
p0139_crossings.count_signed), |u . n| < graze dropped; delta = |sum|, median over the straight path and 6 two-leg
paths (sigma 1.5). Grid tuned on the even-index sites (sorted by stem), reported on the odd ones, as
eval/p0139_xing_sweep.py.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_crossings import mask_normals, orient_normals, sample_paths  # noqa: E402
from p0139_sites import ROOT, load_sites, read_pairs  # noqa: E402
from winding_sites import p0139_summary  # noqa: E402


def ridge_count(f: np.ndarray, cosn: np.ndarray, arc: np.ndarray, skip: float, hi: float, prom: float,
                graze: float) -> np.ndarray:
    """f, cosn, arc: (n, T) profile, u . n at each sample, arclength. Returns |sum of signed entries|."""
    n, T = f.shape
    inpk = f[:, 0] >= hi
    rmax, rmin = f[:, 0].copy(), f[:, 0].copy()
    tot = np.zeros(n)
    for k in range(1, T):
        x = f[:, k]
        rmax = np.where(inpk, np.maximum(rmax, x), rmax)
        drop = inpk & (x <= rmax - prom)
        rmin = np.where(~inpk, np.minimum(rmin, x), rmin)
        rise = ~inpk & (x >= hi) & (x >= rmin + prom)
        rmin = np.where(drop, x, rmin)
        rmax = np.where(rise, x, rmax)
        inpk = (inpk & ~drop) | rise
        c = cosn[:, k]
        tot += np.where(rise & (arc[:, k] >= skip) & (np.abs(c) >= graze), np.sign(c), 0.0)
    return np.abs(tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skips", default="1,2,3")
    ap.add_argument("--his", default="0.3,0.5")
    ap.add_argument("--proms", default="0.1,0.2,0.3")
    ap.add_argument("--grazes", default="0.15")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    fl = lambda x: [float(v) for v in x.split(",")]
    grid = list(itertools.product(fl(args.skips), fl(args.his), fl(args.proms), fl(args.grazes)))
    tags = [f"skip{s}_hi{h}_prom{p}_g{g}" for s, h, p, g in grid]
    sites = sorted(load_sites(ROOT), key=lambda s: s["stem"])[: args.limit or None]
    half = {s["stem"]: k % 2 for k, s in enumerate(sites)}
    rows = [r for r in read_pairs(ROOT) if r["kind"] == "mix" and r["stem"] in half]
    by = {}
    for r in rows:
        r["delta_oracle_pitch"] = r["proj"] / r["pitch_site"]
        by.setdefault(r["stem"], []).append(r)
    for s in sites:
        rr = by.get(s["stem"], [])
        if not rr:
            continue
        prob = np.load(args.prob / f"{s['stem']}.npy").astype(np.float32)
        m = ndi.binary_dilation(prob > 0.5, iterations=1)
        nrm = orient_normals(m, mask_normals(m))
        A = np.array([[r["za"], r["ya"], r["xa"]] for r in rr], float)
        B = np.array([[r["zb"], r["yb"], r["xb"]] for r in rr], float)
        rng = np.random.default_rng(0)
        Ws = [None] + [(A + B) / 2 + rng.normal(0, 1.5, A.shape) for _ in range(6)]
        res = {t: [] for t in tags}
        for W in Ws:
            pts, arc, valid = sample_paths(A, B, W, 3.0)
            f = ndi.map_coordinates(prob, pts.reshape(-1, 3).T, order=1, mode="constant").reshape(arc.shape)
            f = np.where(valid, f, 0.0)
            u = np.gradient(pts, axis=1)
            u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-6)
            p = np.clip(np.rint(pts).astype(int), 0, np.array(prob.shape) - 1)
            cosn = (u * np.moveaxis(nrm[(slice(None),) + tuple(p.reshape(-1, 3).T)], 0, -1).reshape(u.shape)).sum(-1)
            for t, (sk, hi, pr, g) in zip(tags, grid):
                res[t].append(ridge_count(f, cosn, arc, sk, hi, pr, g))
        for t in tags:
            for r, d in zip(rr, np.median(np.stack(res[t]), 0)):
                r[f"delta_{t}"] = float(d)
        print(s["stem"], flush=True)
    tune = [r for r in rows if half[r["stem"]] == 0]
    acc = {t: 100 * np.mean([abs(r[f"delta_{t}"] - r["expected"]) < 0.25 for r in tune]) for t in tags}
    best = max(acc, key=acc.get)
    rep = [r for r in rows if half[r["stem"]] == 1]
    summ = {"prob": str(args.prob), "tune": {t: round(a, 2) for t, a in sorted(acc.items(), key=lambda x: -x[1])},
            "best": best, "report_half": p0139_summary(rep, [best, "oracle_pitch"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(list(summ["tune"].items())[:5])
    for m in (best, "oracle_pitch"):
        v = summ["report_half"][m]["primary_abs_tol_0.25"]
        print(m, v["pooled"], v["by_step_group"], v["by_compression"], summ["report_half"][m].get("vs_oracle_pitch"))


if __name__ == "__main__":
    main()
