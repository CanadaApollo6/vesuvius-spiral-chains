"""Tune the crossing estimator (eval/p0139_crossings.py) on half the PHerc0139 sites, report on the other half.

  uv run --with pandas python eval/p0139_xing_sweep.py --seg data/preds/xing_real350_seg --out data/preds/xing_sweep_real350

Split: sites sorted by stem, even index = tune, odd index = report (fixed, so no peeking). Grid over skip, min_gap,
dilate, and multi-path voting (median crossing count over the straight segment plus jittered two-leg paths A -> W -> B,
W = midpoint + N(0, sigma) per axis; small sigma only, the count is unsigned).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_crossings import count_signed, count_thick, crossings, mask_normals, orient_normals, sample_paths  # noqa: E402
from p0139_sites import ROOT, load_sites, read_pairs  # noqa: E402
from winding_sites import p0139_summary  # noqa: E402


def path_count(mask, A, B, skip, min_gap, paths, sigma, rng) -> np.ndarray:
    """Median crossing count over the straight segment and paths-1 two-leg paths through jittered midpoints."""
    c = [crossings(mask, A, B, skip, min_gap)]
    for _ in range(paths - 1):
        W = (A + B) / 2 + rng.normal(0, sigma, A.shape)
        c.append(crossings(mask, A, B, skip, min_gap, W=W))
    return np.median(np.stack(c), 0)


def thick_sweep(args, sites, half, rows, by, masks, signed: bool = False) -> None:
    """Contact-aware estimator (count_thick) at the frozen path settings (skip 3, gap 2.5, dilate 1, 7 paths s1.5),
    T tuned on the tune half; report half scored against the frozen plain count and oracle pitch."""
    Ts = [float(v) for v in (args.signed if signed else args.thick).split(",")]
    kind = "signed" if signed else "thick"
    tags = ["plain"] + [f"{kind}{T}" for T in Ts]
    for stem, m in masks.items():
        rr = by.get(stem, [])
        if not rr:
            continue
        mm = ndi.binary_dilation(m, iterations=1)
        nrm = mask_normals(mm)
        if signed:
            nrm = orient_normals(mm, nrm)
        A = np.array([[r["za"], r["ya"], r["xa"]] for r in rr], float)
        B = np.array([[r["zb"], r["yb"], r["xb"]] for r in rr], float)
        rng = np.random.default_rng(0)
        Ws = [None] + [(A + B) / 2 + rng.normal(0, 1.5, A.shape) for _ in range(6)]
        res = {t: [] for t in tags}
        for W in Ws:
            res["plain"].append(crossings(mm, A, B, 3.0, 2.5, W=W))
            pts, arc, valid = sample_paths(A, B, W, 3.0)
            for T in Ts:
                f = count_signed if signed else count_thick
                res[f"{kind}{T}"].append(f(mm, nrm, pts, arc, valid, 3.0, 2.5, 0.25, T))
        for t in tags:
            for r, d in zip(rr, np.median(np.stack(res[t]), 0)):
                r[f"delta_{t}"] = float(d)
        print(stem, flush=True)
    tune = [r for r in rows if half[r["stem"]] == 0]
    acc = {t: 100 * np.mean([abs(r[f"delta_{t}"] - r["expected"]) < 0.25 for r in tune]) for t in tags}
    best = max(acc, key=acc.get)
    rep = [r for r in rows if half[r["stem"]] == 1]
    summ = {"seg": str(args.seg), "tune": {t: round(a, 2) for t, a in acc.items()}, "best": best,
            "report_half": p0139_summary(rep, list(dict.fromkeys([best, "plain", "oracle_pitch"])))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(summ["tune"])
    for m in dict.fromkeys([best, "plain", "oracle_pitch"]):
        v = summ["report_half"][m]["primary_abs_tol_0.25"]
        print(m, v["pooled"], v["by_step_group"], v["by_compression"], summ["report_half"][m].get("vs_oracle_pitch"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skips", default="1,2,3")
    ap.add_argument("--gaps", default="0.75,1.5,2.5")
    ap.add_argument("--dils", default="0,1")
    ap.add_argument("--signed", default="", help="comma list of graze thresholds for the signed count; empty = off")
    ap.add_argument("--thick", default="", help="comma list of T (vox) for the contact-aware count; empty = off")
    args = ap.parse_args()
    fl = lambda x: [float(v) for v in x.split(",")]
    sites = sorted(load_sites(ROOT), key=lambda s: s["stem"])
    half = {s["stem"]: k % 2 for k, s in enumerate(sites)}
    rows = [r for r in read_pairs(ROOT) if r["kind"] == "mix"]
    for r in rows:
        r["delta_oracle_pitch"] = r["proj"] / r["pitch_site"]
    by = {}
    for r in rows:
        by.setdefault(r["stem"], []).append(r)
    masks = {s["stem"]: tifffile.imread(args.seg / f"{s['stem']}.tif") > 0 for s in sites}
    if args.signed:
        return thick_sweep(args, sites, half, rows, by, masks, signed=True)
    if args.thick:
        return thick_sweep(args, sites, half, rows, by, masks)
    grid = list(itertools.product(fl(args.skips), fl(args.gaps), [int(v) for v in fl(args.dils)], ((1, 0.0), (7, 1.5), (7, 3.0))))
    res = []
    for skip, gap, dil, (paths, sig) in grid:
        tag = f"skip{skip}_gap{gap}_dil{dil}_p{paths}s{sig}"
        rng = np.random.default_rng(0)
        for stem, m in masks.items():
            mm = ndi.binary_dilation(m, iterations=dil) if dil else m
            rr = by.get(stem, [])
            if not rr:
                continue
            A = np.array([[r["za"], r["ya"], r["xa"]] for r in rr], float)
            B = np.array([[r["zb"], r["yb"], r["xb"]] for r in rr], float)
            for r, d in zip(rr, path_count(mm, A, B, skip, gap, paths, sig, rng)):
                r[f"delta_{tag}"] = float(d)
        tune = [r for r in rows if half[r["stem"]] == 0]
        acc = np.mean([abs(r[f"delta_{tag}"] - r["expected"]) < 0.25 for r in tune]) * 100
        res.append((acc, tag))
        print(f"{tag}: tune {acc:.2f}", flush=True)
    res.sort(reverse=True)
    best = res[0][1]
    rep = [r for r in rows if half[r["stem"]] == 1]
    default = "skip2.0_gap1.5_dil0_p1s0.0"
    if f"delta_{default}" not in rows[0]:
        default = res[-1][1]  # worst of the grid, for contrast
    summ = {"seg": str(args.seg), "split": "sites sorted by stem; even = tune, odd = report",
            "tune_top5": [{"tag": t, "tune": round(a, 2)} for a, t in res[:5]], "best": best,
            "report_half": p0139_summary(rep, list(dict.fromkeys([best, default, "oracle_pitch"])))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    for m in dict.fromkeys([best, default, "oracle_pitch"]):
        v = summ["report_half"][m]
        print(m, v["primary_abs_tol_0.25"]["pooled"], v["primary_abs_tol_0.25"]["by_step_group"], v.get("vs_oracle_pitch"))


if __name__ == "__main__":
    main()
