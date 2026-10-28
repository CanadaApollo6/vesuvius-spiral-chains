"""Global winding assignment on a surface-patch graph, scored on the PHerc0139 pairs.

  uv run --with pandas python eval/p0139_graph.py --seg data/preds/xing_real350_seg --out data/preds/graph_real350

Local crossing counts miss a sheet where two sheets touch (the mask shows one band). A sheet missed at one spot is
usually present a few voxels away, so a winding number per surface patch solved over many overlapping paths can
recover it. Per crop:
  1. Patches: mask (dilated 1, as the frozen estimator) cut into blocks of --block vox; 26-connected components per
     block are patches (< --min-vox dropped). Representative point = patch voxel nearest its centroid.
  2. Edges: each patch to its --k nearest patches within --radius vox; d_ij = signed crossing count from rep_i to
     rep_j (p0139_crossings.count_signed, absolute=False; + along the crop-oriented normal), median over the
     straight path and 2 jittered two-leg paths. Weight 1 / (1 + length / 10).
  3. Solve w (one per patch) minimising sum w_ij |w_j - w_i - d_ij| by IRLS (L1), gauge fixed by a weak prior per
     graph component; then round.
  4. Pair (A, B): nearest patch of A and of B (<= 3 vox, else fall back to the direct signed count);
     delta = |round(w_B - w_A)|.
Grid tuned on the even-index sites (sorted by stem), reported on the odd ones; the direct signed count (frozen
estimator) is scored alongside.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cc3d
import numpy as np
import scipy.sparse as sp
import tifffile
from scipy import ndimage as ndi
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_crossings import count_signed, mask_normals, orient_normals, sample_paths  # noqa: E402
from p0139_sites import ROOT, load_sites, read_pairs  # noqa: E402
from winding_sites import p0139_summary  # noqa: E402

SKIP, GAP, STEP, GRAZE = 3.0, 2.5, 0.25, 0.2  # frozen signed estimator


def split_thick(L: np.ndarray, nrm: np.ndarray, thick: float, min_vox: int, only: np.ndarray | None = None) -> np.ndarray:
    """Split patches whose extent along their mean (oriented) normal (p95 - p5) exceeds `thick` vox into two layers
    at the median projection: two touching sheets in one block form one component, and this separates them.
    `only` (patch labels): split just these patches."""
    idx = np.argwhere(L > 0)
    lab = L[tuple(idx.T)]
    order = np.argsort(lab, kind="stable")
    idx, lab = idx[order], lab[order]
    nv = nrm[(slice(None),) + tuple(idx.T)].T
    bounds = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
    nxt = int(L.max())
    L = L.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2 * min_vox or (only is not None and lab[a] not in only):
            continue
        n = nv[a:b].mean(0)
        n /= max(np.linalg.norm(n), 1e-6)
        pr = idx[a:b] @ n
        lo, hi = np.percentile(pr, (5, 95))
        if hi - lo <= thick:
            continue
        far = pr > np.median(pr)
        if far.sum() < min_vox or (~far).sum() < min_vox:
            continue
        nxt += 1
        L[tuple(idx[a:b][far].T)] = nxt
    return L


def patches(mask: np.ndarray, block: int, min_vox: int, nrm: np.ndarray | None = None,
            thick: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Patch label volume (0 = none) and representative points (n, 3), patch k -> row k - 1. thick > 0 splits
    patches thicker than that along the normal (split_thick)."""
    L = np.zeros(mask.shape, np.int32)
    nxt = 0
    Z, Y, X = mask.shape
    for z in range(0, Z, block):
        for y in range(0, Y, block):
            for x in range(0, X, block):
                sl = (slice(z, z + block), slice(y, y + block), slice(x, x + block))
                m = mask[sl]
                if not m.any():
                    continue
                c = cc3d.connected_components(np.ascontiguousarray(m), connectivity=26)
                cnt = np.bincount(c.ravel())
                keep = np.flatnonzero(cnt >= min_vox)
                keep = keep[keep > 0]
                if len(keep) == 0:
                    continue
                remap = np.zeros(len(cnt), np.int32)
                remap[keep] = np.arange(nxt + 1, nxt + 1 + len(keep))
                L[sl] = np.where(c > 0, remap[c], L[sl])
                nxt += len(keep)
    if thick > 0:
        L = split_thick(L, nrm, thick, min_vox)
    return L, reps(L)


def reps(L: np.ndarray) -> np.ndarray:
    """Representative point per patch (voxel nearest its centroid), patch k -> row k - 1."""
    idx = np.argwhere(L > 0)
    lab = L[tuple(idx.T)]
    n = int(L.max())
    cnt = np.bincount(lab, minlength=n + 1)[1:].astype(float)
    cen = np.stack([np.bincount(lab, idx[:, a], minlength=n + 1)[1:] for a in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    d = np.linalg.norm(idx - cen[lab - 1], axis=1)
    order = np.lexsort((d, lab))
    first = np.r_[True, lab[order][1:] != lab[order][:-1]]
    rep = np.zeros((n, 3))
    rep[lab[order][first] - 1] = idx[order][first]
    return rep


def signed_paths(mask, nrm, A, B, rng, n_jit: int, sigma: float = 1.5) -> np.ndarray:
    c = []
    for W in [None] + [(A + B) / 2 + rng.normal(0, sigma, A.shape) for _ in range(n_jit)]:
        pts, arc, valid = sample_paths(A, B, W, SKIP)
        c.append(count_signed(mask, nrm, pts, arc, valid, SKIP, GAP, STEP, GRAZE, absolute=False))
    return np.median(np.stack(c), 0)


def solve(n: int, i: np.ndarray, j: np.ndarray, d: np.ndarray, wt: np.ndarray, iters: int = 15) -> np.ndarray:
    """IRLS for min sum wt |w_j - w_i - d| with a weak 0-prior per node (gauge)."""
    m = len(d)
    rows = np.r_[np.arange(m), np.arange(m), m + np.arange(n)]
    cols = np.r_[j, i, np.arange(n)]
    w = np.zeros(n)
    r_w = wt.copy()
    for _ in range(iters):
        s = np.sqrt(r_w)
        vals = np.r_[s, -s, np.full(n, 1e-3)]
        M = sp.csr_matrix((vals, (rows, cols)), shape=(m + n, n))
        rhs = np.r_[s * d, np.zeros(n)]
        w = lsqr(M, rhs, x0=w, atol=1e-6, btol=1e-6, iter_lim=2000)[0]
        res = np.abs(w[j] - w[i] - d)
        r_w = wt / np.maximum(res, 0.05)
    return w


def site_graph(mask, nrm, block, min_vox, k, radius, rng, thick: float = 0.0, edges: bool = False,
               rsplit: float = 0.0):
    """rsplit > 0: solve once, split every patch whose mean edge residual exceeds rsplit into two layers along its
    normal (a fused contact gives one patch two windings, so its edges disagree), then solve again."""
    L, rep = patches(mask, block, min_vox, nrm, thick)
    if rsplit > 0:
        _, w0, _, rnode, _ = graph_solve(mask, nrm, rep, k, radius, np.random.default_rng(rng.integers(1 << 31)))
        L = split_thick(L, nrm, 0.0, min_vox, only=set((np.flatnonzero(rnode > rsplit) + 1).tolist()))
        rep = reps(L)
    i, w, ne, rnode, deg = graph_solve(mask, nrm, rep, k, radius, rng)
    if edges:
        return L, w, ne, rnode, deg
    return L, w, ne


def graph_solve(mask, nrm, rep, k, radius, rng):
    n = len(rep)
    tree = cKDTree(rep)
    dist, nb = tree.query(rep, k=k + 1, distance_upper_bound=radius)
    ii = np.repeat(np.arange(n), k + 1)
    jj = nb.ravel()
    ok = (jj < n) & (jj > ii)
    i, j = ii[ok], jj[ok]
    d = signed_paths(mask, nrm, rep[i], rep[j], rng, n_jit=2)
    ln = np.linalg.norm(rep[j] - rep[i], axis=1)
    w = solve(n, i, j, d, 1.0 / (1.0 + ln / 10.0))
    # per-node mean L1 residual |w_j - w_i - d| over incident edges (certifier signal)
    res = np.abs(w[j] - w[i] - d)
    deg = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    rnode = (np.bincount(i, res, n) + np.bincount(j, res, n)) / np.maximum(deg, 1)
    return i, w, len(i), rnode, deg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", default="12,16")
    ap.add_argument("--radii", default="20,32")
    ap.add_argument("--ks", default="30")
    ap.add_argument("--thicks", default="0", help="split_thick thresholds (vox); 0 = no split")
    ap.add_argument("--rsplit", type=float, default=0.0, help="residual-driven split threshold; 0 = off")
    ap.add_argument("--min-vox", type=int, default=20)
    ap.add_argument("--dilate", type=int, default=1, help="mask dilation before patches (frozen: 1)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    grid = list(itertools.product([int(v) for v in args.blocks.split(",")], [float(v) for v in args.radii.split(",")],
                                  [int(v) for v in args.ks.split(",")], [float(v) for v in args.thicks.split(",")]))
    base = [f"graph_b{b}_r{r:g}_k{k}_t{t:g}" for b, r, k, t in grid]
    tags = base + [f"{t}_sp" for t in base]  # _sp: same patch for A and B -> direct count
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
        t0 = time.time()
        mask = tifffile.imread(args.seg / f"{s['stem']}.tif") > 0
        if args.dilate:
            mask = ndi.binary_dilation(mask, iterations=args.dilate)
        nrm = orient_normals(mask, mask_normals(mask))
        A = np.array([[r["za"], r["ya"], r["xa"]] for r in rr], float)
        B = np.array([[r["zb"], r["yb"], r["xb"]] for r in rr], float)
        rng = np.random.default_rng(0)
        direct = signed_paths(mask, nrm, A, B, rng, n_jit=6)
        for r, dd in zip(rr, direct):
            r["delta_direct"] = float(abs(dd))
        info = []
        for tag, (b, rad, kk, th) in zip(base, grid):
            L, w, ne, rnode, deg = site_graph(mask, nrm, b, args.min_vox, kk, rad, np.random.default_rng(1), th, edges=True,
                                             rsplit=args.rsplit)
            dist, ind = ndi.distance_transform_edt(L == 0, return_indices=True)
            ai, bi = [tuple(np.rint(P).astype(int).T) for P in (A, B)]
            pa = L[tuple(ind[(slice(None),) + ai])]
            pb = L[tuple(ind[(slice(None),) + bi])]
            okp = (dist[ai] <= 3) & (dist[bi] <= 3) & (pa > 0) & (pb > 0)
            g = np.abs(np.rint(w[np.maximum(pb, 1) - 1] - w[np.maximum(pa, 1) - 1]))
            ia, ib = np.maximum(pa, 1) - 1, np.maximum(pb, 1) - 1
            fr = lambda v: np.abs(v - np.rint(v))
            for q, (r, ok_, gg, dd, same) in enumerate(zip(rr, okp, g, direct, pa == pb)):
                if tag == base[0]:  # certifier features for the first grid point
                    r["c_ok"], r["c_same"] = int(ok_), int(same)
                    r["c_res"] = float(max(rnode[ia[q]], rnode[ib[q]]))
                    r["c_frac"] = float(max(fr(w[ia[q]]), fr(w[ib[q]])))
                    r["c_wdiff_frac"] = float(fr(w[ib[q]] - w[ia[q]]))
                    r["c_deg"] = int(min(deg[ia[q]], deg[ib[q]]))
                    r["c_disagree"] = float(abs(float(gg) - abs(dd)))
                r[f"delta_{tag}"] = float(gg) if ok_ else float(abs(dd))
                r[f"delta_{tag}_sp"] = float(abs(dd)) if (same or not ok_) else float(gg)
            info.append(f"{tag}: {int(L.max())} patches {ne} edges")
        print(f"{s['stem']} {time.time() - t0:.0f}s " + "; ".join(info), flush=True)
    methods = tags + ["direct"]
    tune = [r for r in rows if half[r["stem"]] == 0]
    acc = {m: 100 * np.mean([abs(r[f"delta_{m}"] - r["expected"]) < 0.25 for r in tune]) for m in methods}
    best = max(tags, key=acc.get)
    rep_rows = [r for r in rows if half[r["stem"]] == 1]
    summ = {"seg": str(args.seg), "tune": {m: round(a, 2) for m, a in acc.items()}, "best": best,
            "report_half": p0139_summary(rep_rows, [best, "direct", "oracle_pitch"])}
    # paired site bootstrap, best graph vs direct, report half
    stems = sorted({r["stem"] for r in rep_rows})
    si = np.array([stems.index(r["stem"]) for r in rep_rows])
    cg = np.array([abs(r[f"delta_{best}"] - r["expected"]) < 0.25 for r in rep_rows], float)
    cd = np.array([abs(r["delta_direct"] - r["expected"]) < 0.25 for r in rep_rows], float)
    bt = np.random.default_rng(1).integers(0, len(stems), (1000, len(stems)))
    num = lambda c: np.bincount(si, c, len(stems))[bt].sum(1) / np.bincount(si, None, len(stems))[bt].sum(1)
    dd = 100 * (num(cg) - num(cd))
    summ["best_vs_direct"] = {"diff_pp": round(100 * (cg.mean() - cd.mean()), 2),
                              "diff_ci95": [round(float(np.percentile(dd, q)), 2) for q in (2.5, 97.5)]}
    import csv
    with args.out.with_name(args.out.name + ".csv").open("w", newline="") as f:
        cols = ["stem", "site_cls", "a", "b", "expected", "dist", "contact", "delta_direct", "delta_oracle_pitch"] + \
            [f"delta_{t}" for t in tags] + ["c_ok", "c_same", "c_res", "c_frac", "c_wdiff_frac", "c_deg", "c_disagree"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_name(args.out.name + ".json").write_text(json.dumps(summ, indent=1))
    print(summ["tune"])
    print("best vs direct (report half):", summ["best_vs_direct"])
    for m in (best, "direct", "oracle_pitch"):
        v = summ["report_half"][m]["primary_abs_tol_0.25"]
        print(m, v["pooled"], v["by_step_group"], v["by_compression"], summ["report_half"][m].get("vs_oracle_pitch"))


if __name__ == "__main__":
    main()
