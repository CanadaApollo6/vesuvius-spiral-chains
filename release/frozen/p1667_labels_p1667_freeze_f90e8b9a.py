"""Winding labels for PHerc1667 from the public merged human mesh (the surface of the full read, 31 wraps w011-w041).

  uv run python eval/p1667_labels.py label     # -> data/eval/p1667/labels_merged_v4.npz (+ .json summary)
  uv run python eval/p1667_labels.py check     # agreement with the 19 per-winding human segments (data/meshes)

Source: PHerc1667/segments/20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4/mesh/
20260612121456-on-20251217075048-2.399um.tifxyz (downloaded to data/dl_p1667/merged_v4_2p399/, untrusted bytes, read
as TIFF arrays only). One continuous tifxyz from the innermost wrap outward (columns run along the spiral, rows ~z).

Frame: the fit frame is level 2 of the 2.399 um volume 20251217075048 (9.596 um, shape 9269 x 3808 x 3808 zyx), so
p_L2 = p_2399 / 4 (OME scale 4, no affine). Everything written here is in L2 voxels.

Labelling (no CT, no fit output):
  1. centre(z) for unwrapping only: per 100-z slab, the centroid of the innermost labelled segment w011 (any point
     inside the innermost loop gives the same unwrapped turn count). The fit/scorer umbilicus is made separately
     (p1667_umbilicus in this file, written by `label`).
  2. theta per vertex around centre(z); unwrapped along each row inside contiguous valid runs (adjacent columns are
     20 2.399-vox = 5 L2 vox apart, i.e. < 6 deg even on the innermost wrap).
  3. runs are tied together through vertical (row r / r+1) neighbours: the integer turn offset between two runs is the
     mode over their shared columns; offsets are propagated breadth-first from the longest run, and conflicting
     links are counted (a topological defect would show up there).
  4. k = K0 + s floor((Phi - S) / 360) (s = +-1), with the sign s, seam S and base K0 chosen to maximise agreement with the 16
     single-wrap human segments (w011, w012, w028-w041; their names are the human winding labels). Multi-wrap
     segments w013 (13-17), w018 (18-22), w023 (23-27) are checked after the fact.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "dl_p1667" / "merged_v4_2p399"
SEGS = REPO / "data" / "meshes"
SEG_TAG = "20251217075048-2.399um"
OUT = REPO / "data" / "eval" / "p1667"
LABELS = OUT / "labels_merged_v4.npz"
UMB = OUT / "umbilicus_p1667_L2.json"
SCALE = 4.0               # 2.399 L0 -> L2
VOL_L2 = (9269, 3808, 3808)


def load_grid(d: Path) -> np.ndarray:
    """(H, W, 3) xyz in L2 voxels, invalid = nan."""
    g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz"], -1)
    ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
    g /= SCALE
    g[~ok] = np.nan
    return g


def segments() -> dict:
    out = {}
    for d in sorted(SEGS.glob("*-w0*_flatboi")):
        w = int(re.search(r"-w(\d+)_", d.name).group(1))
        out[w] = d / SEG_TAG
    return out


def centre_from(g11: np.ndarray, step: float = 100.0) -> np.ndarray:
    p = g11[np.isfinite(g11).all(-1)]
    zb = np.arange(0, VOL_L2[0] + step, step)
    rows = []
    for a in zb:
        s = (p[:, 2] >= a) & (p[:, 2] < a + step)
        if s.sum() > 100:
            rows.append((a + step / 2, p[s, 0].mean(), p[s, 1].mean()))
    return np.array(rows)


def theta(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(p[..., 1] - np.interp(p[..., 2], c[:, 0], c[:, 2]),
                                 p[..., 0] - np.interp(p[..., 2], c[:, 0], c[:, 1])))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def unwrap_grid(g: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, dict]:
    H, W = g.shape[:2]
    ok = np.isfinite(g).all(-1)
    th = np.full((H, W), np.nan, np.float64)
    th[ok] = theta(g[ok].astype(np.float64), c)
    phi = np.full((H, W), np.nan, np.float64)
    run_id = np.full((H, W), -1, np.int64)
    runs = []                                    # (row, c0, c1)
    for r in range(H):
        v = ok[r]
        if not v.any():
            continue
        e = np.diff(np.r_[0, v.astype(np.int8), 0])
        starts, ends = np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]
        for a, b in zip(starts, ends):
            t = th[r, a:b]
            phi[r, a:b] = t[0] + np.r_[0.0, np.cumsum(wrap180(np.diff(t)))]
            run_id[r, a:b] = len(runs)
            runs.append((r, a, b))
    n = len(runs)
    # vertical links
    links = {}
    for r in range(H - 1):
        both = ok[r] & ok[r + 1]
        if not both.any():
            continue
        cols = np.nonzero(both)[0]
        ra, rb = run_id[r, cols], run_id[r + 1, cols]
        d = (phi[r, cols] - phi[r + 1, cols]) - wrap180(th[r, cols] - th[r + 1, cols])
        nt = np.rint(d / 360.0).astype(np.int64)
        key = ra * n + rb
        for kk in np.unique(key):
            m = key == kk
            vals, cnt = np.unique(nt[m], return_counts=True)
            links[int(kk)] = (int(vals[np.argmax(cnt)]), int(cnt.max()), int(m.sum()))
    adj = [[] for _ in range(n)]
    for kk, (nt, cmax, tot) in links.items():
        a, b = divmod(kk, n)
        adj[a].append((b, nt, cmax))      # off_b = off_a + nt
        adj[b].append((a, -nt, cmax))
    off = np.full(n, np.iinfo(np.int64).min, np.int64)
    lens = np.array([b - a for _, a, b in runs])
    comp = np.full(n, -1)
    ncomp = 0
    for seed in np.argsort(-lens):
        if off[seed] != np.iinfo(np.int64).min:
            continue
        off[seed] = 0
        comp[seed] = ncomp
        q = [seed]
        while q:
            a = q.pop()
            # strongest links first keeps weak (few shared columns) links from deciding
            for b, nt, _ in sorted(adj[a], key=lambda x: -x[2]):
                if off[b] == np.iinfo(np.int64).min:
                    off[b] = off[a] + nt
                    comp[b] = ncomp
                    q.append(b)
        ncomp += 1
    conflicts, conflict_cols, total_cols = 0, 0, 0
    for kk, (nt, cmax, tot) in links.items():
        a, b = divmod(kk, n)
        total_cols += tot
        if off[b] != off[a] + nt:
            conflicts += 1
            conflict_cols += tot
        conflict_cols += tot - cmax       # within-link disagreement
    offg = np.zeros((H, W))
    m = run_id >= 0
    offg[m] = off[run_id[m]]
    phi = phi + 360.0 * offg
    compg = np.full((H, W), -1)
    compg[m] = comp[run_id[m]]
    stats = {"runs": n, "links": len(links), "components": int(ncomp),
             "largest_component_vertex_frac": float(np.bincount(compg[m]).max() / m.sum()),
             "conflicting_links": conflicts, "conflict_or_disagree_cols_frac": conflict_cols / max(total_cols, 1)}
    return phi, compg, stats


def seg_phi(segs: dict, g: np.ndarray, phi: np.ndarray, comp: np.ndarray, main: int, stride: int = 4):
    """For each human segment, nearest merged vertex (stride 2) of the main component -> its phi; and distance."""
    from scipy.spatial import cKDTree
    sub = (slice(None, None, 2), slice(None, None, 2))
    gm, pm, cm = g[sub], phi[sub], comp[sub]
    ok = np.isfinite(gm).all(-1) & (cm == main)
    tree = cKDTree(gm[ok])
    pv = pm[ok]
    out = {}
    for w, d in segs.items():
        s = load_grid(d)[::stride, ::stride]
        p = s[np.isfinite(s).all(-1)]
        dist, ii = tree.query(p, distance_upper_bound=8.0, workers=-1)
        hit = np.isfinite(dist)
        out[w] = {"phi": pv[ii[hit]], "dist": dist[hit], "n": len(p), "z": p[hit, 2]}
    return out


def cmd_label(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    segs = segments()
    c = centre_from(load_grid(segs[11]))
    g = load_grid(SRC)
    print(f"merged grid {g.shape[:2]}, valid {np.isfinite(g).all(-1).sum()}", flush=True)
    phi, comp, st = unwrap_grid(g, c)
    print("unwrap:", json.dumps(st), flush=True)
    main = int(np.bincount(comp[comp >= 0]).argmax())
    sp = seg_phi(segs, g, phi, comp, main)
    single = [w for w in segs if w not in (13, 18, 23)]
    # seam S and base K0: maximise agreement on single-wrap segments
    best = None
    allphi = np.concatenate([sp[w]["phi"] for w in single])
    allw = np.concatenate([np.full(len(sp[w]["phi"]), w) for w in single])
    for sg in (1, -1):              # phi may run inward (the merged mesh's column direction is not fixed)
        for S in np.arange(0.0, 360.0, 2.0):
            kk = sg * np.floor((allphi - S) / 360.0).astype(np.int64)
            d = allw - kk
            vals, cnt = np.unique(d, return_counts=True)
            i = np.argmax(cnt)
            if best is None or cnt[i] > best[3]:
                best = (sg, S, int(vals[i]), int(cnt[i]))
    sg, S, K0, hit = best
    lab = lambda ph: sg * np.floor((ph - S) / 360.0) + K0
    print(f"sign {sg}, seam S = {S:.0f} deg (absolute phi), K0 = {K0}: single-wrap agreement "
          f"{hit / len(allphi) * 100:.2f}% of {len(allphi)} segment vertices", flush=True)
    k = np.where(np.isfinite(phi) & (comp == main), lab(phi), -1).astype(np.int16)
    per = {}
    for w in sorted(segs):
        kk = lab(sp[w]["phi"]).astype(int)
        vals, cnt = np.unique(kk, return_counts=True)
        per[int(w)] = {"n": int(sp[w]["n"]), "matched": int(len(kk)), "dist_p50": round(float(np.median(sp[w]["dist"])), 2),
                       "dist_p95": round(float(np.percentile(sp[w]["dist"], 95)), 2),
                       "k_hist": {int(v): int(n) for v, n in zip(vals, cnt)},
                       "agree": round(float((kk == w).mean() * 100), 2) if w in single else None}
        print(f"  w{w:03d}: matched {len(kk)}/{sp[w]['n']} dist p50 {per[w]['dist_p50']} p95 {per[w]['dist_p95']} "
              f"k {dict(list(per[w]['k_hist'].items())[:8])}", flush=True)
    np.savez_compressed(LABELS, k=k, centre=c, seam=S, k0=K0, sign=sg)
    summ = {"source": str(SRC.relative_to(REPO)), "grid": list(g.shape[:2]), "unwrap": st, "sign": sg, "seam_phi": S,
            "k0": K0,
            "k_range": [int(k[k >= 0].min()), int(k[k >= 0].max())],
            "single_wrap_agreement_pct": round(hit / len(allphi) * 100, 2), "segments": per}
    LABELS.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    write_umbilicus(g, k)
    print(f"wrote {LABELS}")


def write_umbilicus(g: np.ndarray, k: np.ndarray, step: float = 64.0) -> None:
    """Umbilicus control points (L2 xyz) from the innermost labelled wrap: per 64-z slab, the least-squares centre of
    the circle through the k = kmin vertices (Kasa fit), median-smoothed over 5 slabs. Human-mesh derived."""
    kmin = int(k[k >= 0].min())
    p = g[(k == kmin)]
    p = p[np.isfinite(p).all(-1)]
    rows = []
    for a in np.arange(0, VOL_L2[0], step):
        s = (p[:, 2] >= a) & (p[:, 2] < a + step)
        if s.sum() < 200:
            continue
        x, y = p[s, 0].astype(np.float64), p[s, 1].astype(np.float64)
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        rows.append((sol[0], sol[1], a + step / 2))
    r = np.array(rows)
    from scipy.ndimage import median_filter
    r[:, 0] = median_filter(r[:, 0], 5, mode="nearest")
    r[:, 1] = median_filter(r[:, 1], 5, mode="nearest")
    UMB.write_text(json.dumps({"control_points": [{"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 2)}
                                                  for x, y, z in r],
                               "metadata": {"frame": "PHerc1667 20251217075048 2.399um level 2 (9.596 um), xyz",
                                            "made_by": "eval/p1667_labels.py write_umbilicus",
                                            "source": f"centre of the innermost labelled wrap (k={kmin}) of the merged "
                                                      "human mesh, Kasa circle fit per 64 z, median of 5"}}, indent=1))
    print(f"umbilicus: {len(r)} control points z {r[0, 2]:.0f}..{r[-1, 2]:.0f} -> {UMB}")


def cmd_check(a) -> None:
    s = json.loads(LABELS.with_suffix(".json").read_text())
    for w, v in s["segments"].items():
        print(w, v["agree"], v["dist_p50"], v["k_hist"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "check"])
    a = ap.parse_args()
    {"label": cmd_label, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
