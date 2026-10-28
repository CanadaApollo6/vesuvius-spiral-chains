"""Loader and pair sampler for the PHerc0139 9.362 um instance test set (data/eval/p0139_9um, built by
scripts/p0139_testset.py; notes/2026-10-18-p0139-testset.md).

sites.json schema (the parts used here):
  {"voxel_um": 9.362, "segments": [{"id", "name", "winding"}...],
   "sites": [{"stem", "cls", "instances": {id: {"winding", "segment"}}, "pairs": [{"a", "b", "wa", "wb", "dw", "n",
              "d_median", ...}], ...}],   # final set only; dropped sites are listed in "not_in_set" with reasons
   "stats": {...gate statistics...}}
  a, b are instance ids in instancesTr/surfacesTr (= global segment ids); pairs list only adjacent windings (|wa-wb| = 1).

pair_points() is the single definition of a winding-pair site, used both for the spacing statistics in sites.json and
for scoring: seed voxels on instance a's 1-vox surface raster (surfacesTr), the local normal from the structure tensor
of that raster, march both ways along the normal (0.25 vox steps, nearest voxel) and keep the first other instance
hit if it is b. Returns (centre a, unit normal u, spacing d) so b = a + u d, as in eval/contact_check.find_sites.
Spacing classes (recto-to-recto, 9.362 vox): conflict d < 1 (coincident meshes, not scored), compressed 1-4 (< 37 um),
intermediate 4-8, wide 8-24.

pair_points() pairs are adjacent windings sampled along the normal, so the expected step is always 1 and the site
class is a function of the distance d: a constant field of fixed pitch P scores ~100% on them under the ratio rule
(delta = d / P). They are kept only as the "legacy" secondary block. The primary scoring pairs come from
scoring_pairs(): any two surface voxels A, B (both in the core, 2 <= |B - A| < 48 vox, readable from one 128^3 tile)
with expected step |w_B - w_A| in {0 (same wrap, displaced along the sheet), 1, 2, 3, 4}. They are balanced per site
and per 4-vox distance bin across the groups {0, 1, >= 2}: every group gets the same number of pairs in every bin, so the
Euclidean distance distribution is identical across groups and carries no information about the step. Each pair also
gets the projected distance along the local sheet normal (mean of |v.n_A| and |v.n_B|) for the pitch baselines and a
local compression flag: contact if either end has another instance within 4 vox (Euclidean EDT to the other surfaces),
open otherwise. write_pairs() freezes them into <root>/scoring_pairs.csv so every scorer uses the same rows.
"""

import csv
import json
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import structure_tensor

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "eval" / "p0139_9um"
CONFLICT_D, COMP_MAX, WIDE_MIN, MAX_D = 1.0, 4.0, 8.0, 24.0
CORE = 24  # seeds stay this far from the crop border
STEP = 0.25
# scoring pairs (expected steps 0..STEP_MAX, distance-balanced)
STEP_MAX = 4
R_MIN, R_MAX, R_BIN = 2.0, 48.0, 4.0
PER_GROUP = 100        # per site and step group {0, 1, >= 2}
N_DRAW = 400_000       # random candidate pairs per site
TILE, STRIDE = 128, 64  # must match eval/winding_sites.py (a pair is read from the one tile most central to its midpoint)
PAIRS_CSV = "scoring_pairs.csv"


def spacing_class(d: float) -> str:
    if d < CONFLICT_D:
        return "conflict"  # < 9.4 um recto-to-recto: thinner than any sheet, so coincident meshes (label error)
    return "compressed" if d < COMP_MAX else ("wide" if d >= WIDE_MIN else "intermediate")


def load_sites(root: Path = ROOT) -> list[dict]:
    """Sites of the final set (sites.json "sites", all passed the gates) with absolute paths."""
    meta = json.loads((root / "sites.json").read_text())
    out = []
    for s in meta["sites"]:
        out.append(s | {"image": root / "imagesTr" / f"{s['stem']}_0000.tif",
                        "instances_tif": root / "instancesTr" / f"{s['stem']}.tif",
                        "surfaces": root / "surfacesTr" / f"{s['stem']}.tif",
                        "labels": root / "labelsTr" / f"{s['stem']}.tif"})
    return out


def instance_normals(surf: np.ndarray, a: int, pts: np.ndarray) -> np.ndarray:
    """Unit normals of instance a's surface raster at pts (zyx int), from its own structure tensor (other sheets unseen)."""
    m = (surf == a).astype(np.float32)
    lo = np.maximum(pts.min(0) - 12, 0)
    hi = np.minimum(pts.max(0) + 13, surf.shape)
    sub = ndi.gaussian_filter(m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 1.0)
    st = structure_tensor(sub, sigma=3, order="rc")
    q = tuple((pts - lo).T)
    T = np.stack([np.stack([st[0][q], st[1][q], st[2][q]], -1), np.stack([st[1][q], st[3][q], st[4][q]], -1),
                  np.stack([st[2][q], st[4][q], st[5][q]], -1)], -2)
    return np.linalg.eigh(T)[1][..., -1]


def ray_hits(surf: np.ndarray, a: int, rng: np.random.Generator, n_seeds: int = 1600,
             core: int = CORE) -> list[tuple[np.ndarray, np.ndarray, float, int]]:
    """From random voxels of instance a, march both ways along a's normal; (centre, u, d, first other instance hit)."""
    n = np.array(surf.shape)
    pts = np.argwhere(surf == a)
    pts = pts[((pts >= core) & (pts < n - core)).all(1)]
    if len(pts) < 20:
        return []
    pts = pts[rng.permutation(len(pts))[:n_seeds]]
    nv = instance_normals(surf, a, pts)
    t = np.arange(STEP * 2, MAX_D + 1e-9, STEP)
    out = []
    for sgn in (1.0, -1.0):
        u = sgn * nv
        q = np.rint(pts[:, None, :] + u[:, None, :] * t[None, :, None]).astype(int)
        inside = ((q >= 0) & (q < n)).all(-1)
        q = np.clip(q, 0, n - 1)
        lab = surf[q[..., 0], q[..., 1], q[..., 2]]
        lab[~inside] = 0
        other = (lab != 0) & (lab != a)
        first = np.argmax(other, 1)
        for k in np.nonzero(other.any(1))[0]:
            j = first[k]
            b = int(lab[k, j])
            e = j  # centre of b's run along the ray (thin surface: 1-6 samples)
            while e + 1 < len(t) and lab[k, e + 1] == b:
                e += 1
            out.append((pts[k].astype(float), u[k], float((t[j] + t[e]) / 2), b))
    return out


def site_pairs(surf: np.ndarray, pairs: list[tuple[int, int]], rng: np.random.Generator, n_max: int = 300,
               core: int = CORE) -> dict:
    """{(a, b): [(centre_a, u, d), ...]} for the requested ordered pairs; one ray cast per source instance."""
    out = {}
    for a in sorted({p[0] for p in pairs}):
        hits = ray_hits(surf, a, rng, 4 * n_max * max(1, sum(p[0] == a for p in pairs)), core)
        for b in [p[1] for p in pairs if p[0] == a]:
            sel = [(c, u, d) for c, u, d, x in hits if x == b]
            out[(a, b)] = [sel[i] for i in rng.permutation(len(sel))[:n_max]]
    return out


def pair_points(surf: np.ndarray, a: int, b: int, rng: np.random.Generator, n_max: int = 300,
                core: int = CORE) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Up to n_max (centre_a, u, d) samples where the nearest other instance along a's normal is b (d <= MAX_D)."""
    return site_pairs(surf, [(a, b)], rng, n_max, core)[(a, b)]


# ------------------------------------------------------------------------------------- distance-balanced scoring pairs
def _starts(n: int) -> list[int]:
    s = list(range(0, max(n - TILE, 0) + 1, STRIDE))
    if s[-1] != max(n - TILE, 0):
        s.append(max(n - TILE, 0))
    return s


def tile_ok(a: np.ndarray, b: np.ndarray, shape) -> np.ndarray:
    """Vectorised copy of winding_sites.site_delta's tile choice: True where both ends lie in the chosen tile."""
    org = np.array([(z, y, x) for z in _starts(shape[0]) for y in _starts(shape[1]) for x in _starts(shape[2])], float)
    m = (a + b) / 2
    cheb = np.abs(m[:, None, :] - (org[None] + (TILE - 1) / 2)).max(-1)
    o = org[np.argmin(cheb, 1)]  # argmin = first minimum, same order as min() over the tile dict
    return (((a - o) >= 0) & ((a - o) <= TILE - 1) & ((b - o) >= 0) & ((b - o) <= TILE - 1)).all(1)


def nn_other(surf: np.ndarray, pts: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Euclidean distance from each surface point to the nearest voxel of any other instance."""
    out = np.full(len(pts), np.inf)
    for a in np.unique(lab):
        other = (surf != 0) & (surf != a)
        if not other.any():
            continue
        m = lab == a
        out[m] = ndi.distance_transform_edt(~other)[tuple(pts[m].T)]
    return out


def scoring_pairs(surf: np.ndarray, wind: dict, rng: np.random.Generator, per_group: int = PER_GROUP,
                  core: int = CORE) -> list[dict]:
    n = np.array(surf.shape)
    pts = np.argwhere(surf > 0)
    pts = pts[((pts >= core) & (pts < n - core)).all(1)]
    lab = surf[tuple(pts.T)].astype(int)
    keep = np.isin(lab, [int(k) for k in wind])
    pts, lab = pts[keep], lab[keep]
    dnn = nn_other(surf, pts, lab)
    keep = dnn >= CONFLICT_D  # coincident meshes: not scorable
    pts, lab, dnn = pts[keep], lab[keep], dnn[keep]
    if len(pts) < 100:
        return []
    w = np.array([wind[int(k)] for k in lab])
    i, j = rng.integers(0, len(pts), (2, N_DRAW))
    v = (pts[j] - pts[i]).astype(float)
    dist = np.linalg.norm(v, axis=1)
    step = np.abs(w[j] - w[i])
    ok = (dist >= R_MIN) & (dist < R_MAX) & (step <= STEP_MAX)
    ok[ok] = tile_ok(pts[i[ok]].astype(float), pts[j[ok]].astype(float), surf.shape)
    i, j, dist, step = i[ok], j[ok], dist[ok], step[ok]
    grp = np.minimum(step, 2)
    rbin = ((dist - R_MIN) // R_BIN).astype(int)
    nb = int(rbin.max()) + 1 if len(rbin) else 0
    cnt = np.zeros((nb, 3), int)
    np.add.at(cnt, (rbin, grp), 1)
    k = cnt.min(1)
    if k.sum() > per_group:
        k = np.floor(k * per_group / k.sum()).astype(int)
    sel = []
    for b in np.nonzero(k)[0]:
        for g in range(3):
            idx = np.nonzero((rbin == b) & (grp == g))[0]
            sel.extend(rng.choice(idx, k[b], replace=False).tolist())
    sel = np.array(sorted(sel), int)
    if not len(sel):
        return []
    ia, ib = i[sel], j[sel]
    nrm = np.zeros((len(pts), 3))
    used = np.unique(np.concatenate([ia, ib]))
    for a in np.unique(lab[used]):
        u = used[lab[used] == a]
        nrm[u] = instance_normals(surf, int(a), pts[u])
    v = (pts[ib] - pts[ia]).astype(float)
    proj = 0.5 * (np.abs((v * nrm[ia]).sum(1)) + np.abs((v * nrm[ib]).sum(1)))
    rows = []
    for q in range(len(sel)):
        A, B = pts[ia[q]], pts[ib[q]]
        dmin = float(min(dnn[ia[q]], dnn[ib[q]]))
        rows.append({"za": int(A[0]), "ya": int(A[1]), "xa": int(A[2]), "zb": int(B[0]), "yb": int(B[1]), "xb": int(B[2]),
                     "a": int(lab[ia[q]]), "b": int(lab[ib[q]]), "wa": int(w[ia[q]]), "wb": int(w[ib[q]]),
                     "expected": int(step[sel[q]]), "dist": round(float(dist[sel[q]]), 3), "proj": round(float(proj[q]), 3),
                     "dnn_min": round(dmin, 3), "contact": bool(dmin < COMP_MAX)})
    return rows


def legacy_pairs(surf: np.ndarray, site: dict, rng: np.random.Generator, per_pair: int = 300) -> list[dict]:
    """The old adjacent-pair normal-ray sites (expected 1; class = spacing of d). Gameable; continuity only."""
    req = [(p["a"], p["b"]) for p in site["pairs"]] + [(p["b"], p["a"]) for p in site["pairs"]]
    smp = site_pairs(surf, req, rng, n_max=per_pair // 2)
    wind = {int(k): v["winding"] for k, v in site["instances"].items()}
    rows = []
    for a, b in req:
        for c, u, d in smp.get((a, b), []):
            if spacing_class(d) == "conflict":
                continue
            B = c + u * d
            rows.append({"za": c[0], "ya": c[1], "xa": c[2], "zb": round(float(B[0]), 3), "yb": round(float(B[1]), 3),
                         "xb": round(float(B[2]), 3), "a": a, "b": b, "wa": wind[a], "wb": wind[b],
                         "expected": abs(wind[a] - wind[b]), "dist": round(d, 3), "proj": round(d, 3),
                         "dnn_min": round(d, 3), "contact": bool(d < COMP_MAX)})
    return rows


FIELDS = ["stem", "site_cls", "kind", "za", "ya", "xa", "zb", "yb", "xb", "a", "b", "wa", "wb", "expected", "dist", "proj",
          "dnn_min", "contact", "pitch_site"]


def write_pairs(root: Path = ROOT, limit: int = 0) -> Path:
    """Freeze the scoring pairs (kind "mix" = primary, "legacy" = adjacent normal rays) into <root>/scoring_pairs.csv."""
    import tifffile

    rows = []
    for s in load_sites(root)[: limit or None]:
        surf = tifffile.imread(s["surfaces"])
        wind = {int(k): v["winding"] for k, v in s["instances"].items()}
        rng = np.random.default_rng(zlib.crc32(s["stem"].encode()))
        extra = {"stem": s["stem"], "site_cls": s["cls"], "pitch_site": round(float(s["spacing"]["d_median_vox"]), 3)}
        mix = scoring_pairs(surf, wind, rng)
        leg = legacy_pairs(surf, s, rng)
        rows += [extra | {"kind": "mix"} | r for r in mix] + [extra | {"kind": "legacy"} | r for r in leg]
        print(f"{s['stem']} ({s['cls']}): {len(mix)} mix pairs "
              f"(steps {np.bincount([r['expected'] for r in mix], minlength=STEP_MAX + 1).tolist()}), {len(leg)} legacy",
              flush=True)
    out = root / PAIRS_CSV
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return out


def read_pairs(root: Path = ROOT) -> list[dict]:
    with (root / PAIRS_CSV).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("za", "ya", "xa", "zb", "yb", "xb", "dist", "proj", "dnn_min", "pitch_site"):
            r[k] = float(r[k])
        for k in ("a", "b", "wa", "wb", "expected"):
            r[k] = int(r[k])
        r["contact"] = r["contact"] == "True"
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="freeze the p0139 scoring pairs")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    print(write_pairs(a.root, a.limit))
