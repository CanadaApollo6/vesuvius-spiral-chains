"""PHerc0139 9.362 um merge-sensitive test set with per-winding instance labels (plan (a), notes/2026-10-18-w1-data-recon.md).

  uv run python scripts/p0139_testset.py mine                 # mesh spacing -> candidate sites (_work/candidates.json)
  uv run python scripts/p0139_testset.py build [--workers 6]  # stream crops, rasterize, quality gates (_work/sites/*.json)
  uv run python scripts/p0139_testset.py finalize             # sites.json, pairs, previews

Geometry. Meshes are the public `-on-20260102150214-2.399um.tifxyz` grids (step 20 vox = 48 um), mapped to the 9.362
scan with the published affine (transform.json next to the 2.399 volume, xyz, p9 = A p24 + t) plus a per-site NCC
translation. The public 9.362 tifxyz are the same affine on a 4x coarser grid (187 um), too coarse to rasterize from.
Coordinates: tifxyz and the affine are xyz; everything else here is zyx in 9.362 L0 voxels.

Output (data/eval/p0139_9um/, local and private: ash2txt licence forbids redistribution):
  imagesTr/<stem>_0000.tif   uint8 CT, N^3 from 9.362 L0 (unchanged; labels are moved onto it)
  instancesTr/<stem>.tif     uint16, 3-vox band per winding, id = global segment id (sites.json "segments"), 0 = none
  labelsTr/<stem>.tif        uint8 {0,1} = instancesTr > 0 (jeong200 convention: 3-vox band on the recto face)
  surfacesTr/<stem>.tif      uint16, 1-vox mesh surface per winding (the raw raster the band is grown from)
  sites.json                 provenance, gates, spacing stats, adjacent pairs (eval/p0139_sites.py loads it)
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_sites import COMP_MAX, CONFLICT_D, WIDE_MIN, ray_hits, spacing_class  # noqa: E402

OUT = REPO / "data" / "eval" / "p0139_9um"
WORK = OUT / "_work"
MESHES = REPO / "data" / "meshes" / "p0139"
S3 = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/"
VOL9 = S3 + "20250728140407-9.362um-1.2m-113keV-masked.zarr"
VOL24 = S3 + "20260102150214-2.399um-0.2m-78keV-masked.zarr"
TF_URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/"
          "20260102150214-2.399um-0.2m-78keV-masked.zarr/transform.json")
TF_LOCAL = REPO / "data" / "eval" / "surface_labels_2026_07_sample" / "transform_0139_2p399_to_9p362.json"
MESH_TAG = "20260102150214-2.399um"
VOX9 = 9.362

N = 192          # crop edge (9.362 vox) = 1.80 mm
CORE = 24        # border excluded from site statistics and gates
CELL = 32        # mining grid cell (vox)
MIN_SEP = N - 2 * CORE  # 144: site cores (where pair samples are seeded) are disjoint; crops overlap by <= 48 vox
SHIFT = 6        # NCC search radius (vox)
L2 = 4           # 2.399 L2 = 4^3 block mean of L0, voxel i centred at L0 4i + 1.5 (verified against L0)
BLUR_L2 = 0.5    # extra blur on L2 before resampling (vox at L2)
SEED = 18

# gates (distances in 9.362 vox to the nearest material voxel: CT, gaussian 0.7, above the crop's Otsu threshold)
NCC_MIN = 0.90
SHIFT_SPREAD_MAX = 2.0   # octant shift deviation from the site shift (vox)
ON_D, OFF_D, DEEP_D = 1.0, 2.0, 4.0  # on papyrus <= 1 vox; off > 2 vox (removed, -> ignore); deep in a gap > 4 vox
DEEP_TILES_MAX = 2       # per site: 24^3 tiles where > 50% of one instance's voxels are > DEEP_D from material
POOLED_ONPAP_MIN = 0.70  # all labelled voxels of the site
INST_ONPAP_MIN = 0.50    # every instance
INST_OFF_MAX = 0.30      # every instance: share of voxels removed as off papyrus
NORMAL_AGREE_MIN = 0.70  # share of surface voxels whose mesh normal is within 45 deg of the CT sheet normal
CROSS_MAX = 0.05         # share of paired points on the minority side (crossing) or closer than 1 vox (coincident)
COVERAGE_MAX_UNL = 0.35  # coverage_ok flag (not a gate): unlabelled material share in the core
# site classes from the cleaned rasters' adjacent-pair samples
COMP_SITE_N, COMP_SITE_FRAC = 100, 0.03
WIDE_SITE_COMP_MAX, WIDE_SITE_FRAC = 0.005, 0.80
MIXED_SITE_N = 40        # neither: "mixed" if >= 40 and >= 0.5% compressed samples (contact zones in an open crop)
CLASSES = ("compressed", "mixed", "wide")


# ----------------------------------------------------------------------------------------------- geometry helpers
def affine():
    tf = json.loads(TF_LOCAL.read_text())
    M = np.array(tf["transformation_matrix"])
    return M[:, :3], M[:, 3]


def segments() -> list[dict]:
    """Global segment table; id = 1-based index (stable: sorted directory names)."""
    out = []
    for d in sorted(MESHES.iterdir()):
        if not (d / MESH_TAG / "x.tif").exists():
            continue
        m = re.search(r"-w(\d+)_", d.name)
        out.append({"id": len(out) + 1, "name": d.name, "winding": int(m.group(1)) if m else None})
    return out


def load_mesh9(name: str, A, t) -> np.ndarray:
    """(H, W, 3) zyx in 9.362 L0 voxels (published affine, no refinement); invalid = nan."""
    d = MESHES / name / MESH_TAG
    g = np.stack([tifffile.imread(d / f"{c}.tif") for c in "xyz"], -1).astype(np.float64)
    ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
    p = g @ A.T + t
    p[~ok] = np.nan
    return p[..., ::-1].copy()


def grid_normals(p: np.ndarray) -> np.ndarray:
    di = np.gradient(p, axis=0)
    dj = np.gradient(p, axis=1)
    n = np.cross(di, dj)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------------------------------------- mine
def mine(args) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    A, t = affine()
    segs = segments()
    pts, nrm, sid = {}, {}, {}
    for s in segs:
        p = load_mesh9(s["name"], A, t)
        n = grid_normals(p)
        ok = np.isfinite(p).all(-1) & np.isfinite(n).all(-1)
        pts[s["id"]], nrm[s["id"]] = p[ok], n[ok]
        print(f"{s['id']:2d} {s['name']}: {ok.sum()} pts, z {np.nanmin(p[..., 0]):.0f}-{np.nanmax(p[..., 0]):.0f}", flush=True)
    trees = {k: cKDTree(v) for k, v in pts.items()}
    wind = {s["id"]: s["winding"] for s in segs}

    # per query point (every 2nd grid point): perpendicular distance to adjacent windings and to any other segment
    q_pts, q_dadj, q_dany, q_any_id, q_sid = [], [], [], [], []
    overlap = {}
    for s in segs:
        a = s["id"]
        P = pts[a][::4]  # ~1/4 of points (grid is flattened; stride 4 ~ every 2nd point in each direction on average)
        d_adj = np.full(len(P), np.inf)
        d_any = np.full(len(P), np.inf)
        any_id = np.zeros(len(P), np.int16)
        lo, hi = P.min(0) - 30, P.max(0) + 30
        for b in pts:
            if b == a or (pts[b].max(0) < lo).any() or (pts[b].min(0) > hi).any():
                continue
            dist, j = trees[b].query(P, distance_upper_bound=25.0)
            hit = np.isfinite(dist)
            if not hit.any():
                continue
            v = P[hit] - pts[b][j[hit]]
            dp = np.abs((v * nrm[b][j[hit]]).sum(1))
            lat = np.sqrt(np.maximum(dist[hit] ** 2 - dp ** 2, 0))
            dp[lat > 4.0] = np.inf  # nearest vertex not over the point: B ends here
            full = np.full(len(P), np.inf)
            full[hit] = dp
            if wind[a] is not None and wind[b] is not None and abs(wind[a] - wind[b]) == 1:
                d_adj = np.minimum(d_adj, full)
            better = full < d_any
            any_id[better] = b
            d_any = np.minimum(d_any, full)
            ov = int((full < 1.0).sum())
            if ov:
                overlap[f"{a}-{b}"] = [ov, len(P)]
        q_pts.append(P), q_dadj.append(d_adj), q_dany.append(d_any), q_any_id.append(any_id), q_sid.append(np.full(len(P), a, np.int16))
        fin = np.isfinite(d_adj)
        print(f"{s['name']}: adj measured {fin.mean():.2f}, d_adj<4 {np.mean(d_adj[fin] < 4) if fin.any() else 0:.3f}, "
              f"median {np.median(d_adj[fin]) if fin.any() else np.nan:.1f}", flush=True)
    P = np.concatenate(q_pts)
    np.savez_compressed(WORK / "mine_points.npz", p=P.astype(np.float32), d_adj=np.concatenate(q_dadj).astype(np.float32),
                        d_any=np.concatenate(q_dany).astype(np.float32), any_id=np.concatenate(q_any_id),
                        sid=np.concatenate(q_sid))
    (WORK / "overlap.json").write_text(json.dumps(overlap, indent=1))
    select(args)


def blacklist() -> dict:
    """Segments whose mesh coincides (< CONFLICT_D) with another segment over > 20% of its sampled points."""
    ov = json.loads((WORK / "overlap.json").read_text())
    segs = {s["id"]: s for s in segments()}
    bad = {}
    for k, (n, tot) in ov.items():
        a, b = map(int, k.split("-"))
        if n / tot > 0.2:
            bad[segs[a]["name"]] = f"coincides with {segs[b]['name']} on {100 * n / tot:.0f}% of points (< 1 vox)"
    return bad


def select(args) -> None:
    """Candidate sites from the mined points: 32^3 cells, crop centre on a cell corner.

    Statistics over the 128^3 core; rejection (blacklisted segment, coincident meshes) over the full 192^3 crop."""
    z = np.load(WORK / "mine_points.npz")
    segs = segments()
    wind = {s["id"]: s["winding"] for s in segs}
    bad = blacklist()
    bad_ids = {s["id"] for s in segs if s["name"] in bad}
    print("blacklisted:", json.dumps(bad, indent=1))
    keep = np.array([wind[int(i)] is not None for i in z["sid"]])
    P, d_adj, d_any, sid = z["p"][keep], z["d_adj"][keep], z["d_any"][keep], z["sid"][keep]
    isbad = np.isin(sid, list(bad_ids))
    cell = np.floor(P / CELL).astype(int)
    c0 = cell.min(0) - 3
    cell -= c0
    shp = cell.max(0) + 4
    fin = np.isfinite(d_adj) & ~isbad
    wlist = sorted({w for w in wind.values() if w is not None})
    widx = np.array([wlist.index(wind[int(i)]) for i in sid])

    def grid(mask, per_w=False):
        g = np.zeros((*shp, len(wlist)) if per_w else shp, np.int32)
        np.add.at(g, (*cell[mask].T, widx[mask]) if per_w else tuple(cell[mask].T), 1)
        return g

    def box(g, k):  # sum over the k^3 cells around each cell corner (k even)
        cs = np.pad(g, [(1, 0)] * 3 + [(0, 0)] * (g.ndim - 3)).cumsum(0).cumsum(1).cumsum(2)
        h = k // 2
        idx = np.indices(shp).reshape(3, -1).T
        lo, hi = np.clip(idx - h, 0, shp), np.clip(idx + h, 0, shp)
        S = lambda a, b, c: cs[a[:, 0], b[:, 1], c[:, 2]]  # noqa: E731
        return (S(hi, hi, hi) - S(lo, hi, hi) - S(hi, lo, hi) - S(hi, hi, lo) + S(lo, lo, hi) + S(lo, hi, lo)
                + S(hi, lo, lo) - S(lo, lo, lo))

    kc, kf = (N - 2 * CORE) // CELL // 2 * 2, N // CELL  # 4 cells (128 core), 6 cells (192 crop)
    comp = fin & (d_adj >= CONFLICT_D) & (d_adj < COMP_MAX)
    cand = {"n_adj": box(grid(fin), kc), "n_comp": box(grid(comp), kc), "n_wide": box(grid(fin & (d_adj >= WIDE_MIN)), kc),
            "n_conflict_crop": box(grid(np.isfinite(d_any) & (d_any < CONFLICT_D)), kf),
            "n_bad_crop": box(grid(isbad), kf)}
    per = box(grid(~isbad, per_w=True), kc) >= 40
    per_full = box(grid(~isbad, per_w=True), kf) >= 1
    # labelled range edge (w023 inner, w059 outer) anywhere in the crop: the unlabelled next winding is likely in it
    cand["range_edge"] = per_full[:, 0] | per_full[:, -1]
    cand["n_windings"] = per.sum(-1)
    first, last = per.argmax(-1), len(wlist) - 1 - per[:, ::-1].argmax(-1)
    cand["winding_gaps"] = np.where(cand["n_windings"] > 0, last - first + 1 - cand["n_windings"], 99)
    cand["frac_comp"] = cand["n_comp"] / np.maximum(cand["n_adj"], 1)
    cand["frac_wide"] = cand["n_wide"] / np.maximum(cand["n_adj"], 1)
    centre = (np.indices(shp).reshape(3, -1).T + c0) * CELL
    ok_vol = ((centre - N // 2 >= 0) & (centre + N // 2 <= np.array([20974, 6621, 6621]))).all(1)
    base = (ok_vol & (cand["n_windings"] >= 4) & (cand["winding_gaps"] == 0) & (cand["n_adj"] >= 300)
            & (cand["n_bad_crop"] == 0) & (cand["n_conflict_crop"] == 0) & ~cand["range_edge"])
    comp_ok = base & (cand["n_comp"] >= 30) & (cand["frac_comp"] >= 0.04)
    wide_ok = base & (cand["frac_comp"] < 0.005) & (cand["frac_wide"] >= 0.9)
    print(f"cells {len(centre)}, base {base.sum()}, compressed-eligible {comp_ok.sum()}, wide-eligible {wide_ok.sum()}")
    for q in (0.5, 0.75, 0.9, 0.99):
        print(f"  frac_comp q{q}: {np.quantile(cand['frac_comp'][base], q):.3f}")

    chosen = []

    def take(mask, order, n_max, cls):
        k = 0
        for i in order:
            if not mask[i]:
                continue
            c = centre[i]
            if all(np.abs(c - d["centre"]).max() >= MIN_SEP for d in chosen):
                chosen.append({"cls": cls, "centre": c, "windings": [wlist[j] for j in np.nonzero(per[i])[0]],
                               **{key: float(v[i]) for key, v in cand.items()}})
                k += 1
                if k == n_max:
                    break
        return k

    order = np.lexsort((-cand["n_comp"], -cand["frac_comp"]))
    nc = take(comp_ok, order, args.n_comp, "compressed")
    rng = np.random.default_rng(SEED)
    nw = take(wide_ok, rng.permutation(len(centre)), args.n_wide, "wide")
    print(f"chosen compressed {nc}, wide {nw}")
    for i, c in enumerate(chosen):
        c["stem"] = f"p0139_{c['cls'][0]}{i:03d}_z{int(c['centre'][0])}_y{int(c['centre'][1])}_x{int(c['centre'][2])}"
        c["origin"] = (np.asarray(c["centre"]) - N // 2).astype(int).tolist()
        c["centre"] = np.asarray(c["centre"]).astype(int).tolist()
    # controls: the three W1 sample sites (notes/2026-10-18-w1-data-recon.md 1a); w037 is the known off-sheet mesh site
    controls = []
    for name, c in (("w037", [6279, 3035, 3786]), ("w047", [6635, 4252, 2057]), ("w023", [4243, 3836, 3761])):
        controls.append({"cls": "control", "stem": f"p0139_ctrl_{name}_z{c[0]}_y{c[1]}_x{c[2]}", "centre": c,
                         "origin": [x - N // 2 for x in c], "windings": [], **{k: float("nan") for k in cand}})
    (WORK / "candidates.json").write_text(json.dumps({"blacklist": bad, "candidates": chosen, "controls": controls}, indent=1))
    print(f"wrote {WORK / 'candidates.json'}")


# ---------------------------------------------------------------------------------------------------------- build
_VOLS = {}


def vols():
    if not _VOLS:
        import zarr
        so = {"anon": True}
        _VOLS["ct9"] = zarr.open_group(VOL9, mode="r", storage_options=so)["0"]
        _VOLS["ct24_l2"] = zarr.open_group(VOL24, mode="r", storage_options=so)["2"]
    return _VOLS


def read_box(arr, lo, hi):
    """Read [lo, hi) (zyx, clipped to the array; zero-filled outside) and the number of 128^3 chunks touched."""
    lo, hi = np.asarray(lo, int), np.asarray(hi, int)
    shp = np.array(arr.shape)
    a, b = np.maximum(lo, 0), np.minimum(hi, shp)
    out = np.zeros(hi - lo, arr.dtype)
    if (b > a).all():
        out[tuple(slice(x - l, y - l) for x, y, l in zip(a, b, lo))] = arr[tuple(slice(x, y) for x, y in zip(a, b))]
    nch = int(np.prod([(y - 1) // 128 - x // 128 + 1 for x, y in zip(a, b)])) if (b > a).all() else 0
    return out, nch


def ncc_masked(a, b, m):
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    if a.size < 1000:
        return float("nan")
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


class Render:
    """2.399 L2 (4^3 block mean, + blur) resampled onto 9.362 voxel grids through the published affine."""

    def __init__(self, lo9, hi9, A, t):
        self.A, self.t, self.Ainv = A, t, np.linalg.inv(A)
        corners = np.array([[z, y, x] for z in (lo9[0], hi9[0]) for y in (lo9[1], hi9[1]) for x in (lo9[2], hi9[2])], float)
        p24 = self.to24(corners)  # zyx L0 2.399
        l2 = (p24 - 1.5) / L2
        self.lo = np.floor(l2.min(0)).astype(int) - 4
        hi = np.ceil(l2.max(0)).astype(int) + 5
        raw, self.nch = read_box(vols()["ct24_l2"], self.lo, hi)
        self.valid = ndi.binary_erosion(raw > 0, iterations=2)
        self.img = ndi.gaussian_filter(raw.astype(np.float32), BLUR_L2)

    def to24(self, zyx9):
        return ((zyx9[:, ::-1] - self.t) @ self.Ainv.T)[:, ::-1]

    def render(self, lo9, shape, shift=(0.0, 0.0, 0.0)):
        """Values at 9.362 voxels lo9 + idx; content at affine position y appears at y + shift, so sample y = x - shift."""
        g = np.indices(shape).reshape(3, -1).T + np.asarray(lo9) - np.asarray(shift)
        q = ((self.to24(g.astype(float)) - 1.5) / L2 - self.lo).T
        v = map_coordinates(self.img, q, order=1, mode="constant", cval=0.0)
        ok = map_coordinates(self.valid.astype(np.float32), q, order=0, mode="constant", cval=0.0) > 0.5
        return np.where(ok, v, 0.0).reshape(shape).astype(np.float32)


def shift_search(rend0, v9m, m, centre, radius, step, sub=2, region=None):
    """Integer shift d (|d - centre| <= radius, grid step) maximizing NCC(rend0[region], v9m[m + d + region])."""
    r_lo, r_hi = (np.zeros(3, int), np.array(rend0.shape)) if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    R = rend0[r_lo[0]:r_hi[0]:sub, r_lo[1]:r_hi[1]:sub, r_lo[2]:r_hi[2]:sub]
    n = r_hi - r_lo
    scores = {}
    rng = range(-radius, radius + 1, step)
    for dz in rng:
        for dy in rng:
            for dx in rng:
                d = np.asarray(centre) + (dz, dy, dx)
                st = m + d + r_lo
                if (st < 0).any() or (st + n > np.array(v9m.shape)).any():
                    continue
                V = v9m[st[0]:st[0] + n[0]:sub, st[1]:st[1] + n[1]:sub, st[2]:st[2] + n[2]:sub]
                scores[tuple(int(x) for x in d)] = ncc_masked(R, V, (R > 0) & (V > 0))
    best = max((k for k in scores if np.isfinite(scores[k])), key=scores.get, default=None)
    return best, scores


def parabolic(scores, d0):
    sub = []
    for ax in range(3):
        e = np.eye(3, dtype=int)[ax]
        sm, s0, sp = (scores.get(tuple(int(x) for x in np.asarray(d0) + k * e), np.nan) for k in (-1, 0, 1))
        den = sm - 2 * s0 + sp
        sub.append(float(np.clip(0.5 * (sm - sp) / den, -0.5, 0.5)) if np.isfinite(den) and den < 0 else 0.0)
    return np.asarray(d0, float) + np.array(sub)


def register(origin, A, t):
    """Site translation (9.362 vox, zyx) by NCC against the rendered 2.399 L2, plus octant consistency."""
    m = SHIFT + 2
    lo9 = np.asarray(origin) - m
    v9m, nch9 = read_box(vols()["ct9"], lo9, lo9 + N + 2 * m)
    R = Render(lo9, lo9 + N + 2 * m, A, t)
    rend0 = R.render(origin, (N, N, N))
    d_c, _ = shift_search(rend0, v9m, m, (0, 0, 0), SHIFT, 2, sub=3)
    d_i, sc = shift_search(rend0, v9m, m, d_c, 2, 1, sub=2)
    shift = parabolic(sc, d_i)
    octs = []
    h = N // 2
    for oz in (0, h):
        for oy in (0, h):
            for ox in (0, h):
                reg = (np.array([oz, oy, ox]), np.array([oz, oy, ox]) + h)
                sub_r = rend0[oz:oz + h, oy:oy + h, ox:ox + h]
                if (sub_r > 0).mean() < 0.2:
                    continue
                d_o, sc_o = shift_search(rend0, v9m, m, d_i, 2, 1, sub=2, region=reg)
                if d_o is not None:
                    octs.append(parabolic(sc_o, d_o).tolist())
    rend = R.render(origin, (N, N, N), shift)
    ct = v9m[m:m + N, m:m + N, m:m + N]
    valid = (rend > 0) & (ct > 0)
    return {"ct": ct, "rend": rend, "shift": shift, "ncc": ncc_masked(rend, ct, valid),
            "ncc_published": ncc_masked(rend0, ct, (rend0 > 0) & (ct > 0)), "ncc_valid_frac": float(valid.mean()),
            "octant_shifts": octs,
            "octant_dev_max": float(np.max(np.abs(np.array(octs) - shift))) if octs else float("nan"),
            "chunks_9p362": nch9, "chunks_2p399_L2": R.nch}


def rasterize(origin, shift, segs, A, t, sub=12):
    """1-vox surface per segment (closest sample wins a contested voxel) + per-voxel mesh normal."""
    o = np.asarray(origin, float) - np.asarray(shift)  # affine-frame position of crop voxel 0
    lo, hi = o - 3, o + N + 3
    surf = np.zeros((N, N, N), np.uint16)
    best = np.full((N, N, N), np.inf, np.float32)
    nrm = np.zeros((N, N, N, 3), np.float16)
    counts = {}
    uu = np.linspace(0, 1, sub + 1)
    wu, wv = np.meshgrid(uu, uu, indexing="ij")
    wu, wv = wu.ravel(), wv.ravel()
    for s in segs:
        meta = json.loads((MESHES / s["name"] / "20250728140407-9.362um" / "meta.json").read_text())
        bb = np.array(meta["bbox"])[:, ::-1]  # zyx 9.362 (published 9.362 mesh = same affine, coarser grid)
        if (bb[1] < lo - 20).any() or (bb[0] > hi + 20).any():
            continue
        g = load_mesh9(s["name"], A, t)
        ins = np.isfinite(g).all(-1) & ((g >= lo) & (g <= hi)).all(-1)
        if not ins.any():
            continue
        q = ins[:-1, :-1] | ins[1:, :-1] | ins[:-1, 1:] | ins[1:, 1:]
        ok = np.isfinite(g).all(-1)
        q &= ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        ii, jj = np.nonzero(q)
        p00, p10, p01, p11 = g[ii, jj], g[ii + 1, jj], g[ii, jj + 1], g[ii + 1, jj + 1]
        n = np.cross(p10 - p00 + p11 - p01, p01 - p00 + p11 - p10)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        P = (p00[:, None] * ((1 - wu) * (1 - wv))[None, :, None] + p10[:, None] * (wu * (1 - wv))[None, :, None]
             + p01[:, None] * ((1 - wu) * wv)[None, :, None] + p11[:, None] * (wu * wv)[None, :, None])
        Nn = np.broadcast_to(n[:, None], P.shape).reshape(-1, 3)
        P = P.reshape(-1, 3) - o
        v = np.rint(P).astype(int)
        keep = ((v >= 0) & (v < N)).all(1)
        if keep.sum() < 50:
            continue
        v, P, Nn = v[keep], P[keep], Nn[keep]
        dist = np.linalg.norm(P - v, axis=1).astype(np.float32)
        order = np.argsort(-dist)  # write far first; the closest sample to each voxel centre is written last
        v, dist, Nn = v[order], dist[order], Nn[order]
        cur = best[tuple(v.T)]
        w = dist < cur
        # within this segment, later (closer) writes win; across segments only if closer than what is there
        idx = tuple(v[w].T)
        best[idx] = dist[w]
        surf[idx] = s["id"]
        nrm[idx] = Nn[w].astype(np.float16)
        counts[s["id"]] = int(keep.sum())
    return surf, nrm, counts


def band_from_surface(surf):
    """jeong200-style band: voxels within 1.0 vox (EDT to the nearest surface voxel) of a surface; id of that voxel."""
    d, ind = ndi.distance_transform_edt(surf == 0, return_indices=True)
    band = d <= 1.0
    inst = np.where(band, surf[tuple(ind)], 0).astype(np.uint16)
    return inst, d


def gates(ct, surf, nrm, segs_by_id, pairs_raw):
    """Per-instance papyrus and orientation statistics on the raw rasters, pair crossing / coincidence / order.

    Returns (stats dict for sites.json, distance-to-material volume)."""
    from skimage.feature import structure_tensor
    from skimage.filters import threshold_otsu

    ctf = ndi.gaussian_filter(ct.astype(np.float32), 0.7)
    vmask = ct > 0
    thr = float(threshold_otsu(ctf[vmask][::5])) if vmask.any() else 0.0
    dmat = ndi.distance_transform_edt(ctf < thr)  # distance to material (vox); 0 on material
    st = structure_tensor(ct.astype(np.float32), sigma=2.0, order="rc")
    c = slice(CORE, N - CORE)
    core = np.zeros(surf.shape, bool)
    core[c, c, c] = True
    out = {"otsu": thr, "instances": {}}
    pooled = []
    for i in [int(x) for x in np.unique(surf) if x]:
        m = (surf == i) & core
        n_vox = int(m.sum())
        rec = {"n_core_vox": n_vox}
        if n_vox < 200:
            rec["skipped"] = "fewer than 200 core voxels"
            out["instances"][i] = rec
            continue
        pts = np.argwhere(m)
        dm = dmat[tuple(pts.T)]
        pooled.append(dm)
        rec["on_papyrus"] = float((dm <= ON_D).mean())
        rec["frac_off"] = float((dm > OFF_D).mean())
        tiles = {}
        for p, x in zip(pts // 24, dm):
            tiles.setdefault(tuple(p), []).append(x)
        tl = [np.array(v) for v in tiles.values() if len(v) >= 40]
        rec["n_tiles"] = len(tl)
        rec["n_tiles_off2"] = int(sum((v > OFF_D).mean() > 0.5 for v in tl))  # strict variant (reported only)
        rec["n_tiles_deep"] = int(sum((v > DEEP_D).mean() > 0.5 for v in tl))
        q = tuple(pts.T)
        T = np.stack([np.stack([st[0][q], st[1][q], st[2][q]], -1), np.stack([st[1][q], st[3][q], st[4][q]], -1),
                      np.stack([st[2][q], st[4][q], st[5][q]], -1)], -2)
        w, V = np.linalg.eigh(T)
        n_ct = V[..., -1]
        coh = (w[:, 2] - w[:, 1]) / (w[:, 2] + w[:, 1] + 1e-9)
        n_m = nrm[q].astype(np.float32)
        cosang = np.abs((n_ct * n_m).sum(1)) / (np.linalg.norm(n_m, axis=1) + 1e-9)
        strong = coh > 0.2
        ok = strong.sum() > 100
        rec["normal_agree"] = float((cosang[strong] >= np.cos(np.radians(45))).mean()) if ok else float("nan")
        rec["normal_angle_median_deg"] = float(np.degrees(np.arccos(np.clip(np.median(cosang[strong]), 0, 1)))) if ok else float("nan")
        out["instances"][i] = rec
    allv = np.concatenate(pooled) if pooled else np.zeros(0)
    out["pooled_on_papyrus"] = float((allv <= ON_D).mean()) if len(allv) else float("nan")
    out["pooled_off"] = float((allv > OFF_D).mean()) if len(allv) else float("nan")
    # pair crossing: share of a's samples where b lies on the minority side of a's (consistently oriented) mesh normal
    out["pairs"] = {}
    for (a, b), smp in pairs_raw.items():
        if len(smp) < 30:
            continue
        side = np.array([np.sign(np.dot(u, nrm[tuple(np.rint(p).astype(int))].astype(np.float32))) for p, u, d in smp])
        side = side[side != 0]
        if len(side) < 30:
            continue
        minority = float(min((side > 0).mean(), (side < 0).mean()))
        out["pairs"][f"{a}-{b}"] = {"n": len(side), "minority_side": minority, "major_side": int(np.sign(side.sum()) or 1),
                                    "conflict_frac": float(np.mean([d < CONFLICT_D for _, _, d in smp]))}
    # winding order: a's neighbours w-1 and w+1 must lie on opposite sides of a
    out["order_flips"] = []
    for a in [int(x) for x in np.unique(surf) if x]:
        wa = segs_by_id[a]["winding"]
        sides = {}
        for key, r in out["pairs"].items():
            x, y = map(int, key.split("-"))
            if x == a and segs_by_id[y]["winding"] is not None and wa is not None:
                sides[segs_by_id[y]["winding"] - wa] = r["major_side"]
        if -1 in sides and 1 in sides and sides[-1] == sides[1]:
            out["order_flips"].append(a)
    return out, dmat


def cast_pairs(surf, ids, rng, n_seeds=2400, cap=400):
    """{(a, b): [(centre, u, d)]} for every instance pair facing each other along a's normal (any windings)."""
    out = {}
    for a in ids:
        for c, u, d, b in ray_hits(surf, a, rng, n_seeds=n_seeds):
            out.setdefault((a, b), []).append((c, u, d))
    return {k: v[:cap] for k, v in out.items()}


def spacing_stats(pairs_s, segs_by_id, ids):
    """Adjacent-winding (|dw| = 1) pair list and pooled spacing statistics (both ray directions pooled)."""
    pairs, alld = [], []
    for a in ids:
        for b in ids:
            wa, wb = segs_by_id[a]["winding"], segs_by_id[b]["winding"]
            if a < b and wa is not None and wb is not None and abs(wa - wb) == 1:
                d = np.array([x[2] for x in pairs_s.get((a, b), []) + pairs_s.get((b, a), [])])
                if len(d) < 30:
                    continue
                alld.append(d)
                cl = [spacing_class(x) for x in d]
                pairs.append({"a": a, "b": b, "wa": wa, "wb": wb, "dw": abs(wa - wb), "n": int(len(d)),
                              "d_median": float(np.median(d)), "d_p10": float(np.percentile(d, 10)),
                              "d_p90": float(np.percentile(d, 90)),
                              **{f"frac_{k}": float(np.mean([c == k for c in cl]))
                                 for k in ("conflict", "compressed", "intermediate", "wide")}})
    if not alld:
        return pairs, {}
    d = np.concatenate(alld)
    comp = (d >= CONFLICT_D) & (d < COMP_MAX)
    return pairs, {"n": int(len(d)), "d_median_vox": float(np.median(d)), "d_p10_p90_vox": np.percentile(d, [10, 90]).tolist(),
                   "d_median_um": float(np.median(d) * VOX9), "frac_conflict": float(np.mean(d < CONFLICT_D)),
                   "frac_compressed": float(np.mean(comp)), "n_compressed": int(comp.sum()),
                   "frac_intermediate": float(np.mean((d >= COMP_MAX) & (d < WIDE_MIN))),
                   "frac_wide": float(np.mean(d >= WIDE_MIN)), "n_wide": int(np.sum(d >= WIDE_MIN))}


def build_site(job):
    cand, segs, A, t = job
    stem = cand["stem"]
    t0 = time.time()
    rec = {"stem": stem, "cls_mined": cand["cls"], "origin_zyx": cand["origin"], "shape": [N] * 3,
           "mined": {k: cand[k] for k in ("frac_comp", "frac_wide", "n_adj", "n_comp", "n_windings", "windings")}}
    try:
        reg = register(cand["origin"], A, t)
        rec |= {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reg.items() if k not in ("ct", "rend")}
        surf_raw, nrm, counts = rasterize(cand["origin"], reg["shift"], segs, A, t)
        segs_by_id = {s["id"]: s for s in segs}
        ids = [int(x) for x in np.unique(surf_raw) if x]
        rec["instances"] = {i: {"segment": segs_by_id[i]["name"], "winding": segs_by_id[i]["winding"],
                                "n_vox_raw": int((surf_raw == i).sum()), "n_samples": counts.get(i, 0)} for i in ids}
        g, dmat = gates(reg["ct"], surf_raw, nrm, segs_by_id, cast_pairs(surf_raw, ids, np.random.default_rng(SEED)))
        rec["gates"] = g
        # clean: drop surface voxels > OFF_D from material (parallel drift into a gap); they go to the ignore mask
        off = (surf_raw > 0) & (dmat > OFF_D)
        surf = np.where(off, 0, surf_raw).astype(np.uint16)
        for i in ids:
            rec["instances"][i]["n_vox"] = int((surf == i).sum())
        rec["cleaning"] = {"removed_vox": int(off.sum()), "removed_frac": float(off.sum() / max(1, (surf_raw > 0).sum()))}
        rec["pairs"], rec["spacing"] = spacing_stats(cast_pairs(surf, ids, np.random.default_rng(SEED)), segs_by_id, ids)
        inst, dist = band_from_surface(surf)
        # coverage: material farther than ~0.7 local pitch from every labelled surface = unlabelled sheet (estimate)
        sp = rec["spacing"].get("d_median_vox", 10.0)
        dmax = float(np.clip(0.7 * sp, 4, 10))
        mat = (ndi.gaussian_filter(reg["ct"].astype(np.float32), 1.0) > g["otsu"]) & (reg["ct"] > 0)
        c = slice(CORE, N - CORE)
        unl = mat & (dist > dmax)
        unl_frac = float(unl[c, c, c].sum() / max(1, mat[c, c, c].sum()))
        rec["coverage"] = {"dmax_vox": dmax, "unlabelled_material_frac": unl_frac, "coverage_ok": unl_frac <= COVERAGE_MAX_UNL}
        ignore = ndi.binary_dilation(unl, iterations=1) | ndi.binary_dilation(off, iterations=2)
        ignore &= inst == 0
        d = WORK / "all"
        for sub in ("imagesTr", "instancesTr", "labelsTr", "surfacesTr", "ignoreTr", "render2p4", "surfacesRaw"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(d / "imagesTr" / f"{stem}_0000.tif", reg["ct"], compression="zlib")
        tifffile.imwrite(d / "instancesTr" / f"{stem}.tif", inst, compression="zlib")
        tifffile.imwrite(d / "labelsTr" / f"{stem}.tif", (inst > 0).astype(np.uint8), compression="zlib")
        tifffile.imwrite(d / "surfacesTr" / f"{stem}.tif", surf, compression="zlib")
        tifffile.imwrite(d / "surfacesRaw" / f"{stem}.tif", surf_raw, compression="zlib")
        tifffile.imwrite(d / "ignoreTr" / f"{stem}.tif", ignore.astype(np.uint8), compression="zlib")
        tifffile.imwrite(d / "render2p4" / f"{stem}.tif", np.clip(reg["rend"], 0, 255).astype(np.uint8), compression="zlib")
        decide(rec)
    except Exception as e:  # log and drop
        import traceback
        rec["passed"] = False
        rec["fail"] = [f"error: {e!r}"]
        rec["traceback"] = traceback.format_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    (WORK / "sites").mkdir(parents=True, exist_ok=True)
    (WORK / "sites" / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def decide(rec):
    """Apply the gates; rec['fail'] lists every reason (empty = pass), rec['fail_gate'] the gate names."""
    fail, names = [], set()

    def f(gate, msg):
        fail.append(f"{gate}: {msg}")
        names.add(gate)

    if not np.isfinite(rec["ncc"]) or rec["ncc"] < NCC_MIN:
        f("ncc", f"{rec['ncc']:.3f} < {NCC_MIN}")
    if np.isfinite(rec["octant_dev_max"]) and rec["octant_dev_max"] > SHIFT_SPREAD_MAX:
        f("ncc", f"octant shift deviates {rec['octant_dev_max']:.2f} > {SHIFT_SPREAD_MAX} vox (one translation does not fit)")
    g = rec["gates"]
    if np.isfinite(g["pooled_on_papyrus"]) and g["pooled_on_papyrus"] < POOLED_ONPAP_MIN:
        f("papyrus", f"pooled on-papyrus {g['pooled_on_papyrus']:.2f} < {POOLED_ONPAP_MIN}")
    strict = True
    for i, r in g["instances"].items():
        if "skipped" in r:
            continue
        w = rec["instances"][i]["winding"]
        strict &= r["on_papyrus"] >= 0.8 and r["n_tiles_off2"] == 0
        if r["on_papyrus"] < INST_ONPAP_MIN:
            f("papyrus", f"w{w} on-papyrus {r['on_papyrus']:.2f} < {INST_ONPAP_MIN}")
        if r["frac_off"] > INST_OFF_MAX:
            f("papyrus", f"w{w} {r['frac_off']:.0%} of voxels > {OFF_D} vox from material (> {INST_OFF_MAX:.0%})")
        if np.isfinite(r["normal_agree"]) and r["normal_agree"] < NORMAL_AGREE_MIN:
            f("crossing", f"w{w} cuts across sheets: normal agreement {r['normal_agree']:.2f} < {NORMAL_AGREE_MIN}")
    deep = {rec["instances"][i]["winding"]: r["n_tiles_deep"] for i, r in g["instances"].items() if r.get("n_tiles_deep")}
    rec["deep_gap_tiles"] = int(sum(deep.values()))
    if rec["deep_gap_tiles"] > DEEP_TILES_MAX:
        f("crossing", f"meshes cross dark gaps in {rec['deep_gap_tiles']} 24^3 tiles (> {DEEP_TILES_MAX}; > 50% of one "
                      f"instance's voxels > {DEEP_D} vox from material): " + ", ".join(f"w{w} x{n}" for w, n in deep.items()))
    rec["strict_papyrus_pass"] = bool(strict)  # every instance >= 0.8 within 1 vox and no 24^3 tile > 50% off by 2 vox
    for key, r in g["pairs"].items():
        a, b = key.split("-")
        wa, wb = rec["instances"][int(a)]["winding"], rec["instances"][int(b)]["winding"]
        if r["minority_side"] > CROSS_MAX:
            f("crossing", f"meshes cross: w{wa}->w{wb} {100 * r['minority_side']:.0f}% on the minority side")
        if r["conflict_frac"] > CROSS_MAX:
            f("crossing", f"meshes coincide: w{wa}->w{wb} {100 * r['conflict_frac']:.0f}% closer than {CONFLICT_D} vox")
    for a in g["order_flips"]:
        f("crossing", f"winding order flip at w{rec['instances'][a]['winding']}")
    if not rec.get("pairs"):
        f("pairs", "no adjacent-winding pair with >= 30 samples after cleaning")
    sp = rec.get("spacing", {})
    if sp and sp["n_compressed"] >= COMP_SITE_N and sp["frac_compressed"] >= COMP_SITE_FRAC:
        rec["cls"] = "compressed"
    elif sp and sp["frac_compressed"] < WIDE_SITE_COMP_MAX and sp["frac_wide"] >= WIDE_SITE_FRAC:
        rec["cls"] = "wide"
    elif sp and sp["n_compressed"] >= MIXED_SITE_N and sp["frac_compressed"] >= WIDE_SITE_COMP_MAX:
        rec["cls"] = "mixed"
    else:
        rec["cls"] = "intermediate"
        f("class", f"neither compressed (>= {COMP_SITE_N} and >= {COMP_SITE_FRAC:.0%} adjacent samples 1-4 vox), mixed "
                   f"(>= {MIXED_SITE_N} and >= {WIDE_SITE_COMP_MAX:.1%}) nor wide "
                   f"(< {WIDE_SITE_COMP_MAX:.1%} compressed, >= {WIDE_SITE_FRAC:.0%} >= 8 vox)")
    rec["fail"] = fail
    rec["fail_gate"] = sorted(names)
    rec["passed"] = not fail


def build(args):
    A, t = affine()
    segs = segments()
    cj = json.loads((WORK / "candidates.json").read_text())
    cands = cj["controls"] + cj["candidates"]
    if args.only:
        cands = [c for c in cands if any(k in c["stem"] for k in args.only.split(","))]
    if args.limit:
        cands = cands[: args.limit]
    todo = [c for c in cands if args.redo or not (WORK / "sites" / f"{c['stem']}.json").exists()]
    print(f"{len(todo)} of {len(cands)} sites to build", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for rec in ex.map(build_site, [(c, segs, A, t) for c in todo]):
            print(f"{rec['stem']}: passed={rec['passed']} ncc={rec.get('ncc', float('nan')):.3f} "
                  f"shift={np.round(rec.get('shift', [np.nan] * 3), 2).tolist()} cls={rec.get('cls')} "
                  f"{rec['seconds']}s fail={rec['fail']}", flush=True)


# ------------------------------------------------------------------------------------------------------- finalize
def finalize(args):
    """Select the final set (all passing compressed sites, wide controls capped at the compressed count), lay it out
    in OUT/{imagesTr,instancesTr,labelsTr,surfacesTr,ignoreTr} and write sites.json with provenance and gate stats."""
    import os
    import shutil

    recs = [json.loads(f.read_text()) for f in sorted((WORK / "sites").glob("*.json"))]
    for r in recs:
        if "gates" in r:
            r["instances"] = {int(k): v for k, v in r["instances"].items()}  # json keys are str
            r["gates"]["order_flips"] = [int(a) for a in r["gates"]["order_flips"]]
            r["gates"]["instances"] = {int(k): v for k, v in r["gates"]["instances"].items()}
            decide(r)  # re-apply the current gate constants to the stored measurements
            r["instances"] = {str(k): v for k, v in r["instances"].items()}
            r["gates"]["instances"] = {str(k): v for k, v in r["gates"]["instances"].items()}
    controls = [r for r in recs if r["cls_mined"] == "control"]
    recs = [r for r in recs if r["cls_mined"] != "control"]
    comp = [r for r in recs if r["passed"] and r["cls"] == "compressed"]
    mixed = [r for r in recs if r["passed"] and r["cls"] == "mixed"]
    wide = [r for r in recs if r["passed"] and r["cls"] == "wide"]
    rng = np.random.default_rng(SEED)
    n_wide = min(len(wide), len(comp) + len(mixed), max(0, 100 - len(comp) - len(mixed)))
    keep_wide = {wide[i]["stem"] for i in rng.permutation(len(wide))[:n_wide]}
    for r in recs:
        r["in_set"] = bool(r["passed"] and (r["cls"] in ("compressed", "mixed") or r["stem"] in keep_wide))
        if r["passed"] and not r["in_set"]:
            r["not_selected"] = "wide control beyond the contact-site count (class balance)"
    for sub in ("imagesTr", "instancesTr", "labelsTr", "surfacesTr", "ignoreTr"):
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for r in recs:
        if not r["in_set"]:
            continue
        for sub, name in (("imagesTr", f"{r['stem']}_0000.tif"), ("instancesTr", f"{r['stem']}.tif"),
                          ("labelsTr", f"{r['stem']}.tif"), ("surfacesTr", f"{r['stem']}.tif"), ("ignoreTr", f"{r['stem']}.tif")):
            os.link(WORK / "all" / sub / name, OUT / sub / name)

    def q(v, qs=(0.1, 0.5, 0.9)):
        v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
        return {f"p{int(100 * k)}": round(float(np.quantile(v, k)), 4) for k in qs} if len(v) else {}

    built = [r for r in recs if "gates" in r]
    gate_names = ("ncc", "papyrus", "crossing", "pairs", "class")
    stats = {
        "candidates": len(recs), "built": len(built), "errors": sum("error" in " ".join(r["fail"]) for r in recs),
        "passed": sum(r["passed"] for r in recs), "in_set": sum(r["in_set"] for r in recs),
        "in_set_by_class": {c: sum(r["in_set"] and r["cls"] == c for r in recs) for c in CLASSES},
        "passed_by_class": {c: sum(r["passed"] and r["cls"] == c for r in recs) for c in CLASSES},
        "by_mined_class": {c: {"candidates": sum(r["cls_mined"] == c for r in recs),
                               "passed": sum(r["cls_mined"] == c and r["passed"] for r in recs),
                               **{f"fail_{g}": sum(r["cls_mined"] == c and g in r.get("fail_gate", []) for r in recs)
                                  for g in gate_names}} for c in ("compressed", "wide")},
        "fail_by_gate": {g: sum(g in r.get("fail_gate", []) for r in recs) for g in gate_names},
        "fail_only_gate": {g: sum(r.get("fail_gate", []) == [g] for r in recs) for g in gate_names},
        "quality_pass_ignoring_class": sum(not (set(r.get("fail_gate", ["x"])) - {"class"}) for r in built),
        "strict_papyrus_pass": sum(r.get("strict_papyrus_pass", False) for r in built),
        "ncc": q([r.get("ncc") for r in built]), "ncc_published_affine": q([r.get("ncc_published") for r in built]),
        "shift_um": q([np.linalg.norm(r["shift"]) * VOX9 for r in built]),
        "octant_dev_vox": q([r.get("octant_dev_max") for r in built]),
        "pooled_on_papyrus": q([r["gates"]["pooled_on_papyrus"] for r in built]),
        "removed_off_papyrus_frac": q([r["cleaning"]["removed_frac"] for r in built]),
        "unlabelled_material_frac_in_set": q([r["coverage"]["unlabelled_material_frac"] for r in recs if r["in_set"]]),
        "coverage_ok_in_set": {c: sum(r["in_set"] and r["cls"] == c and r["coverage"]["coverage_ok"] for r in recs)
                               for c in CLASSES},
        "adjacent_pair_samples_in_set": {k: int(sum(r["spacing"].get(f"n_{k}", 0) for r in recs if r["in_set"]))
                                         for k in ("compressed", "wide")},
        "pairs_in_set": int(sum(len(r["pairs"]) for r in recs if r["in_set"])),
        "download_GB": round(sum(r.get("chunks_9p362", 0) + r.get("chunks_2p399_L2", 0) for r in recs + controls)
                             * 128 ** 3 / 1e9, 2),
        "controls": {r["stem"]: {"passed": r["passed"], "ncc": r.get("ncc"), "shift": r.get("shift"), "fail": r["fail"]}
                     for r in controls},
    }
    A, t = affine()
    cj = json.loads((WORK / "candidates.json").read_text())
    meta = {
        "dataset": "p0139_9um", "created": time.strftime("%Y-%m-%d"), "voxel_um": VOX9, "crop_shape": [N] * 3,
        "licence": "derived from PHerc0139 data (ash2txt data agreement: no redistribution). Local and private.",
        "volume_9p362": VOL9, "level": 0, "volume_2p399_for_ncc": VOL24 + " (level 2 = 4^3 block mean)",
        "transform": {"url": TF_URL, "convention": "p9_xyz = A @ p2399_xyz + t (moving = 2.399 scan)",
                      "A": A.tolist(), "t": t.tolist()},
        "meshes": f"s3://vesuvius-challenge-open-data/PHerc0139/segments/<segment>/mesh/<id>-on-{MESH_TAG}.tifxyz",
        "shift_convention": "per-site translation (9.362 vox, zyx) added to affine-mapped mesh coordinates; CT is not resampled",
        "label_convention": {
            "surfacesTr": "uint16, 1-vox raster of each segment mesh (id = segments[].id), voxels > 2 vox from material removed",
            "instancesTr": "uint16, surfacesTr grown to a band: voxels within 1.0 vox (EDT) of a surface voxel, id of the nearest",
            "labelsTr": "uint8 {0,1} = instancesTr > 0; jeong200 convention (binary, ~3-vox band on the recto face)",
            "ignoreTr": "uint8 1 = removed off-papyrus label voxels (dilated 2) or CT material > max(4, min(10, 0.7 pitch)) vox "
                        "from every labelled surface (unlabelled sheets); 0 elsewhere. Optional for scorers.",
        },
        "site_classes": {"compressed": f">= {COMP_SITE_N} adjacent-pair ray samples at 1-4 vox and >= {COMP_SITE_FRAC:.0%} of them",
                         "mixed": f"not compressed; >= {MIXED_SITE_N} samples at 1-4 vox and >= {WIDE_SITE_COMP_MAX:.1%} (contact zones "
                                  "inside an open crop)",
                         "wide": f"< {WIDE_SITE_COMP_MAX:.1%} samples at 1-4 vox and >= {WIDE_SITE_FRAC:.0%} at >= 8 vox",
                         "note": "site class is a stratum only; scoring stratifies every pair by its own local spacing"},
        "spacing_classes_vox": {"conflict": [0, CONFLICT_D], "compressed": [CONFLICT_D, COMP_MAX],
                                "intermediate": [COMP_MAX, WIDE_MIN], "wide": [WIDE_MIN, 24.0]},
        "gates": {"NCC_MIN": NCC_MIN, "SHIFT_SPREAD_MAX": SHIFT_SPREAD_MAX, "ON_D": ON_D, "OFF_D": OFF_D, "DEEP_D": DEEP_D,
                  "DEEP_TILES_MAX": DEEP_TILES_MAX, "POOLED_ONPAP_MIN": POOLED_ONPAP_MIN, "INST_ONPAP_MIN": INST_ONPAP_MIN,
                  "INST_OFF_MAX": INST_OFF_MAX, "NORMAL_AGREE_MIN": NORMAL_AGREE_MIN, "CROSS_MAX": CROSS_MAX,
                  "COVERAGE_MAX_UNL": COVERAGE_MAX_UNL, "COMP_SITE": [COMP_SITE_N, COMP_SITE_FRAC],
                  "WIDE_SITE": [WIDE_SITE_COMP_MAX, WIDE_SITE_FRAC], "MIXED_SITE": [MIXED_SITE_N, WIDE_SITE_COMP_MAX]},
        "segments": [s | ({"blacklisted": cj["blacklist"][s["name"]]} if s["name"] in cj["blacklist"] else {})
                     for s in segments()],
        "stats": stats,
        "sites": [r for r in recs if r["in_set"]],
        "not_in_set": [{k: r.get(k) for k in ("stem", "cls_mined", "cls", "passed", "fail", "not_selected", "ncc")}
                       for r in recs if not r["in_set"]],
    }
    (OUT / "sites.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(stats, indent=1, default=float))


# ---------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "select", "build", "finalize"])
    ap.add_argument("--n-comp", type=int, default=200)
    ap.add_argument("--n-wide", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    {"mine": mine, "select": select, "build": build, "finalize": finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
