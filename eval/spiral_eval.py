"""Score a spiral fit of PHerc0139 (per-winding tifxyz surfaces, 9.362 um frame of 20250728140407) against the human
PHerc0139 meshes and the frozen p0139 scoring pairs (notes/2026-10-20-spiral-eval.md).

  uv run python eval/spiral_eval.py --fit <dir> [--fit-b <dir>] --z0 3000 --z1 4500 --tol 4 --out <json>
  uv run python eval/spiral_eval.py build-oracle --out <dir> --z0 3000 --z1 4500
  uv run python eval/spiral_eval.py corrupt --src <dir> --out <dir> --kind shift|relabel|merge|drop [--winding W]
        [--sector A0 A1]

Fit format. <dir> is searched recursively for tifxyz surfaces (x.tif, y.tif, z.tif, meta.json; optional mask.tif;
invalid vertices -1, nan or mask 0). Coordinates are 9.362 L0 voxels, xyz. Winding index per surface: meta "winding"
(or "winding_index"), else a combined villa surface (meta "winding_column_ranges" + "component_winding_ids", split into
column blocks), else the last "w<int>" in the directory name. Several surfaces may share a winding index.

Nearest surface: exact point-to-triangle distance (each valid grid quad = 2 triangles; candidates = the 8 quads with
the nearest centroids, per surface). A query point is matched if its nearest surface is within --tol vox; j = that
surface's winding.

Metrics (all with a paired bootstrap, 1000 resamples, seed 1; units = test sites for pairs, angle x z blocks for
vertices; strata contact/open and site class):
 1. pairs (scoring_pairs.csv kind "mix"; "legacy" adjacent normal-ray pairs as a secondary block): global endpoint =
    site origin_zyx + local voxel. correct = both ends matched and |j_B - j_A| == expected. acc_matched (over pairs with
    both ends matched), coverage, acc_all (unmatched = wrong). The 172 pairs that skip a winding absent from their
    site are dropped (labels unreliable, notes/2026-10-20-p0139-label-audit.md).
 2. vertices of the human meshes (2.399 tifxyz -> published affine; stride --stride on the 48 um grid) with z in
    [z0, z1): d = j - s k (s = +-1, the orientation that maximises the mode share). global: offset = mode of d;
    seam: offset = mode of d per (10 deg around the umbilicus x 250 z) bin, because human seams sit at ~140 deg (w<=34)
    and ~250 deg (w>=36) and a fit's own seam can be anywhere, so a correct fit has d piecewise constant in angle.
    switches per 1000 adjacent grid pairs (both matched) where d - offset changes; wrong_area = share of matched
    vertices with d != offset; unmatched share.
 3. villa #2167: per 50-vox z slab the largest empty angular run of w035 (umbilicus 20260102150214 mapped by the
    published affine) is the gap sector; pairs (either end) and vertices inside it are flagged. Every metric is reported
    for all / excl_flag_touch (drop flagged & touching w033-w037) / excl_flag (drop all flagged).

--scroll p1667 (default p0139, unchanged): PHerc1667, fit frame = level 2 of the 2.399 um volume 20251217075048
(9.596 um; p_L2 = p_2399 / 4, no affine). Human labels = the merged human mesh of the full read cut into windings
k 11-41 by eval/p1667_labels.py (data/eval/p1667/labels_merged_v4.npz); umbilicus = data/eval/p1667/
umbilicus_p1667_L2.json (L2 xyz, from the same labeller). No frozen pair sites: the pair block is replaced by ray
pairs (pairs_ray): on rays every --ray-dz z x --ray-dtheta deg from the umbilicus, each labelled winding's crossing is
the median of its vertices within +-3 z and +-dtheta/2 (radius spread > 6 vox = crosses twice: dropped there);
pairs = crossings on the same ray with |dk| <= 3 (dk = 0 excluded); correct = both ends matched and
|j_B - j_A| == |dk|; contact = the two crossings closer than 4 vox for dk = 1; bootstrap units = 30 deg x 250 z blocks.
No #2167 gap (flags empty, so the excl_flag variants equal "all"); site classes are "outside".
  uv run python eval/spiral_eval.py --scroll p1667 --fit <dir> [--fit-b <dir>] --z0 6500 --z1 8000 --tol 4 --out <json>
  uv run python eval/spiral_eval.py build-oracle --scroll p1667 --out <dir> --z0 6500 --z1 8000 [--source segs]

--scroll paris4: PHercParis4 (Scroll 1), fit frame = level 2 of the 2.400 um volume 20260411134726 (9.6 um, no affine).
Reference labels = the organisers' 20260701 winding-group meshes (tifxyz_original, L2 xyz on a 20-vox grid, in
data/dl_paris4/w0701_orig/) with per-vertex k from data/eval/paris4/labels_w0701.npz (data/scratch/third_region/
paris4_labels.py); vertices whose k lies outside their group's named range (seam spill, ~0.5 %) are dropped, so each k
comes from one group. The 20-vox grids are upsampled 4x bilinearly (PARIS4_UP; 5-vox spacing, the spacing of the
PHerc1667 merged-mesh grid; the source is a smooth spiral fit sampled every 20 vox), with each new vertex taking the k of
its nearest source vertex. One surface per k (sid = k). Umbilicus = data/eval/paris4/umbilicus_paris4_L2.json (public,
/ 4). Pairs and flags as p1667 (ray pairs, no #2167 gap). Default --stride 2 (10-vox vertex sample, as PHerc1667).
"""

import argparse
import csv
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "eval" / "p0139_9um"
MESHES = REPO / "data" / "meshes" / "p0139"
MESH_TAG = "20260102150214-2.399um"
TF_LOCAL = REPO / "data" / "eval" / "surface_labels_2026_07_sample" / "transform_0139_2p399_to_9p362.json"
UMB = REPO / "data" / "eval" / "p0139_umbilicus" / "umbilicus_20260102150214.json"
CACHE = REPO / "data" / "scratch" / "spiral_eval" / "cache"
P1667 = REPO / "data" / "eval" / "p1667"
P1667_MERGED = REPO / "data" / "dl_p1667" / "merged_v4_2p399"
P1667_SEGS_TAG = "20251217075048-2.399um"
PARIS4 = REPO / "data" / "eval" / "paris4"
PARIS4_GROUPS = REPO / "data" / "dl_paris4" / "w0701_orig"
SCROLLS = ("p0139", "p1667", "paris4")
PARIS4_UP = 4
SCROLL = "p0139"         # set_scroll(); every p0139 path / output below is unchanged by this switch
BAND = (33, 37)          # windings questioned by villa #2167
GAP_W = 35
NOLAB = -10**6
VARIANTS = ("all", "excl_flag_touch", "excl_flag")
N_BOOT, SEED = 1000, 1


# ------------------------------------------------------------------------------------------------- geometry inputs
def set_scroll(name: str) -> None:
    global SCROLL
    assert name in SCROLLS, name
    SCROLL = name


def ref_path(z0: float, z1: float, stride: int = 2, margin: float = 0.0) -> Path:
    """Cache file of the human vertex reference sample (p0139 names unchanged)."""
    tag = "" if SCROLL == "p0139" else f"{SCROLL}_"
    return CACHE / f"ref_{tag}z{z0:g}-{z1:g}_s{stride}_m{margin:g}_v2.npz"


def affine():
    M = np.array(json.loads(TF_LOCAL.read_text())["transformation_matrix"])
    return M[:, :3], M[:, 3]


class Umbilicus:
    """Public 2.399 umbilicus (xyz control points) mapped to 9.362 with p9 = A p + t; yx(z) by linear interpolation."""

    def __init__(self):
        if SCROLL in ("p1667", "paris4"):   # already in the fit frame (L2 xyz)
            f = P1667 / "umbilicus_p1667_L2.json" if SCROLL == "p1667" else PARIS4 / "umbilicus_paris4_L2.json"
            p = np.array([[q["x"], q["y"], q["z"]] for q in json.loads(f.read_text())["control_points"]], float)
            p = p[np.argsort(p[:, 2])]
            self.z, self.y, self.x = p[:, 2], p[:, 1], p[:, 0]
            return
        A, t = affine()
        c = np.array([[p["x"], p["y"], p["z"]] for p in json.loads(UMB.read_text())["control_points"]], float)
        p = c @ A.T + t
        p = p[np.argsort(p[:, 2])]
        self.z, self.y, self.x = p[:, 2], p[:, 1], p[:, 0]

    def angle(self, zyx: np.ndarray) -> np.ndarray:
        z = zyx[..., 0]
        return np.degrees(np.arctan2(zyx[..., 1] - np.interp(z, self.z, self.y),
                                     zyx[..., 2] - np.interp(z, self.z, self.x))) % 360.0


def read_tifxyz(d: Path) -> tuple[np.ndarray, dict]:
    """(H, W, 3) zyx float32, invalid = nan."""
    g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "zyx"], -1)
    ok = np.isfinite(g).all(-1) & (g >= 0).all(-1) & (g > 0).any(-1)
    if (d / "mask.tif").exists():
        m = tifffile.imread(d / "mask.tif")
        ok &= (m[..., 0] if m.ndim == 3 else m) != 0
    g[~ok] = np.nan
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return g, meta


def write_tifxyz(d: Path, g: np.ndarray, meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(g), g, -1.0).astype(np.float32)
    for k, c in enumerate("zyx"):
        tifffile.imwrite(d / f"{c}.tif", out[..., k])
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "scale": [0.05, 0.05]} | meta, indent=1))


def crop_band(g: np.ndarray, zlo: float, zhi: float) -> np.ndarray:
    """Smallest row/col box holding every vertex with zlo <= z < zhi (others outside the box are dropped, not masked)."""
    z = g[..., 0]
    inb = np.isfinite(z) & (z >= zlo) & (z < zhi)
    if not inb.any():
        return g[:0, :0]
    r, c = np.nonzero(inb.any(1))[0], np.nonzero(inb.any(0))[0]
    return g[r[0]:r[-1] + 2, c[0]:c[-1] + 2]


def discover_fit(fit: Path, zlo: float, zhi: float) -> list[dict]:
    dirs = sorted({p.parent for p in fit.rglob("x.tif") if (p.parent / "z.tif").exists()})
    if not dirs:
        sys.exit(f"no tifxyz surfaces under {fit}")
    out = []
    for d in dirs:
        g, meta = read_tifxyz(d)
        if "winding_column_ranges" in meta and "component_winding_ids" in meta:
            for (a, b), w in zip(meta["winding_column_ranges"], meta["component_winding_ids"]):
                sub = crop_band(g[:, a:min(b + 1, g.shape[1])], zlo, zhi)  # +1 column: keep the seam quad
                if sub.size:
                    out.append({"label": int(w), "grid": sub, "name": f"{d.name}[{a}:{b}]"})
            continue
        w = meta.get("winding", meta.get("winding_index"))
        if w is None:
            m = re.findall(r"w(-?\d+)", d.name) or re.findall(r"w(-?\d+)", d.parent.name)
            if not m:
                sys.exit(f"{d}: no winding index (meta 'winding' or 'w<int>' in the directory name)")
            w = m[-1]
        sub = crop_band(g, zlo, zhi)
        if sub.size:
            out.append({"label": int(w), "grid": sub, "name": str(d.relative_to(fit))})
    return out


_P1667_CACHE = {}


def _p1667_merged():
    """(H, W, 3) zyx float32 L2 voxels (nan invalid) and (H, W) int16 winding labels (-1 none) of the merged mesh."""
    if not _P1667_CACHE:
        g = np.stack([tifffile.imread(P1667_MERGED / f"{c}.tif") for c in "zyx"], -1).astype(np.float32)
        ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
        g /= 4.0
        k = np.load(P1667 / "labels_merged_v4.npz")["k"]
        assert k.shape == ok.shape
        g[~ok | (k < 0)] = np.nan
        _P1667_CACHE["g"], _P1667_CACHE["k"] = g, k
    return _P1667_CACHE["g"], _P1667_CACHE["k"]


def p1667_segments(zlo: float, zhi: float) -> list[dict]:
    """One labelled surface per winding k of the merged PHerc1667 human mesh (sid = k), cropped to [zlo, zhi)."""
    g, k = _p1667_merged()
    z = g[..., 0]
    rows = np.nonzero((np.isfinite(z) & (z >= zlo) & (z < zhi)).any(1))[0]
    if not len(rows):
        return []
    r0, r1 = rows[0], rows[-1] + 2
    gb, kb = g[r0:r1], k[r0:r1]
    out = []
    for kk in np.unique(kb[kb >= 0]):
        m = kb == kk
        cols = np.nonzero(m.any(0))[0]
        sub = gb[:, cols[0]:cols[-1] + 1].copy()
        sub[~m[:, cols[0]:cols[-1] + 1]] = np.nan
        sub = crop_band(sub, zlo, zhi)
        if sub.size:
            out.append({"sid": int(kk), "label": int(kk), "name": f"merged_v4:k{int(kk)}", "blacklisted": False,
                        "grid": sub})
    return out


def _upsample_grid(g: np.ndarray, k: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear f-x upsampling of a (H, W, 3) grid (nan invalid; a new vertex is valid iff its cell's corners are) and
    nearest-source-vertex labels k (-1 where the grid is invalid)."""
    H, W = g.shape[:2]
    if min(H, W) < 2:
        return g, k
    u = np.arange((H - 1) * f + 1) / f
    v = np.arange((W - 1) * f + 1) / f
    i0 = np.minimum(np.floor(u).astype(int), H - 2)
    j0 = np.minimum(np.floor(v).astype(int), W - 2)
    a, b = (u - i0)[:, None, None], (v - j0)[None, :, None]
    out = np.empty((len(u), len(v), 3), np.float32)
    for c0 in range(0, len(u), 256):
        ii, aa = i0[c0:c0 + 256], a[c0:c0 + 256]
        out[c0:c0 + 256] = ((1 - aa) * (1 - b) * g[ii][:, j0] + (1 - aa) * b * g[ii][:, j0 + 1]
                            + aa * (1 - b) * g[ii + 1][:, j0] + aa * b * g[ii + 1][:, j0 + 1])
    kn = k[np.rint(u).astype(int)][:, np.rint(v).astype(int)]
    kn = np.where(np.isfinite(out).all(-1), kn, -1)
    return out, kn


def paris4_segments(zlo: float, zhi: float) -> list[dict]:
    """One labelled surface per winding k of the PHercParis4 20260701 group meshes (sid = k), cropped to [zlo, zhi).
    Group wAAA-BBB keeps only vertices with AAA <= k <= BBB (seam spill dropped)."""
    L = np.load(PARIS4 / "labels_w0701.npz")
    out = []
    for d in sorted(PARIS4_GROUPS.glob("20260701*-w*")):
        m = re.search(r"-w(\d+)-(\d+)$", d.name)
        if not m or not (d / "z.tif").exists():
            continue
        w0, w1 = int(m.group(1)), int(m.group(2))
        g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "zyx"], -1)
        ok = (g >= 0).all(-1) & np.isfinite(g).all(-1) & (g[..., 0] > 0)
        k = L[f"k_{w0:03d}_{w1:03d}"]
        assert k.shape == ok.shape, (d, k.shape, ok.shape)
        ok &= (k >= w0) & (k <= w1)
        g[~ok] = np.nan
        z = g[..., 0]
        rows = np.nonzero((np.isfinite(z) & (z >= zlo) & (z < zhi)).any(1))[0]
        if not len(rows):
            continue
        r0, r1 = max(rows[0] - 1, 0), rows[-1] + 3
        gb, kb = _upsample_grid(g[r0:r1], np.where(ok, k, -1)[r0:r1], PARIS4_UP)
        for kk in np.unique(kb[kb >= 0]):
            mk = kb == kk
            cols = np.nonzero(mk.any(0))[0]
            sub = gb[:, cols[0]:cols[-1] + 1].copy()
            sub[~mk[:, cols[0]:cols[-1] + 1]] = np.nan
            sub = crop_band(sub, zlo, zhi)
            if sub.size:
                out.append({"sid": int(kk), "label": int(kk), "name": f"w0701_{w0:03d}-{w1:03d}:k{int(kk)}",
                            "blacklisted": False, "grid": sub})
    return out


def human_segments(zlo: float, zhi: float) -> list[dict]:
    """sites.json segments with a winding and a 2.399 mesh, mapped to 9.362 (published affine), cropped to the band.
    (--scroll p1667: p1667_segments; --scroll paris4: paris4_segments.)"""
    if SCROLL == "p1667":
        return p1667_segments(zlo, zhi)
    if SCROLL == "paris4":
        return paris4_segments(zlo, zhi)
    A, t = affine()
    meta = json.loads((ROOT / "sites.json").read_text())
    out = []
    for s in meta["segments"]:
        d = MESHES / s["name"] / MESH_TAG
        if s["winding"] is None or not (d / "x.tif").exists():
            continue
        g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
        ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
        p = g @ A.T + t
        p[~ok] = np.nan
        p = p[..., ::-1].astype(np.float32)
        out.append({"sid": s["id"], "label": s["winding"], "name": s["name"], "blacklisted": "blacklisted" in s,
                    "grid": crop_band(p, zlo, zhi)})
    return out


# -------------------------------------------------------------------------------------------- nearest-surface index
def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(_dot(p - a, ab) / np.maximum(_dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - a - t[..., None] * ab, axis=-1)


def tri_dist(p, a, b, c):
    """Exact point-triangle distance (broadcast over leading dims)."""
    ab, ac, ap = b - a, c - a, p - a
    n = np.cross(ab, ac)
    nn = _dot(n, n)
    ok = nn > 1e-12
    nn = np.where(ok, nn, 1.0)
    tt = _dot(ap, n) / nn
    v2 = ap - tt[..., None] * n
    d00, d01, d11 = _dot(ab, ab), _dot(ab, ac), _dot(ac, ac)
    d20, d21 = _dot(v2, ab), _dot(v2, ac)
    den = np.where(ok, d00 * d11 - d01 * d01, 1.0)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = ok & (v >= 0) & (w >= 0) & (v + w <= 1)
    d_in = np.abs(tt) * np.sqrt(nn)
    d_ed = np.minimum(np.minimum(_seg_dist(p, a, b), _seg_dist(p, b, c)), _seg_dist(p, c, a))
    return np.where(inside, d_in, d_ed)


class Surfaces:
    """Per-surface quad sets (valid 2x2 vertex cells) with a centroid KD-tree, restricted to [zlo, zhi]."""

    def __init__(self, items: list[dict], zlo: float, zhi: float, min_edge: float = 2.5, rad_factor: float = 4.0):
        self.surfs, self.stats = [], {"surfaces": 0, "quads": 0, "dropped_quads": 0, "decimation": []}
        for it in items:
            g = it["grid"]
            if min(g.shape[:2]) < 2:
                continue
            e = np.nanmedian(np.concatenate([np.linalg.norm(np.diff(g, axis=0), axis=-1).ravel(),
                                             np.linalg.norm(np.diff(g, axis=1), axis=-1).ravel()]))
            st = int(np.ceil(min_edge / e)) if np.isfinite(e) and e < min_edge else 1
            g = g[::st, ::st]
            c = np.stack([g[:-1, :-1], g[:-1, 1:], g[1:, 1:], g[1:, :-1]], 2)  # (H-1, W-1, 4, 3)
            ok = np.isfinite(c).all((-1, -2))
            cz = np.where(ok[..., None], c[..., 0], 0.0)
            ok &= (cz.max(-1) >= zlo) & (cz.min(-1) <= zhi)
            c = c[ok]
            if not len(c):
                continue
            cen = c.mean(1)
            rad = np.linalg.norm(c - cen[:, None], axis=-1).max(1)
            keep = rad <= rad_factor * max(np.median(rad), 0.5)
            self.stats["dropped_quads"] += int((~keep).sum())
            c, cen, rad = c[keep], cen[keep], rad[keep]
            self.surfs.append({"label": it["label"], "sid": it.get("sid", len(self.surfs)), "corners": c,
                               "tree": cKDTree(cen), "rmax": float(rad.max()), "lo": cen.min(0), "hi": cen.max(0)})
            self.stats["surfaces"] += 1
            self.stats["quads"] += len(c)
            self.stats["decimation"].append(st)
        self.stats["decimation"] = int(max(self.stats["decimation"], default=1))
        self.stats["windings"] = sorted({s["label"] for s in self.surfs})

    def query(self, pts: np.ndarray, tol: float, own_sid: np.ndarray | None = None, k: int = 8,
              chunk: int = 60000) -> tuple[np.ndarray, np.ndarray]:
        """(label, distance) of the nearest surface within tol (label NOLAB, distance inf otherwise). own_sid: per-point
        surface id to ignore (distance to the nearest *other* surface)."""
        best_d = np.full(len(pts), np.inf)
        best_l = np.full(len(pts), NOLAB, np.int64)
        pts = np.asarray(pts, np.float64)
        for s in self.surfs:
            r = tol + s["rmax"]
            m = ((pts >= s["lo"] - r) & (pts <= s["hi"] + r)).all(1)
            if own_sid is not None:
                m &= own_sid != s["sid"]
            idx = np.nonzero(m)[0]
            n = len(s["corners"])
            for i0 in range(0, len(idx), chunk):
                sub = idx[i0:i0 + chunk]
                _, ii = s["tree"].query(pts[sub], k=min(k, n), distance_upper_bound=r, workers=-1)
                ii = ii.reshape(len(sub), -1)
                has = ii[:, 0] < n
                sub, ii = sub[has], ii[has]
                if not len(sub):
                    continue
                val = ii < n
                C = s["corners"][np.where(val, ii, 0)].astype(np.float64)  # (N, k, 4, 3)
                p = pts[sub][:, None, :]
                d = np.minimum(tri_dist(p, C[:, :, 0], C[:, :, 1], C[:, :, 2]),
                               tri_dist(p, C[:, :, 0], C[:, :, 2], C[:, :, 3]))
                d = np.where(val, d, np.inf).min(1)
                upd = d < best_d[sub]
                best_d[sub[upd]] = d[upd]
                best_l[sub[upd]] = s["label"]
        miss = best_d > tol
        best_l[miss] = NOLAB
        return best_l, best_d


# --------------------------------------------------------------------------------------------------- villa #2167
def gap_sectors(segs: list[dict], umb: Umbilicus, z0: float, z1: float, slab: float = 50.0, min_gap: float = 20.0):
    """Per z slab: the largest circular run of 1-degree bins with no w035 vertex (start, width in degrees)."""
    g = np.concatenate([s["grid"][np.isfinite(s["grid"]).all(-1)] for s in segs if s["label"] == GAP_W])
    edges = np.arange(np.floor(z0 / slab) * slab, z1 + slab, slab)
    out = []
    th = umb.angle(g)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (g[:, 0] >= a) & (g[:, 0] < b)
        if sel.sum() < 100:
            continue
        occ = np.bincount(th[sel].astype(int) % 360, minlength=360) > 0
        if occ.all():
            continue
        roll = int(np.argmax(occ))  # start the scan on an occupied bin so no run wraps
        e = ~np.roll(occ, -roll)
        best, cur, start, bs = 0, 0, 0, 0
        for i, v in enumerate(e):
            if v:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best:
                    best, bs = cur, start
            else:
                cur = 0
        if best >= min_gap:
            out.append({"z0": float(a), "z1": float(b), "start": float((bs + roll) % 360), "width": float(best)})
    return out


def in_gap(zyx: np.ndarray, ang: np.ndarray, gaps: list[dict], margin: float) -> np.ndarray:
    f = np.zeros(len(zyx), bool)
    for g in gaps:
        sel = (zyx[:, 0] >= g["z0"]) & (zyx[:, 0] < g["z1"])
        rel = (ang[sel] - (g["start"] - margin)) % 360.0
        f[np.nonzero(sel)[0]] = rel < g["width"] + 2 * margin
    return f


# ------------------------------------------------------------------------------------------------------- inputs
def load_pairs(z0: float, z1: float, pair_frame: str) -> dict:
    meta = json.loads((ROOT / "sites.json").read_text())
    sites = {s["stem"]: s for s in meta["sites"]}
    with (ROOT / "scoring_pairs.csv").open() as f:
        rows = list(csv.DictReader(f))
    stem = np.array([r["stem"] for r in rows])
    org = np.array([sites[s]["origin_zyx"] for s in stem], float)
    if pair_frame == "affine":  # diagnostic: undo the per-site NCC shift (frame of the affine-mapped meshes)
        org = org - np.array([sites[s]["shift"] for s in stem], float)
    A = org + np.array([[float(r["za"]), float(r["ya"]), float(r["xa"])] for r in rows])
    B = org + np.array([[float(r["zb"]), float(r["yb"]), float(r["xb"])] for r in rows])
    wa = np.array([int(r["wa"]) for r in rows])
    wb = np.array([int(r["wb"]) for r in rows])
    inst = {s: {v["winding"] for v in sites[s]["instances"].values()} for s in sites}
    skip = np.array([any(w not in inst[s] for w in range(min(a, b) + 1, max(a, b))) for s, a, b in zip(stem, wa, wb)])
    inb = (np.minimum(A[:, 0], B[:, 0]) >= z0) & (np.maximum(A[:, 0], B[:, 0]) < z1)
    kind = np.array([r["kind"] for r in rows])
    return {"stem": stem, "A": A, "B": B, "wa": wa, "wb": wb, "expected": np.array([int(r["expected"]) for r in rows]),
            "contact": np.array([r["contact"] == "True" for r in rows]), "cls": np.array([r["site_cls"] for r in rows]),
            "kind": kind, "skip": skip, "inb": inb, "row": np.arange(len(rows))}


def sample_vertices(segs: list[dict], z0: float, z1: float, stride: int) -> dict:
    pts, lab, sid, rc, adj_a, adj_b = [], [], [], [], [], []
    off = 0
    for s in segs:
        if s["blacklisted"]:
            continue
        g = s["grid"][::stride, ::stride]
        ok = np.isfinite(g).all(-1) & (g[..., 0] >= z0) & (g[..., 0] < z1)
        ind = np.full(ok.shape, -1, np.int64)
        ind[ok] = off + np.arange(ok.sum())
        for a, b in ((ind[:-1], ind[1:]), (ind[:, :-1], ind[:, 1:])):
            m = (a >= 0) & (b >= 0)
            adj_a.append(a[m])
            adj_b.append(b[m])
        pts.append(g[ok])
        lab.append(np.full(ok.sum(), s["label"]))
        sid.append(np.full(ok.sum(), s["sid"]))
        off += int(ok.sum())
    return {"pts": np.concatenate(pts).astype(np.float64), "k": np.concatenate(lab), "sid": np.concatenate(sid),
            "adj": np.stack([np.concatenate(adj_a), np.concatenate(adj_b)], 1)}


def ray_pairs(segs: list[dict], umb: Umbilicus, z0: float, z1: float, dz: float, dth: float, maxdk: int = 3) -> dict:
    """Ray pairs for scrolls without frozen pair sites (--scroll p1667; see the module docstring). Returned in the
    load_pairs layout (stem = bootstrap block, cls = 'outside', kind = 'ray', skip/inb as for the frozen pairs)."""
    zl = np.arange(z0 + dz / 2, z1, dz)
    nth = int(round(360 / dth))
    cross = {}                                   # (iz, ia) -> list of (k, r, zyx)
    for s in segs:
        if s["blacklisted"]:
            continue
        g = s["grid"][np.isfinite(s["grid"]).all(-1)].astype(np.float64)
        if not len(g):
            continue
        iz = np.floor((g[:, 0] - z0) / dz).astype(np.int64)
        near = np.abs(g[:, 0] - (z0 + (iz + 0.5) * dz)) <= 3.0
        a = umb.angle(g)
        ia = np.floor(a / dth).astype(np.int64) % nth
        near &= np.abs(((a - (ia + 0.5) * dth) + 180) % 360 - 180) <= dth / 2
        ok = near & (iz >= 0) & (iz < len(zl))
        g, iz, ia = g[ok], iz[ok], ia[ok]
        uy, ux = np.interp(g[:, 0], umb.z, umb.y), np.interp(g[:, 0], umb.z, umb.x)
        r = np.hypot(g[:, 1] - uy, g[:, 2] - ux)
        key = iz * nth + ia
        o = np.argsort(key, kind="stable")
        key, g, r = key[o], g[o], r[o]
        u, st = np.unique(key, return_index=True)
        for kk, a0, a1 in zip(u, st, np.r_[st[1:], len(key)]):
            rr = r[a0:a1]
            if a1 - a0 < 2 or np.ptp(rr) > 6.0:   # crosses the ray twice (or too few points): not used here
                continue
            i = a0 + int(np.argsort(rr)[len(rr) // 2])
            cross.setdefault(int(kk), []).append((s["label"], float(rr.mean()), g[i]))
    A, B, wa, wb, stem, cont = [], [], [], [], [], []
    for kk, lst in cross.items():
        lst.sort(key=lambda t: t[1])
        if len({t[0] for t in lst}) != len(lst):  # two surfaces with one label on a ray: drop the ray
            continue
        iz, ia = divmod(kk, nth)
        blk = f"z{int(np.floor((z0 + (iz + 0.5) * dz) / 250.0))}_a{int((ia + 0.5) * dth // 30)}"
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                dk = abs(lst[j][0] - lst[i][0])
                if 1 <= dk <= maxdk:
                    A.append(lst[i][2]); B.append(lst[j][2]); wa.append(lst[i][0]); wb.append(lst[j][0])
                    stem.append(blk)
                    cont.append(dk == 1 and abs(lst[j][1] - lst[i][1]) < 4.0)
    n = len(A)
    wa, wb = np.array(wa, int), np.array(wb, int)
    return {"stem": np.array(stem), "A": np.array(A, float).reshape(-1, 3), "B": np.array(B, float).reshape(-1, 3),
            "wa": wa, "wb": wb, "expected": np.abs(wb - wa), "contact": np.array(cont, bool),
            "cls": np.full(n, "outside"), "kind": np.full(n, "ray"), "skip": np.zeros(n, bool),
            "inb": np.ones(n, bool), "row": np.arange(n)}


def site_class_of(pts: np.ndarray) -> np.ndarray:
    if SCROLL != "p0139":
        return np.full(len(pts), "outside")
    meta = json.loads((ROOT / "sites.json").read_text())
    out = np.full(len(pts), "outside", dtype=object)
    for s in meta["sites"]:
        o = np.array(s["origin_zyx"], float)
        m = ((pts >= o) & (pts < o + np.array(s["shape"] if "shape" in s else [192] * 3))).all(1)
        out[m] = s["cls"]
    return out.astype(str)


def reference(args, segs: list[dict], umb: Umbilicus, gaps: list[dict]) -> dict:
    """Human vertex sample + contact + flags + bins (cached; independent of the fit)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = ref_path(args.z0, args.z1, args.stride, args.gap_margin)
    if key.exists() and not args.no_cache:
        return dict(np.load(key, allow_pickle=False))
    v = sample_vertices(segs, args.z0, args.z1, args.stride)
    hs = Surfaces(segs, args.z0 - 10, args.z1 + 10)
    _, dother = hs.query(v["pts"], 4.0, own_sid=v["sid"])
    ang = umb.angle(v["pts"])
    v |= {"contact": dother < 4.0, "ang": ang, "flag": in_gap(v["pts"], ang, gaps, args.gap_margin),
          "cls": site_class_of(v["pts"])}
    np.savez(key, **v)
    return v


# ---------------------------------------------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, units: np.ndarray):
        self.names, self.u = np.unique(units, return_inverse=True)
        self.idx = np.random.default_rng(SEED).integers(0, len(self.names), (N_BOOT, len(self.names)))

    def ratio(self, num: np.ndarray, den: np.ndarray, mask: np.ndarray):
        """Point estimate and per-resample ratios of sum(num[mask]) / sum(den[mask]) over resampled units."""
        nu = np.bincount(self.u[mask], num[mask].astype(float), len(self.names))
        de = np.bincount(self.u[mask], den[mask].astype(float), len(self.names))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (nu.sum() / de.sum() if de.sum() else np.nan), nu[self.idx].sum(1) / de[self.idx].sum(1)


def summ(est, res, scale=100.0):
    lo, hi = np.nanpercentile(res, [2.5, 97.5]) if np.isfinite(res).any() else (np.nan, np.nan)
    f = lambda x: None if not np.isfinite(x) else round(float(x) * scale, 3)
    return {"est": f(est), "ci": [f(lo), f(hi)]}


# ----------------------------------------------------------------------------------------------------- scoring
def pair_metrics(P: dict, sel: np.ndarray, fits: list[dict], strata: dict, boot: Boot, variants: dict) -> dict:
    """fits: [{jA, jB, kA_seam, kB_seam}] (one or two). Returns per variant/stratum metrics (+ B - A if two)."""
    exp = P["expected"]
    per = []
    for f in fits:
        mA, mB = f["jA"] != NOLAB, f["jB"] != NOLAB
        both = mA & mB
        corr = both & (np.abs(f["jB"] - f["jA"]) == exp)
        corr_s = both & (np.abs(f["kB_seam"] - f["kA_seam"]) == exp)
        one = np.ones(len(exp))
        per.append({"acc_matched": (corr, both), "coverage": (both, one), "acc_all": (corr, one),
                    "acc_matched_seam": (corr_s, both), "acc_all_seam": (corr_s, one)})
    out = {}
    for vn, vm in variants.items():
        out[vn] = {}
        for sn, smask in strata.items():
            m = sel & vm & smask
            row = {"n": int(m.sum())}
            res_by = []
            for i, mets in enumerate(per):
                tag = "" if i == 0 else "_b"
                rr = {}
                for name, (num, den) in mets.items():
                    est, res = boot.ratio(num, den, m)
                    row[name + tag] = summ(est, res)
                    rr[name] = (est, res)
                res_by.append(rr)
            if len(per) == 2:
                for name in per[0]:
                    (ea, ra), (eb, rb) = res_by[0][name], res_by[1][name]
                    row[name + "_diff"] = summ(eb - ea, rb - ra)
            out[vn][sn] = row
    return out


def vertex_scores(V: dict, lab: np.ndarray, zbin: float = 250.0, abin: float = 10.0) -> dict:
    """Orientation, offsets and residuals of one fit on the human vertex sample."""
    m = lab != NOLAB
    best = None
    for s in (1, -1):
        d = lab - s * V["k"]
        vals, cnt = np.unique(d[m], return_counts=True)
        if len(vals) and (best is None or cnt.max() > best[2]):
            best = (s, int(vals[np.argmax(cnt)]), int(cnt.max()))
    s, c = (best[0], best[1]) if best else (1, 0)
    d = lab - s * V["k"]
    b = (np.floor(V["pts"][:, 0] / zbin).astype(np.int64) * 1000 + (V["ang"] // abin).astype(np.int64))
    off = np.full(len(d), c)
    ub, inv = np.unique(b, return_inverse=True)
    bin_off = {}
    for i in range(len(ub)):
        mm = (inv == i) & m
        if mm.sum() >= 50:
            vals, cnt = np.unique(d[mm], return_counts=True)
            off[inv == i] = vals[np.argmax(cnt)]
            bin_off[int(ub[i])] = int(vals[np.argmax(cnt)])
    return {"sign": s, "global_offset": c, "d": d, "res_g": d - c, "res_s": d - off, "matched": m,
            "bin_off": bin_off, "zbin": zbin, "abin": abin}


def seam_k(j: np.ndarray, zyx: np.ndarray, ang: np.ndarray, vs: dict) -> np.ndarray:
    """Fit winding mapped into the human label frame with the per-bin offset (falls back to the global offset)."""
    b = np.floor(zyx[:, 0] / vs["zbin"]).astype(np.int64) * 1000 + (ang // vs["abin"]).astype(np.int64)
    off = np.array([vs["bin_off"].get(int(x), vs["global_offset"]) for x in b])
    k = vs["sign"] * (j - off)
    return np.where(j == NOLAB, NOLAB, k)


def vertex_metrics(V: dict, fits: list[dict], strata: dict, boot_v: Boot, variants: dict) -> dict:
    a, b = V["adj"][:, 0], V["adj"][:, 1]
    n = len(V["k"])
    per = []
    for vs in fits:
        m = vs["matched"]
        pm = m[a] & m[b]
        sw_g = pm & (vs["res_g"][a] != vs["res_g"][b])
        sw_s = pm & (vs["res_s"][a] != vs["res_s"][b])
        per.append({"vert": {"unmatched": (~m, np.ones(n)), "wrong_area_global": (m & (vs["res_g"] != 0), m),
                             "wrong_area_seam": (m & (vs["res_s"] != 0), m)},
                    "adj": {"switch_per1000_global": (sw_g * 1000.0, pm), "switch_per1000_seam": (sw_s * 1000.0, pm)}})
    out = {}
    for vn, vm in variants.items():
        out[vn] = {}
        for sn, smask in strata.items():
            mv = vm & smask
            ma = mv[a] & mv[b]
            row = {"n_vertices": int(mv.sum()), "n_adjacent": int(ma.sum())}
            res_by = []
            for i, mets in enumerate(per):
                tag = "" if i == 0 else "_b"
                rr = {}
                for name, (num, den) in mets["vert"].items():
                    est, res = boot_v.ratio(num, den, mv)
                    row[name + tag] = summ(est, res)
                    rr[name] = (est, res, 100.0)
                for name, (num, den) in mets["adj"].items():
                    est, res = boot_v.ratio_adj(num, den, ma)
                    row[name + tag] = summ(est, res, 1.0)
                    rr[name] = (est, res, 1.0)
                res_by.append(rr)
            if len(per) == 2:
                for name in res_by[0]:
                    (ea, ra, sc), (eb, rb, _) = res_by[0][name], res_by[1][name]
                    row[name + "_diff"] = summ(eb - ea, rb - ra, sc)
            out[vn][sn] = row
    return out


class BootV(Boot):
    """Vertex bootstrap over spatial blocks (30 deg x 250 z); adjacent pairs are assigned to their first vertex's block."""

    def __init__(self, V: dict):
        blk = np.floor(V["pts"][:, 0] / 250.0).astype(np.int64) * 100 + (V["ang"] // 30.0).astype(np.int64)
        super().__init__(blk)
        self.ua = self.u[V["adj"][:, 0]]

    def ratio_adj(self, num, den, mask):
        nu = np.bincount(self.ua[mask], num[mask].astype(float), len(self.names))
        de = np.bincount(self.ua[mask], den[mask].astype(float), len(self.names))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (nu.sum() / de.sum() if de.sum() else np.nan), nu[self.idx].sum(1) / de[self.idx].sum(1)


def peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)


def score(args) -> None:
    t0 = time.time()
    zlo, zhi = args.z0 - args.tol - 2, args.z1 + args.tol + 2
    umb = Umbilicus()
    segs = human_segments(zlo - 50, zhi + 50)
    gaps = gap_sectors(segs, umb, args.z0, args.z1) if SCROLL == "p0139" else []
    V = reference(args, segs, umb, gaps)
    if SCROLL != "p0139":
        P = ray_pairs(segs, umb, args.z0, args.z1, args.ray_dz, args.ray_dtheta)
    del segs
    print(f"[ref] {len(V['k'])} human vertices (stride {args.stride}), {len(V['adj'])} adjacent pairs, "
          f"{V['flag'].mean() * 100:.1f}% in the #2167 gap, {V['contact'].mean() * 100:.1f}% contact "
          f"({time.time() - t0:.0f}s)", flush=True)
    if SCROLL == "p0139":
        P = load_pairs(args.z0, args.z1, args.pair_frame)
    angA, angB = umb.angle(P["A"]), umb.angle(P["B"])
    P["flag"] = in_gap(P["A"], angA, gaps, args.gap_margin) | in_gap(P["B"], angB, gaps, args.gap_margin)
    P["touch"] = (np.minimum(P["wa"], P["wb"]) <= BAND[1]) & (np.maximum(P["wa"], P["wb"]) >= BAND[0])
    kinds = (["mix", "legacy"] if args.legacy else ["mix"]) if SCROLL == "p0139" else ["ray"]
    q = P["inb"] & np.isin(P["kind"], kinds)
    qpts = np.concatenate([P["A"][q], P["B"][q]])

    fit_info, vfits, pfits = [], [], []
    for fd in [args.fit] + ([args.fit_b] if args.fit_b else []):
        t1 = time.time()
        S = Surfaces(discover_fit(fd, zlo, zhi), zlo, zhi, args.min_edge)
        lab_v, _ = S.query(V["pts"], args.tol)
        lab_p, dist_p = S.query(qpts, args.tol)
        del S.surfs
        vs = vertex_scores(V, lab_v)
        jA = np.full(len(P["stem"]), NOLAB)
        jB = jA.copy()
        nq = int(q.sum())
        jA[q], jB[q] = lab_p[:nq], lab_p[nq:]
        pf = {"jA": jA, "jB": jB, "kA_seam": seam_k(jA, P["A"], angA, vs), "kB_seam": seam_k(jB, P["B"], angB, vs)}
        vfits.append(vs)
        pfits.append(pf)
        fit_info.append({"path": str(fd), **S.stats, "orientation_sign": vs["sign"], "global_offset": vs["global_offset"],
                         "offset_bins": len(vs["bin_off"]),
                         "offset_bins_off_global": int(sum(v != vs["global_offset"] for v in vs["bin_off"].values())),
                         "seconds": round(time.time() - t1, 1)})
        print(f"[fit] {fd}: {S.stats['surfaces']} surfaces, windings {S.stats['windings'][:3]}..{S.stats['windings'][-3:]}, "
              f"{S.stats['quads']} quads, sign {vs['sign']}, offset {vs['global_offset']} ({time.time() - t1:.0f}s)",
              flush=True)
        if args.dump_pairs:
            dp = Path(args.dump_pairs if len(pfits) == 1 else str(args.dump_pairs).replace(".csv", "_b.csv"))
            with dp.open("w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["row", "stem", "kind", "wa", "wb", "expected", "contact", "flag", "touch", "skip", "jA", "jB",
                             "kA_seam", "kB_seam", "angA"])
                for i in np.nonzero(q)[0]:
                    wr.writerow([i, P["stem"][i], P["kind"][i], P["wa"][i], P["wb"][i], P["expected"][i],
                                 int(P["contact"][i]), int(P["flag"][i]), int(P["touch"][i]), int(P["skip"][i]),
                                 jA[i], jB[i], pf["kA_seam"][i], pf["kB_seam"][i], round(float(angA[i]), 1)])

    out = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
           "scroll": SCROLL, "fits": fit_info,
           "gap2167": {"slabs": len(gaps), "z_range": [min((g["z0"] for g in gaps), default=None),
                                                       max((g["z1"] for g in gaps), default=None)],
                       "width_deg_median": float(np.median([g["width"] for g in gaps])) if gaps else None,
                       "start_deg_median": float(np.median([g["start"] for g in gaps])) if gaps else None,
                       "margin_deg": args.gap_margin, "per_slab": gaps}}
    # pairs
    for kind in kinds:
        km = q & (P["kind"] == kind)
        dropped = km & P["skip"]
        sel = km & ~P["skip"]
        boot = Boot(P["stem"][sel]) if sel.any() else None
        if boot is None:
            continue
        # Boot units are indexed on the selected rows; expand to full length with a dummy unit for unselected rows
        full_u = np.zeros(len(sel), np.int64)
        full_u[sel] = boot.u
        boot.u = full_u
        e = P["expected"]
        strata = {"all": np.ones(len(sel), bool), "contact": P["contact"], "open": ~P["contact"],
                  "compressed": P["cls"] == "compressed", "mixed": P["cls"] == "mixed", "wide": P["cls"] == "wide",
                  "step0": e == 0, "step1": e == 1, "step2plus": e >= 2}
        variants = {"all": np.ones(len(sel), bool), "excl_flag_touch": ~(P["flag"] & P["touch"]), "excl_flag": ~P["flag"]}
        out[f"pairs_{kind}"] = {"n_in_band": int(km.sum()), "dropped_skip": int(dropped.sum()),
                                "dropped_skip_total_csv": int(((P["kind"] == kind) & P["skip"]).sum()),
                                "sites": int(len(boot.names)), "flagged": int((sel & P["flag"]).sum()),
                                "flagged_touch": int((sel & P["flag"] & P["touch"]).sum()),
                                "metrics": pair_metrics(P, sel, pfits, strata, boot, variants)}
    # vertices
    bv = BootV(V)
    touch_v = (V["k"] >= BAND[0]) & (V["k"] <= BAND[1])
    vstrata = {"all": np.ones(len(V["k"]), bool), "contact": V["contact"], "open": ~V["contact"]}
    for c in ("compressed", "mixed", "wide"):
        vstrata[f"site_{c}"] = V["cls"] == c
    vvariants = {"all": np.ones(len(V["k"]), bool), "excl_flag_touch": ~(V["flag"] & touch_v), "excl_flag": ~V["flag"]}
    out["vertices"] = {"n": int(len(V["k"])), "stride": args.stride, "blocks": int(len(bv.names)),
                       "flagged": int(V["flag"].sum()), "flagged_touch": int((V["flag"] & touch_v).sum()),
                       "metrics": vertex_metrics(V, vfits, vstrata, bv, vvariants)}
    out["runtime_s"] = round(time.time() - t0, 1)
    out["peak_rss_gb"] = peak_gb()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    report(out)


def report(out: dict) -> None:
    two = len(out["fits"]) == 2
    f = lambda r: "n/a" if r["est"] is None else f"{r['est']:.2f} [{r['ci'][0]}, {r['ci'][1]}]"
    for key in [k for k in out if k.startswith("pairs_")]:
        p = out[key]
        print(f"\n{key}: {p['n_in_band']} in band, {p['dropped_skip']} skip pairs dropped, {p['sites']} sites, "
              f"{p['flagged']} flagged ({p['flagged_touch']} touching w033-w037)")
        for vn in VARIANTS:
            for sn in ("all", "contact", "open"):
                r = p["metrics"][vn][sn]
                s = (f"  {vn:16s} {sn:8s} n={r['n']:6d} acc_matched {f(r['acc_matched'])}  coverage {f(r['coverage'])}  "
                     f"acc_all {f(r['acc_all'])}  acc_matched_seam {f(r['acc_matched_seam'])}")
                if two:
                    s += f"\n{'':36s}B-A: acc_matched {f(r['acc_matched_diff'])} coverage {f(r['coverage_diff'])} " \
                         f"acc_all {f(r['acc_all_diff'])}"
                print(s)
    v = out["vertices"]
    print(f"\nvertices: {v['n']} (stride {v['stride']}), {v['blocks']} blocks, {v['flagged']} flagged")
    for vn in VARIANTS:
        for sn in ("all", "contact", "open"):
            r = v["metrics"][vn][sn]
            s = (f"  {vn:16s} {sn:8s} n={r['n_vertices']:8d} switch/1000 g {f(r['switch_per1000_global'])} "
                 f"s {f(r['switch_per1000_seam'])}  wrong% g {f(r['wrong_area_global'])} s {f(r['wrong_area_seam'])}  "
                 f"unmatched% {f(r['unmatched'])}")
            if two:
                s += f"\n{'':36s}B-A: switch g {f(r['switch_per1000_global_diff'])} wrong g " \
                     f"{f(r['wrong_area_global_diff'])} unmatched {f(r['unmatched_diff'])}"
            print(s)
    print(f"\nruntime {out['runtime_s']} s, peak RSS {out['peak_rss_gb']} GB")


# ----------------------------------------------------------------------------------------------------- builders
def build_oracle(args) -> None:
    """Human meshes as a 'fit': one tifxyz per segment (all segments with a winding, blacklisted included).
    source 2um: the 48 um 2.399 meshes mapped by the published affine (the same grids the vertex sample comes from);
    9um: the public 187 um 9.362 meshes (independent, 4x coarser grids of the same segments)."""
    out = Path(args.out)
    if SCROLL == "p1667" and args.source == "segs":
        # independent human tracings: the single-wrap per-winding segments (winding = their name), L2 frame
        import re as _re
        for d in sorted((REPO / "data" / "meshes").glob("*-w0*_flatboi")):
            w = int(_re.search(r"-w(\d+)_", d.name).group(1))
            if w in (13, 18, 23):          # multi-wrap segments carry no per-wrap label
                continue
            g = np.stack([tifffile.imread(d / P1667_SEGS_TAG / f"{c}.tif").astype(np.float32) for c in "zyx"], -1)
            ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
            g /= 4.0
            g[~ok] = np.nan
            g = crop_band(g, args.z0 - args.pad, args.z1 + args.pad)
            if g.size:
                write_tifxyz(out / f"w{w:03d}_seg", g, {"winding": w, "source": f"p1667 segment {d.name}"})
        print(f"oracle (p1667 single-wrap segments) written to {out}")
        return
    if args.source == "9um":
        meta = json.loads((ROOT / "sites.json").read_text())
        for s in meta["segments"]:
            d = MESHES / s["name"] / "20250728140407-9.362um"
            if s["winding"] is None or not (d / "x.tif").exists():
                continue
            g = crop_band(read_tifxyz(d)[0], args.z0 - args.pad, args.z1 + args.pad)
            if g.size:
                write_tifxyz(out / f"w{s['winding']:03d}_s{s['id']:02d}", g,
                             {"winding": s["winding"], "source": f"oracle9:{s['name']}"})
        print(f"oracle (9um meshes) written to {out}")
        return
    for s in human_segments(args.z0 - args.pad, args.z1 + args.pad):
        if s["grid"].size:
            write_tifxyz(out / f"w{s['label']:03d}_s{s['sid']:02d}", s["grid"],
                         {"winding": s["label"], "source": f"oracle:{s['name']}"})
    print(f"oracle written to {out}")


def corrupt(args) -> None:
    """Controlled corruptions of a per-surface fit (meta 'winding'):
    shift: every winding +1. relabel: winding W inside the sector -> W+1 (split surface). merge: inside the sector W and
    W+1 become one index (W) and every winding > W+1 is decremented. drop: W's vertices in a 180 deg range removed."""
    src, out = Path(args.src), Path(args.out)
    umb = Umbilicus()
    a0, a1 = args.sector
    for d in sorted({p.parent for p in src.rglob("x.tif")}):
        g, meta = read_tifxyz(d)
        w = int(meta["winding"])
        name = d.name
        ang = umb.angle(np.nan_to_num(g))
        insec = ((ang - a0) % 360.0) < ((a1 - a0) % 360.0)
        parts = []
        if args.kind == "shift":
            parts = [(name, g, w + 1)]
        elif args.kind == "relabel" and w == args.winding:
            parts = [(name, np.where(insec[..., None], np.nan, g), w), (name + "_relab", np.where(insec[..., None], g, np.nan), w + 1)]
        elif args.kind == "merge" and w > args.winding:
            parts = [(name, np.where(insec[..., None], np.nan, g), w), (name + "_merge", np.where(insec[..., None], g, np.nan), w - 1)]
        elif args.kind == "drop" and w == args.winding:
            half = ((ang - a0) % 360.0) < 180.0
            parts = [(name, np.where(half[..., None], np.nan, g), w)]
        else:
            parts = [(name, g, w)]
        for nm, gg, ww in parts:
            if np.isfinite(gg).all(-1).sum() >= 4:
                write_tifxyz(out / nm, gg, meta | {"winding": int(ww), "corruption": args.kind})
    print(f"{args.kind} corruption of {src} written to {out}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("build-oracle", "corrupt"):
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd", required=True)
        b = sub.add_parser("build-oracle")
        b.add_argument("--scroll", choices=list(SCROLLS), default="p0139")
        b.add_argument("--out", type=Path, required=True)
        b.add_argument("--z0", type=float, default=3000)
        b.add_argument("--z1", type=float, default=4500)
        b.add_argument("--pad", type=float, default=40)
        b.add_argument("--source", choices=["2um", "9um", "segs"], default="2um",
                       help="p0139: 2um / 9um meshes; p1667: 2um = merged labelled mesh, segs = single-wrap segments")
        c = sub.add_parser("corrupt")
        c.add_argument("--scroll", choices=list(SCROLLS), default="p0139")
        c.add_argument("--src", type=Path, required=True)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--kind", choices=["shift", "relabel", "merge", "drop"], required=True)
        c.add_argument("--winding", type=int, default=36)
        c.add_argument("--sector", type=float, nargs=2, default=[0.0, 90.0], help="degrees around the umbilicus")
        a = ap.parse_args()
        set_scroll(a.scroll)
        (build_oracle if a.cmd == "build-oracle" else corrupt)(a)
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scroll", choices=list(SCROLLS), default="p0139")
    ap.add_argument("--ray-dz", type=float, default=25.0, help="p1667 ray pairs: z spacing of the rays")
    ap.add_argument("--ray-dtheta", type=float, default=3.0, help="p1667 ray pairs: angular spacing (deg)")
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--fit-b", type=Path)
    ap.add_argument("--z0", type=float, default=3000)
    ap.add_argument("--z1", type=float, default=4500)
    ap.add_argument("--tol", type=float, default=4.0, help="match tolerance (9.362 vox)")
    ap.add_argument("--stride", type=int, default=2, help="human mesh vertex stride (48 um grid)")
    ap.add_argument("--gap-margin", type=float, default=0.0, help="degrees added on each side of the #2167 gap")
    ap.add_argument("--min-edge", type=float, default=2.5, help="decimate fit grids finer than this (vox)")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false", help="skip the legacy adjacent-pair block")
    ap.add_argument("--pair-frame", choices=["ct", "affine"], default="ct",
                    help="ct: pair endpoints as frozen (CT frame); affine: per-site NCC shift removed (diagnostic)")
    ap.add_argument("--dump-pairs", type=Path, help="per-pair CSV of matched windings")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    set_scroll(a.scroll)
    score(a)


if __name__ == "__main__":
    main()
