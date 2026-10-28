"""Dataset111_Contact0172: contact-aware surface labels from PHerc0172 traced windings (w052-w095), 192^3 at 7.91 um.

  uv run --with "zarr==2.18.4" --with "numcodecs<0.16" --with s3fs --with tifffile --with scipy --with matplotlib \
      python scripts/contact_set.py [--n-contact 240 --n-open 80 --workers 5] [--only 3 --out data/scratch/contact_proto]

Why: binary surface masks fuse touching sheets into one band, so every surface model (m7, ours) sits at ~56-58 %
winding accuracy on point pairs where two sheets touch. Here each traced winding is rasterized on its own, and touching
DISTINCT windings get an explicit 1-voxel background separator, to fine-tune m7 (labels 0 bg / 1 surface / 2 ignore).

Meshes: s3 PHerc0172/segments/*-wNNN_*/mesh/*-on-20241024131839-7.91um.tifxyz (20-vox quad grid, xyz), cached under
data/meshes/PHerc0172/. auto_grown_* traces are skipped (overlapping auto traces, not distinct windings).
CT: PHerc0172/volumes/20241024131839-7.910um-53keV-masked.zarr level 0 (zyx), streamed per crop.

Crops: centred on points where consecutive windings k, k+1 come within 6 vox (point-to-plane, global search on the
mesh grids), greedy non-overlapping in order of local contact density; plus open crops (no contact) on random winding
points. Crops with > 10 % masked-zero CT are dropped.

Snapping: the published meshes are only coarsely on the sheets (surface volumes show the sheet wandering +-5-10 vox
from the mesh; mesh CT barely above the crop mean), so per crop each winding is moved along its normal onto the CT
sheet ridge (offset field h, |h| <= 8, coarse-to-fine smoothing 8/4/2 vox, bounded by half the gap to the traced
neighbours). Samples are flagged (treated like mesh holes) where the ridge is weak, the raw ridge strays > 1 vox from
the smooth h, h hits its bound, the CT structure-tensor normal disagrees (|cos| < 0.7), or two windings snap onto
the same ridge (< 2.5 vox).

Labels per crop (signed distance sd_k to each snapped winding, + towards the next winding outward):
  1 surface: |sd| <= 1.5 of the nearest winding (band ~3 vox); the centre line |sd| <= 0.5 always stays 1.
  0 separator: voxels whose two nearest windings differ, |d1 - d2| < 1 and d2 <= 2.5 (bands overlap or touch).
  0 background: the slab between consecutive traced windings k, k+1 (sd_k > 0 > sd_k+1, both projections on
    unflagged mesh); nothing else.
  2 ignore: everything else (beyond the first/last traced winding, between non-consecutive windings), within 4 vox of
    a mesh boundary, hole or flagged sample, bright untraced material in the slabs (most slabs hold an untraced
    layer), mesh crossings, coincident surfaces (<= 2 vox apart: a trace conflict, not a contact), and windings with
    < 20 % confident samples or snapped surface not brighter than 6 vox off (per crop).
Writes imagesTr/<stem>_0000.tif, labelsTr/<stem>.tif (uint8), dataset.json, manifest.json, qc.json and preview PNGs.
"""

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
BUCKET = "vesuvius-challenge-open-data/PHerc0172"
TAG = "20241024131839-7.91um"
VOL = f"s3://{BUCKET}/volumes/20241024131839-7.910um-53keV-masked.zarr"
MESHES = REPO / "data/meshes/PHerc0172"
C, PAD, M_CT, R_SNAP = 192, 8, 16, 8  # crop edge, EDT pad, CT margin for snapping, max snap offset
CONTACT, BAND, CENTRE, EDGE_IGN = 6.0, 1.5, 0.5, 4.0
G = {}  # global meshes, filled before forking workers


def fetch_meshes() -> dict[int, Path]:
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    out = {}
    for seg in fs.ls(f"{BUCKET}/segments"):
        m = re.search(r"-w(\d{3})_", seg)
        if not m:
            continue
        w = int(m.group(1))
        d = MESHES / f"w{w:03d}"
        d.mkdir(parents=True, exist_ok=True)
        src = [q for q in fs.ls(f"{seg}/mesh") if q.endswith(f"-on-{TAG}.tifxyz")][0]
        for c in ("x", "y", "z", "meta.json"):
            f = d / (c if c.endswith("json") else f"{c}.tif")
            if not f.exists():
                fs.get(f"{src}/{f.name}", str(f))
        out[w] = d
    return dict(sorted(out.items()))


def load_grid(d: Path):
    g = np.stack([tifffile.imread(d / f"{c}.tif") for c in "xyz"], -1).astype(np.float32)
    v = (g > 0).all(-1) & np.isfinite(g).all(-1)
    return g, v


def grid_normals(g: np.ndarray) -> np.ndarray:
    n = np.cross(np.gradient(g, axis=0), np.gradient(g, axis=1))
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def densify(g, v, f):
    """Bilinear upsample of the valid quads by factor f: returns points (n,3) and normals."""
    qv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    i, j = np.nonzero(qv)
    t = (np.arange(f) + 0.5) / f
    u, w = [a.ravel() for a in np.meshgrid(t, t, indexing="ij")]
    p00, p10, p01, p11 = g[i, j], g[i + 1, j], g[i, j + 1], g[i + 1, j + 1]
    P = ((1 - u)[None, :, None] * ((1 - w)[None, :, None] * p00[:, None] + w[None, :, None] * p01[:, None])
         + u[None, :, None] * ((1 - w)[None, :, None] * p10[:, None] + w[None, :, None] * p11[:, None]))
    du = (1 - w)[None, :, None] * (p10 - p00)[:, None] + w[None, :, None] * (p11 - p01)[:, None]
    dv = (1 - u)[None, :, None] * (p01 - p00)[:, None] + u[None, :, None] * (p11 - p10)[:, None]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return P.reshape(-1, 3), n.reshape(-1, 3)


def load_all():
    """Load every winding, orient normals outward (towards the next winding), find contacts with the next winding."""
    from scipy.spatial import cKDTree

    cache = REPO / "data/scratch/contact_set/contacts.npz"
    dirs = fetch_meshes()
    grids = {w: load_grid(d) for w, d in dirs.items()}
    ws = list(grids)
    if cache.exists():
        z = np.load(cache)
        orient = dict(zip(ws, z["orient"].tolist()))
    else:
        orient, rows = {}, []
        for w in ws:  # outward = away from the axis (xy centroid of this full wrap per 500-vox z bin)
            g, v = grids[w]
            n = grid_normals(g)
            p, nn = g[v], n[v]
            zb = (p[:, 2] // 500).astype(int)
            cen = np.zeros_like(p)
            for b in np.unique(zb):
                cen[zb == b] = p[zb == b].mean(0)
            r = (p - cen)[:, :2]
            orient[w] = 1 if np.median(np.sign((r * nn[:, :2]).sum(1))) >= 0 else -1
        for w in ws:  # contacts: vertices of w within CONTACT of w+1 (signed gap along w's outward normal)
            if w + 1 not in grids:
                continue
            g, v = grids[w]
            n = grid_normals(g) * orient[w]
            P2, N2 = densify(*grids[w + 1], 4)
            N2 *= orient[w + 1]
            tree = cKDTree(P2)
            p, nn = g[v], n[v]
            ij = np.argwhere(v)
            d, k = tree.query(p, distance_upper_bound=40)
            ok = np.isfinite(d)
            q = P2[np.minimum(k, len(P2) - 1)]
            gap = np.abs(((q - p) * N2[np.minimum(k, len(P2) - 1)]).sum(1))  # point-to-plane distance on w+1
            along = ((q - p) * nn).sum(1)
            lat = np.linalg.norm((q - p) - along[:, None] * nn, axis=1)
            ok &= lat < 0.5 * np.abs(along) + 3  # the w+1 point is roughly along w's normal, not off to the side
            gap = np.where(ok, gap * np.sign(along), np.inf)
            sel = ok & (gap < 12)
            rows.append(np.c_[np.full(sel.sum(), w), p[sel], gap[sel], ij[sel]])
            print(f"w{w:03d}-w{w + 1:03d}: verts {len(p)} matched {ok.sum()} gap 2-6 {((gap >= 2) & (gap <= CONTACT)).sum()} "
                  f"|gap|<2 {(np.abs(gap) < 2).sum()} crossing(gap<-2) {(gap < -2).sum()} median {np.median(gap[ok]):.1f}",
                  flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, orient=np.array([orient[w] for w in ws]), rows=np.concatenate(rows).astype(np.float32))
        z = np.load(cache)
    qv = {w: v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:] for w, (g, v) in grids.items()}
    G.update(grids=grids, orient=orient, qvalid=qv, contacts=z["rows"])
    return G


def lattices(w, lo, hi, step=0.7):
    """Pieces of winding w near the zyx box [lo, hi] as fine UV lattices (spacing <= step vox): points P (H,W,3) zyx,
    outward normals N, valid V (inside a valid quad), boundary B (valid pixels next to holes / the grid edge)."""
    from scipy import ndimage as ndi
    from scipy.ndimage import map_coordinates

    g, v = G["grids"][w]
    qv = G["qvalid"][w]
    gz = g[..., ::-1]
    inb = v & (gz >= lo - 24).all(-1) & (gz <= hi + 24).all(-1)
    sel = qv & (inb[:-1, :-1] | inb[1:, :-1] | inb[:-1, 1:] | inb[1:, 1:])
    out = []
    if not sel.any():
        return out
    lab, n = ndi.label(ndi.binary_dilation(sel, iterations=2), np.ones((3, 3)))
    for k in range(1, n + 1):
        ii, jj = np.nonzero(lab == k)
        a0, a1, b0, b1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
        sub = gz[a0:a1 + 1, b0:b1 + 1].astype(np.float64)
        q = qv[a0:a1, b0:b1]
        qe = np.pad(q, 1)
        e0 = np.linalg.norm(np.diff(sub, axis=0), axis=-1)[(qe[1:-1, 1:] | qe[1:-1, :-1])[:, :]]
        e1 = np.linalg.norm(np.diff(sub, axis=1), axis=-1)[(qe[1:, 1:-1] | qe[:-1, 1:-1])]
        f = int(min(np.ceil(max(e0.max(initial=1), e1.max(initial=1)) / step), 60))
        S, T = np.meshgrid((np.arange((a1 - a0) * f) + 0.5) / f, (np.arange((b1 - b0) * f) + 0.5) / f, indexing="ij")
        P = np.stack([map_coordinates(sub[..., c], [S, T], order=1) for c in range(3)], -1)
        V = q[S.astype(int), T.astype(int)]
        Vp = np.pad(V, 1)
        if a0 > 0:  # lattice sides inside the grid continue: not a boundary
            Vp[0, 1:-1] = V[0]
        if a1 < qv.shape[0]:
            Vp[-1, 1:-1] = V[-1]
        if b0 > 0:
            Vp[1:-1, 0] = V[:, 0]
        if b1 < qv.shape[1]:
            Vp[1:-1, -1] = V[:, -1]
        B = V & ~ndi.binary_erosion(Vp, np.ones((3, 3)))[1:-1, 1:-1]
        N = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
        N *= -G["orient"][w]  # zyx cross product flips handedness relative to xyz
        out.append(dict(w=w, P=P.astype(np.float32), N=N.astype(np.float32), V=V, B=B, pxv=f / 20))
    return out


def msmooth(a, V, s):
    """Gaussian smoothing over the valid lattice pixels (last two axes)."""
    from scipy import ndimage as ndi

    num = ndi.gaussian_filter(np.where(V, a, 0).astype(np.float32), (0, s, s) if a.ndim == 3 else s, truncate=3)
    return num / np.maximum(ndi.gaussian_filter(V.astype(np.float32), s, truncate=3), 1e-3)


def snap(L, ctm, om, lo_b, hi_b):
    """Normal offset h (smooth) moving the lattice onto the CT sheet ridge, coarse to fine, within [lo_b, hi_b].
    Returns h, ridge prominence at h (vs the darkest point 2-6 vox away), |raw ridge - h| at 1-vox smoothing, and the raw profiles."""
    from scipy.ndimage import map_coordinates

    P, N, V, pxv = L["P"], L["N"], L["V"], L["pxv"]
    ts = np.arange(-R_SNAP, R_SNAP + 1, dtype=np.float32)
    prof = np.stack([map_coordinates(ctm, (P + t * N - om).reshape(-1, 3).T, order=1, mode="nearest").reshape(V.shape)
                     for t in ts])
    inwin = (ts[:, None, None] >= lo_b[None]) & (ts[:, None, None] <= hi_b[None])
    h = np.zeros(V.shape, np.float32)
    for s_vox, win in ((8, R_SNAP), (4, 3), (2, 2)):
        ps = msmooth(prof, V, s_vox * pxv)
        ok = (np.abs(ts[:, None, None] - h[None]) <= win) & inwin
        h = msmooth(ts[np.argmax(np.where(ok, ps, -1e9), 0)], V, s_vox * pxv).astype(np.float32)
    ps = msmooth(prof, V, 1 * pxv)
    ii = np.clip(np.round(h + R_SNAP).astype(int), 0, 2 * R_SNAP)
    at = lambda k: np.take_along_axis(ps, np.clip(k, 0, 2 * R_SNAP)[None], 0)[0]
    lo_side = np.min([at(ii - k) for k in range(2, 7)], 0)
    hi_side = np.min([at(ii + k) for k in range(2, 7)], 0)
    prom = at(ii) - np.minimum(lo_side, hi_side)  # brighter than a dip on at least one side (contacts have one)
    raw = ts[np.argmax(np.where((np.abs(ts[:, None, None] - h[None]) <= 1.5) & inwin, ps, -1e9), 0)]
    return h, prom, np.abs(raw - h), prof


def structure_tensor(ctm):
    from scipy import ndimage as ndi

    gr = [ndi.gaussian_filter(ctm, 1.0, order=[int(i == k) for i in range(3)]) for k in range(3)]
    return {(i, j): ndi.gaussian_filter(gr[i] * gr[j], 3.0) for i in range(3) for j in range(i, 3)}


def st_cos(T, Q, N):
    """|cos| between the CT sheet normal (structure tensor major axis) and N at points Q (ctm frame)."""
    from scipy.ndimage import map_coordinates

    M = np.empty((len(Q), 3, 3), np.float32)
    for (i, j), a in T.items():
        M[:, i, j] = M[:, j, i] = map_coordinates(a, Q.T, order=1, mode="nearest")
    return np.abs((np.linalg.eigh(M)[1][:, :, -1] * N).sum(1))


def neighbour_bound(P, N, tree, sgn):
    """Half the gap (minus 1.5 vox) to a neighbouring winding along +-N, R_SNAP where it is absent."""
    p, n = P.reshape(-1, 3), N.reshape(-1, 3)
    d, k = tree.query(p, distance_upper_bound=40)
    q = tree.data[np.minimum(k, tree.n - 1)]
    al = ((q - p) * n).sum(1) * sgn
    lat = np.linalg.norm(q - p - (al * sgn)[:, None] * n, axis=1)
    ok = np.isfinite(d) & (al > 0) & (lat < 0.5 * al + 3)
    return np.where(ok, np.clip(al / 2 - 1.5, 0, R_SNAP), R_SNAP).reshape(P.shape[:2]).astype(np.float32)


def snapped_windings(o, ctm, om):
    """Snap every winding near the crop to the CT; returns {w: dict(P, N, flag, stats)} (flag = boundary or low
    confidence: weak ridge, unstable offset, offset at its bound, sheet normal disagrees, or same ridge as a neighbour)."""
    from scipy.spatial import cKDTree

    lo, hi = (o - PAD - 24).astype(np.float32), (o + C + PAD + 23).astype(np.float32)
    Ls = {}
    for w in G["grids"]:
        x = lattices(w, lo + 24, hi - 24)
        if x:
            Ls[w] = x
    trees = {w: cKDTree(np.concatenate([L["P"][L["V"]][::2] for L in x])) for w, x in Ls.items()}
    inner = ctm[M_CT:-M_CT, M_CT:-M_CT, M_CT:-M_CT]
    prom_thr = 0.15 * (np.percentile(inner, 95) - np.percentile(inner, 5))
    T = structure_tensor(ctm)
    for w, x in Ls.items():
        for L in x:
            P, N, V = L["P"], L["N"], L["V"]
            hi_b = neighbour_bound(P, N, trees[w + 1], 1) if w + 1 in trees else np.full(V.shape, R_SNAP, np.float32)
            lo_b = -neighbour_bound(P, N, trees[w - 1], -1) if w - 1 in trees else np.full(V.shape, -R_SNAP, np.float32)
            h, prom, resid, prof = snap(L, ctm, om, lo_b, hi_b)
            Q = P + h[..., None] * N
            Nq = np.cross(np.gradient(Q, axis=0), np.gradient(Q, axis=1))
            Nq /= np.linalg.norm(Nq, axis=-1, keepdims=True) + 1e-9
            Nq *= np.sign((Nq * N).sum(-1, keepdims=True) + 1e-9)
            cos = np.ones(V.shape, np.float32)
            cos[V] = st_cos(T, Q[V] - om, Nq[V])
            cos = msmooth(cos, V, 2 * L["pxv"])
            atb = ((h > hi_b - 0.3) & (hi_b < R_SNAP)) | ((h < lo_b + 0.3) & (lo_b > -R_SNAP)) | (np.abs(h) > R_SNAP - 0.5)
            tests = dict(prom=prom > prom_thr, resid=resid <= 1.0, cos=cos >= 0.7, bound=~atb)
            good = V.copy()
            for t in tests.values():
                good &= t
            L["fails"] = {k: int((V & ~t).sum()) for k, t in tests.items()}
            ii = np.clip(np.round(h + R_SNAP).astype(int), 0, 2 * R_SNAP)
            L.update(Q=Q, Nq=Nq, good=good, h=h, c_snap=np.take_along_axis(prof, ii[None], 0)[0], prof0=prof)
    gp = [(L["Q"][L["good"]][::2], w) for w, x in Ls.items() for L in x]
    allw = np.concatenate([np.full(len(q), w, np.int16) for q, w in gp])
    rtree = cKDTree(np.concatenate([q for q, _ in gp])) if len(allw) else None
    from scipy import ndimage as ndi
    from skimage.morphology import disk, remove_small_holes, remove_small_objects

    out = {}
    for w, x in Ls.items():
        P_, N_, F_, P0, N0 = [], [], [], [], []
        st = dict(n=0, good=0, h=[], fails=dict(prom=0, resid=0, cos=0, bound=0, same_ridge=0, n=0))
        for L in x:
            V = L["V"]
            if rtree is not None:  # snapped onto the same ridge as another winding (< 2.5 vox)
                d, k = rtree.query(L["Q"].reshape(-1, 3), k=6, distance_upper_bound=2.5)
                other = np.isfinite(d) & (allw[np.minimum(k, len(allw) - 1)] != w)
                same = other.any(1).reshape(V.shape)
                L["fails"]["same_ridge"] = int((V & same).sum())
                L["good"] &= ~same
            for k, v in L["fails"].items():
                st["fails"][k] += v
            st["fails"]["n"] += int(V.sum())
            px = L["pxv"]  # clean the confidence mask: fill specks, open (r 2 vox), drop islands < 12x12 vox
            g = remove_small_holes(L["good"], max(int(36 * px * px), 1))
            g = ndi.binary_opening(g, disk(max(int(round(2 * px)), 1)))
            L["good"] = remove_small_objects(g, max(int(144 * px * px), 1)) & V
            V = L["V"]
            inc = V & ((L["Q"] - o >= 0) & (L["Q"] - o <= C - 1)).all(-1)
            st["n"] += int(inc.sum())
            st["good"] += int((inc & L["good"]).sum())
            st["h"].append(L["h"][inc & L["good"]])
            P_.append(L["Q"][V])
            N_.append(L["Nq"][V])
            F_.append((~L["good"] | L["B"])[V])
            P0.append(L["P"][inc] - o)
            N0.append(L["N"][inc])
        hh = np.concatenate(st.pop("h"))
        st["med_abs_h"] = round(float(np.median(np.abs(hh))), 2) if len(hh) else None
        out[w] = dict(P=np.concatenate(P_), N=np.concatenate(N_), flag=np.concatenate(F_), P0=np.concatenate(P0),
                      N0=np.concatenate(N0), stats=st)
    return out


def raster(pts, origin, shape):
    """Voxel grid of sample ids (-1 = empty), keeping per voxel the sample closest to the voxel centre."""
    q = pts - origin
    r = np.round(q).astype(np.int32)
    ok = ((r >= 0) & (r < np.array(shape))).all(1)
    ids = np.nonzero(ok)[0]
    r, d = r[ok], np.linalg.norm(q[ok] - r[ok], axis=1)
    o = np.argsort(-d)
    grid = np.full(shape, -1, np.int32)
    grid[r[o, 0], r[o, 1], r[o, 2]] = ids[o]
    return grid


def winding_fields(w, o, P, N, flag):
    """Signed distance to the snapped winding on the crop grid (+ outward), projection confidence (nearest point not
    within 2 vox of a flagged sample), and the near-flag mask (<= EDGE_IGN)."""
    from scipy.ndimage import distance_transform_edt as edt

    S = (C + 2 * PAD,) * 3
    op = o - PAD
    sid = raster(P, op, S)
    if (sid[PAD:PAD + C, PAD:PAD + C, PAD:PAD + C] >= 0).sum() < 50:
        return None
    inds = edt(sid < 0, return_distances=False, return_indices=True)
    cz = slice(PAD, PAD + C)
    nz, ny, nx = (a[cz, cz, cz] for a in inds)
    del inds
    near = sid[nz, ny, nx]
    vox = np.stack(np.meshgrid(*[np.arange(C, dtype=np.float32) + PAD] * 3, indexing="ij"), -1)
    d = vox - (P[near] - op)
    del vox
    sd = (d * N[near]).sum(-1)
    eu = np.linalg.norm(d, axis=-1)
    sd = np.where(eu < 3, sd, np.sign(sd) * eu).astype(np.float32)
    b = raster(P[flag], op, S) >= 0 if flag.any() else np.zeros(S, bool)
    db = edt(~b).astype(np.float32) if b.any() else np.full(S, np.inf, np.float32)
    return dict(w=w, sd=sd, conf=db[nz, ny, nx] > 2.0, nearb=db[cz, cz, cz] <= EDGE_IGN)


def contrast(ct, P, N, k=4):
    """CT along the normal at subsampled surface points (crop frame): profile t=-6..6 (mean), c3 / c6 = I0 - mean(I(+-t))."""
    from scipy.ndimage import map_coordinates

    P, N = P[::k], N[::k]
    if len(P) == 0:
        return np.zeros(13), None, None
    prof = np.stack([map_coordinates(ct, (P + t * N).T, order=1, mode="nearest") for t in range(-6, 7)])
    return prof.mean(1), float(np.median(prof[6] - (prof[3] + prof[9]) / 2)), float(np.median(prof[6] - (prof[0] + prof[12]) / 2))


_CT = {}


def read_ct(o, m=0):
    import zarr

    if "a" not in _CT:
        _CT["a"] = zarr.open_group(VOL, mode="r", storage_options={"anon": True})["0"]
    a = o - m
    return np.asarray(_CT["a"][a[0]:a[0] + C + 2 * m, a[1]:a[1] + C + 2 * m, a[2]:a[2] + C + 2 * m])


def build_crop(job):
    """Label one crop; returns its manifest row (or None if rejected)."""
    from scipy import ndimage as ndi

    o, kind, out, prev = np.array(job["origin"]), job["kind"], Path(job["out"]), job.get("preview")
    t0 = time.time()
    stem = f"p0172_z{o[0]}_y{o[1]}_x{o[2]}"
    big = read_ct(o, M_CT)
    ct = big[M_CT:-M_CT, M_CT:-M_CT, M_CT:-M_CT]
    if big.shape != (C + 2 * M_CT,) * 3 or (ct == 0).mean() > 0.10:
        return None
    ctm = ndi.gaussian_filter(big.astype(np.float32), 1.0)
    ctf = ct.astype(np.float32)
    W = snapped_windings(o, ctm, o - M_CT)
    F, qc = [], {}
    for w, sw in sorted(W.items()):
        f = winding_fields(w, o, sw["P"], sw["N"], sw["flag"])
        if f is None:
            continue
        inc = ~sw["flag"] & ((sw["P"] - o >= 0) & (sw["P"] - o <= C - 1)).all(1)
        _, c3o, c6o = contrast(ctf, sw["P0"], sw["N0"])
        prof, c3s, c6s = contrast(ctf, sw["P"][inc] - o, sw["N"][inc])
        st = sw["stats"]
        gfrac = st["good"] / max(st["n"], 1)
        fl = st["fails"]
        qc[w] = dict(c3_mesh=c3o, c6_mesh=c6o, c3_snap=c3s, c6_snap=c6s, good_frac=round(gfrac, 3),
                     med_abs_h=st["med_abs_h"], n=st["n"],
                     fail={k: round(v / max(fl["n"], 1), 3) for k, v in fl.items() if k != "n"})
        f["ok"] = gfrac >= 0.2 and c6s is not None and c6s > 0
        F.append(f)
    good = [f for f in F if f["ok"]]
    if not good:
        return None
    lab = np.full((C,) * 3, 2, np.uint8)
    d1 = np.full((C,) * 3, np.inf, np.float32)
    d2 = d1.copy()
    i1 = np.full((C,) * 3, -1, np.int16)
    ncen = np.zeros((C,) * 3, np.uint8)
    slab = np.zeros((C,) * 3, bool)
    inv = np.zeros((C,) * 3, bool)
    ign = np.zeros((C,) * 3, bool)
    prevf = None
    for f in F:
        if not f["ok"]:
            ign |= np.abs(f["sd"]) <= 3  # unreliable trace: its neighbourhood is unknown
            prevf = None
            continue
        a = np.abs(f["sd"])
        m = a < d1
        d2 = np.where(m, d1, np.minimum(d2, a))
        i1 = np.where(m, f["w"], i1)
        d1 = np.where(m, a, d1)
        ncen += a <= CENTRE
        ign |= f["nearb"]
        if prevf is not None and prevf["w"] == f["w"] - 1:
            both = prevf["conf"] & f["conf"]
            slab |= (prevf["sd"] > 0) & (f["sd"] < 0) & both
            inv |= (prevf["sd"] < 0) & (f["sd"] > 0) & both & (np.abs(prevf["sd"]) < 15) & (a < 15)
        prevf = f
    band = d1 <= BAND
    sep = (d2 - d1 < 1.0) & (d2 <= BAND + 1.0) & (ncen == 0)
    hard = sep & (d1 + d2 < 3.0)  # distinct surfaces < 3 vox apart
    # surfaces <= 2 vox apart: distinct 4-6 vox sheets cannot be that close, so this is a trace conflict -> ignore
    coinc = (d2 <= 1.0) & (i1 >= 0)
    coinc = ndi.binary_dilation(coinc, iterations=1) if coinc.any() else coinc
    # bright untraced material inside the slabs (> 4.5 vox from every traced surface)
    sm = ctm[M_CT:-M_CT, M_CT:-M_CT, M_CT:-M_CT]
    far = slab & (d1 > 4.5)
    unt = np.zeros_like(slab)
    if far.any() and (ncen > 0).any():
        unt = far & (sm > (np.median(sm[ncen > 0]) + np.median(sm[far])) / 2)
        cc, n = ndi.label(unt, np.ones((3, 3, 3)))
        if n:
            sz = np.bincount(cc.ravel())
            sz[0] = 0
            unt = ndi.binary_dilation((sz >= 200)[cc], iterations=2)
    invd = ndi.binary_dilation(inv, iterations=3) if inv.any() else inv
    lab[slab] = 0
    lab[band] = 1
    lab[sep] = 0
    lab[ncen > 0] = 1
    lab[ign | invd | coinc | (unt & ~band)] = 2
    contact_vox = int(((d2 - d1 < 1.0) & (d1 + d2 <= CONTACT) & (lab != 2)).sum())
    tifffile.imwrite(out / "imagesTr" / f"{stem}_0000.tif", ct, compression="zlib")
    tifffile.imwrite(out / "labelsTr" / f"{stem}.tif", lab, compression="zlib")
    on = lab == 1
    row = dict(stem=stem, origin_zyx=o.tolist(), kind=kind, windings=[int(f["w"]) for f in good],
               dropped_windings=[int(f["w"]) for f in F if not f["ok"]], contact_vox=contact_vox,
               separator_vox=int((sep & (lab == 0)).sum()), hard_contact_vox=int((hard & (lab == 0)).sum()),
               coincident_vox=int(coinc.sum()), crossing_vox=int(invd.sum()),
               untraced_vox=int(unt.sum()),
               frac={k: round(float((lab == v).mean()), 4) for k, v in (("bg", 0), ("surf", 1), ("ign", 2))},
               ct_surf=round(float(ctf[on].mean()), 1) if on.any() else None,
               ct_bg=round(float(ctf[lab == 0].mean()), 1) if (lab == 0).any() else None,
               qc={int(k): v for k, v in qc.items()}, sec=round(time.time() - t0, 1))
    if prev:
        preview(ct, lab, i1, ncen, sep, prev, stem, [W[f["w"]]["N"] for f in good])
    return row


def preview(ct, lab, i1, ncen, sep, path, stem, normals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = np.abs(np.concatenate(normals).mean(0)) if normals else np.array([1, 0, 0])
    ax = int(np.argmin(n))  # slice so the sheets cross the image
    best = np.argmax([np.take(sep, k, ax).sum() for k in range(C)])  # the slice with the most separator voxels
    take = lambda a: np.take(a, best, ax)
    c, l_, w, s = take(ct).astype(float), take(lab), take(np.where(ncen > 0, i1, -1)), take(sep & (lab == 0))
    rgb = np.repeat((c / 255)[..., None], 3, -1)
    ov = rgb.copy()
    ov[l_ == 2] *= 0.45
    ov[l_ == 1] = 0.5 * ov[l_ == 1] + 0.5 * np.array([1, 0, 0])
    ov[s] = [0.2, 0.4, 1.0]
    cols = plt.get_cmap("tab10")
    wi = rgb.copy()
    for k, u in enumerate(np.unique(w[w >= 0])):
        wi[w == u] = cols(k % 10)[:3]
    from scipy.ndimage import uniform_filter

    dens = uniform_filter(s.astype(np.float32), 32)[32:-32, 32:-32]  # zoom on the densest separator window
    cy, cx = (int(v) + 32 for v in np.unravel_index(np.argmax(dens), dens.shape))
    fig, axs = plt.subplots(2, 3, figsize=(21, 14.5))
    for r_, zoom in enumerate((None, (slice(cy - 32, cy + 32), slice(cx - 32, cx + 32)))):
        for a_, im, t in zip(axs[r_], (rgb, ov, wi), ("CT", "label: surface red, separator blue, ignore dim",
                                                     "snapped winding centre lines")):
            a_.imshow(im if zoom is None else im[zoom], interpolation="nearest")
            a_.set_title(t + ("" if zoom is None else f"  (zoom 64^2 at {cy},{cx})"))
            a_.axis("off")
    fig.suptitle(f"{stem}  axis {'zyx'[ax]}={best}")
    plt.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def select(n_contact, n_open, seed=0):
    """Crop origins: contact-dense non-overlapping crops first, then open crops away from any contact or crossing."""
    from scipy.spatial import cKDTree

    r = G["contacts"]
    xyz, gap = r[:, 1:4], r[:, 4]
    con = xyz[(gap >= 2) & (gap <= CONTACT)][:, ::-1]
    bad = xyz[(gap < -2) | (np.abs(gap) < 2)][:, ::-1]
    cell = np.floor(con / 64).astype(int)
    key, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    cen = np.stack([np.bincount(inv.ravel(), con[:, k]) / cnt for k in range(3)], 1)
    tb = cKDTree(bad)
    tc = cKDTree(con)
    score = np.array([len(x) for x in tc.query_ball_point(cen, 96, p=np.inf)]) \
        - np.array([len(x) for x in tb.query_ball_point(cen, 96, p=np.inf)])
    picked = []
    for k in np.argsort(-score):
        if len(picked) >= n_contact:
            break
        if all(np.abs(cen[k] - q).max() >= C for q in picked):
            picked.append(cen[k])
    jobs = [dict(origin=(np.round(c).astype(int) - C // 2).tolist(), kind="contact") for c in picked]
    rng = np.random.default_rng(seed)
    allc = cKDTree(np.concatenate([con, bad]))
    opens = []
    ws = [w for w in G["grids"] if w - 1 in G["grids"] and w + 1 in G["grids"]]
    while len(opens) < n_open:
        g, v = G["grids"][ws[rng.integers(len(ws))]]
        ii, jj = np.nonzero(v)
        k = rng.integers(len(ii))
        c = g[ii[k], jj[k]][::-1]
        if allc.query_ball_point(c, 110, p=np.inf) or any(np.abs(c - q).max() < C for q in picked + opens):
            continue
        opens.append(c)
    jobs += [dict(origin=(np.round(c).astype(int) - C // 2).tolist(), kind="open") for c in opens]
    return jobs


def init_worker():
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-contact", type=int, default=240)
    ap.add_argument("--n-open", type=int, default=80)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", type=int, default=0, help="prototype: build only the first N contact + N//2 open crops")
    ap.add_argument("--out", type=Path, default=REPO / "data/nnunet/nnUNet_raw/Dataset111_Contact0172")
    ap.add_argument("--previews", type=int, default=3)
    args = ap.parse_args()
    load_all()
    over = 1.1  # extra candidates for crops rejected on masked CT
    jobs = select(int(args.n_contact * over), int(args.n_open * over))
    if args.only:
        jobs = [j for j in jobs if j["kind"] == "contact"][: args.only] + \
            [j for j in jobs if j["kind"] == "open"][: max(1, args.only // 2)]
    for d in ("imagesTr", "labelsTr"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    pv = REPO / "data/scratch"
    for k, j in enumerate(jobs):
        j["out"] = str(args.out)
        if j["kind"] == "contact" and k < args.previews:
            j["preview"] = str(pv / f"contact_set_preview_{k}.png")
    import multiprocessing as mp

    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as ex:
        for k, r in enumerate(ex.map(build_crop, jobs)):
            if r:
                rows.append(r)
                print(f"[{k + 1}/{len(jobs)}] {r['stem']} {r['kind']} w{r['windings']} drop{r['dropped_windings']} "
                      f"contact {r['contact_vox']} sep {r['separator_vox']} hard {r['hard_contact_vox']} "
                      f"cross {r['crossing_vox']} unt {r['untraced_vox']} {r['frac']} {r['sec']}s", flush=True)
            else:
                print(f"[{k + 1}/{len(jobs)}] rejected {jobs[k]['origin']}", flush=True)
    keep = [r for r in rows if r["kind"] == "contact"][: args.n_contact] + [r for r in rows if r["kind"] == "open"][: args.n_open]
    ks = {r["stem"] for r in keep}
    for d in ("imagesTr", "labelsTr"):
        for f in (args.out / d).glob("*.tif"):
            if f.name.replace("_0000.tif", "").replace(".tif", "") not in ks:
                f.unlink()
    (args.out / "manifest.json").write_text(json.dumps(keep, indent=1))
    med = lambda v: round(float(np.median([x for x in v if x is not None])), 2) if any(x is not None for x in v) else None
    per = {}
    for r in keep:
        for w, q in r["qc"].items():
            per.setdefault(int(w), []).append(dict(q, dropped=int(w) in r["dropped_windings"]))
    qc = dict(
        per_winding={w: dict(crops=len(v), dropped=sum(x["dropped"] for x in v),
                             **{k: med([x[k] for x in v]) for k in ("c3_mesh", "c6_mesh", "c3_snap", "c6_snap",
                                                                    "good_frac", "med_abs_h")}) for w, v in sorted(per.items())},
        label_on_material=dict(ct_surface=med([r["ct_surf"] for r in keep]), ct_background=med([r["ct_bg"] for r in keep])),
        totals={k: int(sum(r[k] for r in keep)) for k in ("contact_vox", "separator_vox", "hard_contact_vox",
                                                          "coincident_vox", "crossing_vox", "untraced_vox")},
        frac={k: med([r["frac"][k] for r in keep]) for k in ("bg", "surf", "ign")},
        crops=len(keep), contact_crops=sum(r["kind"] == "contact" for r in keep),
        contact_crops_with_separator=sum(r["kind"] == "contact" and r["separator_vox"] > 0 for r in keep))
    (args.out / "qc.json").write_text(json.dumps(qc, indent=1))
    print(json.dumps({k: v for k, v in qc.items() if k != "per_winding"}))
    (args.out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": len(keep), "file_ending": ".tif", "overwrite_image_reader_writer": "SimpleTiffIO",
        "name": "Contact0172: PHerc0172 traced windings w052-w095, per-winding bands with background separators "
                "between touching windings; 7.91 um, 192^3"}, indent=1))
    print("kept", len(keep), "contact", sum(r["kind"] == "contact" for r in keep))


if __name__ == "__main__":
    main()
