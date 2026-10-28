"""CT audit of human winding labels: count the physical turns between labelled meshes along rays from the umbilicus.

Generalises eval/ct_gap_audit.py (which stays byte-identical: its sha256 is in notes/2026-10-21-second-band-plan.md 7b)
to any z range and any scroll that has (1) human winding meshes (tifxyz, one winding label per mesh), (2) a binary
surface mask zarr at the volume's L0 (e.g. the bucket m7 prediction) and (3) a lasagna grad_mag (winding density)
zarr, plus an umbilicus. Only those inputs are read; no fit output.

Method (the calibrated band-1/band-2 method of notes/2026-10-21-fit-figures.md "Label audit, band 1"):
  rays     z levels every --z-step (default 48, offset 24: every level sits >= 8 slices inside a 128- and a 192-chunk
           row) x angles every --angle-step deg around the umbilicus.
  crossing where each mesh crosses the ray: the mesh's contour at plane z (marching squares on its vertex grid)
           intersected with the ray. A mesh that crosses one ray at radii > 5 vox apart (its own >360 deg overlap or a
           fold) is dropped on that ray.
  m7       mask sampled on 3 z x 3 tangential lines (+-2 vox), step 0.5 vox, max over the 9 lines (> 0 = sheet).
           Sheets along the ray = mask runs. Each mesh crossing is assigned to the run that contains it or lies
           within 4 vox (the old anchor rule); a crossing with no run nearby gets its own virtual sheet; two meshes on
           one run but > 8 vox apart are split. Turns between two meshes = difference of their sheet indices
           (= "m7 runs strictly between + 1" of ct_gap_audit, 0 for meshes on the same sheet).
  gm       lasagna grad_mag (u8/1000 windings per working voxel) on the centre line, integrated between the two
           crossings; scaled per z slab so that the median label-adjacent (dk = 1) gap of the calibration half of the
           rays is 1 turn.
Calibration / validation, verdicts and the corrected mapping are in `analyse` (docstring there).

  # PHerc0139 preset (sites.json meshes 2.399 -> published affine, bucket m7, local lasagna stores)
  uv run python eval/winding_audit.py extract --scroll PHerc0139 --z0 1750 --z1 8500 --out data/scratch/label_audit/full
  uv run python eval/winding_audit.py analyse --out data/scratch/label_audit/full --anchor 29
  uv run python eval/winding_audit.py report --out data/scratch/label_audit/full \
      --corrections data/scratch/label_audit/p0139_label_corrections.json
  # any scroll
  uv run python eval/winding_audit.py audit --meshes DIR --umbilicus umb.json --mask s3://.../surface.zarr/0 \
      --gm path/grad_mag.ome.zarr/2 --gm-scale 4 --z0 Z0 --z1 Z1 --out OUTDIR --anchor K
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
STEP, PAD, ANCHOR, MULTI_GAP, SPLIT = 0.5, 14.0, 4.0, 5.0, 8.0
DZ, DT = (-2, 0, 2), (-2.0, 0.0, 2.0)

PRESETS = {
    "PHerc0139": {
        "mask": "s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/"
                "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr/0",
        "ct": "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0",
        "gm": [str(REPO / "data/p0139/inputs/PHerc0139_grad_mag.ome.zarr/2"),
               str(REPO / "data/p0139/inputs_audit/PHerc0139_grad_mag.ome.zarr/2")],
        "gm_scale": 4.0,
        "seams": [(115.0, 145.0), (220.0, 265.0)],
        "sectors": {"villa2167": (145.0, 260.0)},
    },
    # PHercParis4 (Scroll 1), L2 frame of volume 20260411134726 (9.6 um; notes/2026-10-24-paris4-label-audit.md).
    # Labels: organiser 20260701 winding-group meshes, one k per vertex (data/eval/paris4/labels_w0701.npz, built by
    # data/scratch/third_region/paris4_labels.py); each k (inside its group's named range) is one mesh here.
    # Local byte copies of the m7 rows / resampled lasagna rows are used where they cover a whole chunk row,
    # else the public bucket is streamed. z 10800-11800 is a fresh confirmation region: no level may fall there.
    "PHercParis4": {
        "mask": "s3://vesuvius-challenge-open-data/PHercParis4/representations/predictions/surfaces/"
                "20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0",
        "mask_local": [(str(REPO / "data/paris4/masks/m7_L2_th0.2.zarr/0"), 8640, 10944),
                       (str(REPO / "data/paris4_z7000/masks/m7_L2_th0.2.zarr/0"), 6528, 9024)],
        "gm": [],
        "gm_local": [(str(REPO / "data/paris4/inputs/PHercParis4_grad_mag.ome.zarr/2"), 8704, 10880),
                     (str(REPO / "data/paris4_z7000/inputs/PHercParis4_grad_mag.ome.zarr/2"), 6656, 8832)],
        "gm_remote": "s3://vesuvius-challenge-open-data/PHercParis4/representations/predictions/lasagna/"
                     "20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c_grad_mag"
                     ".ome.zarr/4",
        "gm_scale": 4.0,
        "ct": "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2",
        "ct_profile": True,
        "umbilicus": str(REPO / "data/eval/paris4/umbilicus_paris4_L2.json"),
        "labels_npz": str(REPO / "data/eval/paris4/labels_w0701.npz"),
        "grids": str(REPO / "data/dl_paris4/w0701_orig"),
        "forbidden_z": [(10800.0, 11800.0)],
        "forbidden_margin": 48.0,
        "seams": [],
        "sectors": {},
    },
}


# ------------------------------------------------------------------------------------------------------------ inputs
def load_affine(path):
    M = np.array(json.loads(Path(path).read_text())["transformation_matrix"], float)
    return M[:3, :3], M[:3, 3]


def umbilicus_from(path, affine=None):
    """control points xyz (json {"control_points": [{x,y,z}]} or [[x,y,z], ...]); returns z, y, x arrays sorted by z."""
    d = json.loads(Path(path).read_text())
    pts = d["control_points"] if isinstance(d, dict) else d
    c = np.array([[p["x"], p["y"], p["z"]] if isinstance(p, dict) else p for p in pts], float)
    if affine is not None:
        c = c @ affine[0].T + affine[1]
    c = c[np.argsort(c[:, 2])]
    return c[:, 2], c[:, 1], c[:, 0]


def iter_meshes_p0139():
    """sites.json segments with a winding (blacklisted included, flagged) as zyx grids in the 9.362 frame."""
    sys.path.insert(0, str(REPO / "eval"))
    import spiral_eval as se
    import tifffile
    A, t = se.affine()
    meta = json.loads((se.ROOT / "sites.json").read_text())
    for s in meta["segments"]:
        d = se.MESHES / s["name"] / se.MESH_TAG
        if s["winding"] is None or not (d / "x.tif").exists():
            continue
        g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
        ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
        p = g @ A.T + t
        p[~ok] = np.nan
        yield {"label": int(s["winding"]), "name": s["name"], "blacklisted": "blacklisted" in s,
               "note": s.get("blacklisted", ""), "grid": p[..., ::-1].astype(np.float32)}


def iter_meshes_paris4(pre):
    """One mesh per winding k of the organiser 20260701 group meshes (L2 xyz tifxyz_original, invalid -1). k comes
    from labels_w0701.npz; only k inside the group's named range wAAA-BBB is kept (the spill is dropped, as in
    spiral_eval --scroll paris4). Grids are cropped to the columns / rows that hold k."""
    import tifffile
    L = np.load(pre["labels_npz"])
    for d in sorted(Path(pre["grids"]).glob("20260701*-w*")):
        m = re.search(r"-w(\d+)-(\d+)$", d.name)
        if not m or not (d / "z.tif").exists():
            continue
        w0, w1 = int(m.group(1)), int(m.group(2))
        g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float32) for c in "xyz"], -1)
        ok = (g >= 0).all(-1) & np.isfinite(g).all(-1) & (g[..., 2] > 0)
        g[~ok] = np.nan
        kk = L[f"k_{w0:03d}_{w1:03d}"]
        assert kk.shape == g.shape[:2], d
        for k in range(w0, w1 + 1):
            sel = (kk == k) & ok
            if not sel.any():
                continue
            rr, cc = np.nonzero(sel.any(1))[0], np.nonzero(sel.any(0))[0]
            r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
            sub = g[r0:r1, c0:c1].copy()
            sub[~sel[r0:r1, c0:c1]] = np.nan
            yield {"label": k, "name": f"{d.name}/k{k:03d}", "blacklisted": False, "note": "",
                   "grid": sub[..., ::-1].copy()}


def iter_meshes_dir(root, affine=None):
    """Every directory under root holding x.tif/y.tif/z.tif (tifxyz, xyz). Label: meta.json 'winding' /
    'winding_index', else the last 'w<int>' in the directory or parent name."""
    import tifffile
    for d in sorted({p.parent for p in Path(root).rglob("x.tif")}):
        if not ((d / "y.tif").exists() and (d / "z.tif").exists()):
            continue
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        w = meta.get("winding", meta.get("winding_index"))
        if w is None:
            m = re.findall(r"w(-?\d+)", d.name) or re.findall(r"w(-?\d+)", d.parent.name)
            if not m:
                print(f"skip {d}: no winding label", file=sys.stderr)
                continue
            w = m[-1]
        g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
        ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
        if (d / "mask.tif").exists():
            mk = tifffile.imread(d / "mask.tif")
            ok &= (mk[..., 0] if mk.ndim == 3 else mk) != 0
        if affine is not None:
            g = g @ affine[0].T + affine[1]
        g[~ok] = np.nan
        yield {"label": int(w), "name": str(d.relative_to(root)), "blacklisted": False, "note": "",
               "grid": g[..., ::-1].astype(np.float32)}


# ----------------------------------------------------------------------------------------------------- geometry
def contour_segments(g, levels, block=256):
    """Marching squares of a zyx vertex grid at planes z = levels (uniformly spaced). Returns (level_idx, y0, x0, y1,
    x1) arrays. A quad crossing more than one level (taller than the level step) only yields its lowest level."""
    off, step = levels[0], levels[1] - levels[0] if len(levels) > 1 else 1.0
    out = []
    for r0 in range(0, g.shape[0] - 1, block):
        sub = g[r0:r0 + block + 1]
        c = [sub[:-1, :-1], sub[:-1, 1:], sub[1:, 1:], sub[1:, :-1]]     # quad corners in cyclic order
        z = np.stack([q[..., 0] for q in c], -1)
        valid = np.isfinite(z).all(-1)
        if not valid.any():
            continue
        zmin, zmax = np.where(valid, z.min(-1), np.inf), np.where(valid, z.max(-1), -np.inf)
        n = np.ceil((zmin - off) / step)
        zl = off + step * n
        sel = valid & (zl <= zmax) & (n >= 0) & (n < len(levels))
        if not sel.any():
            continue
        ii, jj = np.nonzero(sel)
        C = np.stack([q[ii, jj] for q in c], 1).astype(np.float64)        # (m, 4, 3)
        L = zl[ii, jj][:, None]
        s = C[..., 0] - L                                                   # (m, 4)
        sa, sb = s, np.roll(s, -1, 1)
        Ca, Cb = C, np.roll(C, -1, 1)
        cross = (sa < 0) != (sb < 0)
        f = np.where(cross, sa / np.where(cross, sa - sb, 1.0), 0.0)
        P = Ca + (Cb - Ca) * f[..., None]                                   # edge crossing points
        nc = cross.sum(1)
        lev = n[ii, jj].astype(np.int32)
        for want, pairs in ((2, None), (4, ((0, 1), (2, 3)))):
            m = nc == want
            if not m.any():
                continue
            if want == 2:
                e = np.argsort(~cross[m], 1, kind="stable")[:, :2]
                p0 = P[m][np.arange(m.sum()), e[:, 0]]
                p1 = P[m][np.arange(m.sum()), e[:, 1]]
                out.append(np.column_stack([lev[m], p0[:, 1], p0[:, 2], p1[:, 1], p1[:, 2]]))
            else:
                for a, b in pairs:
                    out.append(np.column_stack([lev[m], P[m][:, a, 1], P[m][:, a, 2], P[m][:, b, 1], P[m][:, b, 2]]))
    return np.concatenate(out) if out else np.zeros((0, 5))


def ray_crossings(segs, seg_label, labels, uy, ux, angs):
    """R[angle, label] radius of each mesh on each ray (nan: absent), multi[angle, label]: crosses > MULTI_GAP apart."""
    R = np.full((len(angs), len(labels)), np.nan, np.float32)
    multi = np.zeros((len(angs), len(labels)), bool)
    if len(segs) == 0:
        return R, multi
    p0 = np.column_stack([segs[:, 2] - ux, segs[:, 1] - uy])                # (x, y) relative to the umbilicus
    e = np.column_stack([segs[:, 4] - segs[:, 2], segs[:, 3] - segs[:, 1]])
    li = np.searchsorted(labels, seg_label)
    for ia, a in enumerate(np.radians(angs)):
        d = np.array([np.cos(a), np.sin(a)])
        ced = e[:, 0] * d[1] - e[:, 1] * d[0]
        ok = np.abs(ced) > 1e-9
        ced = np.where(ok, ced, 1.0)
        s = -(p0[:, 0] * d[1] - p0[:, 1] * d[0]) / ced
        t = (p0[:, 0] * e[:, 1] - p0[:, 1] * e[:, 0]) / -ced
        hit = ok & (s >= 0) & (s <= 1) & (t > 0)
        if not hit.any():
            continue
        hl, ht = li[hit], t[hit]
        o = np.lexsort((ht, hl))
        hl, ht = hl[o], ht[o]
        starts = np.r_[0, np.nonzero(np.diff(hl))[0] + 1]
        ends = np.r_[starts[1:], len(hl)]
        for s0, e0 in zip(starts, ends):
            v = ht[s0:e0]
            if v[-1] - v[0] > MULTI_GAP:
                multi[ia, hl[s0]] = True
            else:
                R[ia, hl[s0]] = np.median(v)
    return R, multi


def ray_xy(uy, ux, a, t):
    """(3 tangential, n, 2) yx points along the ray."""
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.stack([np.stack([uy + t * s + dt * c, ux + t * c - dt * s], -1) for dt in DT])


# ------------------------------------------------------------------------------------------------------- extract
class SparsePlanes:
    """Slices z of a (remote) zarr array, filled chunk by chunk only where needed; order-0 sampling."""

    def __init__(self, arr, zs, threads):
        self.arr, self.zs, self.threads = arr, list(zs), threads
        self.cz, self.cy, self.cx = arr.chunks
        self.planes = {z: np.zeros(arr.shape[1:], arr.dtype) for z in self.zs}   # lazy pages (calloc)

    def fill(self, yx):
        """yx: (m, 2) float points that will be sampled on every slice."""
        q = np.rint(yx).astype(np.int64)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < self.arr.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < self.arr.shape[2])]
        keys = np.unique(np.column_stack([q[:, 0] // self.cy, q[:, 1] // self.cx]), axis=0)
        rows = sorted({z // self.cz for z in self.zs})
        jobs = [(r, int(a), int(b)) for r in rows for a, b in keys]

        def get(job):
            r, a, b = job
            blk = np.asarray(self.arr[r * self.cz:(r + 1) * self.cz, a * self.cy:(a + 1) * self.cy,
                                      b * self.cx:(b + 1) * self.cx])
            for z in self.zs:
                if z // self.cz == r:
                    self.planes[z][a * self.cy:a * self.cy + blk.shape[1], b * self.cx:b * self.cx + blk.shape[2]] = \
                        blk[z - r * self.cz]
            return blk.nbytes
        with ThreadPoolExecutor(self.threads) as ex:
            nb = sum(ex.map(get, jobs))
        return len(jobs), nb

    def sample(self, z, yx):
        q = np.rint(yx).astype(np.int64)
        q[..., 0] = np.clip(q[..., 0], 0, self.arr.shape[1] - 1)
        q[..., 1] = np.clip(q[..., 1], 0, self.arr.shape[2] - 1)
        return self.planes[z][q[..., 0], q[..., 1]]


def open_zarr(url):
    import zarr
    so = {"anon": True} if url.startswith("s3://") else None
    return zarr.open_array(url, mode="r", storage_options=so) if so else zarr.open_array(url, mode="r")


def gm_plane(stores, z, scale):
    """Two grad_mag slices bracketing z / scale (float32) from the first store that has data there."""
    k = int(np.floor(z / scale))
    for a in stores:
        if k + 1 >= a.shape[0]:
            continue
        sl = np.asarray(a[k:k + 2]).astype(np.float32)
        if sl.any():
            return sl, k
    return None, k


def extract(a) -> None:
    pre = PRESETS.get(a.scroll, {})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    aff = load_affine(a.mesh_affine) if a.mesh_affine else None
    if a.scroll == "PHerc0139" and not a.meshes:
        sys.path.insert(0, str(REPO / "eval"))
        import spiral_eval as se
        u = se.Umbilicus()
        uz, uy_, ux_ = u.z, u.y, u.x
        meshes = iter_meshes_p0139()
    elif a.scroll == "PHercParis4" and not a.meshes:
        uz, uy_, ux_ = umbilicus_from(a.umbilicus or pre["umbilicus"])
        meshes = iter_meshes_paris4(pre)
    else:
        uz, uy_, ux_ = umbilicus_from(a.umbilicus, aff if a.umbilicus_frame == "mesh" else None)
        meshes = iter_meshes_dir(a.meshes, aff)
    mask_url = a.mask or pre.get("mask")
    gm_urls = a.gm or pre.get("gm", [])
    gm_scale = a.gm_scale or pre.get("gm_scale", 4.0)
    ct_url = None if a.no_ct else (a.ct or (pre.get("ct") if pre.get("ct_profile") else None))
    levels = np.arange(a.z0 + ((a.z_offset - a.z0) % a.z_step), a.z1, a.z_step).astype(float)
    # fresh (confirmation) regions: no level, and no +-dz sample line, may fall inside them (with a margin)
    fz = pre.get("forbidden_z", [])
    fm = pre.get("forbidden_margin", 0.0)
    bad = np.zeros(len(levels), bool)
    for f0, f1 in fz:
        bad |= (levels >= f0 - fm) & (levels < f1 + fm)
    if bad.any():
        print(f"dropping {int(bad.sum())} levels inside forbidden z ranges {fz} (margin {fm}): "
              f"{levels[bad].astype(int).tolist()}", flush=True)
        levels = levels[~bad]
    for f0, f1 in fz:
        assert not ((levels + max(DZ) >= f0 - fm) & (levels + min(DZ) < f1 + fm)).any()
    angs = np.arange(0.0, 360.0, a.angle_step)
    print(f"{len(levels)} z levels {levels[0]:.0f}..{levels[-1]:.0f} x {len(angs)} angles", flush=True)

    # contours of every mesh at every level (one mesh in memory at a time)
    t0 = time.time()
    segs, seg_lab, info = [], [], {}
    for m in meshes:
        s = contour_segments(m["grid"], levels)
        if len(s):
            segs.append(s); seg_lab.append(np.full(len(s), m["label"]))
        if m["label"] in info:
            print(f"warning: label {m['label']} has more than one mesh ({info[m['label']]['name']}, {m['name']}); "
                  f"they are merged", file=sys.stderr)
        info[m["label"]] = {"name": m["name"], "blacklisted": m["blacklisted"], "note": m["note"],
                            "z": [float(np.nanmin(m["grid"][..., 0])), float(np.nanmax(m["grid"][..., 0]))]}
        del m
    segs, seg_lab = np.concatenate(segs), np.concatenate(seg_lab)
    labels = np.array(sorted(info))
    print(f"contours: {len(segs)} segments from {len(labels)} meshes [{time.time() - t0:.0f} s]", flush=True)

    R = np.full((len(levels), len(angs), len(labels)), np.nan, np.float32)
    multi = np.zeros(R.shape, bool)
    umb = np.zeros((len(levels), 2))
    for il, z in enumerate(levels):
        uy, ux = np.interp(z, uz, uy_), np.interp(z, uz, ux_)
        umb[il] = uy, ux
        m = segs[:, 0] == il
        R[il], multi[il] = ray_crossings(segs[m], seg_lab[m], labels, uy, ux, angs)
    del segs, seg_lab

    mask = open_zarr(mask_url)
    # local byte copies of whole chunk rows (path, z_lo, z_hi) are preferred where they cover the row
    mask_local = [(open_zarr(p), lo, hi) for p, lo, hi in pre.get("mask_local", []) if Path(p).exists()] \
        if not a.mask else []
    gms = []
    for u in gm_urls:
        if u.startswith("s3://") or Path(u).exists():
            gms.append(open_zarr(u))
        else:
            print(f"warning: grad_mag store {u} missing; skipped", file=sys.stderr)
    gm_local = [(open_zarr(p), lo, hi) for p, lo, hi in pre.get("gm_local", []) if Path(p).exists()] \
        if not a.gm else []
    gm_remote = open_zarr(pre["gm_remote"]) if (pre.get("gm_remote") and not a.gm) else None
    ct = open_zarr(ct_url) if ct_url else None
    src_log = {"mask_rows": {}, "gm_levels": {}}
    rows = {}
    for il, z in enumerate(levels):
        rows.setdefault(int(z) // mask.chunks[0], []).append(il)
    prof = {"lev": [], "ia": [], "r0": [], "n": [], "off": []}
    M7, GM, CT = [], [], []
    off = 0
    t0 = time.time()
    nbytes = 0
    for r, ils in sorted(rows.items()):
        zs = sorted({int(levels[il]) + dz for il in ils for dz in DZ})
        cz = mask.chunks[0]
        marr, msrc = mask, mask_url
        for arr_l, lo, hi in mask_local:
            if r * cz >= lo and (r + 1) * cz <= hi and arr_l.chunks[0] == cz:
                marr, msrc = arr_l, "local"
                break
        src_log["mask_rows"][int(r)] = msrc if msrc == "local" else "remote"
        sp = SparsePlanes(marr, zs, a.threads)
        rays = []
        for il in ils:
            uy, ux = umb[il]
            for ia, ang in enumerate(angs):
                rr = R[il, ia][np.isfinite(R[il, ia])]
                if len(rr) < 2:
                    continue
                t = np.arange(max(rr.min() - PAD, 0.0), rr.max() + PAD, STEP)
                rays.append((il, ia, ang, t, ray_xy(uy, ux, ang, t)))
        if not rays:
            continue
        allyx = np.concatenate([p.reshape(-1, 2) for *_, p in rays])
        nch, nb = sp.fill(allyx)
        nbytes += nb
        gmp = {}
        remote_ils = []
        for il in ils:
            z = levels[il]
            loc = [arr_l for arr_l, lo, hi in gm_local if lo <= z - gm_scale and z + 2 * gm_scale <= hi]
            gmp[il] = gm_plane(loc + gms, z, gm_scale)
            if gmp[il][0] is not None:
                src_log["gm_levels"][int(z)] = "local" if loc else "listed"
            elif gm_remote is not None:
                remote_ils.append(il)
        if remote_ils:      # stream the remote lasagna slices only where the rays pass
            gz = sorted({int(np.floor(levels[il] / gm_scale)) + d for il in remote_ils for d in (0, 1)})
            gsp = SparsePlanes(gm_remote, gz, a.threads)
            q = allyx / gm_scale
            gsp.fill(np.concatenate([q + np.array([dy, dx]) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]))
            for il in remote_ils:
                k0 = int(np.floor(levels[il] / gm_scale))
                gmp[il] = (np.stack([gsp.planes[k0], gsp.planes[k0 + 1]]).astype(np.float32), k0)
                src_log["gm_levels"][int(levels[il])] = "remote"
            del gsp
        if ct is not None:
            csp = SparsePlanes(ct, zs, a.threads)
            nchc, nbc = csp.fill(allyx)
        for il, ia, ang, t, P in rays:
            z = int(levels[il])
            m7 = np.max([sp.sample(z + dz, P[k]) for dz in DZ for k in range(3)], 0)
            sl, k0 = gmp[il]
            if sl is None:
                g = np.full(len(t), np.nan, np.float32)
            else:
                q = np.stack([np.full(len(t), levels[il] / gm_scale - k0), P[1, :, 0] / gm_scale, P[1, :, 1] / gm_scale])
                g = ndi.map_coordinates(sl, q, order=1, mode="nearest")
            if ct is not None:      # CT profile: mean over the same 9 lines (3 z x 3 tangential)
                c = np.mean([csp.sample(z + dz, P[k]).astype(np.float32) for dz in DZ for k in range(3)], 0)
                CT.append(np.clip(np.rint(c), 0, 255).astype(np.uint8))
            prof["lev"].append(il); prof["ia"].append(ia); prof["r0"].append(t[0]); prof["n"].append(len(t))
            prof["off"].append(off)
            off += len(t)
            M7.append((m7 > 0).astype(np.uint8)); GM.append(g.astype(np.float16))
        del sp
        msg = ""
        if ct is not None:
            del csp
            nbytes += nbc
            msg = f", CT {nchc} chunks"
        print(f"mask row {r} (z {r * mask.chunks[0]}, {src_log['mask_rows'][int(r)]}): levels "
              f"{[int(levels[i]) for i in ils]}, {len(rays)} rays, {nch} chunks{msg} "
              f"[{time.time() - t0:.0f} s, {nbytes / 1e9:.1f} GB decoded]", flush=True)
    extra = {"ct": np.concatenate(CT)} if CT else {}
    np.savez_compressed(out / "rays.npz", levels=levels, angles=angs, labels=labels, R=R, multi=multi, umb=umb,
                        lev=np.array(prof["lev"]), ia=np.array(prof["ia"]), r0=np.array(prof["r0"], np.float32),
                        n=np.array(prof["n"]), off=np.array(prof["off"]), m7=np.concatenate(M7),
                        gm=np.concatenate(GM), **extra)
    meta = {"scroll": a.scroll, "z": [a.z0, a.z1], "z_step": a.z_step, "angle_step": a.angle_step,
            "mask": mask_url, "gm": gm_urls, "gm_scale": gm_scale,
            "meshes": a.meshes or ("sites.json (PHerc0139)" if a.scroll == "PHerc0139" else
                                   pre.get("labels_npz", "") + " + " + pre.get("grids", "")),
            "mesh_affine": a.mesh_affine, "labels": {int(k): v for k, v in info.items()},
            "seams": pre.get("seams", []), "sectors": pre.get("sectors", {}),
            "ct": ct_url, "mask_local": pre.get("mask_local", []) if not a.mask else [],
            "gm_local": pre.get("gm_local", []) if not a.gm else [], "gm_remote": pre.get("gm_remote"),
            "forbidden_z": fz, "forbidden_margin": fm, "sources": src_log,
            "method": {"step": STEP, "pad": PAD, "anchor": ANCHOR, "multi_gap": MULTI_GAP, "split": SPLIT,
                       "dz": DZ, "dt": DT}}
    (out / "rays_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out / 'rays.npz'} ({len(prof['lev'])} rays)")


# ------------------------------------------------------------------------------------------------------- analyse
def sheet_index(t, m7, r):
    """Sheet index of each crossing radius r along one ray (see the module docstring)."""
    lab, n = ndi.label(m7 > 0)
    runs = ndi.find_objects(lab)
    lo = np.array([t[s[0].start] for s in runs]) if n else np.zeros(0)
    hi = np.array([t[s[0].stop - 1] for s in runs]) if n else np.zeros(0)
    return sheet_index_runs(lo, hi, r)


CT_WIN, CT_LEVEL, CT_SMOOTH = 40.0, 0.5, 1.0
CT_PROMS = (0.2, 0.35, 0.5, 0.7)


def ct_runs(t, ct, prom):
    """CT sheets along one ray (no labels): the profile (mean of the 9 sample lines) is smoothed (gaussian, CT_SMOOTH
    vox) and normalised by its local 10th / 90th percentiles over +-CT_WIN vox, pn = (p - p10) / max(p90 - p10, 10);
    sheets are the runs pn > CT_LEVEL inside the scroll (CT > 0), each split at the minimum between two peaks of pn
    of prominence >= prom (a contact with a visible dip). Returns run extents lo, hi (vox) and centres."""
    from scipy.signal import find_peaks
    p = ndi.gaussian_filter1d(ct.astype(np.float32), CT_SMOOTH / STEP)
    w = int(2 * CT_WIN / STEP) + 1
    a = ndi.percentile_filter(p, 10, size=w, mode="nearest")
    b = ndi.percentile_filter(p, 90, size=w, mode="nearest")
    pn = (p - a) / np.maximum(b - a, 10.0)
    on = (pn > CT_LEVEL) & (ct > 0)
    pk = find_peaks(pn, prominence=prom)[0]
    cut = np.zeros(len(p), bool)
    for i0, i1 in zip(pk[:-1], pk[1:]):
        if on[i0] and on[i1] and on[i0:i1 + 1].all():
            cut[i0 + int(np.argmin(pn[i0:i1 + 1]))] = True
    on &= ~cut
    lab, n = ndi.label(on)
    runs = ndi.find_objects(lab)
    lo = np.array([t[s[0].start] for s in runs]) if n else np.zeros(0)
    hi = np.array([t[s[0].stop - 1] for s in runs]) if n else np.zeros(0)
    return lo, hi


def sheet_index_runs(lo, hi, r):
    """Sheet index of each crossing radius r given sheet runs [lo, hi] along one ray."""
    n = len(lo)
    assign = np.full(len(r), -1)
    if n:
        for i, x in enumerate(r):
            d = np.maximum(np.maximum(lo - x, x - hi), 0.0)
            j = int(np.argmin(d))
            if d[j] <= ANCHOR:
                assign[i] = j
    sheets = []                                     # (centre radius, member crossing indices)
    for j in range(n):
        mem = np.nonzero(assign == j)[0]
        if len(mem) == 0:
            sheets.append(((lo[j] + hi[j]) / 2, []))
            continue
        mem = mem[np.argsort(r[mem])]
        grp = np.r_[0, np.cumsum(np.diff(r[mem]) > SPLIT)]
        for g in range(grp[-1] + 1):
            mm = mem[grp == g]
            sheets.append(((lo[j] + hi[j]) / 2 if grp[-1] == 0 else float(r[mm].mean()), list(mm)))
    un = np.nonzero(assign < 0)[0]
    if len(un):
        un = un[np.argsort(r[un])]
        grp = np.r_[0, np.cumsum(np.diff(r[un]) > ANCHOR)]
        for g in range(grp[-1] + 1):
            mm = un[grp == g]
            sheets.append((float(r[mm].mean()), list(mm)))
    sheets.sort(key=lambda s: s[0])
    idx = np.zeros(len(r), int)
    for si, (_, mem) in enumerate(sheets):
        idx[mem] = si
    return idx


def per_ray(D, il, ia, k):
    """t, m7, gm of stored ray k."""
    o, n = D["off"][k], D["n"][k]
    t = D["r0"][k] + STEP * np.arange(n)
    return t, D["m7"][o:o + n], D["gm"][o:o + n].astype(np.float32)


def ray_tables(D):
    """Per ray: label list, radii, m7 sheet index, cumulative gm at each crossing; with a CT profile also the CT
    sheet index per prominence in CT_PROMS (dict) and the local CT sheet spacing (median distance between
    consecutive CT sheet centres within +-50 vox of each crossing, prominence 0.35; label-free packing)."""
    labels = D["labels"]
    has_ct = "ct" in D
    out = []
    for k in range(len(D["lev"])):
        il, ia = int(D["lev"][k]), int(D["ia"][k])
        r = D["R"][il, ia]
        ok = np.nonzero(np.isfinite(r))[0]
        t, m7, g = per_ray(D, il, ia, k)
        rr = r[ok].astype(float)
        idx = sheet_index(t, m7, rr)
        G = np.concatenate([[0.0], np.cumsum((g[1:] + g[:-1]) / 2 / 1000.0 * STEP)])
        Gr = np.interp(rr, t, G) if np.isfinite(g).all() else np.full(len(rr), np.nan)
        row = (il, ia, labels[ok], rr, idx, Gr)
        if has_ct:
            o, n = D["off"][k], D["n"][k]
            ct = D["ct"][o:o + n]
            ict, sp = {}, np.full(len(rr), np.nan)
            for pr in CT_PROMS:
                lo, hi = ct_runs(t, ct, pr)
                ict[pr] = sheet_index_runs(lo, hi, rr)
                if pr == 0.35 and len(lo) > 2:
                    c = (lo + hi) / 2
                    d = np.diff(c)
                    mid = (c[1:] + c[:-1]) / 2
                    for i, x in enumerate(rr):
                        s = d[np.abs(mid - x) <= 50]
                        if len(s):
                            sp[i] = np.median(s)
            row = row + (ict, sp)
        out.append(row)
    return out


def _runs(b):
    """Circular runs of True in b: list of (start, length)."""
    n = len(b)
    if b.all():
        return [(0, n)]
    if not b.any():
        return []
    o = int(np.argmin(b))                 # a False position: unroll from there
    bb = np.roll(b, -o)
    out, i = [], 0
    while i < n:
        if bb[i]:
            j = i
            while j < n and bb[j]:
                j += 1
            out.append(((i + o) % n, j - i))
            i = j
        else:
            i += 1
    return out


def seam_angles(R, multi, angs, min_cover=0.3):
    """theta[level, label] (deg): a mesh's seam = centre of its largest self-overlap run (it crosses the ray twice);
    without overlap, the centre of its largest absence run (partial 'bridge' meshes such as PHerc0139 w035 / w050,
    or a seam gap); gaps filled from the same label at +-2 levels, then from labels +-2 at the level."""
    nl, na, nk = R.shape
    step = angs[1] - angs[0]
    th = np.full((nl, nk), np.nan)
    for il in range(nl):
        for j in range(nk):
            single = np.isfinite(R[il, :, j])
            if single.mean() < min_cover:
                continue
            for b in (multi[il, :, j], ~single):
                rr = _runs(b)
                if rr:
                    st, ln = max(rr, key=lambda x: x[1])
                    th[il, j] = (angs[st] + (ln - 1) * step / 2) % 360
                    break
    filled = th.copy()
    for il in range(nl):
        for j in range(nk):
            if np.isfinite(filled[il, j]) or not np.isfinite(R[il, :, j]).any():
                continue
            v = th[max(0, il - 2):il + 3, j]
            v = v[np.isfinite(v)]
            if not len(v):
                v = th[il, max(0, j - 2):j + 3]
                v = v[np.isfinite(v)]
            if len(v):
                filled[il, j] = np.degrees(np.angle(np.exp(1j * np.radians(v)).mean())) % 360
    return filled


def seam_term(a, th, aref, outward_cw=True):
    """c = 1 if ray angle a lies before the mesh's seam th when walking outward from the reference angle aref."""
    if outward_cw:
        da, dt = (aref - a) % 360, (aref - th) % 360
    else:
        da, dt = (a - aref) % 360, (th - aref) % 360
    return (da < dt).astype(int)


def circ_dist(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def boot_ci(vals, clusters, stat, n=1000, seed=0):
    """Cluster (angle) bootstrap percentile CI of stat(vals)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    groups = [np.nonzero(inv == i)[0] for i in range(len(u))]
    res = []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        res.append(stat(vals[np.concatenate([groups[i] for i in pick])]))
    return np.percentile(res, [2.5, 97.5]), np.array(res)


def analyse(a) -> None:
    """Pairs, verdicts, validation, corrected mapping.

    Pairs: on every ray, consecutive labels among the meshes present (label order). Per ray and pair: signed gap,
    m7 turns T (integer, negative if the outer label lies inside on another sheet) and gm turns.
    gm scale per slab: 1 / median gm of dk = 1 pairs (|gap| >= 4 vox) on the calibration half (even angle index).
    Verdict per (slab [, sector], pair) with >= --min-rays rays: T* = modal m7 T if it holds on >= 50 % of rays and
    the gm median is within 0.5 of it, else 'ambiguous'. ok: T* = dk; duplicate: T* = 0; order: T* < 0;
    overcount (labels skip a sheet): 0 < T* < dk; untraced (a sheet between has no label): T* > dk.
    Confidence = share of angle-cluster bootstrap replicates (1000) that reproduce T*.
    Seam convention: each human mesh is one turn cut at its own seam (the angle where its z-contour is open), and
    seams differ between groups of meshes, so the label difference is the physical turn count only on one side of
    both seams. Expected turns E = dk + c(k2) - c(k), c = [ray lies before the mesh's seam walking outward from the
    reference angle aref]; T above is the label-convention count T_raw - (E - dk), i.e. it equals dk where labels
    are right. aref = one angle for all slabs (best ray agreement of T_raw with E) unless --aref; rays within
    --seam-margin deg of either seam are dropped. Partial bridge meshes (PHerc0139 w035, w050) follow from this.
    Mapping per (slab, sector) and per slab ('all'): robust least squares x_k' - x_k = edge value over the observed
    pairs (weights = rays; modal T for dk <= 2, rounded median for longer edges), x_anchor = anchor; new label =
    round(x_k); confidence = angle-bootstrap share reproducing it. x_gm_int = the same solve on per-edge rounded gm
    turns; 'trusted' (report) = conf >= --min-conf and x_gm_int == new.
    """
    out = Path(a.out)
    D = dict(np.load(out / "rays.npz"))
    meta = json.loads((out / "rays_meta.json").read_text())
    levels, angs, labels = D["levels"], D["angles"], D["labels"]
    seams = [tuple(s) for s in meta.get("seams", [])]
    in_seam = np.zeros(len(angs), bool)
    for s0, s1 in seams:
        in_seam |= (angs >= s0) & (angs < s1)
    edges = np.arange(a.z0 if a.z0 is not None else meta["z"][0], (a.z1 or meta["z"][1]) + 1e-6, a.slab)
    if edges[-1] < meta["z"][1]:
        edges = np.r_[edges, meta["z"][1]]
    slab_of = np.clip(np.searchsorted(edges, levels, side="right") - 1, 0, len(edges) - 2)
    nsec = int(round(360 / a.sector))
    sec_of = (angs // a.sector).astype(int) % nsec

    t0 = time.time()
    tabs = ray_tables(D)
    print(f"{len(tabs)} rays tabulated [{time.time() - t0:.0f} s]", flush=True)
    # pair rows
    has_ct = "ct" in D
    P = {k: [] for k in ("lev", "ia", "k", "k2", "gap", "T", "G", "ra", "rb")}
    if has_ct:
        P |= {f"Tc{i}": [] for i in range(len(CT_PROMS))} | {"sp": []}
    for row in tabs:
        il, ia, lab, rr, idx, Gr = row[:6]
        for j in range(len(lab) - 1):
            P["lev"].append(il); P["ia"].append(ia); P["k"].append(lab[j]); P["k2"].append(lab[j + 1])
            P["gap"].append(rr[j + 1] - rr[j]); P["T"].append(idx[j + 1] - idx[j]); P["G"].append(Gr[j + 1] - Gr[j])
            P["ra"].append(rr[j]); P["rb"].append(rr[j + 1])
            if has_ct:
                ict, sp = row[6], row[7]
                for i, pr in enumerate(CT_PROMS):
                    P[f"Tc{i}"].append(ict[pr][j + 1] - ict[pr][j])
                P["sp"].append(np.nanmean([sp[j], sp[j + 1]]) if np.isfinite([sp[j], sp[j + 1]]).any() else np.nan)
    P = {k: np.array(v) for k, v in P.items()}
    P["dk"] = P["k2"] - P["k"]
    P["slab"], P["sec"] = slab_of[P["lev"]], sec_of[P["ia"]]
    P["half"] = P["ia"] % 2                     # 0 = calibration, 1 = held out
    # ---- seam-aware expected turns. Each mesh is one turn cut at its seam; a mesh's ray index is k or k+1
    # depending on which side of its seam the ray lies, seen from a reference angle aref where the numbering is
    # taken literally. Expected turns E = dk + c(k2) - c(k). Partial bridge meshes (no sheet in their gap) follow.
    li = {int(k): i for i, k in enumerate(labels)}
    TH = seam_angles(D["R"], D["multi"], angs)
    ang_r = angs[P["ia"]]
    thk = TH[P["lev"], [li[int(k)] for k in P["k"]]]
    thk2 = TH[P["lev"], [li[int(k)] for k in P["k2"]]]
    P["theta_k"], P["theta_k2"] = thk, thk2
    P["seam"] = ~(np.isfinite(thk) & np.isfinite(thk2)) | (circ_dist(ang_r, thk) < a.seam_margin) | \
        (circ_dist(ang_r, thk2) < a.seam_margin)
    if a.no_seams:
        P["seam"] = in_seam[P["ia"]]
    T_raw = P["T"].copy()
    aref = np.full(len(edges) - 1, np.nan)
    agree = {}
    for s in range(len(edges) - 1):
        m = (P["slab"] == s) & ~P["seam"]
        if not m.any() or a.no_seams:
            continue
        best = None
        for cand in angs:
            E = P["dk"][m] + seam_term(ang_r[m], thk2[m], cand, a.outward_cw) - seam_term(ang_r[m], thk[m], cand, a.outward_cw)
            sc = float(np.mean(T_raw[m] == E))
            agree.setdefault(s, {})[float(cand)] = round(sc, 4)
            if best is None or sc > best[0] + 1e-9:
                best = (sc, cand)
        aref[s] = best[1]
    if not a.no_seams:
        # one reference angle for the whole scroll (per-slab fits are near-ties and make low slabs unstable):
        # the candidate with the best ray-weighted agreement over all slabs, unless --aref is given
        mm = ~P["seam"]
        tot = {}
        for cand in angs:
            E = P["dk"][mm] + seam_term(ang_r[mm], thk2[mm], cand, a.outward_cw) - \
                seam_term(ang_r[mm], thk[mm], cand, a.outward_cw)
            tot[float(cand)] = float(np.mean(T_raw[mm] == E))
        agree["global"] = {k: round(v, 4) for k, v in tot.items()}
        g_best = max(tot, key=tot.get) if a.aref is None else float(a.aref)
        agree["best_per_slab"] = {str(s): (None if not np.isfinite(x) else float(x)) for s, x in enumerate(aref)}
        aref = np.where(np.isfinite(aref), g_best, np.nan)
        print(f"reference angle {g_best:.0f} deg (agreement {tot.get(g_best, float('nan')):.4f})", flush=True)
    P["E"] = P["dk"].copy()
    if not a.no_seams:
        ok = np.isfinite(aref[P["slab"]])
        ar = np.where(ok, aref[P["slab"]], 0.0)
        P["E"] = P["dk"] + np.where(ok, seam_term(ang_r, thk2, ar, a.outward_cw) - seam_term(ang_r, thk, ar, a.outward_cw), 0)
    P["T_raw"] = T_raw
    P["T"] = T_raw - (P["E"] - P["dk"])         # label-convention turns: equals dk where the labels are right
    ct_cal = None
    if has_ct:
        # CT dip prominence: the one of CT_PROMS whose per-ray CT turns equal 1 most often on the calibration half
        # (even angles) of the label-only clean controls (dk = 1); the odd half stays held out
        ctrl = control_cells(P, edges)
        cm = np.zeros(len(P["k"]), bool)
        for s, k in ctrl:
            cm |= (P["slab"] == s) & (P["k"] == k) & (P["dk"] == 1) & ~P["seam"] & (P["half"] == 0)
        sc = {}
        for i, pr in enumerate(CT_PROMS):
            Tci = P[f"Tc{i}"] - (P["E"] - P["dk"])
            sc[pr] = {"p1": round(float(np.mean(Tci[cm] == 1)), 4), "p0": round(float(np.mean(Tci[cm] == 0)), 4),
                      "p2plus": round(float(np.mean(Tci[cm] >= 2)), 4)}
        ib = int(np.argmax([sc[pr]["p1"] for pr in CT_PROMS]))
        P["Tc_raw"] = P[f"Tc{ib}"]
        P["Tc"] = P["Tc_raw"] - (P["E"] - P["dk"])
        ct_cal = {"prominence": CT_PROMS[ib], "grid": {str(k): v for k, v in sc.items()},
                  "calibration_rays": int(cm.sum()), "controls": len(ctrl),
                  "detector": ct_runs.__doc__}
        print(f"CT prominence {CT_PROMS[ib]} (calibration controls: {sc})", flush=True)

    # gm calibration per slab on the calibration half
    scale = np.full(len(edges) - 1, np.nan)
    for s in range(len(edges) - 1):
        m = (P["slab"] == s) & (P["half"] == 0) & (P["dk"] == 1) & (P["E"] == 1) & ~P["seam"] & \
            (np.abs(P["gap"]) >= 4) & np.isfinite(P["G"])
        if m.sum() >= 20:
            scale[s] = 1.0 / np.median(P["G"][m])
    P["Tg"] = P["G"] * scale[P["slab"]] - (P["E"] - P["dk"])
    np.savez_compressed(out / "pairs_rays.npz", **P)

    def verdict(m, seed=0):
        T, Tg, cl = P["T"][m], P["Tg"][m], P["ia"][m]
        Tc = P["Tc"][m] if has_ct else None
        dk = int(np.bincount(P["dk"][m]).argmax())
        vals, cnt = np.unique(T, return_counts=True)
        mode, share = int(vals[cnt.argmax()]), cnt.max() / len(T)
        gmed = float(np.nanmedian(Tg)) if np.isfinite(Tg).any() else np.nan
        gci, _ = boot_ci(Tg[np.isfinite(Tg)], cl[np.isfinite(Tg)], np.median, 300, seed) if np.isfinite(Tg).sum() > 5 \
            else (np.array([np.nan, np.nan]), None)
        mci, _ = boot_ci(T.astype(float), cl, np.mean, 300, seed)

        def call(Tb, Tgb):
            v, c = np.unique(Tb, return_counts=True)
            md, sh = int(v[c.argmax()]), c.max() / len(Tb)
            gm = np.nanmedian(Tgb) if np.isfinite(Tgb).any() else np.nan
            if sh < 0.5 or not (np.isfinite(gm) and abs(gm - md) < 0.5):
                return None
            return md

        def comb(c_m7, c_ct):
            """Paris4 rule: m7+gm and CT+gm calls; equal -> that count; one ambiguous -> the other; conflict -> None."""
            if c_m7 is None:
                return c_ct
            if c_ct is None or c_ct == c_m7:
                return c_m7
            return None
        Tm7 = call(T, Tg)
        Tct = call(Tc, Tg) if has_ct else None
        Tstar = comb(Tm7, Tct) if has_ct else Tm7
        rng = np.random.default_rng(seed + 1)
        u, inv = np.unique(cl, return_inverse=True)
        groups = [np.nonzero(inv == i)[0] for i in range(len(u))]
        agree, agree_m7 = 0, 0
        for _ in range(a.boot):
            ii = np.concatenate([groups[i] for i in rng.integers(0, len(u), len(u))])
            bm = call(T[ii], Tg[ii])
            agree_m7 += bm == Tm7
            agree += (comb(bm, call(Tc[ii], Tg[ii])) if has_ct else bm) == Tstar

        def cls_of(x):
            return ("ambiguous" if x is None else "ok" if x == dk else "duplicate" if x == 0 else
                    "order" if x < 0 else "overcount" if x < dk else "untraced")
        out = {"dk": dk, "rays": int(m.sum()), "angles": int(len(u)), "gap_med": round(float(np.median(P["gap"][m])), 1),
               "m7_mean": round(float(T.mean()), 3), "m7_ci": [round(float(x), 3) for x in mci],
               "m7_mode": mode, "m7_mode_share": round(float(share), 3),
               "m7_p": {str(v): round(float(c / len(T)), 3) for v, c in zip(vals, cnt)},
               "gm_med": round(gmed, 3) if np.isfinite(gmed) else None,
               "gm_ci": [round(float(x), 3) for x in gci] if np.isfinite(gci).all() else None,
               "T": Tstar, "verdict": cls_of(Tstar), "conf": round(agree / a.boot, 3)}
        if has_ct:
            vc, cc = np.unique(Tc, return_counts=True)
            sp = P["sp"][m]
            out |= {"ct_mean": round(float(Tc.mean()), 3), "ct_mode": int(vc[cc.argmax()]),
                    "ct_mode_share": round(float(cc.max() / len(Tc)), 3),
                    "ct_p": {str(v): round(float(c / len(Tc)), 3) for v, c in zip(vc, cc)},
                    "T_m7": Tm7, "verdict_m7": cls_of(Tm7), "conf_m7": round(agree_m7 / a.boot, 3),
                    "T_ct": Tct, "verdict_ct": cls_of(Tct),
                    "ct_spacing_med": round(float(np.nanmedian(sp)), 1) if np.isfinite(sp).any() else None,
                    "contact_share": round(float(np.mean(np.abs(P["gap"][m]) < 8)), 3)}
        return out

    # ---- verdicts per slab (all angles, seams excluded) and per (slab, sector) (seams included)
    pair_rows = []
    keys = sorted({(int(x), int(y)) for x, y in zip(P["k"], P["k2"])})
    for s in range(len(edges) - 1):
        for (k, k2) in keys:
            base = (P["slab"] == s) & (P["k"] == k) & (P["k2"] == k2)
            m = base & ~P["seam"]
            if m.sum() >= a.min_rays and len(np.unique(P["ia"][m])) >= 4:
                pair_rows.append({"z": [float(edges[s]), float(edges[s + 1])], "sector": "all", "k": k, "k2": k2,
                                  **verdict(m)})
            for c in range(nsec):
                m = base & (P["sec"] == c)
                if m.sum() >= a.min_rays and len(np.unique(P["ia"][m])) >= 3:
                    pair_rows.append({"z": [float(edges[s]), float(edges[s + 1])],
                                      "sector": [c * a.sector, (c + 1) * a.sector], "k": k, "k2": k2, **verdict(m)})
        print(f"slab {edges[s]:.0f}-{edges[s + 1]:.0f}: gm scale {scale[s]:.3f} [{time.time() - t0:.0f} s]", flush=True)

    # ---- validation: label-only clean controls, held-out half
    val = validation(P, edges, a)

    # ---- mapping per (slab, sector) and per slab (all)
    corr = mapping(P, edges, nsec, a, labels)
    res = {"inputs": meta, "anchor": a.anchor, "slab_edges": edges.tolist(), "sector_deg": a.sector,
           "seam_aware": not a.no_seams, "outward_cw": a.outward_cw, "seam_margin_deg": a.seam_margin,
           "aref_per_slab": [None if not np.isfinite(x) else float(x) for x in aref],
           "aref_agreement": {str(k): v for k, v in agree.items()},
           "seam_deg_per_label": {str(int(k)): [None if not np.isfinite(x) else round(float(x), 1) for x in TH[:, j]]
                                  for j, k in enumerate(labels)}, "levels": levels.tolist(),
           "gm_scale_per_slab": [None if not np.isfinite(x) else round(float(x), 4) for x in scale],
           "rules": analyse.__doc__, "validation": val, "pairs": pair_rows, "mapping": corr,
           **({"ct_calibration": ct_cal} if ct_cal else {})}
    (out / "audit.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"saved {out / 'audit.json'} [{time.time() - t0:.0f} s]")


def validation(P, edges, a):
    """Label-only controls (no CT): adjacent dk = 1 pairs whose per-angle z-median spacing is within 0.6-1.6 x the
    local median (pairs within k +- 6 at the same angle +- 1 step) on >= 80 % of angles and whose |gap| >= 4 vox on
    >= 95 % of rays. Calibration = even angle index, evaluation = odd. Reports the m7 / gm turn distribution of the
    held-out control rays and the false-flag rate of per-(slab, pair) verdicts made from the held-out half only."""
    rows = []
    ctrl = control_cells(P, edges)
    has_ct = "Tc" in P
    for s, k in ctrl:
        m = (P["slab"] == s) & (P["k"] == k) & (P["dk"] == 1) & ~P["seam"] & (P["half"] == 1)
        if m.sum() < 5:
            continue
        T, Tg = P["T"][m], P["Tg"][m]
        v, c = np.unique(T, return_counts=True)
        md, sh = int(v[c.argmax()]), c.max() / len(T)
        gm = float(np.nanmedian(Tg))
        call = md if (sh >= 0.5 and abs(gm - md) < 0.5) else None
        row = {"z": [float(edges[s]), float(edges[s + 1])], "k": k, "k2": k + 1, "rays": int(m.sum()),
               "m7_mean": round(float(T.mean()), 3), "m7_share1": round(float((T == 1).mean()), 3),
               "gm_med": round(gm, 3), "T": call, "flagged": call != 1}
        if has_ct:
            Tc = P["Tc"][m]
            vc, cc = np.unique(Tc, return_counts=True)
            mc, shc = int(vc[cc.argmax()]), cc.max() / len(Tc)
            callc = mc if (shc >= 0.5 and abs(gm - mc) < 0.5) else None
            cmb = call if callc is None else (callc if call is None else (call if call == callc else None))
            row |= {"ct_mean": round(float(Tc.mean()), 3), "ct_share1": round(float((Tc == 1).mean()), 3),
                    "T_ct": callc, "T_comb": cmb, "flagged_ct": callc != 1, "flagged_comb": cmb != 1}
        rows.append(row)
    return validation_tail(P, edges, a, ctrl, rows)


def control_cells(P, edges):
    """Label-only clean controls (see validation): list of (slab, k) with k -> k + 1 adjacent and evenly spaced."""
    ctrl = []
    for s in range(len(edges) - 1):
        S = P["slab"] == s
        adj = S & (P["dk"] == 1) & ~P["seam"]
        if not adj.any():
            continue
        ks = np.unique(P["k"][adj])
        angs = np.unique(P["ia"][S])
        med = {}
        for k in ks:
            m = adj & (P["k"] == k)
            med[k] = {ia: np.median(P["gap"][m & (P["ia"] == ia)]) for ia in np.unique(P["ia"][m])}
        for k in ks:
            ratios = []
            for ia, g in med[k].items():
                pool = [med[j].get(ib) for j in ks if abs(j - k) <= 6 for ib in (ia - 1, ia, ia + 1) if med[j].get(ib)]
                pool = [p for p in pool if p and p > 0]
                if len(pool) >= 3:
                    ratios.append(g / np.median(pool))
            m = adj & (P["k"] == k)
            if len(ratios) >= 10 and np.mean((np.array(ratios) >= 0.6) & (np.array(ratios) <= 1.6)) >= 0.8 and \
                    np.mean(np.abs(P["gap"][m]) >= 4) >= 0.95:
                ctrl.append((s, int(k)))
    return ctrl


def validation_tail(P, edges, a, ctrl, rows):
    has_ct = "Tc" in P
    # split half: verdict of each (slab, pair) from the calibration half, re-measured on the held-out half
    def call(m):
        T, Tg = P["T"][m], P["Tg"][m]
        v, c = np.unique(T, return_counts=True)
        md, sh = int(v[c.argmax()]), c.max() / len(T)
        gm = np.nanmedian(Tg) if np.isfinite(Tg).any() else np.nan
        return md if (sh >= 0.5 and np.isfinite(gm) and abs(gm - md) < 0.5) else None
    split = {"pairs": 0, "ok_cal": 0, "flag_cal": 0, "ok_reproduced": 0, "flag_reproduced": 0}
    okT, okG = [], []
    for s in range(len(edges) - 1):
        S = (P["slab"] == s) & ~P["seam"]
        for k, k2 in sorted({(int(x), int(y)) for x, y in zip(P["k"][S], P["k2"][S])}):
            m = S & (P["k"] == k) & (P["k2"] == k2)
            m0, m1 = m & (P["half"] == 0), m & (P["half"] == 1)
            if m0.sum() < a.min_rays // 2 or m1.sum() < a.min_rays // 2:
                continue
            c0, c1 = call(m0), call(m1)
            if c0 is None:
                continue
            split["pairs"] += 1
            if c0 == k2 - k:
                split["ok_cal"] += 1
                split["ok_reproduced"] += c1 == c0
                okT.append(P["T"][m1]); okG.append(P["Tg"][m1])
            else:
                split["flag_cal"] += 1
                split["flag_reproduced"] += c1 == c0
    okT, okG = np.concatenate(okT), np.concatenate(okG)
    split["ok_heldout_m7_mean"] = round(float(okT.mean()), 3)
    split["ok_heldout_m7_p"] = {str(v): round(float((okT == v).mean()), 4) for v in np.unique(okT)}
    split["ok_heldout_gm_median"] = round(float(np.nanmedian(okG)), 3)
    split["ok_heldout_gm_mean"] = round(float(np.nanmean(okG)), 3)
    split["ok_heldout_rays"] = int(len(okT))
    allm = np.zeros(len(P["k"]), bool)
    for s, k in ctrl:
        allm |= (P["slab"] == s) & (P["k"] == k) & (P["dk"] == 1) & ~P["seam"] & (P["half"] == 1)
    T, Tg = P["T"][allm], P["Tg"][allm]
    fl = [r["flagged"] for r in rows]
    extra = {}
    if has_ct:
        def call_c(m, key):
            T, Tg = P[key][m], P["Tg"][m]
            v, c = np.unique(T, return_counts=True)
            md, sh = int(v[c.argmax()]), c.max() / len(T)
            gm = np.nanmedian(Tg) if np.isfinite(Tg).any() else np.nan
            return md if (sh >= 0.5 and np.isfinite(gm) and abs(gm - md) < 0.5) else None

        def call_comb(m):
            cm, cc = call_c(m, "T"), call_c(m, "Tc")
            return cc if cm is None else (cm if cc is None or cc == cm else None)
        sc = {"pairs": 0, "ok_cal": 0, "flag_cal": 0, "ok_reproduced": 0, "flag_reproduced": 0,
              "ok_heldout_ambiguous": 0, "flag_heldout_ambiguous": 0}
        for s in range(len(edges) - 1):
            S = (P["slab"] == s) & ~P["seam"]
            for k, k2 in sorted({(int(x), int(y)) for x, y in zip(P["k"][S], P["k2"][S])}):
                m = S & (P["k"] == k) & (P["k2"] == k2)
                m0, m1 = m & (P["half"] == 0), m & (P["half"] == 1)
                if m0.sum() < a.min_rays // 2 or m1.sum() < a.min_rays // 2:
                    continue
                c0, c1 = call_comb(m0), call_comb(m1)
                if c0 is None:
                    continue
                sc["pairs"] += 1
                tag = "ok" if c0 == k2 - k else "flag"
                sc[f"{tag}_cal"] += 1
                sc[f"{tag}_reproduced"] += c1 == c0
                sc[f"{tag}_heldout_ambiguous"] += c1 is None
        Tc = P["Tc"][allm]
        extra = {"heldout_ct_turns": {"mean": round(float(Tc.mean()), 3),
                                      "p": {str(v): round(float((Tc == v).mean()), 4) for v in np.unique(Tc)}},
                 "false_flag_rate_ct": round(float(np.mean([r["flagged_ct"] for r in rows])), 4) if rows else None,
                 "false_flag_rate_comb": round(float(np.mean([r["flagged_comb"] for r in rows])), 4) if rows else None,
                 "false_flags_comb": int(np.sum([r["flagged_comb"] for r in rows])),
                 "false_flags_comb_ambiguous": int(np.sum([r["T_comb"] is None for r in rows])),
                 "split_half_comb": sc}
    return {"controls": len(rows), "control_rays_heldout": int(allm.sum()), **extra,
            "heldout_m7_turns": {"mean": round(float(T.mean()), 3),
                                 "p": {str(v): round(float((T == v).mean()), 4) for v in np.unique(T)}},
            "heldout_gm_turns": {"median": round(float(np.nanmedian(Tg)), 3),
                                 "p10_90": [round(float(x), 3) for x in np.nanpercentile(Tg, [10, 90])]},
            "false_flag_rate": round(float(np.mean(fl)), 4) if fl else None, "false_flags": int(np.sum(fl)),
            "split_half": split, "rows": rows}


def _lsq(E, lab, anchor):
    li = {k: i for i, k in enumerate(lab)}
    A = np.zeros((len(E) + 1, len(lab)))
    b = np.zeros(len(E) + 1)
    for i, (k, k2, T, w) in enumerate(E):
        sw = np.sqrt(w)
        A[i, li[k2]], A[i, li[k]], b[i] = sw, -sw, sw * T
    A[-1, li[anchor]], b[-1] = 1e3, 1e3 * anchor
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return {k: float(x[li[k]]) for k in lab}


def _component(E, lab, anchor):
    adj = {k: set() for k in lab}
    for k, k2, _, _ in E:
        adj[k].add(k2); adj[k2].add(k)
    comp, stack = {anchor}, [anchor]
    while stack:
        u = stack.pop()
        for v in adj[u] - comp:
            comp.add(v); stack.append(v)
    return comp


def solve(edge_vals, labels_present, anchor, robust=True):
    """Least squares positions x_k (x_anchor = anchor) from edges (k, k2, T, w): x_k2 - x_k = T. Robust: while an
    edge disagrees with the solution by > 0.5, drop the worst one if the anchor's component keeps its labels.
    Returns (x, max residual, dropped edges)."""
    lab = sorted(labels_present)
    if anchor not in lab:
        return None, None, []
    comp = _component(edge_vals, lab, anchor)
    lab = sorted(comp)
    E = [e for e in edge_vals if e[0] in comp and e[1] in comp]
    dropped = []
    while True:
        x = _lsq(E, lab, anchor)
        res = [abs(x[k2] - x[k] - T) for k, k2, T, _ in E]
        if not E or max(res) <= 0.5 or not robust:
            break
        order = np.argsort(res)[::-1]
        for i in order:
            if res[i] <= 0.5:
                i = None
                break
            E2 = E[:i] + E[i + 1:]
            if _component(E2, lab, anchor) == comp:
                dropped.append(list(E[i][:3]) + [E[i][3]])
                E = E2
                break
        else:
            i = None
        if i is None:
            break
    res = [abs(x[k2] - x[k] - T) for k, k2, T, _ in E]
    return x, (max(res) if res else 0.0), dropped


def cell_edges(P, m, use_gm=False, key="T"):
    """(k, k2, value, rays) per pair in the cell: modal m7 T (key 'Tc': CT turns; or median gm turns)."""
    out = []
    for k, k2 in sorted({(int(x), int(y)) for x, y in zip(P["k"][m], P["k2"][m])}):
        mm = m & (P["k"] == k) & (P["k2"] == k2)
        if mm.sum() < 3:
            continue
        if use_gm:
            g = P["Tg"][mm]
            if np.isfinite(g).sum() < 3:
                continue
            out.append((k, k2, float(np.nanmedian(g)), int(mm.sum())))
        else:
            out.append((k, k2, edge_value(P[key][mm], np.ones(int(mm.sum())), k2 - k), int(mm.sum())))
    return out


def edge_value(T, w, dk):
    """Integer turns of one pair in a cell from its rays (weights w): the weighted mode for dk <= 2, the rounded
    weighted median for long edges (dk >= 3), whose per-ray counts spread over several values."""
    T, w = np.asarray(T), np.asarray(w, float)
    if dk >= 3:
        o = np.argsort(T, kind="stable")
        cw = np.cumsum(w[o])
        return int(T[o][np.searchsorted(cw, 0.5 * cw[-1])])
    v = np.unique(T)
    return int(v[int(np.argmax([w[T == u].sum() for u in v]))])


def mapping(P, edges, nsec, a, labels):
    rng = np.random.default_rng(7)
    cells = []
    for s in range(len(edges) - 1):
        for c in list(range(nsec)) + ["all"]:
            m = (P["slab"] == s) if c == "all" else (P["slab"] == s) & (P["sec"] == c)
            m &= ~P["seam"]                     # rays near a seam carry an uncertain expectation
            if m.sum() < 20:
                continue
            present = set(P["k"][m].tolist()) | set(P["k2"][m].tolist())
            ev = cell_edges(P, m)
            x, resid, dropped = solve(ev, present, a.anchor)
            if x is None:
                continue
            # gm positions: same robust solve on median gm turns, edges dropped above excluded
            dk_ = {(d[0], d[1]) for d in dropped}
            evg = [e for e in cell_edges(P, m, True) if (e[0], e[1]) not in dk_]
            xg, _, _ = solve(evg, present, a.anchor)
            # integer gm positions: each gm edge rounded first, so per-edge calibration drift does not accumulate
            xgi, _, _ = solve([(k, k2, int(round(v)), w) for k, k2, v, w in evg], present, a.anchor)
            xci = None
            if "Tc" in P:       # CT positions: the same robust solve on modal CT turns (dropped m7 edges excluded)
                xci, _, _ = solve([e for e in cell_edges(P, m, key="Tc") if (e[0], e[1]) not in dk_], present,
                                  a.anchor)
            ia_u = np.unique(P["ia"][m])
            loc = np.nonzero(m)[0]
            lk, lk2, lT, lia = P["k"][loc], P["k2"][loc], P["T"][loc], np.searchsorted(ia_u, P["ia"][loc])
            emask = [((lk == k) & (lk2 == k2)) for k, k2, _, _ in ev]
            boots = {k: [] for k in x}
            for _ in range(a.boot_map):
                cnt = np.bincount(rng.integers(0, len(ia_u), len(ia_u)), minlength=len(ia_u))
                w = cnt[lia]
                evb = []
                for (k, k2, _, _), em in zip(ev, emask):
                    ww = w[em]
                    if ww.sum() == 0:
                        continue
                    evb.append((k, k2, edge_value(lT[em], ww, k2 - k), int(ww.sum())))
                xb, _, _ = solve(evb, present, a.anchor)
                for k in x:
                    boots[k].append(round(xb[k]) if xb and k in xb else None)
            mp = {}
            for k in sorted(x):
                new = int(round(x[k]))
                nr = int((m & ((P["k"] == k) | (P["k2"] == k))).sum())
                mp[str(k)] = {"new": new, "shift": new - k, "x_m7": round(x[k], 2),
                              "x_gm": round(xg[k], 2) if xg and k in xg else None,
                              "x_gm_int": int(round(xgi[k])) if xgi and k in xgi else None,
                              "conf": round(float(np.mean([b == new for b in boots[k]])), 3), "rays": nr}
                if "Tc" in P:
                    mp[str(k)]["x_ct_int"] = int(round(xci[k])) if xci and k in xci else None
            cells.append({"z": [float(edges[s]), float(edges[s + 1])],
                          "sector": "all" if c == "all" else [c * a.sector, (c + 1) * a.sector],
                          "max_resid": round(float(resid), 2), "dropped_edges": dropped, "map": mp})
        print(f"mapping slab {edges[s]:.0f}-{edges[s + 1]:.0f} done", flush=True)
    return cells


# -------------------------------------------------------------------------------------------------------- report
def report(a) -> None:
    """audit.json -> corrections json + csv (per cell old -> new), pair verdict csv, summary tables on stdout.

    Shift of a mesh in a cell is trusted when conf >= --min-conf and the integer gm position agrees (x_gm_int).
    Affected share = crossings (rays x meshes, weighted by radius ~ mesh area) whose trusted new label != old."""
    import csv
    out = Path(a.out)
    A = json.loads((out / "audit.json").read_text())
    P = dict(np.load(out / "pairs_rays.npz"))
    D = np.load(out / "rays.npz")
    info = {int(k): v for k, v in A["inputs"]["labels"].items()}
    edges = np.array(A["slab_edges"])
    levels, angs, R = D["levels"], D["angles"], D["R"]
    sec_w = A["sector_deg"]
    corr_path = Path(a.corrections) if a.corrections else out / "label_corrections.json"

    def trusted(m):     # with a CT profile the CT solve must agree too
        return m["conf"] >= a.min_conf and m.get("x_gm_int") == m["new"] and m.get("x_ct_int", m["new"]) == m["new"]

    cells = A["mapping"]
    # per crossing: corrected label from its (slab, sector) cell; affected share weighted by radius
    cell_idx = {}
    for c in cells:
        if c["sector"] != "all":
            cell_idx[(c["z"][0], c["sector"][0])] = c
    slab_of = np.clip(np.searchsorted(edges, levels, side="right") - 1, 0, len(edges) - 2)
    tot = {"w": 0.0, "w_changed": 0.0, "w_untrusted": 0.0, "w_nocell": 0.0}
    per_label = {}
    for il in range(len(levels)):
        for ia, ang in enumerate(angs):
            c = cell_idx.get((float(edges[slab_of[il]]), float((ang // sec_w) * sec_w)))
            for j, k in enumerate(D["labels"]):
                r = R[il, ia, j]
                if not np.isfinite(r):
                    continue
                k = int(k)
                pl = per_label.setdefault(k, {"w": 0.0, "changed": 0.0, "untrusted": 0.0, "nocell": 0.0, "shifts": {}})
                tot["w"] += r; pl["w"] += r
                m = c["map"].get(str(k)) if c else None
                if m is None:
                    tot["w_nocell"] += r; pl["nocell"] += r
                elif not trusted(m):
                    tot["w_untrusted"] += r; pl["untrusted"] += r
                elif m["shift"] != 0:
                    tot["w_changed"] += r; pl["changed"] += r
                if m is not None and trusted(m):
                    pl["shifts"][m["shift"]] = pl["shifts"].get(m["shift"], 0.0) + r
    # pair verdicts (all-angle rows) -> issues
    issues = {"duplicate": [], "untraced": [], "overcount": [], "order": []}
    for row in A["pairs"]:
        if row["verdict"] in issues and row["conf"] >= a.min_conf:
            issues[row["verdict"]].append(row)

    def zr(rows):
        return sorted({tuple(r["z"]) for r in rows})

    def group(rows, key):
        g = {}
        for r in rows:
            g.setdefault(key(r), []).append(r)
        return g
    dup = []
    for (k, k2), rr in sorted(group(issues["duplicate"], lambda r: (r["k"], r["k2"])).items()):
        allr = [r for r in rr if r["sector"] == "all"]
        dup.append({"labels": [k, k2], "meshes": [info[k]["name"], info[k2]["name"]],
                    "organiser_note": [info[k]["note"], info[k2]["note"]],
                    "slabs_all_angles": [list(z) for z in zr(allr)], "sector_cells": len(rr) - len(allr),
                    "median_gap_vox": float(np.median([r["gap_med"] for r in rr])),
                    "m7_mean": float(np.mean([r["m7_mean"] for r in rr]))})
    def issue_list(kind):
        outl = []
        for (k, k2), rr in sorted(group(issues[kind], lambda r: (r["k"], r["k2"])).items()):
            allr = [r for r in rr if r["sector"] == "all"]
            secs = [r for r in rr if r["sector"] != "all"]
            outl.append({"between": [k, k2], "labels_dk": rr[0]["dk"],
                         "slabs_all_angles": [{"z": r["z"], "turns_m7": r["m7_mean"], "turns_gm": r["gm_med"],
                                               "gm_ci": r["gm_ci"], "T": r["T"], "conf": r["conf"], "rays": r["rays"]}
                                              for r in allr],
                         "sector_cells": [{"z": r["z"], "sector": r["sector"], "T": r["T"], "conf": r["conf"],
                                           "rays": r["rays"]} for r in secs]})
        return outl
    # per-mesh shift summary (trusted cells only)
    mesh = []
    for k in sorted(per_label):
        pl = per_label[k]
        sh = {int(x): v / pl["w"] for x, v in pl["shifts"].items()}
        modal = max(sh, key=sh.get) if sh else None
        cl = [c for c in cells if c["sector"] != "all" and str(k) in c["map"] and trusted(c["map"][str(k)])]
        by = {}
        for c in cl:
            by.setdefault(c["map"][str(k)]["shift"], []).append((c["z"][0], c["sector"][0]))
        mesh.append({"label": k, "mesh": info[k]["name"], "blacklisted_by_organisers": info[k]["blacklisted"],
                     "area_share_changed": round(pl["changed"] / pl["w"], 3),
                     "area_share_untrusted": round((pl["untrusted"] + pl["nocell"]) / pl["w"], 3),
                     "modal_shift": modal, "shift_area_shares": {str(x): round(v, 3) for x, v in sorted(sh.items())},
                     "cells_by_shift": {str(x): len(v) for x, v in sorted(by.items())},
                     "slabs_by_shift": {str(x): sorted({z for z, _ in v}) for x, v in sorted(by.items())}})
    # local (pair-level) view, independent of how errors chain along the numbering: ray-weighted share of
    # label-adjacent pairs in (slab, sector) cells with a confident verdict, and the split of those by verdict
    pw = {"rays": 0, "confident": 0}
    by_v = {}
    for row in A["pairs"]:
        if row["sector"] == "all":
            continue
        pw["rays"] += row["rays"]
        if row["verdict"] != "ambiguous" and row["conf"] >= a.min_conf:
            pw["confident"] += row["rays"]
            by_v[row["verdict"]] = by_v.get(row["verdict"], 0) + row["rays"]
    pair_view = {"rays": pw["rays"], "confident_share": round(pw["confident"] / max(pw["rays"], 1), 4),
                 "verdict_share_of_confident": {k: round(v / max(pw["confident"], 1), 4) for k, v in sorted(by_v.items())}}
    res = {"title": f"CT-consistent winding numbering proposal for the {A['inputs']['scroll']} human meshes",
           "generated_by": "eval/winding_audit.py report", "anchor_label": A["anchor"],
           "convention": ("new label = anchor + physical turns from the anchor mesh's sheet, counted per (z slab, "
                          "angle sector) on rays from the umbilicus: m7 sheet-index differences per label-adjacent "
                          "pair (modal over rays), solved by robust least squares; x_gm = the same from lasagna "
                          "grad_mag turns. Relative numbering: the anchor keeps its label."),
           "trusted_rule": f"conf >= {a.min_conf} (angle bootstrap) and x_gm_int == new (gm turns rounded per edge, "
                           "then solved the same way)" + ("; with a CT profile also x_ct_int == new (modal CT-sheet "
                                                          "turns, same solve)" if "ct_calibration" in A else ""),
           "inputs": {k: A["inputs"][k] for k in ("scroll", "z", "z_step", "angle_step", "mask", "gm", "meshes")},
           "slab_edges": A["slab_edges"], "sector_deg": sec_w,
           "affected": {"crossings_radius_weighted": {
               "changed": round(tot["w_changed"] / tot["w"], 4), "untrusted": round(tot["w_untrusted"] / tot["w"], 4),
               "no_cell": round(tot["w_nocell"] / tot["w"], 4)}, "pairs_ray_weighted": pair_view},
           "validation": {k: v for k, v in A["validation"].items() if k != "rows"},
           "duplicates": dup, "untraced_sheets": issue_list("untraced"), "overcounts": issue_list("overcount"),
           "order_errors": issue_list("order"), "meshes": mesh,
           "cells": [{"z": c["z"], "sector": c["sector"], "max_resid": c["max_resid"],
                      "dropped_edges": c["dropped_edges"], "map": c["map"]} for c in cells]}
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    corr_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(corr_path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        hct = any("x_ct_int" in m for c in cells for m in c["map"].values())
        w.writerow(["z0", "z1", "sector_deg0", "sector_deg1", "old_label", "mesh", "new_label", "shift", "conf",
                    "x_m7", "x_gm", "x_gm_int"] + (["x_ct_int"] if hct else []) + ["rays", "trusted"])
        for c in cells:
            s0, s1 = ("all", "all") if c["sector"] == "all" else c["sector"]
            for k, m in c["map"].items():
                w.writerow([c["z"][0], c["z"][1], s0, s1, k, info[int(k)]["name"], m["new"], m["shift"], m["conf"],
                            m["x_m7"], m["x_gm"], m.get("x_gm_int")] + ([m.get("x_ct_int")] if hct else []) +
                           [m["rays"], int(trusted(m))])
    with open(out / "audit_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        cx = ["ct_mean", "ct_mode", "ct_mode_share", "T_m7", "verdict_m7", "conf_m7", "T_ct", "verdict_ct",
              "ct_spacing_med", "contact_share"]
        hct = bool(A["pairs"]) and "ct_mean" in A["pairs"][0]
        w.writerow(["z0", "z1", "sector_deg0", "sector_deg1", "k", "k2", "dk", "rays", "angles", "gap_med_vox",
                    "m7_mean", "m7_ci_lo", "m7_ci_hi", "m7_mode", "m7_mode_share", "gm_med", "gm_ci_lo", "gm_ci_hi",
                    "T", "verdict", "conf"] + (cx if hct else []))
        for r in A["pairs"]:
            s0, s1 = ("all", "all") if r["sector"] == "all" else r["sector"]
            gci = r["gm_ci"] or [None, None]
            w.writerow([r["z"][0], r["z"][1], s0, s1, r["k"], r["k2"], r["dk"], r["rays"], r["angles"], r["gap_med"],
                        r["m7_mean"], *r["m7_ci"], r["m7_mode"], r["m7_mode_share"], r["gm_med"], *gci, r["T"],
                        r["verdict"], r["conf"]] + ([r.get(c) for c in cx] if hct else []))
    # stdout summary
    print(f"saved {corr_path} and {corr_path.with_suffix('.csv')}, {out / 'audit_pairs.csv'}")
    print("affected (radius-weighted crossings):", res["affected"])
    v = res["validation"]
    print(f"validation: {v['controls']} label-clean controls, held-out m7 {v['heldout_m7_turns']}, gm "
          f"{v['heldout_gm_turns']}, false flags {v['false_flags']} ({v['false_flag_rate']})")
    print("split half:", v["split_half"])
    sym = {"ok": ".", "duplicate": "D", "untraced": "U", "overcount": "O", "order": "X", "ambiguous": "?"}
    allrows = [r for r in A["pairs"] if r["sector"] == "all"]
    slabs = sorted({tuple(r["z"]) for r in allrows})
    keys = sorted({(r["k"], r["k2"]) for r in allrows})
    print("pair verdicts, all angles (seams excluded): . ok, D duplicate, U untraced sheet, O overcount, X order, "
          "? ambiguous, lower case = conf < min-conf")
    print(" " * 9 + " ".join(f"{int(z0):>5d}" for z0, _ in slabs))
    for k, k2 in keys:
        line = []
        for z in slabs:
            r = next((r for r in allrows if (r["k"], r["k2"]) == (k, k2) and tuple(r["z"]) == z), None)
            if r is None:
                line.append("     ")
            else:
                c = sym[r["verdict"]]
                line.append(f"{(c if r['conf'] >= a.min_conf else c.lower()):>5s}")
        print(f"w{k:03d}-w{k2:03d}" + " ".join(line))
    print("pooled (all-angle) map, shift per label and slab (trusted; '~' untrusted):")
    allc = {tuple(c["z"]): c for c in cells if c["sector"] == "all"}
    labs = sorted(per_label)
    print(" " * 5 + " ".join(f"{int(z0):>5d}" for z0, _ in slabs))
    for k in labs:
        line = []
        for z in slabs:
            m = allc.get(z, {}).get("map", {}).get(str(k)) if z in allc else None
            line.append("     " if m is None else f"{m['shift']:>+4d}{'' if trusted(m) else '~'}".rjust(5))
        print(f"w{k:03d}" + " ".join(line))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("cmd", choices=["extract", "analyse", "report", "audit"])
    ap.add_argument("--out", required=True, help="output directory (rays.npz, audit.json, ...)")
    g = ap.add_argument_group("inputs (extract)")
    g.add_argument("--scroll", default=None, help=f"preset: {list(PRESETS)} (defaults for the other inputs)")
    g.add_argument("--meshes", help="directory of tifxyz human meshes (one winding each); default for PHerc0139: "
                                    "sites.json 2.399 meshes + published affine")
    g.add_argument("--mesh-affine", help="json with 'transformation_matrix' (xyz, mesh frame -> volume L0)")
    g.add_argument("--umbilicus", help="umbilicus json (control_points x,y,z)")
    g.add_argument("--umbilicus-frame", choices=["volume", "mesh"], default="volume")
    g.add_argument("--mask", help="binary surface mask zarr at volume L0 (local path or s3://)")
    g.add_argument("--gm", nargs="+", help="lasagna grad_mag zarr(s); the first with data at a z is used")
    g.add_argument("--gm-scale", type=float, help="L0 voxels per grad_mag voxel (PHerc0139 stores: 4)")
    g.add_argument("--ct", help="CT zarr array in the mask frame (default: preset 'ct'; PHercParis4 = level 2)")
    g.add_argument("--no-ct", action="store_true", help="do not sample the CT profile")
    g = ap.add_argument_group("sampling (extract)")
    g.add_argument("--z0", type=float)
    g.add_argument("--z1", type=float)
    g.add_argument("--z-step", type=float, default=48.0)
    g.add_argument("--z-offset", type=float, default=24.0, help="levels at z = offset + k * step")
    g.add_argument("--angle-step", type=float, default=5.0)
    g.add_argument("--threads", type=int, default=16)
    g = ap.add_argument_group("analysis")
    g.add_argument("--slab", type=float, default=500.0, help="z slab height for verdicts and the mapping")
    g.add_argument("--sector", type=float, default=30.0, help="angle sector width (deg)")
    g.add_argument("--anchor", type=int, required=False, help="label kept as is (corrected labels are relative to it)")
    g.add_argument("--min-rays", type=int, default=12)
    g.add_argument("--seam-margin", type=float, default=10.0, help="drop rays within this many deg of a pair's seams")
    g.add_argument("--aref", type=float, default=None, help="reference angle where labels are literal (default: "
                   "one angle for all slabs, the one that maximises agreement of raw m7 turns with the seam-aware expectation)")
    g.add_argument("--outward-ccw", dest="outward_cw", action="store_false",
                   help="the spiral winds outward counter-clockwise in the umbilicus angle convention (default cw)")
    g.add_argument("--no-seams", action="store_true", help="label difference = expected turns (old behaviour)")
    g.add_argument("--boot", type=int, default=300, help="bootstrap replicates for verdict confidence")
    g.add_argument("--boot-map", type=int, default=200, help="bootstrap replicates for mapping confidence")
    g.add_argument("--min-conf", type=float, default=0.8, help="report: confidence needed to trust a verdict / shift")
    g.add_argument("--corrections", help="report: corrections json path (csv next to it); default OUT/label_corrections.json")
    a = ap.parse_args()
    if a.cmd in ("extract", "audit"):
        if a.z0 is None or a.z1 is None:
            ap.error("--z0 and --z1 are required")
        if a.scroll not in PRESETS and not (a.meshes and a.umbilicus and a.mask and a.gm):
            ap.error("without --scroll preset, --meshes --umbilicus --mask --gm are required")
        extract(a)
    if a.cmd in ("analyse", "audit"):
        if a.anchor is None:
            ap.error("--anchor is required for analyse")
        analyse(a)
    if a.cmd in ("report", "audit"):
        report(a)


if __name__ == "__main__":
    main()
