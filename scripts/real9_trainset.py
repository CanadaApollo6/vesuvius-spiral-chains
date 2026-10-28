"""Real ~9 um surface TRAINING set from public segment meshes of scrolls other than PHerc0139
(notes/2026-10-18-real9-trainset.md). Generalizes scripts/p0139_testset.py (imports its helpers) to any scroll with an
ESRF 9.362 / 8.64 um volume and tifxyz meshes.

  uv run python scripts/real9_trainset.py fetch  [--scroll PHerc0814,...]   # meshes -> data/meshes/<scroll>/
  uv run python scripts/real9_trainset.py mine   [--scroll ...] [--n-max 150]  # candidates (_work/<scroll>/candidates.json)
  uv run python scripts/real9_trainset.py build  [--scroll ...] [--workers 6]  # stream crops, register, rasterize, gates
  uv run python scripts/real9_trainset.py finalize                            # per-scroll sets, manifest, Dataset103, preview

Geometry. Each segment is used through its mesh on the volume it was traced on (`<id>-on-<original_volume>.tifxyz`,
`original_volume_id` in the bucket metadata.json) and mapped to the target 9.362 / 8.64 um volume with the published
affine `samples.<scroll>.sample.properties.volume_transforms[from=original][to=target]` (xyz, p_tgt = A p_src + t; the
same file format and convention as transform_0139_2p399_to_9p362.json, whose matrix it reproduces for PHerc0139).
Meshes traced on the target volume itself use the identity. Segments with no tifxyz or no published transform are skipped.
Non-native groups get a per-crop NCC translation against the source volume rendered at ~9 um (as for p0139); native
meshes are used as they are (shift 0). Coordinates: tifxyz and affines are xyz; everything else zyx in target L0 voxels.
PHerc0139 is excluded (it is the held-out test scroll).

Output (local and private: CC BY-NC / ash2txt data agreement; never upload or commit):
  data/train/real9/<scroll>/imagesTr/<stem>_0000.tif  uint8 CT, 192^3 from the target L0 (not resampled)
  data/train/real9/<scroll>/labelsTr/<stem>.tif       uint8 {0,1}: 3-vox band (EDT <= 1 around the 1-vox mesh raster)
  data/train/real9/<scroll>/ignoreTr/<stem>.tif       uint8 {0,1}: off-papyrus / rejected mesh voxels and unlabelled material
  data/train/real9/manifest.json                      provenance, transforms, gates, per-crop stats
  data/nnunet/nnUNet_raw/Dataset103_Real9/            labels 0 background, 1 surface, 2 ignore (= ignoreTr)
"""

import argparse
import json
import os
import re
import shutil
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
sys.path.insert(0, str(REPO / "scripts"))
from p0139_testset import (CELL, CORE, DEEP_D, MIN_SEP, N, OFF_D, ON_D, band_from_surface, cast_pairs,  # noqa: E402
                           gates, grid_normals, ncc_masked, parabolic, read_box, shift_search)

BUCKET = "vesuvius-challenge-open-data"
S3 = f"s3://{BUCKET}/"
OUT = REPO / "data" / "train" / "real9"
WORK = OUT / "_work"
META = OUT / "_meta" / "metadata.json"
META_URL = f"https://{BUCKET}.s3.amazonaws.com/metadata.json"
MESHES = REPO / "data" / "meshes"
DS = REPO / "data" / "nnunet" / "nnUNet_raw" / "Dataset103_Real9"
PREVIEW = REPO / "data" / "scratch" / "real9_preview.png"
EXCLUDE = {"PHerc0139"}  # held-out test scroll: never in training
SCROLLS = {              # target = the ESRF ~9 um scan of each scroll
    "PHerc0814": "20250804134230",
    "PHerc0500P2": "20250820143440",
    "PHerc0009B": "20250521125136",
    "PHerc1447": "20250521151220",
}
SHORT = {"PHerc0814": "p0814", "PHerc0500P2": "p0500p2", "PHerc0009B": "p0009b", "PHerc1447": "p1447"}
SEED = 18

# mining
MINE_STEP = 8.0      # mesh grids are bilinearly densified to <= ~8 target vox spacing for mining
QUERY_SPACING = 10.0 # mining query points ~ every 10 vox along the sheet
AREA_MIN = 16384.0   # >= one full 128^2 sheet-equivalent of mesh area in the 128^3 core
FRAME_OUT_MAX = 0.01      # segment: share of mesh points outside the target volume
FRAME_MASKED_MAX = 0.25   # segment: share of mesh points on masked (zero) CT in the L4 overview
DEDUP_D = 4.0        # surface of a smaller segment within 4 vox of a larger one = the same sheet traced twice (traces
                     # wander 2-4 vox); removed voxels go to ignore, so a genuine < 4 vox contact is never a false negative
# registration (non-native groups)
SHIFT = {"default": 6, "PHerc0500P2": 10}
BLUR = 0.5
# training gates (looser than the p0139 test set; failing instances go to ignore instead of failing the crop)
NCC_MIN = 0.80
SHIFT_SPREAD_MAX = 2.5
INST_ONPAP_MIN, INST_OFF_MAX, NORMAL_AGREE_MIN, DEEP_TILES_MAX, CROSS_MAX = 0.50, 0.30, 0.70, 2, 0.05
FRAG_MIN = 500           # instance (connected piece) smaller than this: fragment -> ignore
POOLED_ONPAP_MIN = 0.65   # crop: kept label voxels <= 1 vox from material
LABEL_CORE_MIN = 3000     # crop: kept 1-vox surface voxels in the core
CT_VALID_MIN = 0.30       # crop: share of non-zero (unmasked) CT voxels
LABEL_ON_CT_MIN = 0.95    # crop: share of label voxels on non-zero CT
MATERIAL_MIN = 0.05       # crop: material (Otsu) share of the core
UNL_MAX = 0.80            # crop: unlabelled-material share in the core (rest is ignore anyway; > 0.8 = little signal)


# --------------------------------------------------------------------------------------------------------- metadata
def meta() -> dict:
    if not META.exists():
        import gzip
        import urllib.request
        META.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(META_URL).read()
        META.write_bytes(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    return json.loads(META.read_text())


def tag(long_id: str) -> str:
    """'20250804134230-9.362um-1.2m-113keV-masked.zarr' -> '20250804134230-9.362um'."""
    return "-".join(long_id.split("-")[:2])


def scroll_cfg(scroll: str) -> dict:
    """Target volume, usable segments with their transforms, and skip reasons, from the bucket metadata."""
    assert scroll not in EXCLUDE, f"{scroll} is excluded from training"
    s = meta()["samples"][scroll]
    vols = s["volumes"]
    tgt = SCROLLS[scroll]
    tfs = {}
    for vt in s["sample"]["properties"].get("volume_transforms") or []:
        for t in vt["transforms"]:
            tfs[(vt["from_volume_id"], t["to_volume_id"])] = t
    segs, skipped = [], {}
    for k in sorted(s["segments"]):
        v = s["segments"][k]
        ov = v.get("original_volume_id")
        types = {d["type"] for d in v.get("data", [])}
        if ov not in vols:
            skipped[v["long_id"]] = f"original volume {ov} not in the bucket"
            continue
        # the mesh on the volume it was traced on: a tifxyz-transformed entry "<id>-on-<original_volume>-<um>.tifxyz"
        on_src = [o["path"] for d in v.get("data", []) if d["type"] == "tifxyz-transformed" for o in d["origins"]
                  if re.search(rf"-on-{ov}-[^/]*\.tifxyz/?$", o["path"])]
        if not on_src:
            skipped[v["long_id"]] = f"no -on-{tag(vols[ov]['long_id'])} tifxyz mesh (types: {sorted(types)})"
            continue
        if ov == tgt:
            M, how = np.hstack([np.eye(3), np.zeros((3, 1))]), "identity (traced on the target volume)"
        elif (ov, tgt) in tfs:
            M, how = np.array(tfs[(ov, tgt)]["matrix"]), f"published affine {tfs[(ov, tgt)]['derivation_path']}"
        else:
            skipped[v["long_id"]] = f"no published transform {tag(vols[ov]['long_id'])} -> {tag(vols[tgt]['long_id'])}"
            continue
        segs.append({"name": v["long_id"], "seg_id": v["id"], "src_vol": ov, "src_tag": tag(vols[ov]["long_id"]),
                     "src_long": vols[ov]["long_id"], "src_um": vols[ov]["properties"]["pixel_size_um"],
                     "native": ov == tgt, "A": M[:, :3].tolist(), "t": M[:, 3].tolist(), "transform": how,
                     "mesh_path": on_src[0].rstrip("/")})
    for i, sg in enumerate(segs):
        sg["id"] = i + 1
    return {"scroll": scroll, "target": tgt, "target_long": vols[tgt]["long_id"], "target_tag": tag(vols[tgt]["long_id"]),
            "vox_um": vols[tgt]["properties"]["pixel_size_um"], "url": S3 + f"{scroll}/volumes/{vols[tgt]['long_id']}",
            "segments": segs, "skipped": skipped}


def vol_url(scroll: str, long_id: str) -> str:
    return S3 + f"{scroll}/volumes/{long_id}"


def mesh_dir(scroll: str, sg: dict) -> Path:
    return MESHES / scroll / sg["name"] / sg["src_tag"]


# ------------------------------------------------------------------------------------------------------------ fetch
def fetch(args) -> None:
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    for scroll in args.scrolls:
        cfg = scroll_cfg(scroll)
        print(f"{scroll}: {len(cfg['segments'])} usable segments, skipped {len(cfg['skipped'])}", flush=True)
        for name, why in cfg["skipped"].items():
            print(f"  skip {name}: {why}")
        for sg in cfg["segments"]:
            d = mesh_dir(scroll, sg)
            if all((d / f).exists() for f in ("x.tif", "y.tif", "z.tif", "meta.json")):
                continue
            hit = [f"{BUCKET}/{sg['mesh_path']}"]
            d.mkdir(parents=True, exist_ok=True)
            for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
                fs.get(f"{hit[0]}/{f}", str(d / f))
            print(f"  {sg['name']} <- {hit[0].split('/')[-1]}", flush=True)


def load_mesh(scroll: str, sg: dict) -> np.ndarray:
    """(H, W, 3) zyx in target L0 voxels; invalid = nan."""
    d = mesh_dir(scroll, sg)
    g = np.stack([tifffile.imread(d / f"{c}.tif") for c in "xyz"], -1).astype(np.float64)
    ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
    p = g @ np.asarray(sg["A"]).T + np.asarray(sg["t"])
    p[~ok] = np.nan
    return p[..., ::-1].copy()


def grid_step(p: np.ndarray) -> float:
    e = np.concatenate([np.linalg.norm(np.diff(p, axis=a), axis=-1).ravel() for a in (0, 1)])
    e = e[np.isfinite(e)]
    return float(np.median(e)) if len(e) else float("nan")


def densify(p: np.ndarray, z: int) -> np.ndarray:
    if z <= 1:
        return p
    ii = np.arange(0, p.shape[0] - 1 + 1e-9, 1 / z)
    jj = np.arange(0, p.shape[1] - 1 + 1e-9, 1 / z)
    I, J = np.meshgrid(ii, jj, indexing="ij")
    return np.stack([map_coordinates(p[..., c], [I, J], order=1, mode="nearest") for c in range(3)], -1)


# ------------------------------------------------------------------------------------------------------------- mine
def box_sum(g: np.ndarray, k: int) -> np.ndarray:
    """Sum of the k^3 cells around every cell corner (k even); output shape = g.shape + 1."""
    cs = np.pad(g, [(1, 0)] * 3).cumsum(0).cumsum(1).cumsum(2)
    h = k // 2
    idx = [(np.clip(np.arange(n + 1) - h, 0, n), np.clip(np.arange(n + 1) + h, 0, n)) for n in g.shape]
    out = np.zeros([n + 1 for n in g.shape], np.float64)
    for sz, iz in ((1, idx[0][1]), (-1, idx[0][0])):
        for sy, iy in ((1, idx[1][1]), (-1, idx[1][0])):
            for sx, ix in ((1, idx[2][1]), (-1, idx[2][0])):
                out += sz * sy * sx * cs[np.ix_(iz, iy, ix)]
    return out


def mine(args) -> None:
    import zarr
    for scroll in args.scrolls:
        t0 = time.time()
        cfg = scroll_cfg(scroll)
        work = WORK / scroll
        work.mkdir(parents=True, exist_ok=True)
        zg = zarr.open_group(cfg["url"], mode="r", storage_options={"anon": True})
        shape = zg["0"].shape
        cfg["shape"] = list(shape)
        lv = 4 if "4" in zg else 3
        ovw, fo = zg[str(lv)][:], 2 ** lv  # overview for the frame check
        dense, q = {}, {}
        for sg in cfg["segments"]:
            if not (mesh_dir(scroll, sg) / "z.tif").exists():
                sg["missing"] = True
                continue
            p = load_mesh(scroll, sg)
            # frame check: a mesh that fits the target volume lies inside it and on unmasked CT
            pv = p.reshape(-1, 3)
            pv = pv[np.isfinite(pv).all(1)][::7]
            outside = ~((pv >= 0) & (pv < np.array(shape))).all(1)
            qi = np.clip((pv[~outside] / fo).astype(int), 0, np.array(ovw.shape) - 1)
            sg["frac_outside_volume"] = float(outside.mean())
            sg["frac_on_masked_ct"] = float((ovw[tuple(qi.T)] == 0).mean()) if len(qi) else 1.0
            if sg["frac_outside_volume"] > FRAME_OUT_MAX or sg["frac_on_masked_ct"] > FRAME_MASKED_MAX:
                sg["frame_bad"] = (f"mesh does not fit the target volume frame: {100 * sg['frac_outside_volume']:.1f}% of points "
                                   f"outside the volume, {100 * sg['frac_on_masked_ct']:.0f}% on masked (zero) CT at L{lv}")
                cfg["skipped"][sg["name"]] = sg["frame_bad"]
                print(f"  skip {sg['name']}: {sg['frame_bad']}", flush=True)
                continue
            st = grid_step(p)
            z = max(1, int(round(st / MINE_STEP)))
            pd = densify(p, z)
            nd = grid_normals(pd)
            ok = np.isfinite(pd).all(-1) & np.isfinite(nd).all(-1)
            P, Nn = pd[ok], nd[ok]
            sp = st / z
            stride = max(1, int(round((QUERY_SPACING / sp) ** 2)))
            sg |= {"grid_step_vox": round(st, 3), "mine_spacing_vox": round(sp, 3), "n_valid": int(np.isfinite(p).all(-1).sum()),
                   "bbox_zyx": [np.nanmin(p.reshape(-1, 3), 0).tolist(), np.nanmax(p.reshape(-1, 3), 0).tolist()],
                   "area_vox2": float(ok.sum() * sp * sp)}
            dense[sg["id"]] = (P, Nn, cKDTree(P), sp)
            q[sg["id"]] = (P[::stride], sp * sp * stride)
            print(f"  {sg['id']:2d} {sg['name'][:50]:50s} {sg['src_tag']:24s} step {st:5.2f} vox, {len(P)} pts, "
                  f"z {sg['bbox_zyx'][0][0]:.0f}-{sg['bbox_zyx'][1][0]:.0f}", flush=True)
        ids = sorted(dense)
        segs_by_id = {s["id"]: s for s in cfg["segments"]}
        rank = {a: r for r, a in enumerate(sorted(ids, key=lambda a: -segs_by_id[a]["area_vox2"]))}  # 0 = largest
        # pairwise perpendicular distances (query points of a against densified b); d_big = to any LARGER segment
        d_big = {a: np.full(len(q[a][0]), np.inf) for a in ids}
        pair = {}
        for a in ids:
            Pq = q[a][0]
            lo, hi = Pq.min(0) - 30, Pq.max(0) + 30
            for b in ids:
                Pb, Nb, Tb, spb = dense[b]
                if b == a or (Pb.max(0) < lo).any() or (Pb.min(0) > hi).any():
                    continue
                dist, j = Tb.query(Pq, distance_upper_bound=25.0)
                hit = np.isfinite(dist)
                if not hit.any():
                    continue
                v = Pq[hit] - Pb[j[hit]]
                dp = np.abs((v * Nb[j[hit]]).sum(1))
                lat = np.sqrt(np.maximum(dist[hit] ** 2 - dp ** 2, 0))
                dp[lat > 0.8 * spb + 1.0] = np.inf
                full = np.full(len(Pq), np.inf)
                full[hit] = dp
                fin = np.isfinite(full)
                if fin.sum():
                    pair[(a, b)] = {"n": len(Pq), "n_meas": int(fin.sum()), "n_lt1": int((full < 1).sum()),
                                    "n_lt2": int((full < DEDUP_D).sum()), "median": float(np.median(full[fin]))}
                if rank[b] < rank[a]:
                    d_big[a] = np.minimum(d_big[a], full)
        # overlapping traces of the same sheet are common (auto-grown strips): no segment is dropped; points within
        # DEDUP_D of a larger segment do not count as new area here, and build() removes them per crop (local dedupe)
        for a in ids:
            sg = segs_by_id[a]
            sg["frac_overlapping_larger"] = float(np.mean(d_big[a] < DEDUP_D)) if len(d_big[a]) else 0.0
            big = [(b, r) for (x, b), r in pair.items() if x == a and rank[b] < rank[a] and r["n_lt2"] / r["n"] > 0.05]
            if big:
                sg["overlaps"] = {segs_by_id[b]["name"]: round(r["n_lt2"] / r["n"], 3) for b, r in big}
            print(f"  {sg['name'][:50]:50s} {100 * sg['frac_overlapping_larger']:5.1f}% of points within {DEDUP_D} vox of a "
                  f"larger segment", flush=True)
        keep = ids
        allP = np.concatenate([q[a][0] for a in keep])
        c0 = np.floor(allP.min(0) / CELL).astype(int) - 3
        shp = np.floor(allP.max(0) / CELL).astype(int) - c0 + 4
        total = np.zeros(shp + 1)
        best = np.zeros(shp + 1)
        dom = np.zeros(shp + 1, np.int32)
        kc = (N - 2 * CORE) // CELL // 2 * 2
        for a in keep:
            Pq, w = q[a]
            new = ~(d_big[a] < DEDUP_D)
            cell = np.floor(Pq[new] / CELL).astype(int) - c0
            g = np.zeros(shp, np.float32)
            np.add.at(g, tuple(cell.T), w)
            bs = box_sum(g, kc)
            total += bs
            m = bs > best
            best[m], dom[m] = bs[m], a
        centre = (np.indices(shp + 1).reshape(3, -1).T + c0) * CELL
        tot, dm = total.ravel(), dom.ravel()
        cf = np.zeros_like(tot)
        inside = ((centre - N // 2 - 12 >= 0) & (centre + N // 2 + 12 <= np.array(shape))).all(1)
        ok = inside & (tot >= AREA_MIN)
        idx = np.nonzero(ok)[0]
        print(f"  cells {len(tot)}, inside {inside.sum()}, area-ok {(inside & (tot >= AREA_MIN)).sum()}, eligible {len(idx)}")
        # round-robin over dominant segments, best area first, disjoint cores
        order = idx[np.argsort(-tot[idx])]
        by_seg = {a: [i for i in order if dm[i] == a] for a in keep}
        ptr = {a: 0 for a in keep}
        chosen = []
        cent = np.zeros((0, 3))
        seg_order = sorted(keep, key=lambda a: -segs_by_id[a]["area_vox2"])
        while len(chosen) < args.n_max and any(ptr[a] < len(by_seg[a]) for a in keep):
            for a in seg_order:
                lst = by_seg[a]
                while ptr[a] < len(lst):
                    i = lst[ptr[a]]
                    ptr[a] += 1
                    c = centre[i]
                    if len(cent) == 0 or np.abs(cent - c).max(1).min() >= MIN_SEP:
                        chosen.append({"centre": c.astype(int).tolist(), "area_core": float(tot[i]), "dom_seg": int(a),
                                       "conflict_area_crop": float(cf[i])})
                        cent = np.vstack([cent, c])
                        break
                if len(chosen) >= args.n_max:
                    break
        for k, c in enumerate(chosen):
            z, y, x = c["centre"]
            c["stem"] = f"{SHORT[scroll]}_{k:03d}_z{z}_y{y}_x{x}"
            c["origin"] = [v - N // 2 for v in c["centre"]]
        cnt = {}
        for c in chosen:
            cnt[c["dom_seg"]] = cnt.get(c["dom_seg"], 0) + 1
        print(f"  chosen {len(chosen)} candidates over {len(cnt)} dominant segments ({time.time() - t0:.0f}s)")
        cfg["pairs_overlap"] = {f"{a}-{b}": r for (a, b), r in pair.items()}
        (work / "candidates.json").write_text(json.dumps({"cfg": cfg, "candidates": chosen}, indent=1, default=float))
        print(f"  wrote {work / 'candidates.json'}", flush=True)


# ------------------------------------------------------------------------------------------------------------ build
_VOLS = {}


def zarr_level(url: str, level: int):
    key = (url, level)
    if key not in _VOLS:
        import zarr
        _VOLS[key] = zarr.open_group(url, mode="r", storage_options={"anon": True})[str(level)]
    return _VOLS[key]


class Render:
    """Source volume at level L (2^L block mean, voxel i centred at L0 2^L i + (2^L - 1)/2), blurred, resampled onto
    target voxel grids through the published affine (generalizes p0139_testset.Render)."""

    def __init__(self, arr, level, lo, hi, A, t):
        self.A, self.t, self.Ainv, self.f = A, t, np.linalg.inv(A), 2 ** level
        corners = np.array([[z, y, x] for z in (lo[0], hi[0]) for y in (lo[1], hi[1]) for x in (lo[2], hi[2])], float)
        ls = (self.to_src(corners) - (self.f - 1) / 2) / self.f
        self.lo = np.floor(ls.min(0)).astype(int) - 4
        h = np.ceil(ls.max(0)).astype(int) + 5
        raw, self.nch = read_box(arr, self.lo, h)
        self.valid = ndi.binary_erosion(raw > 0, iterations=2)
        self.img = ndi.gaussian_filter(raw.astype(np.float32), BLUR)

    def to_src(self, zyx):
        return ((zyx[:, ::-1] - self.t) @ self.Ainv.T)[:, ::-1]

    def render(self, lo, shape, shift=(0.0, 0.0, 0.0)):
        g = np.indices(shape).reshape(3, -1).T + np.asarray(lo) - np.asarray(shift)
        q = ((self.to_src(g.astype(float)) - (self.f - 1) / 2) / self.f - self.lo).T
        v = map_coordinates(self.img, q, order=1, mode="constant", cval=0.0)
        ok = map_coordinates(self.valid.astype(np.float32), q, order=0, mode="constant", cval=0.0) > 0.5
        return np.where(ok, v, 0.0).reshape(shape).astype(np.float32)


def register_group(v9m, m, origin, scroll, sg, tgt_um, radius):
    """Translation (target vox, zyx) of one source-volume group by NCC (as p0139_testset.register)."""
    level = max(0, int(round(np.log2(tgt_um / sg["src_um"]))))
    lo = np.asarray(origin) - m
    R = Render(zarr_level(vol_url(scroll, sg["src_long"]), level), level, lo, lo + N + 2 * m,
               np.asarray(sg["A"]), np.asarray(sg["t"]))
    rend0 = R.render(origin, (N, N, N))
    d_c, _ = shift_search(rend0, v9m, m, (0, 0, 0), radius, 2, sub=3)
    if d_c is None:
        raise RuntimeError("no overlap between render and CT")
    d_i, sc = shift_search(rend0, v9m, m, d_c, 2, 1, sub=2)
    shift = parabolic(sc, d_i)
    octs = []
    h = N // 2
    for oz in (0, h):
        for oy in (0, h):
            for ox in (0, h):
                reg = (np.array([oz, oy, ox]), np.array([oz, oy, ox]) + h)
                if (rend0[oz:oz + h, oy:oy + h, ox:ox + h] > 0).mean() < 0.2:
                    continue
                d_o, sc_o = shift_search(rend0, v9m, m, d_i, 2, 1, sub=2, region=reg)
                if d_o is not None:
                    octs.append(parabolic(sc_o, d_o).tolist())
    rend = R.render(origin, (N, N, N), shift)
    ct = v9m[m:m + N, m:m + N, m:m + N]
    return {"src": sg["src_tag"], "level": level, "shift": shift.tolist(), "ncc": ncc_masked(rend, ct, (rend > 0) & (ct > 0)),
            "ncc_published": ncc_masked(rend0, ct, (rend0 > 0) & (ct > 0)), "at_search_edge": bool(np.abs(d_c).max() >= radius),
            "octant_shifts": octs, "octant_dev_max": float(np.max(np.abs(np.array(octs) - shift))) if octs else float("nan"),
            "chunks_src": R.nch}


def rasterize(scroll, origin, shifts, segs):
    """1-vox surface per segment (closest sample wins) + per-voxel mesh normal; shifts = {src_vol: zyx shift}.
    Quads are subdivided so samples are <= 0.5 vox apart whatever the mesh grid step."""
    surf = np.zeros((N, N, N), np.uint16)
    best = np.full((N, N, N), np.inf, np.float32)
    nrm = np.zeros((N, N, N, 3), np.float16)
    counts = {}
    for s in segs:
        if s.get("duplicate") or s.get("missing") or s.get("frame_bad") or "bbox_zyx" not in s:
            continue
        sh = np.asarray(shifts.get(s["src_vol"], (0.0, 0.0, 0.0)), float)
        o = np.asarray(origin, float) - sh
        lo, hi = o - 3, o + N + 3
        bb = np.array(s["bbox_zyx"])
        if (bb[1] < lo - 5).any() or (bb[0] > hi + 5).any():
            continue
        g = load_mesh(scroll, s)
        ins = np.isfinite(g).all(-1) & ((g >= lo - s["grid_step_vox"]) & (g <= hi + s["grid_step_vox"])).all(-1)
        if not ins.any():
            continue
        qd = ins[:-1, :-1] | ins[1:, :-1] | ins[:-1, 1:] | ins[1:, 1:]
        ok = np.isfinite(g).all(-1)
        qd &= ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        ii, jj = np.nonzero(qd)
        if not len(ii):
            continue
        sub = int(np.ceil(2 * s["grid_step_vox"] * 1.2)) + 1
        uu = np.linspace(0, 1, sub + 1)
        wu, wv = np.meshgrid(uu, uu, indexing="ij")
        wu, wv = wu.ravel(), wv.ravel()
        for c0 in range(0, len(ii), max(1, 4_000_000 // len(wu))):  # chunk quads to bound memory
            i_, j_ = ii[c0:c0 + max(1, 4_000_000 // len(wu))], jj[c0:c0 + max(1, 4_000_000 // len(wu))]
            p00, p10, p01, p11 = g[i_, j_], g[i_ + 1, j_], g[i_, j_ + 1], g[i_ + 1, j_ + 1]
            n = np.cross(p10 - p00 + p11 - p01, p01 - p00 + p11 - p10)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            P = (p00[:, None] * ((1 - wu) * (1 - wv))[None, :, None] + p10[:, None] * (wu * (1 - wv))[None, :, None]
                 + p01[:, None] * ((1 - wu) * wv)[None, :, None] + p11[:, None] * (wu * wv)[None, :, None])
            Nn = np.broadcast_to(n[:, None], P.shape).reshape(-1, 3)
            P = P.reshape(-1, 3) - o
            v = np.rint(P).astype(int)
            keep = ((v >= 0) & (v < N)).all(1)
            if not keep.any():
                continue
            v, P, Nn = v[keep], P[keep], Nn[keep]
            dist = np.linalg.norm(P - v, axis=1).astype(np.float32)
            order = np.argsort(-dist)
            v, dist, Nn = v[order], dist[order], Nn[order]
            w = dist < best[tuple(v.T)]
            idx = tuple(v[w].T)
            best[idx] = dist[w]
            surf[idx] = s["id"]
            nrm[idx] = Nn[w].astype(np.float16)
            counts[s["id"]] = counts.get(s["id"], 0) + int(keep.sum())
    for k in [k for k, c in counts.items() if c < 50]:
        surf[surf == k] = 0
        del counts[k]
    return surf, nrm, counts


def instance_verdicts(rec, g, src_of, reg_bad):
    """Instances whose mesh does not fit the CT -> to ignore. Returns {id: reason}."""
    bad = {}
    for i, r in g["instances"].items():
        i = int(i)
        if src_of[i] in reg_bad:
            bad[i] = f"registration: {reg_bad[src_of[i]]}"
            continue
        if rec["instances"][i]["n_vox_raw"] < FRAG_MIN:
            bad[i] = f"fragment ({rec['instances'][i]['n_vox_raw']} < {FRAG_MIN} vox)"
            continue
        if "skipped" in r:  # < 200 core voxels: a sheet piece in the crop margin, not gated
            continue
        if r["on_papyrus"] < INST_ONPAP_MIN:
            bad[i] = f"on-papyrus {r['on_papyrus']:.2f} < {INST_ONPAP_MIN}"
        elif r["frac_off"] > INST_OFF_MAX:
            bad[i] = f"{r['frac_off']:.0%} > {OFF_D} vox off material"
        elif np.isfinite(r["normal_agree"]) and r["normal_agree"] < NORMAL_AGREE_MIN:
            bad[i] = f"normal agreement {r['normal_agree']:.2f} < {NORMAL_AGREE_MIN} (cuts across sheets)"
        elif r["n_tiles_deep"] > DEEP_TILES_MAX:
            bad[i] = f"crosses dark gaps in {r['n_tiles_deep']} 24^3 tiles"
    onp = {int(i): r.get("on_papyrus", 1.0) for i, r in g["instances"].items()}
    nv = {int(i): r.get("n_core_vox", 0) for i, r in g["instances"].items()}
    for key, r in g["pairs"].items():
        a, b = map(int, key.split("-"))
        if a in bad or b in bad:
            continue
        if r["minority_side"] > CROSS_MAX:
            if min(nv.get(a, 0), nv.get(b, 0)) < 2000:  # a small piece crossing a sheet: the piece is the error
                w = a if nv.get(a, 0) < nv.get(b, 0) else b
            else:
                w = a if onp.get(a, 1) < onp.get(b, 1) else b
            bad[w] = f"crosses instance {b if w == a else a} ({100 * r['minority_side']:.0f}% minority side)"
        elif r["conflict_frac"] > CROSS_MAX:
            w = a if nv.get(a, 0) < nv.get(b, 0) else b
            bad[w] = f"coincides with instance {b if w == a else a} ({100 * r['conflict_frac']:.0f}% < 1 vox)"
    return bad


DEDUP_CT = False       # CT-checked dedupe (Dataset105/106): keep a close second tracing if a dark gap separates it
DARK_F, BRIGHT_F = 0.4, 0.6   # dark / bright levels as fractions between the crop's air and sheet CT medians
DIST_MIN = 1.5         # closer than this to a kept raster: always a duplicate (no room for a gap)
EXT = 1.5              # the profile extends this far beyond both rasters (each sheet's material may lie outside the line)


def ct_levels(ct):
    from skimage.filters import threshold_otsu
    ctf = ndi.gaussian_filter(ct.astype(np.float32), 0.7)
    v = ct > 0
    t = float(threshold_otsu(ctf[v][::5]))
    air, sheet = float(np.median(ctf[v & (ctf < t)])), float(np.median(ctf[v & (ctf >= t)]))
    return ctf, air + DARK_F * (sheet - air), air + BRIGHT_F * (sheet - air)


def distinct_sheet(ctf, q, p, dark, bright, K=41):
    """Per point pair (kept raster voxel q, candidate voxel p): True if the CT profile on the segment q->p (extended by
    EXT beyond both ends) has a dark sample strictly between the rasters (>= 0.5 vox from both) with bright material
    on BOTH sides of it: sheet - gap - sheet. A duplicate trace offset into air (no material beyond p) or into the
    same sheet (no dark sample) fails."""
    L = np.linalg.norm(p - q, axis=1)
    u = (p - q) / L[:, None]
    sgrid = -EXT + (L[:, None] + 2 * EXT) * np.linspace(0, 1, K)[None, :]          # (n, K) vox along the line
    pos = q[:, None, :] + u[:, None, :] * sgrid[..., None]
    prof = map_coordinates(ctf, pos.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(q), K)
    pre = np.maximum.accumulate(prof, axis=1)
    suf = np.maximum.accumulate(prof[:, ::-1], axis=1)[:, ::-1]
    pre = np.concatenate([np.full((len(q), 1), -np.inf), pre[:, :-1]], 1)      # max strictly before k
    suf = np.concatenate([suf[:, 1:], np.full((len(q), 1), -np.inf)], 1)       # max strictly after k
    inner = (sgrid >= 0.5) & (sgrid <= L[:, None] - 0.5)
    return (inner & (prof < dark) & (pre > bright) & (suf > bright)).any(1)


def dedupe(surf, segs_by_id, ct=None):
    """Local dedupe: voxels of a smaller segment within DEDUP_D of an already kept (larger) segment are removed.

    With DEDUP_CT, such voxels are kept when the CT shows a distinct sheet (sheet - dark gap - sheet on the line to
    the nearest kept voxel), decided per voxel and smoothed over a 5^3 neighbourhood (majority)."""
    ids = sorted([int(x) for x in np.unique(surf) if x], key=lambda i: -segs_by_id[i]["area_vox2"])
    out = surf.copy()
    kept = np.zeros(surf.shape, bool)
    removed, distinct = {}, {}
    use_ct = DEDUP_CT and ct is not None
    if use_ct:
        ctf, dark, bright = ct_levels(ct)
    for i in ids:
        m = surf == i
        if kept.any():
            pts = np.argwhere(m)
            lo, hi = np.maximum(pts.min(0) - 6, 0), np.minimum(pts.max(0) + 7, N)
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            dk, ik = ndi.distance_transform_edt(~kept[sl], return_indices=True)
            near = (dk <= DEDUP_D) & m[sl]
            if use_ct and near.any():
                cand = near & (dk >= DIST_MIN)
                pc = np.argwhere(cand)
                if len(pc):
                    qc = np.stack([ik[k][tuple(pc.T)] for k in range(3)], 1)
                    ok = distinct_sheet(ctf, qc + lo, pc + lo, dark, bright)
                    D = np.zeros(cand.shape, np.float32)
                    D[tuple(pc[ok].T)] = 1
                    frac = ndi.uniform_filter(D, 5) / np.maximum(ndi.uniform_filter(cand.astype(np.float32), 5), 1e-6)
                    keep = cand & (frac >= 0.5)
                    if keep.any():
                        distinct[i] = int(keep.sum())
                    near &= ~keep
            if near.any():
                o = out[sl]
                o[near] = 0
                removed[i] = int(near.sum())
        kept |= out == i
    if use_ct:
        return out, {"removed": removed, "kept_distinct": distinct}
    return out, removed


def reg_verdict(r, radius):
    if not np.isfinite(r["ncc"]) or r["ncc"] < NCC_MIN:
        return f"NCC {r['ncc']:.3f} < {NCC_MIN}"
    if np.isfinite(r["octant_dev_max"]) and r["octant_dev_max"] > SHIFT_SPREAD_MAX:
        return f"octant shift deviation {r['octant_dev_max']:.2f} > {SHIFT_SPREAD_MAX}"
    if r["at_search_edge"]:
        return f"shift at the search edge ({radius} vox)"
    return None


def build_site(job, local=False):
    """Build one crop. local=True: re-derive the cleaned 1-vox raster from the saved CT crop and the saved per-group
    registration (no streaming), write surfacesTr only, and return it together with the band/ignore checks."""
    cand, cfg = job
    scroll, stem = cfg["scroll"], cand["stem"]
    work = WORK / scroll
    t0 = time.time()
    rec = {"stem": stem, "scroll": scroll, "origin_zyx": cand["origin"], "shape": [N] * 3, "vox_um": cfg["vox_um"],
           "mined": {k: cand[k] for k in ("area_core", "dom_seg", "conflict_area_crop")}}
    try:
        segs = cfg["segments"]
        segs_by_id = {s["id"]: s for s in segs}
        o = np.asarray(cand["origin"])
        near = [s for s in segs if "bbox_zyx" in s and not s.get("duplicate") and not s.get("frame_bad")
                and not ((np.array(s["bbox_zyx"][1]) < o - 20).any() or (np.array(s["bbox_zyx"][0]) > o + N + 20).any())]
        groups = {s["src_vol"]: s for s in near if not s["native"]}
        radius = SHIFT.get(scroll, SHIFT["default"])
        m = radius + 2 if groups else 0
        rec["registration"], shifts, reg_bad = {}, {}, {}
        if local:
            prev = json.loads((work / "sites" / f"{stem}.json").read_text())
            ct = tifffile.imread(work / "all" / "imagesTr" / f"{stem}_0000.tif")
            assert set(prev.get("registration", {})) == set(groups), "registration groups changed"
            for src, r in prev.get("registration", {}).items():
                rec["registration"][src] = r
                shifts[src] = r["shift"]
                if (v := reg_verdict(r, radius)):
                    reg_bad[src] = v
        else:
            v9m, nch = read_box(zarr_level(cfg["url"], 0), o - m, o + N + m)
            ct = v9m[m:m + N, m:m + N, m:m + N].copy()
            rec["chunks_target"] = nch
            for src, sg in groups.items():
                r = register_group(v9m, m, o, scroll, sg, cfg["vox_um"], radius)
                rec["registration"][src] = r
                shifts[src] = r["shift"]
                if (v := reg_verdict(r, radius)):
                    reg_bad[src] = v
            del v9m
        surf_all, nrm, counts = rasterize(scroll, o, shifts, near)
        surf_seg, dedup = dedupe(surf_all, segs_by_id, ct)
        dupm = (surf_all > 0) & (surf_seg == 0)
        if DEDUP_CT and ct is not None:
            rec["dedupe_kept_distinct_vox"] = {int(k): v for k, v in dedup["kept_distinct"].items()}
            dedup = dedup["removed"]
        rec["dedupe_removed_vox"] = {int(k): v for k, v in dedup.items()}
        # instances = connected pieces of each segment's raster: multi-wrap strips cross a crop several times
        surf_raw = np.zeros_like(surf_seg)
        seg_of = {}
        for sid in [int(x) for x in np.unique(surf_seg) if x]:
            msk = surf_seg == sid
            lab, n = ndi.label(ndi.binary_dilation(msk), structure=np.ones((3, 3, 3), bool))  # bridge 1-2 vox gaps
            lab[~msk] = 0
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            for k in range(1, n + 1):
                if sizes[k] == 0:
                    continue
                if sizes[k] < 50:  # speckle left by the dedupe threshold: to ignore
                    dupm |= lab == k
                    continue
                seg_of[len(seg_of) + 1] = sid
                surf_raw[lab == k] = len(seg_of)
        ids = [int(x) for x in np.unique(surf_raw) if x]
        if not ids:
            raise RuntimeError("no mesh voxels in the crop")
        sbi = {i: {"winding": None, **segs_by_id[seg_of[i]]} for i in ids}
        rec["instances"] = {i: {"segment": segs_by_id[seg_of[i]]["name"], "seg_id": seg_of[i],
                                "src": segs_by_id[seg_of[i]]["src_tag"], "n_vox_raw": int((surf_raw == i).sum())}
                            for i in ids}
        g, dmat = gates(ct, surf_raw, nrm, sbi, cast_pairs(surf_raw, ids, np.random.default_rng(SEED)))
        rec["gates"] = g
        bad = instance_verdicts(rec, g, {i: segs_by_id[seg_of[i]]["src_vol"] for i in ids}, reg_bad)
        rec["rejected_instances"] = {i: bad[i] for i in bad}
        badm = np.isin(surf_raw, list(bad)) if bad else np.zeros(surf_raw.shape, bool)
        off = (surf_raw > 0) & (dmat > OFF_D) & ~badm
        surf = np.where(off | badm, 0, surf_raw).astype(np.uint16)
        kept = [i for i in ids if i not in bad]
        c = slice(CORE, N - CORE)
        rec["cleaning"] = {"removed_off_vox": int(off.sum()), "rejected_vox": int(badm.sum()),
                           "removed_frac": float((off.sum() + badm.sum()) / max(1, (surf_raw > 0).sum())),
                           "label_core_vox": int((surf[c, c, c] > 0).sum())}
        # pooled on-papyrus over kept instances (raw raster, core)
        kv = (np.isin(surf_raw, kept) if kept else np.zeros(surf_raw.shape, bool))
        kv[:CORE], kv[N - CORE:], kv[:, :CORE], kv[:, N - CORE:], kv[:, :, :CORE], kv[:, :, N - CORE:] = (False,) * 6
        rec["pooled_on_papyrus_kept"] = float((dmat[kv] <= ON_D).mean()) if kv.any() else float("nan")
        # local pitch: first other instance along the normal, any pair (kept instances only)
        hits = cast_pairs(surf, kept, np.random.default_rng(SEED)) if len(kept) > 1 else {}
        dd = np.array([d for v in hits.values() for _, _, d in v if d >= 1.0])
        pitch = float(np.median(dd)) if len(dd) >= 30 else 10.0
        inst, dist = band_from_surface(surf)
        dmax = float(np.clip(0.7 * pitch, 4, 10))
        mat = (ndi.gaussian_filter(ct.astype(np.float32), 1.0) > g["otsu"]) & (ct > 0)
        unl = mat & (dist > dmax)
        unl_frac = float(unl[c, c, c].sum() / max(1, mat[c, c, c].sum()))
        rec["coverage"] = {"pitch_vox": pitch, "pitch_n": int(len(dd)), "dmax_vox": dmax, "unlabelled_material_frac": unl_frac,
                           "material_frac_core": float(mat[c, c, c].mean())}
        ignore = (ndi.binary_dilation(unl, iterations=1) | ndi.binary_dilation(off, iterations=2)
                  | ndi.binary_dilation(badm, iterations=3) | ndi.binary_dilation(dupm, iterations=1))
        ignore &= inst == 0
        rec["label_frac"] = float((inst > 0).mean())
        rec["ignore_frac"] = float(ignore.mean())
        d = work / "all"
        if local:
            ct_checks(rec, ct, inst)
            decide(rec)
            sd = d / (local if isinstance(local, str) else "surfacesTr")
            sd.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(sd / f"{stem}.tif", surf, compression="zlib")
            lab0 = tifffile.imread(d / "labelsTr" / f"{stem}.tif")
            ign0 = tifffile.imread(d / "ignoreTr" / f"{stem}.tif")
            return {"stem": stem, "surf": surf, "ignore": ignore, "instances": rec["instances"],
                    "inst": inst, "dd": dd, "pitch": pitch, "kept_distinct": rec.get("dedupe_kept_distinct_vox", {}),
                    "gates_rec": rec,
                    "label_mismatch_vox": int(((inst > 0) != (lab0 > 0)).sum()),
                    "ignore_mismatch_vox": int((ignore != (ign0 > 0)).sum())}
        for sub in ("imagesTr", "labelsTr", "ignoreTr", "instancesTr"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(d / "imagesTr" / f"{stem}_0000.tif", ct, compression="zlib")
        tifffile.imwrite(d / "labelsTr" / f"{stem}.tif", (inst > 0).astype(np.uint8), compression="zlib")
        tifffile.imwrite(d / "ignoreTr" / f"{stem}.tif", ignore.astype(np.uint8), compression="zlib")
        tifffile.imwrite(d / "instancesTr" / f"{stem}.tif", inst, compression="zlib")
        ct_checks(rec, ct, inst)
        decide(rec)
    except Exception as e:  # log and drop
        if local:
            raise
        import traceback
        rec["passed"] = False
        rec["fail"] = [f"error: {e!r}"]
        rec["traceback"] = traceback.format_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    (work / "sites").mkdir(parents=True, exist_ok=True)
    (work / "sites" / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def ct_checks(rec, img, lab):
    """CT validity: an empty (masked) crop makes Otsu meaningless and every label 'on papyrus'."""
    v = img > 0
    rec["ct_valid_frac"] = float(v.mean())
    rec["label_on_valid_ct"] = float(v[lab > 0].mean()) if (lab > 0).any() else float("nan")


def decide(rec):
    """Crop-level training gates (instance-level failures were already moved to ignore)."""
    fail = []
    if rec.get("ct_valid_frac", 1.0) < CT_VALID_MIN or rec.get("label_on_valid_ct", 1.0) < LABEL_ON_CT_MIN:  # nan (no labels) is the label gate's job
        fail.append(f"ct: valid (non-zero) CT {rec.get('ct_valid_frac', float('nan')):.2f} (min {CT_VALID_MIN}), labels on "
                    f"valid CT {rec.get('label_on_valid_ct', float('nan')):.2f} (min {LABEL_ON_CT_MIN})")
    if rec["coverage"].get("material_frac_core", 1.0) < MATERIAL_MIN:
        fail.append(f"ct: material {rec['coverage']['material_frac_core']:.3f} of the core < {MATERIAL_MIN}")
    if rec["cleaning"]["label_core_vox"] < LABEL_CORE_MIN:
        fail.append(f"label: {rec['cleaning']['label_core_vox']} kept surface voxels in the core < {LABEL_CORE_MIN}")
    p = rec["pooled_on_papyrus_kept"]
    if not np.isfinite(p) or p < POOLED_ONPAP_MIN:
        fail.append(f"papyrus: pooled on-papyrus of kept instances {p:.2f} < {POOLED_ONPAP_MIN}")
    u = rec["coverage"]["unlabelled_material_frac"]
    if u > UNL_MAX:
        fail.append(f"coverage: unlabelled material {u:.2f} > {UNL_MAX}")
    rec["fail"] = fail
    rec["fail_gate"] = sorted({f.split(":")[0] for f in fail})
    rec["passed"] = not fail


SHELL_D = 1.5   # thin labels: ignore shell around the raster
GAP_D = 2.5     # thin labels: shell voxels within this EDT of ANOTHER instance's raster are taught as background (gap)


def thin_label(surf, ignore103):
    """Dataset104 label from the cleaned 1-vox raster (uint16 instance ids) and the Dataset103 ignore mask.

    1 = raster; 2 = Dataset103 ignore, plus the 1.5-vox EDT shell around the raster; 0 = background, including shell
    voxels within GAP_D of another instance's raster (the shells of two sheets touch or overlap: teach the gap)."""
    raster = surf > 0
    d1, ind = ndi.distance_transform_edt(~raster, return_indices=True)
    nearest = surf[tuple(ind)]
    shell = (d1 <= SHELL_D) & ~raster
    gap = np.zeros(surf.shape, bool)
    for j in [int(x) for x in np.unique(surf) if x]:
        pts = np.argwhere(surf == j)
        lo, hi = np.maximum(pts.min(0) - 4, 0), np.minimum(pts.max(0) + 5, surf.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        dj = ndi.distance_transform_edt(surf[sl] != j)
        gap[sl] |= shell[sl] & (dj <= GAP_D) & (nearest[sl] != j)
    lab = np.zeros(surf.shape, np.uint8)
    lab[ignore103 | (shell & ~gap)] = 2
    lab[gap & ~ignore103] = 0
    lab[raster] = 1
    return lab, gap


def _thin_init(dedup_d):
    global DEDUP_D
    DEDUP_D = dedup_d


def thin_site(job):
    cand, cfg, sub, write = job
    r = build_site((cand, cfg), local=sub)
    work = WORK / cfg["scroll"] / "all"
    lab, gap = thin_label(r["surf"], r["ignore"])
    l103 = tifffile.imread(DS / "labelsTr" / f"{r['stem']}.tif")
    if write:
        (work / "labelsThinTr").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(work / "labelsThinTr" / f"{r['stem']}.tif", lab, compression="zlib")
    # gap voxels between different SEGMENTS (vs different connected pieces of one segment)
    seg = np.zeros(r["surf"].shape, np.int32)
    for i, v in r["instances"].items():
        seg[r["surf"] == int(i)] = v["seg_id"]
    _, ind = ndi.distance_transform_edt(seg == 0, return_indices=True)
    nseg = seg[tuple(ind)]
    gseg = np.zeros(seg.shape, bool)
    for sid in [int(x) for x in np.unique(seg) if x]:
        pts = np.argwhere(seg == sid)
        lo, hi = np.maximum(pts.min(0) - 4, 0), np.minimum(pts.max(0) + 5, seg.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        gseg[sl] |= gap[sl] & (ndi.distance_transform_edt(seg[sl] != sid) <= GAP_D) & (nseg[sl] != sid)
    gap_seg = int(gseg.sum())
    f = lambda a, v: float((a == v).mean())
    return {"stem": r["stem"], "scroll": cfg["scroll"], "label_mismatch_vox": r["label_mismatch_vox"],
            "ignore_mismatch_vox": r["ignore_mismatch_vox"], "gap_vox": int(gap.sum()), "gap_vox_between_segments": gap_seg,
            "n_instances": len(r["instances"]),
            "frac104": {"surface": f(lab, 1), "ignore": f(lab, 2), "background": f(lab, 0)},
            "frac103": {"surface": f(l103, 1), "ignore": f(l103, 2), "background": f(l103, 0)}}


def thin(args):
    """Dataset104_Real9Thin: same crops and images as Dataset103, labels = 1-vox raster + 1.5-vox ignore shell + gaps."""
    m = json.loads((OUT / "manifest.json").read_text())
    stems = {c["stem"] for c in m["crops"] if c["passed"]}
    write = args.dedup_d == DEDUP_D
    sub = "surfacesTr" if write else f"surfacesTr_dedup{args.dedup_d}"
    jobs = []
    for scroll in args.scrolls:
        cj = json.loads((WORK / scroll / "candidates.json").read_text())
        jobs += [(c, cj["cfg"], sub, write) for c in cj["candidates"] if c["stem"] in stems]
    print(f"{len(jobs)} crops, DEDUP_D={args.dedup_d}, write={write}", flush=True)
    res = []
    with ProcessPoolExecutor(args.workers, initializer=_thin_init, initargs=(args.dedup_d,)) as ex:
        for r in ex.map(thin_site, jobs):
            res.append(r)
            print(f"{r['stem']}: gap={r['gap_vox']} (between segments {r['gap_vox_between_segments']}) "
                  f"mismatch lab={r['label_mismatch_vox']} ign={r['ignore_mismatch_vox']} "
                  f"surf {r['frac104']['surface']:.4f} ign {r['frac104']['ignore']:.3f}", flush=True)
    tag = "" if write else f"_dedup{args.dedup_d}"
    (OUT / f"thin_stats{tag}.json").write_text(json.dumps(res, indent=1))
    if not write:
        return
    D4 = DS.parent / "Dataset104_Real9Thin"
    for s in ("imagesTr", "labelsTr"):
        if (D4 / s).exists():
            shutil.rmtree(D4 / s)
        (D4 / s).mkdir(parents=True)
    for r in res:
        st, sc = r["stem"], r["scroll"]
        os.link(OUT / sc / "imagesTr" / f"{st}_0000.tif", D4 / "imagesTr" / f"{st}_0000.tif")
        os.link(WORK / sc / "all" / "labelsThinTr" / f"{st}.tif", D4 / "labelsTr" / f"{st}.tif")
    dj = json.loads((DS / "dataset.json").read_text())
    dj["name"] = ("thin-label variant of Dataset103_Real9: surface = cleaned 1-vox mesh raster; ignore = Dataset103 "
                  f"ignore + {SHELL_D}-vox EDT shell around the raster, except shell voxels within {GAP_D} vox of another "
                  "instance's raster (touching sheets: background); not PHerc0139")
    dj["numTraining"] = len(res)
    (D4 / "dataset.json").write_text(json.dumps(dj, indent=1))
    print(f"Dataset104: {len(res)} cases -> {D4}", flush=True)


def _gap_init():
    global DEDUP_CT
    DEDUP_CT = True


def pair_stats(surf, seg_of_inst):
    """Close pairs: raster voxels within 4 vox of ANOTHER instance's raster (and of another segment's)."""
    raster = surf > 0
    d1, ind = ndi.distance_transform_edt(~raster, return_indices=True)
    nearest = surf[tuple(ind)]
    seg = np.zeros(surf.shape, np.int32)
    for i, sid in seg_of_inst.items():
        seg[surf == i] = sid
    between = np.zeros(surf.shape, bool)       # non-raster voxels within GAP_D of two different instances
    close_i = np.zeros(surf.shape, bool)       # raster voxels within 4 vox of another instance
    close_s = np.zeros(surf.shape, bool)       # ... of another segment
    for j in [int(x) for x in np.unique(surf) if x]:
        pts = np.argwhere(surf == j)
        lo, hi = np.maximum(pts.min(0) - 5, 0), np.minimum(pts.max(0) + 6, surf.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        dj = ndi.distance_transform_edt(surf[sl] != j)
        between[sl] |= ~raster[sl] & (d1[sl] <= GAP_D) & (dj <= GAP_D) & (nearest[sl] != j)
        close_i[sl] |= raster[sl] & (surf[sl] != j) & (dj < 4.0)
        close_s[sl] |= raster[sl] & (seg[sl] != seg_of_inst[j]) & (dj < 4.0)
    return d1, between, close_i, close_s


def gap_site(job):
    cand, cfg = job
    r = build_site((cand, cfg), local="surfacesTr_ctdedup")
    work = WORK / cfg["scroll"] / "all"
    surf, ign, band = r["surf"], r["ignore"], r["inst"] > 0
    soi = {int(i): v["seg_id"] for i, v in r["instances"].items() if int(i) in set(np.unique(surf).tolist())}
    d1, between, close_i, close_s = pair_stats(surf, soi)
    raster = surf > 0
    shell = (d1 <= SHELL_D) & ~raster
    # Dataset106 (104b): thin raster, shell ignore, explicit gap background (overrides any ignore)
    l6 = np.zeros(surf.shape, np.uint8)
    l6[ign | shell] = 2
    l6[between] = 0
    l6[raster] = 1
    # Dataset105: 3-vox band (Dataset103 convention), gap voxels between distinct sheets background (not ignore)
    l5 = np.zeros(surf.shape, np.uint8)
    l5[ign] = 2
    l5[between & ~band] = 0
    l5[band] = 1
    for sub, lab in (("labelsThinGapTr", l6), ("labelsGapTr", l5)):
        (work / sub).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(work / sub / f"{r['stem']}.tif", lab, compression="zlib")
    l3 = tifffile.imread(DS / "labelsTr" / f"{r['stem']}.tif")
    l4 = tifffile.imread(DS.parent / "Dataset104_Real9Thin" / "labelsTr" / f"{r['stem']}.tif")
    f = lambda a: {"surface": float((a == 1).mean()), "ignore": float((a == 2).mean()), "background": float((a == 0).mean())}
    dd = r["dd"]
    g = r["gates_rec"]
    return {"stem": r["stem"], "scroll": cfg["scroll"], "pitch_vox": r["pitch"], "n_hits": int(len(dd)),
            "frac_hits_lt4": float((dd < 4).mean()) if len(dd) else float("nan"),
            "hits_lt4": int((dd < 4).sum()), "kept_distinct_vox": int(sum(r["kept_distinct"].values())),
            "dedupe_removed_vox": int(sum(g["dedupe_removed_vox"].values())),
            "close_pair_vox_instances": int(close_i.sum()), "close_pair_vox_segments": int(close_s.sum()),
            "gap_vox_106": int((between & (l6 == 0)).sum()), "gap_vox_106_in_shell": int((between & shell).sum()),
            "gap_vox_105": int((between & ~band).sum()),
            "passed_gates": g["passed"], "fail": g["fail"], "n_instances_kept": len(soi),
            "frac": {"103": f(l3), "104": f(l4), "105": f(l5), "106": f(l6)}}


def gap(args):
    """Dataset105_Real9Gap (3-vox band) and Dataset106_Real9ThinGap (= "104b", thin): CT-checked dedupe, same crops."""
    m = json.loads((OUT / "manifest.json").read_text())
    stems = {c["stem"] for c in m["crops"] if c["passed"]}
    jobs = []
    for scroll in args.scrolls:
        cj = json.loads((WORK / scroll / "candidates.json").read_text())
        jobs += [(c, cj["cfg"]) for c in cj["candidates"] if c["stem"] in stems]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} crops, CT-checked dedupe", flush=True)
    res = []
    with ProcessPoolExecutor(args.workers, initializer=_gap_init) as ex:
        for r in ex.map(gap_site, jobs):
            res.append(r)
            print(f"{r['stem']}: kept_distinct={r['kept_distinct_vox']} removed={r['dedupe_removed_vox']} "
                  f"close<4 inst={r['close_pair_vox_instances']} seg={r['close_pair_vox_segments']} "
                  f"gap106={r['gap_vox_106']} gap105={r['gap_vox_105']} pitch={r['pitch_vox']:.1f} "
                  f"passed={r['passed_gates']}", flush=True)
    (OUT / "gap_stats.json").write_text(json.dumps(res, indent=1))
    if args.limit:
        return
    names = {"Dataset105_Real9Gap": ("labelsGapTr", "3-vox band (Dataset103 convention) with CT-checked dedupe: a second "
                                     "tracing within 4 vox is kept if a dark gap (sheet-gap-sheet CT profile) separates it; "
                                     "voxels between distinct touching sheets = background"),
             "Dataset106_Real9ThinGap": ("labelsThinGapTr", "thin labels (1-vox raster, 1.5-vox shell ignore; = '104b') with "
                                         "CT-checked dedupe; voxels between distinct touching sheets = background")}
    dj0 = json.loads((DS / "dataset.json").read_text())
    for name, (sub, desc) in names.items():
        D = DS.parent / name
        for s_ in ("imagesTr", "labelsTr"):
            if (D / s_).exists():
                shutil.rmtree(D / s_)
            (D / s_).mkdir(parents=True)
        for r in res:
            st, sc = r["stem"], r["scroll"]
            os.link(OUT / sc / "imagesTr" / f"{st}_0000.tif", D / "imagesTr" / f"{st}_0000.tif")
            os.link(WORK / sc / "all" / sub / f"{st}.tif", D / "labelsTr" / f"{st}.tif")
        dj = dict(dj0, name=desc + "; not PHerc0139", numTraining=len(res))
        (D / "dataset.json").write_text(json.dumps(dj, indent=1))
        print(f"{name}: {len(res)} cases -> {D}", flush=True)


def build(args):
    jobs = []
    for scroll in args.scrolls:
        cj = json.loads((WORK / scroll / "candidates.json").read_text())
        cands = cj["candidates"]
        if args.only:
            cands = [c for c in cands if any(k in c["stem"] for k in args.only.split(","))]
        if args.limit:
            cands = cands[: args.limit]
        todo = [c for c in cands if args.redo or not (WORK / scroll / "sites" / f"{c['stem']}.json").exists()]
        print(f"{scroll}: {len(todo)} of {len(cands)} crops to build", flush=True)
        jobs += [(c, cj["cfg"]) for c in todo]
    with ProcessPoolExecutor(args.workers) as ex:
        for rec in ex.map(build_site, jobs):
            regs = " ".join(f"{k[:14]}:ncc={v['ncc']:.3f},sh={np.round(v['shift'], 1).tolist()}"
                            for k, v in rec.get("registration", {}).items())
            print(f"{rec['stem']}: passed={rec['passed']} inst={len(rec.get('instances', {}))} "
                  f"rej={len(rec.get('rejected_instances', {}))} onpap={rec.get('pooled_on_papyrus_kept', float('nan')):.2f} "
                  f"unl={rec.get('coverage', {}).get('unlabelled_material_frac', float('nan')):.2f} {regs} "
                  f"{rec['seconds']}s fail={rec['fail']}", flush=True)


# --------------------------------------------------------------------------------------------------------- finalize
def q(v, qs=(0.1, 0.5, 0.9)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {f"p{int(100 * k)}": round(float(np.quantile(v, k)), 4) for k in qs} if len(v) else {}


def finalize(args):
    manifest = {"dataset": "real9", "created": time.strftime("%Y-%m-%d"), "crop_shape": [N] * 3,
                "licence": "derived from Vesuvius Challenge open data (CC BY-NC 4.0 / ash2txt data agreement). Local and "
                           "private: do not upload, publish or commit.",
                "excluded_scrolls": sorted(EXCLUDE),
                "label_convention": {
                    "labelsTr": "uint8 {0,1}: voxels within EDT 1.0 of the 1-vox mesh raster (~3-vox band; p0139 / jeong200 "
                                "convention), after removing surface voxels > 2 vox from material and rejected instances",
                    "ignoreTr": "uint8 {0,1}: removed off-papyrus voxels (dilated 2), rejected instances (dilated 3), locally deduplicated voxels of overlapping traces (dilated 1), CT material "
                                "> clip(0.7 pitch, 4, 10) vox from every kept label (unlabelled sheets, dilated 1); "
                                "restricted to unlabelled voxels",
                    "Dataset103": "label = labelsTr, set to 2 where ignoreTr (nnU-Net ignore label)"},
                "gates": {"NCC_MIN": NCC_MIN, "SHIFT_SPREAD_MAX": SHIFT_SPREAD_MAX, "INST_ONPAP_MIN": INST_ONPAP_MIN,
                          "INST_OFF_MAX": INST_OFF_MAX, "NORMAL_AGREE_MIN": NORMAL_AGREE_MIN, "DEEP_TILES_MAX": DEEP_TILES_MAX,
                          "CROSS_MAX": CROSS_MAX, "FRAG_MIN": FRAG_MIN, "POOLED_ONPAP_MIN": POOLED_ONPAP_MIN, "LABEL_CORE_MIN": LABEL_CORE_MIN,
                          "UNL_MAX": UNL_MAX, "CT_VALID_MIN": CT_VALID_MIN, "LABEL_ON_CT_MIN": LABEL_ON_CT_MIN,
                          "MATERIAL_MIN": MATERIAL_MIN, "ON_D": ON_D, "OFF_D": OFF_D, "DEEP_D": DEEP_D,
                          "mining": {"AREA_MIN": AREA_MIN, "DEDUP_D": DEDUP_D, "MIN_SEP": MIN_SEP, "FRAME_OUT_MAX": FRAME_OUT_MAX,
                                     "FRAME_MASKED_MAX": FRAME_MASKED_MAX}},
                "scrolls": {}, "crops": []}
    for sub in ("imagesTr", "labelsTr"):
        if (DS / sub).exists():
            shutil.rmtree(DS / sub)
        (DS / sub).mkdir(parents=True)
    n_ds = 0
    for scroll in SCROLLS:
        work = WORK / scroll
        if not (work / "candidates.json").exists():
            continue
        cj = json.loads((work / "candidates.json").read_text())
        cfg = cj["cfg"]
        cur = {c["stem"] for c in cj["candidates"]}
        recs = [json.loads(f.read_text()) for f in sorted((work / "sites").glob("*.json"))]
        recs = [r for r in recs if r["stem"] in cur]
        for r in recs:
            if "gates" in r:
                if "ct_valid_frac" not in r:
                    ct_checks(r, tifffile.imread(work / "all" / "imagesTr" / f"{r['stem']}_0000.tif"),
                              tifffile.imread(work / "all" / "labelsTr" / f"{r['stem']}.tif"))
                decide(r)
        passed = [r for r in recs if r["passed"]]
        od = OUT / scroll
        for sub in ("imagesTr", "labelsTr", "ignoreTr"):
            if (od / sub).exists():
                shutil.rmtree(od / sub)
            (od / sub).mkdir(parents=True)
        for r in passed:
            st = r["stem"]
            for sub, name in (("imagesTr", f"{st}_0000.tif"), ("labelsTr", f"{st}.tif"), ("ignoreTr", f"{st}.tif")):
                os.link(work / "all" / sub / name, od / sub / name)
            os.link(od / "imagesTr" / f"{st}_0000.tif", DS / "imagesTr" / f"{st}_0000.tif")  # hardlink: a real file for tar
            lab = tifffile.imread(od / "labelsTr" / f"{st}.tif")
            ign = tifffile.imread(od / "ignoreTr" / f"{st}.tif")
            tifffile.imwrite(DS / "labelsTr" / f"{st}.tif", np.where(ign > 0, 2, lab).astype(np.uint8), compression="zlib")
            n_ds += 1
        built = [r for r in recs if "gates" in r]
        regs = [v for r in built for v in r.get("registration", {}).values()]
        used = sorted({v["seg_id"] for r in passed for i, v in r["instances"].items() if i not in r["rejected_instances"]})
        segs_by_id = {s["id"]: s for s in cfg["segments"]}
        transforms = {}
        for s in cfg["segments"]:
            transforms.setdefault(s["src_tag"], {"mesh": f"segments/<seg>/mesh/<id>-on-{s['src_tag']}.tifxyz",
                                                 "transform": s["transform"], "A": s["A"], "t": s["t"],
                                                 "n_segments": 0})["n_segments"] += not (s.get("frame_bad") or s.get("missing"))
        manifest["scrolls"][scroll] = {
            "target_volume": cfg["url"], "vox_um": cfg["vox_um"], "target_shape_zyx": cfg["shape"],
            "transforms_by_mesh_volume": transforms,
            "segments_total": len(cfg["segments"]) + sum(not any(x["name"] == k for x in cfg["segments"]) for k in cfg["skipped"]),
            "segments_usable": sum(not (x.get("frame_bad") or x.get("missing")) for x in cfg["segments"]),
            "skipped_segments": cfg["skipped"],
            "segment_frame_check": {s["name"]: {"frac_outside_volume": round(s.get("frac_outside_volume", float("nan")), 4),
                                                "frac_on_masked_ct": round(s.get("frac_on_masked_ct", float("nan")), 4)}
                                    for s in cfg["segments"]},
            "segment_overlaps": {s["name"]: {"frac_within_4vox_of_larger": round(s.get("frac_overlapping_larger", 0), 3),
                                             **({"overlaps": s["overlaps"]} if s.get("overlaps") else {})}
                                 for s in cfg["segments"]},
            "candidates": len(cj["candidates"]), "built": len(built),
            "errors": sum("error" in " ".join(r["fail"]) for r in recs), "passed": len(passed),
            "fail_by_gate": {k: sum(k in r.get("fail_gate", []) for r in built) for k in ("ct", "label", "papyrus", "coverage")},
            "segments_in_passed_crops": len(used), "segments_in_passed_crops_names": [segs_by_id[i]["name"] for i in used],
            "registration": {"n": len(regs), "ncc": q([v["ncc"] for v in regs]), "ncc_published": q([v["ncc_published"] for v in regs]),
                             "shift_vox": q([np.linalg.norm(v["shift"]) for v in regs]),
                             "octant_dev_vox": q([v["octant_dev_max"] for v in regs]),
                             "groups_rejected": sum(1 for r in built for k in r.get("rejected_instances", {}).values()
                                                    if k.startswith("registration"))},
            "instances_total": sum(len(r["instances"]) for r in built),
            "instances_rejected": sum(len(r["rejected_instances"]) for r in built),
            "passed_stats": {"pooled_on_papyrus_kept": q([r["pooled_on_papyrus_kept"] for r in passed]),
                             "removed_frac": q([r["cleaning"]["removed_frac"] for r in passed]),
                             "unlabelled_material_frac": q([r["coverage"]["unlabelled_material_frac"] for r in passed]),
                             "label_frac": q([r["label_frac"] for r in passed]), "ignore_frac": q([r["ignore_frac"] for r in passed]),
                             "pitch_vox": q([r["coverage"]["pitch_vox"] for r in passed]),
                             "instances_per_crop": q([len(r["instances"]) - len(r["rejected_instances"]) for r in passed])},
            "download_GB_upper": round(sum(r.get("chunks_target", 0) + sum(v["chunks_src"] for v in r.get("registration", {}).values())
                                           for r in recs) * 128 ** 3 / 1e9, 2),
        }
        for r in recs:
            manifest["crops"].append({k: r.get(k) for k in ("stem", "scroll", "vox_um", "origin_zyx", "passed", "fail", "mined",
                                                             "registration", "rejected_instances", "pooled_on_papyrus_kept",
                                                             "cleaning", "coverage", "label_frac", "ignore_frac", "seconds")}
                                     | {"instances": {i: v["segment"] for i, v in r.get("instances", {}).items()},
                                        "gates_instances": r.get("gates", {}).get("instances")})
        print(scroll, json.dumps({k: manifest["scrolls"][scroll][k] for k in ("candidates", "built", "errors", "passed",
                                                                             "fail_by_gate", "segments_in_passed_crops")}))
    manifest["n_passed_total"] = n_ds
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    (DS / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "surface": 1, "ignore": 2}, "numTraining": n_ds,
        "file_ending": ".tif", "overwrite_image_reader_writer": "Tiff3DIO",
        "name": "real ~9 um crops (PHerc0814/0500P2 9.362 um, PHerc0009B/1447 8.64 um) from public segment meshes; "
                "not PHerc0139"}, indent=1))
    print(f"Dataset103: {n_ds} cases -> {DS}")
    preview(manifest)


def preview(manifest, n_per=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    pick = []
    for scroll in SCROLLS:
        ps = [c for c in manifest["crops"] if c["scroll"] == scroll and c["passed"]]
        pick += [ps[i] for i in rng.permutation(len(ps))[:n_per]]
    if not pick:
        return
    fig, axs = plt.subplots(len(pick), 3, figsize=(13, 4.3 * len(pick)))
    axs = np.atleast_2d(axs)
    for row, c in zip(axs, pick):
        d = OUT / c["scroll"]
        img = tifffile.imread(d / "imagesTr" / f"{c['stem']}_0000.tif")
        lab = tifffile.imread(d / "labelsTr" / f"{c['stem']}.tif")
        ign = tifffile.imread(d / "ignoreTr" / f"{c['stem']}.tif")
        h = N // 2
        for ax, (im, lb, ig, nm) in zip(row, ((img[h], lab[h], ign[h], "z"), (img[:, h], lab[:, h], ign[:, h], "y"),
                                              (img[:, :, h], lab[:, :, h], ign[:, :, h], "x"))):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.imshow(np.ma.masked_where(ig == 0, ig), cmap="winter", alpha=0.35, vmin=0, vmax=1)
            if lb.any():
                ax.contour(lb, levels=[0.5], colors="red", linewidths=0.6)
            ax.set_title(f"{c['stem']} {nm}={h} ({c['vox_um']} um)", fontsize=8)
            ax.axis("off")
    fig.suptitle("real9 crops: CT, label band contour (red), ignore (blue)", fontsize=10)
    fig.tight_layout()
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW, dpi=90)
    print(f"preview -> {PREVIEW}")


# ------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "mine", "build", "finalize", "cfg", "thin", "gap"])
    ap.add_argument("--scroll", default=",".join(SCROLLS))
    ap.add_argument("--n-max", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--dedup-d", type=float, default=DEDUP_D)
    args = ap.parse_args()
    args.scrolls = [s for s in args.scroll.split(",") if s]
    assert not set(args.scrolls) & EXCLUDE, "PHerc0139 must stay out of training"
    if args.cmd == "cfg":
        for s in args.scrolls:
            c = scroll_cfg(s)
            print(s, c["target_long"], c["vox_um"], len(c["segments"]), "usable;", json.dumps(c["skipped"], indent=1))
            for k in sorted({x["src_tag"] for x in c["segments"]}):
                ex = next(x for x in c["segments"] if x["src_tag"] == k)
                print(f"  {k}: {sum(x['src_tag'] == k for x in c['segments'])} segs, {ex['transform']}")
        return
    {"fetch": fetch, "mine": mine, "build": build, "finalize": finalize, "thin": thin, "gap": gap}[args.cmd](args)


if __name__ == "__main__":
    main()
