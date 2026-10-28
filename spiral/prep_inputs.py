#!/usr/bin/env python3
"""prep_inputs.py - PHerc0139 spiral-fit inputs that are published only in the 2.399 um frame (20260102150214),
moved into the 9.362 um frame of volume 20250728140407 with the published affine.

  umbilicus   PHerc0139/representations/umbilicus/20260102150214-umbilicus-*.json (391 control points, xyz in 2.399
              voxels) -> umbilicus.json in 9.362 voxels (p9 = A p2 + t, xyz).
  fetch       mirror the z chunk rows of the lasagna nx / ny / grad_mag stores (group 4 = 16x of 2.399) that the band
              needs, into <dl>/lasagna_2p399/ (downloaded data, its own directory, read as untrusted bytes).
  resample    write PHerc0139_{nx,ny,grad_mag}.ome.zarr in the 9.362 frame at lasagna_scale 4 (group "2", the same
              layout as the PHerc0125 store), only the chunk rows of the band (rest absent = no data):
                - nearest neighbour: target voxel i (zyx) sits at working coordinate 4 i (the fitter samples with
                  round(p / lasagna_scale)); source voxel j at 16 j in 2.399 coordinates (same convention);
                - normals are covectors: n9 = normalize(L^-T n2), L = linear part of A. The matrix is derived from
                  the json, not assumed (it is a 177.4 deg rotation about an axis 0.35 deg off z, scale 0.2571, so
                  nx and ny change sign). The stores keep only nx, ny and the fitter takes nz = +sqrt(1 - nx^2 - ny^2),
                  so a vector with n9z < 0 is negated (normals are lines; every loss uses |cos|);
                - grad_mag is a winding density. lasagna writes u8 = 1000 x windings per model-input voxel, and
                  <run>.lasagna.json gives grad_mag_factor: windings per base (L0) voxel = u8 / 1000 x factor
                  (PHerc0139 2.399 store: 0.25, the model ran on L2; PHerc0125 9.362 store: 1.0). The fitter decodes
                  u8 x dense_grad_mag_factor (0.25) / 1000 x lasagna_scale (4) = u8 / 1000 per working voxel, i.e.
                  the PHerc0125 convention. A density is |grad w|, which maps with |L^-T n2|, so
                  u8_9 = u8_2 x 0.25 x |L^-T n2| (~0.97 x u8_2). Note: dense_spacing_mode "phase" (the recipe) does
                  not read grad_mag at all; it is written for completeness and for grad_mag-mode arms.
Writes plain zarr v2 (blosc lz4, "/" separator, 32^3 chunks, only non-empty chunks), so pack_resident_pools.py and the
fitter read it as they read the published stores.

    uv run python spiral/prep_inputs.py all --z0 3000 --z1 4500 --dl data/dl_p0139 --out data/p0139/inputs
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AFFINE = os.path.join(ROOT, "data/eval/surface_labels_2026_07_sample/transform_0139_2p399_to_9p362.json")
UMB_PREFIX = "PHerc0139/representations/umbilicus/"
LAS_PREFIX = "PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/"
LAS_RUN = "PHerc0139-20260102150214-lasagna-20260724"
SRC_GROUP, SRC_SCALE = "4", 16          # 2.399 voxels per source voxel
TGT_GROUP, TGT_SCALE = "2", 4           # 9.362 voxels per target voxel
TGT_CHUNK = 32
VOL9_SHAPE = (20974, 6621, 6621)        # 20250728140407 level 0, zyx
ARRAYS = ("nx", "ny", "grad_mag")
SCROLL = "PHerc0139"
# --scroll PHerc1667: the fit frame is level 2 (9.596 um) of the 2.399 um volume 20251217075048 itself, so the
# "affine" is p_L2 = p_2399 / 4 (L = 0.25 I, t = 0): source group 4 maps 1:1 onto target group 2 (identity resample,
# normals unchanged, grad_mag u8 x 0.25 x 4 = unchanged). The umbilicus is the human-mesh derived one
# (eval/p1667_labels.py), already in the L2 frame; no published umbilicus exists for this scroll.
SCROLLS = {
    "PHerc1667": {
        "LAS_PREFIX": "PHerc1667/representations/predictions/lasagna/20251217075048-lasagna-20260419180421-L2/",
        "LAS_RUN": "PHerc1667-20251217075048-lasagna-20260724",
        "VOL9_SHAPE": (9269, 3808, 3808),   # 20251217075048 level 2, zyx
        "AFFINE": None,
        "UMB_LOCAL": os.path.join(ROOT, "data/eval/p1667/umbilicus_p1667_L2.json"),
        "VOLUME": "PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr (level 2)",
        "CT_URL": "PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr",
        "CT_SRC_LEVEL": 5,                  # 2.399 level 5 = L2-frame level 3; max-pooled 2x -> L2-frame level 4
    },
    # PHercParis4 (Scroll 1, third confirmation region, notes/2026-10-23-third-region-prereg.md): level 2 (9.6 um) of
    # the 2.400 um volume 20260411134726, same structure as PHerc1667 (lasagna group 4 = scale 16 in the 2.400 frame).
    # The umbilicus is the public one (representations/umbilicus/20260411134726-umbilicus-20260524235033.json) / 4.
    "PHercParis4": {
        "LAS_PREFIX": "PHercParis4/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/",
        "LAS_RUN": "PHercParis4-20260411134726-las-sd2-5b17ff6c",
        "VOL9_SHAPE": (18946, 8174, 8174),  # 20260411134726 level 2, zyx
        "AFFINE": None,
        "UMB_LOCAL": os.path.join(ROOT, "data/eval/paris4/umbilicus_paris4_L2.json"),
        "VOLUME": "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr (level 2)",
        "CT_URL": "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr",
        "CT_SRC_LEVEL": 5,
    },
}


def set_scroll(name):
    global SCROLL, LAS_PREFIX, LAS_RUN, VOL9_SHAPE, AFFINE
    SCROLL = name
    if name in SCROLLS:
        c = SCROLLS[name]
        LAS_PREFIX, LAS_RUN, VOL9_SHAPE, AFFINE = c["LAS_PREFIX"], c["LAS_RUN"], c["VOL9_SHAPE"], c["AFFINE"]


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                raise
        except Exception:
            if k == tries - 1:
                raise
        time.sleep(2 * (k + 1))


def s3_keys(prefix):
    keys, tok = [], None
    while True:
        u = f"{S3}/?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        r = ET.fromstring(get(u))
        keys += [c.find("s3:Key", NS).text for c in r.findall("s3:Contents", NS)]
        t = r.find("s3:NextContinuationToken", NS)
        if t is None:
            return keys
        tok = t.text


def load_affine():
    if AFFINE is None:                                  # PHerc1667: L2 frame of the source volume itself
        return np.eye(3) * 0.25, np.zeros(3)
    T = json.load(open(AFFINE))
    A = np.array(T["transformation_matrix"], dtype=np.float64)
    assert A.shape == (3, 4)
    L, t = A[:, :3], A[:, 3]
    return L, t


def affine_report(L):
    s = float(np.cbrt(np.linalg.det(L)))
    R = L / s
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    w, v = np.linalg.eig(R)
    axis = v[:, np.argmin(abs(w - 1))].real
    M = np.linalg.inv(L).T
    return {"scale": s, "det": float(np.linalg.det(L)), "rotation_deg": ang,
            "axis_xyz": (axis * np.sign(axis[2])).round(5).tolist(),
            "normal_map_times_scale": (M * s).round(5).tolist(),
            "singular_values": np.linalg.svd(L)[1].round(6).tolist()}


# ---------------------------------------------------------------------------------------------------- umbilicus
def cmd_umbilicus(a):
    if SCROLL in SCROLLS:
        c = SCROLLS[SCROLL]
        d = json.load(open(c["UMB_LOCAL"]))
        d.setdefault("metadata", {}).update({"source_volume": c["VOLUME"], "derived_from": os.path.relpath(
            c["UMB_LOCAL"], ROOT), "made_by": "spiral/prep_inputs.py --scroll " + SCROLL})
        os.makedirs(a.out, exist_ok=True)
        json.dump(d, open(os.path.join(a.out, "umbilicus.json"), "w"), indent=1)
        zs = np.array([float(p["z"]) for p in d["control_points"]])
        print(f"umbilicus ({SCROLL}, human-mesh derived): {len(zs)} points, z {zs.min():.0f}..{zs.max():.0f}, "
              f"band points: {int(((zs >= a.z0) & (zs < a.z1)).sum())}")
        return
    os.makedirs(a.dl, exist_ok=True)
    keys = [k for k in s3_keys(UMB_PREFIX) if k.split("/")[-1].startswith("20260102150214") and k.endswith(".json")]
    src = os.path.join(a.dl, "umbilicus_2p399.json")
    if not os.path.exists(src):
        open(src, "wb").write(get(f"{S3}/{sorted(keys)[-1]}"))
    d = json.load(open(src))
    L, t = load_affine()
    pts = d["control_points"]
    p2 = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in pts])
    p9 = p2 @ L.T + t
    out = {"control_points": [{"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 2),
                               **({"score": p["score"]} if "score" in p else {})}
                              for (x, y, z), p in zip(p9, pts)],
           "metadata": {"source_volume": "PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr",
                        "derived_from": f"{S3}/{sorted(keys)[-1]}",
                        "transform": os.path.relpath(AFFINE, ROOT), "made_by": "spiral/prep_inputs.py"}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "umbilicus.json"), "w"), indent=1)
    zs = p9[:, 2]
    print(f"umbilicus: {len(pts)} points, 9.362 z {zs.min():.0f}..{zs.max():.0f}, monotonic z: "
          f"{bool(np.all(np.diff(zs[np.argsort(p2[:, 2])]) > 0))}; band points: "
          f"{int(((zs >= a.z0) & (zs < a.z1)).sum())}")


# ---------------------------------------------------------------------------------------------------- lasagna
def target_rows(z0, z1, margin):
    """Target chunk rows [r0, r1) covering working z [z0 - margin, z1 + margin)."""
    lo = max(0, (z0 - margin) // TGT_SCALE)
    hi = -(-(z1 + margin) // TGT_SCALE)
    return lo // TGT_CHUNK, -(-hi // TGT_CHUNK)


def target_shape():
    return tuple(-(-s // TGT_SCALE) for s in VOL9_SHAPE)


def src_z_range(L, t, z9lo, z9hi, src_shape):
    """Source group-4 z rows needed for target working z [z9lo, z9hi) over the whole yx plane."""
    Li = np.linalg.inv(L)
    corners = np.array([[x, y, z] for x in (0, VOL9_SHAPE[2]) for y in (0, VOL9_SHAPE[1]) for z in (z9lo, z9hi)], float)
    z2 = ((corners - t) @ Li.T)[:, 2] / SRC_SCALE
    return max(0, int(np.floor(z2.min())) - 2), min(src_shape[0], int(np.ceil(z2.max())) + 3)


def cmd_fetch(a):
    L, t = load_affine()
    dst = os.path.join(a.dl, "lasagna_2p399")
    os.makedirs(dst, exist_ok=True)
    meta = get(f"{S3}/{LAS_PREFIX}{LAS_RUN}.lasagna.json")
    open(os.path.join(dst, f"{LAS_RUN}.lasagna.json"), "wb").write(meta)
    r0, r1 = target_rows(a.z0, a.z1, a.margin)
    for name in ARRAYS:
        store = f"{LAS_PREFIX}{LAS_RUN}_{name}.ome.zarr"
        sd = os.path.join(dst, f"{LAS_RUN}_{name}.ome.zarr")
        os.makedirs(os.path.join(sd, SRC_GROUP), exist_ok=True)
        for f in (".zattrs", ".zgroup"):
            open(os.path.join(sd, f), "wb").write(get(f"{S3}/{store}/{f}"))
        za = json.loads(get(f"{S3}/{store}/{SRC_GROUP}/.zarray"))
        open(os.path.join(sd, SRC_GROUP, ".zarray"), "w").write(json.dumps(za))
        zlo, zhi = src_z_range(L, t, r0 * TGT_CHUNK * TGT_SCALE, r1 * TGT_CHUNK * TGT_SCALE, za["shape"])
        c = za["chunks"][0]
        rows = range(zlo // c, -(-zhi // c))
        done = os.path.join(sd, f".rows_{rows.start}-{rows.stop}.done")
        if os.path.exists(done):
            print(f"fetch {name}: rows {rows.start}..{rows.stop} present")
            continue
        keys = []
        for r in rows:
            keys += s3_keys(f"{store}/{SRC_GROUP}/{r}/")
        def dl(k):
            p = os.path.join(sd, SRC_GROUP, k.split(f"/{SRC_GROUP}/", 1)[1])
            if os.path.exists(p):
                return 0
            os.makedirs(os.path.dirname(p), exist_ok=True)
            b = get(f"{S3}/{k}")
            open(p + ".part", "wb").write(b)
            os.replace(p + ".part", p)
            return len(b)
        t0 = time.time()
        with ThreadPoolExecutor(32) as ex:
            n = sum(ex.map(dl, keys))
        open(done, "w").write("")
        print(f"fetch {name}: source rows {rows.start}..{rows.stop} (z {zlo}..{zhi} of group {SRC_GROUP}), "
              f"{len(keys)} chunks, {n / 1e6:.0f} MB in {time.time() - t0:.0f} s")


class ChunkReader:
    """Read a box of a local zarr v2 uint8 array (absent chunks = fill 0)."""

    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, ".zarray")))
        self.shape, self.chunks = self.meta["shape"], self.meta["chunks"]
        self.codec = numcodecs.get_codec(self.meta["compressor"]) if self.meta["compressor"] else None
        self.sep = self.meta.get("dimension_separator", ".")

    def chunk(self, idx):
        p = os.path.join(self.path, self.sep.join(map(str, idx)))
        if not os.path.exists(p):
            return None
        b = open(p, "rb").read()
        if self.codec:
            b = self.codec.decode(b)
        return np.frombuffer(b, np.uint8).reshape(self.chunks)

    def box(self, z0, z1):
        """Full yx planes for z rows [z0, z1)."""
        out = np.zeros((z1 - z0, self.shape[1], self.shape[2]), np.uint8)
        c = self.chunks
        for cz in range(z0 // c[0], -(-z1 // c[0])):
            for cy in range(-(-self.shape[1] // c[1])):
                for cx in range(-(-self.shape[2] // c[2])):
                    a = self.chunk((cz, cy, cx))
                    if a is None:
                        continue
                    zs, ze = max(z0, cz * c[0]), min(z1, (cz + 1) * c[0])
                    ys, ye = cy * c[1], min(self.shape[1], (cy + 1) * c[1])
                    xs, xe = cx * c[2], min(self.shape[2], (cx + 1) * c[2])
                    out[zs - z0:ze - z0, ys:ye, xs:xe] = a[zs - cz * c[0]:ze - cz * c[0], :ye - ys, :xe - xs]
        return out


def write_zarr_meta(path, group, shape, scale, name):
    os.makedirs(os.path.join(path, group), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(path, ".zgroup"), "w"))
    json.dump({"multiscales": [{"version": "0.4", "name": name,
                                "axes": [{"name": k, "type": "space", "unit": "pixel"} for k in "zyx"],
                                "datasets": [{"path": group, "coordinateTransformations": [
                                    {"type": "scale", "scale": [float(scale)] * 3}]}]}],
               "p0139_resampled_from": f"{S3}/{LAS_PREFIX}{LAS_RUN}_{name}.ome.zarr/{SRC_GROUP}",
               "p0139_transform": os.path.relpath(AFFINE, ROOT) if AFFINE else "p_L2 = p_2399 / 4"},
              open(os.path.join(path, ".zattrs"), "w"), indent=1)
    json.dump({"zarr_format": 2, "shape": list(shape), "chunks": [TGT_CHUNK] * 3, "dtype": "|u1", "fill_value": 0,
               "order": "C", "filters": None, "dimension_separator": "/",
               "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}},
              open(os.path.join(path, group, ".zarray"), "w"), indent=1)


def resample_slab(src, L, t, kz0, kz1, tshape):
    """Target rows [kz0, kz1) -> dict name -> uint8 (kz1-kz0, Y, X); also returns stats."""
    Li = np.linalg.inv(L)
    M = Li.T                                            # normal (covector) map 2.399 -> 9.362, xyz
    ny_, nx_ = tshape[1], tshape[2]
    jj, ii = np.meshgrid(np.arange(ny_, dtype=np.float64), np.arange(nx_, dtype=np.float64), indexing="ij")
    out = {k: np.zeros((kz1 - kz0, ny_, nx_), np.uint8) for k in ARRAYS}
    z2rows = src["z0"], src["z0"] + src["nx"].shape[0]
    for k in range(kz0, kz1):
        p9 = np.stack([ii * TGT_SCALE, jj * TGT_SCALE, np.full_like(ii, k * TGT_SCALE)], -1)  # xyz working
        q = np.rint(((p9 - t) @ Li.T) / SRC_SCALE).astype(np.int64)                        # source idx xyz
        sx, sy, sz = q[..., 0], q[..., 1], q[..., 2] - z2rows[0]
        inb = ((sx >= 0) & (sx < src["nx"].shape[2]) & (sy >= 0) & (sy < src["nx"].shape[1])
               & (sz >= 0) & (sz < src["nx"].shape[0]))
        if q[..., 2][inb].size and (q[..., 2][inb].min() < z2rows[0] or q[..., 2][inb].max() >= z2rows[1]):
            raise RuntimeError("source slab too small")
        sx, sy, sz = np.where(inb, sx, 0), np.where(inb, sy, 0), np.where(inb, sz, 0)
        unx = src["nx"][sz, sy, sx]
        uny = src["ny"][sz, sy, sx]
        ug = src["grad_mag"][sz, sy, sx]
        valid = inb & ((unx != 0) | (uny != 0))
        n2x = (unx.astype(np.float64) - 128.0) / 127.0
        n2y = (uny.astype(np.float64) - 128.0) / 127.0
        n2z = np.sqrt(np.clip(1.0 - n2x * n2x - n2y * n2y, 0.0, None))
        n2 = np.stack([n2x, n2y, n2z], -1)
        n2 /= np.maximum(np.linalg.norm(n2, axis=-1, keepdims=True), 1e-9)
        n9 = n2 @ M.T
        g = np.linalg.norm(n9, axis=-1)
        n9 /= np.maximum(g[..., None], 1e-12)
        n9 *= np.where(n9[..., 2:3] < 0, -1.0, 1.0)
        enc = lambda v: np.clip(np.rint(128.0 + 127.0 * v), 1, 255).astype(np.uint8)
        out["nx"][k - kz0] = np.where(valid, enc(n9[..., 0]), 0)
        out["ny"][k - kz0] = np.where(valid, enc(n9[..., 1]), 0)
        gv = inb & (ug != 0)
        gm = np.rint(ug.astype(np.float64) * src["grad_factor"] * np.where(valid, g, np.median(g[valid]) if valid.any() else (1 / 0.2571 if AFFINE else 4.0)))
        out["grad_mag"][k - kz0] = np.where(gv, np.clip(gm, 1, 255), 0).astype(np.uint8)
    return out


def cmd_resample(a):
    L, t = load_affine()
    rep = affine_report(L)
    print("affine:", json.dumps(rep))
    dl = os.path.join(a.dl, "lasagna_2p399")
    lmeta = json.load(open(os.path.join(dl, f"{LAS_RUN}.lasagna.json")))
    gfac = float(lmeta["grad_mag_factor"])            # windings per base voxel = u8 / encode_scale * factor
    assert float(lmeta["grad_mag_encode_scale"]) == 1000.0
    # the fitter (and the PHerc0125 store) use u8 / 1000 = windings per working voxel at lasagna_scale 4 with its
    # default dense_grad_mag_factor 0.25: target u8 = 1000 * (u8_src / 1000 * gfac) * |L^-T n|
    readers = {k: ChunkReader(os.path.join(dl, f"{LAS_RUN}_{k}.ome.zarr", SRC_GROUP)) for k in ARRAYS}
    tshape = target_shape()
    r0, r1 = target_rows(a.z0, a.z1, a.margin)
    codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    outs = {}
    for k in ARRAYS:
        p = os.path.join(a.out, f"{SCROLL}_{k}.ome.zarr")
        write_zarr_meta(p, TGT_GROUP, tshape, TGT_SCALE, k)
        outs[k] = p
    stats = {"affine": rep, "target_shape": tshape, "target_rows": [r0, r1], "grad_mag_src_factor": gfac,
             "rows": {}}
    t00 = time.time()
    for r in range(r0, r1):
        done = os.path.join(a.out, f".resample_row{r}.done")
        if os.path.exists(done):
            continue
        kz0, kz1 = r * TGT_CHUNK, min(tshape[0], (r + 1) * TGT_CHUNK)
        zlo, zhi = src_z_range(L, t, kz0 * TGT_SCALE, kz1 * TGT_SCALE, readers["nx"].shape)
        src = {k: readers[k].box(zlo, zhi) for k in ARRAYS}
        src["z0"] = zlo
        src["grad_factor"] = gfac
        o = resample_slab(src, L, t, kz0, kz1, tshape)
        nchunks = 0
        for k in ARRAYS:
            arr = o[k]
            for cy in range(-(-tshape[1] // TGT_CHUNK)):
                for cx in range(-(-tshape[2] // TGT_CHUNK)):
                    blk = np.zeros((TGT_CHUNK,) * 3, np.uint8)
                    v = arr[:, cy * TGT_CHUNK:(cy + 1) * TGT_CHUNK, cx * TGT_CHUNK:(cx + 1) * TGT_CHUNK]
                    if not v.any():
                        continue
                    blk[:v.shape[0], :v.shape[1], :v.shape[2]] = v
                    d = os.path.join(outs[k], TGT_GROUP, str(r), str(cy))
                    os.makedirs(d, exist_ok=True)
                    open(os.path.join(d, str(cx)), "wb").write(codec.encode(blk))
                    nchunks += 1
        sg = src["grad_mag"][src["grad_mag"] > 0]
        tg = o["grad_mag"][o["grad_mag"] > 0]
        stats["rows"][r] = {"src_z": [zlo, zhi], "chunks": nchunks,
                            "valid_frac": float((o["nx"] > 0).mean()),
                            "grad_src_p50": float(np.median(sg)) if sg.size else None,
                            "grad_tgt_p50": float(np.median(tg)) if tg.size else None,
                            "grad_tgt_sat_frac": float((tg == 255).mean()) if tg.size else None}
        open(done, "w").write(json.dumps(stats["rows"][r]))
        print(f"row {r} (z9 {kz0 * TGT_SCALE}..{kz1 * TGT_SCALE}): {json.dumps(stats['rows'][r])} "
              f"[{time.time() - t00:.0f} s]", flush=True)
    for r in range(r0, r1):
        stats["rows"][r] = json.load(open(os.path.join(a.out, f".resample_row{r}.done")))
    meta = {"version": 2, "source_to_base": 1.0, "grad_mag_encode_scale": 1000.0, "grad_mag_factor": 1.0,
            "groups": {k: {"zarr": f"{SCROLL}_{k}.ome.zarr/{TGT_GROUP}", "scaledown": 2, "channels": [k]}
                       for k in ARRAYS},
            "base_shape_zyx": list(VOL9_SHAPE), "p0139": stats}
    json.dump(meta, open(os.path.join(a.out, f"{SCROLL}.lasagna.json"), "w"), indent=1)
    print(f"resample done: rows {r0}..{r1} of {tshape}; grad_mag u8 median src -> tgt per row: "
          + ", ".join(f"{v['grad_src_p50']}->{v['grad_tgt_p50']}" for v in list(stats["rows"].values())[::3]))


# ---------------------------------------------------------------------------------------------------- CT level 4
def cmd_ct(a):
    """PHerc1667 only: ct.zarr group "4" in the L2 frame (scale 16 = the recipe's ct_mask_level 4, as the 9.362 um
    PHerc0139 volume's own level 4). The 2.399 volume has no level 6, so 2.399 level 5 rows of the band (+-512 L2
    slices) are max-pooled 2x (exact for make_surf_sdt's CT > 0 occupancy test); rows outside are absent (= 0)."""
    if SCROLL not in SCROLLS:
        return
    c = SCROLLS[SCROLL]
    url = f"{S3}/{c['CT_URL']}/{c['CT_SRC_LEVEL']}"
    za = json.loads(get(f"{url}/.zarray"))
    assert za["compressor"] is None and za["dtype"] == "|u1", za
    f = 2 ** (c["CT_SRC_LEVEL"] - 2)                    # L2 voxels per source voxel (8)
    zlo, zhi = max(0, (a.z0 - 512) // f), min(za["shape"][0], -(-(a.z1 + 512) // f))
    ch = za["chunks"]
    src_dir = os.path.join(a.dl, f"ct_L{c['CT_SRC_LEVEL']}")
    os.makedirs(src_dir, exist_ok=True)
    jobs = [(cz, cy, cx) for cz in range(zlo // ch[0], -(-zhi // ch[0]))
            for cy in range(-(-za["shape"][1] // ch[1])) for cx in range(-(-za["shape"][2] // ch[2]))]

    def dl(k):
        p = os.path.join(src_dir, "_".join(map(str, k)))
        if not os.path.exists(p):
            b = get(f"{url}/{k[0]}/{k[1]}/{k[2]}")
            open(p + ".part", "wb").write(b if b is not None else b"")
            os.replace(p + ".part", p)
        return p
    with ThreadPoolExecutor(16) as ex:
        paths = dict(zip(jobs, ex.map(dl, jobs)))
    vol = np.zeros(((-(-za["shape"][0] // ch[0])) * ch[0], za["shape"][1] + ch[1], za["shape"][2] + ch[2]), np.uint8)
    for (cz, cy, cx), p in paths.items():
        b = open(p, "rb").read()
        if b:
            vol[cz * ch[0]:(cz + 1) * ch[0], cy * ch[1]:(cy + 1) * ch[1], cx * ch[2]:(cx + 1) * ch[2]] = \
                np.frombuffer(b, np.uint8).reshape(ch)
    vol = vol[:za["shape"][0], :za["shape"][1], :za["shape"][2]]
    vol[:zlo] = 0
    vol[zhi:] = 0
    pad = [(0, n % 2) for n in vol.shape]
    v = np.pad(vol, pad)
    l4 = v.reshape(v.shape[0] // 2, 2, v.shape[1] // 2, 2, v.shape[2] // 2, 2).max(axis=(1, 3, 5))
    out = os.path.join(a.out, "ct.zarr")
    os.makedirs(os.path.join(out, "4"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(out, ".zgroup"), "w"))
    json.dump({"multiscales": [{"version": "0.4", "name": "ct", "axes": [{"name": k, "type": "space"} for k in "zyx"],
                                "datasets": [{"path": "4", "coordinateTransformations": [
                                    {"type": "scale", "scale": [16.0] * 3}]}]}],
               "made_by": f"spiral/prep_inputs.py ct --scroll {SCROLL}",
               "source": f"{url} rows z {zlo}..{zhi}, max-pooled 2x; frame = level 2 of the 2.399 um volume"},
              open(os.path.join(out, ".zattrs"), "w"), indent=1)
    C = 128
    json.dump({"zarr_format": 2, "shape": list(l4.shape), "chunks": [C] * 3, "dtype": "|u1", "fill_value": 0,
               "order": "C", "filters": None, "dimension_separator": "/",
               "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}},
              open(os.path.join(out, "4", ".zarray"), "w"), indent=1)
    codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    for cz in range(-(-l4.shape[0] // C)):
        for cy in range(-(-l4.shape[1] // C)):
            for cx in range(-(-l4.shape[2] // C)):
                blk = np.zeros((C,) * 3, np.uint8)
                sub = l4[cz * C:(cz + 1) * C, cy * C:(cy + 1) * C, cx * C:(cx + 1) * C]
                if not sub.any():
                    continue
                blk[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                os.makedirs(os.path.join(out, "4", str(cz), str(cy)), exist_ok=True)
                open(os.path.join(out, "4", str(cz), str(cy), str(cx)), "wb").write(codec.encode(blk))
    print(f"ct: {out}/4 shape {l4.shape} from {url} rows {zlo}..{zhi} ({len(jobs)} chunks); "
          f"L2 z {zlo * f}..{zhi * f} occupied {float((l4[zlo // 2:zhi // 2] > 0).mean()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["umbilicus", "fetch", "resample", "ct", "all"])
    ap.add_argument("--z0", type=int, default=3000)
    ap.add_argument("--z1", type=int, default=4500)
    ap.add_argument("--margin", type=int, default=128, help="working slices of lasagna beyond the band")
    ap.add_argument("--dl", default=os.path.join(ROOT, "data/dl_p0139"), help="downloads (untrusted)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/p0139/inputs"))
    ap.add_argument("--scroll", default="PHerc0139", choices=["PHerc0139", *SCROLLS])
    a = ap.parse_args()
    set_scroll(a.scroll)
    for c in (["umbilicus", "fetch", "resample", "ct"] if a.cmd == "all" else [a.cmd]):
        globals()[f"cmd_{c}"](a)


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  (s3_keys)
    main()
