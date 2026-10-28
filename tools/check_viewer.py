"""check_viewer.py - local blind viewer for human checks of uncertain winding steps (PHerc0139, PHerc1667).

Each queue item is one chain step: two points A and B on a radial line from the umbilicus, same z, in the scroll's fit
frame: PHerc0139 = 9.362 um L0 voxels (volume 20250728140407); PHerc1667 = 9.596 um level 2 of the 2.399 um volume
20251217075048 (queue field "scroll": "p1667", or --scroll p1667; crops cached under data/scratch/hitl/cache_p1667/). The user answers "how many sheets lie strictly between A and B?" (0/1/2/3/u) and the answer,
time taken and whether the 2.4 um view was opened go to a jsonl log. Notes: notes/2026-10-22-check-viewer.md.

  # one-time / when the queue changes: render crops for the first 300 items (shuffled order) into the cache
  uv run python tools/check_viewer.py precache --queue data/scratch/hitl/check_queue_band1.json --n 300
  # session (opens the browser; resumable: run the same command again to continue)
  uv run python tools/check_viewer.py --queue data/scratch/hitl/check_queue_band1.json \
      --out data/scratch/hitl/answers_2026-10-25.jsonl
  # PHerc1667 gm calibration session (notes/2026-10-22-p1667-calib-session.md)
  uv run python tools/check_viewer.py --queue data/scratch/hitl/check_queue_p1667_calib.json \
      --out data/scratch/hitl/answers_p1667_calib.jsonl

Other modes: `standin` (stand-in queue from band_m7_ray_gmxc gm-raised steps), `tutorial-pick` (dev: candidate
tutorial steps from z 4500-6000 + a contact sheet), `shots` (headless end-to-end test + screenshots).

Views, all in one rotated frame (u = along A->B, A on the left; v = in-plane perpendicular; w = z):
  - axial 9.362 slice at the step's z; ray-z plane (u, z) through the line; intensity profile along A->B
    (perpendicular mean over +-1.5 vox, 3 z planes);
  - toggle (h): the same at 2.399 um (axial L0 slice; ray-z plane from L1 = 4.8 um), registered to the 9.362 frame
    with the published affine plus a per-item NCC shift search (+-3 vox). PHerc1667: the fit frame is level 2 of the
    same 2.399 um volume, so the map is the pyramid scale (L0 = 4 p + 1.5, voxel centres) plus the same NCC search.
CT is streamed as byte ranges of the uncompressed 128^3 zarr chunks (only the z planes needed); nothing else is read.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
HITL = REPO / "data" / "scratch" / "hitl"
CACHE = HITL / "cache"
UI_HTML = Path(__file__).resolve().parent / "check_viewer_ui.html"
TUTORIAL_JSON = HITL / "tutorial.json"
BUCKET = "vesuvius-challenge-open-data/PHerc0139/volumes/"
VOL9 = BUCKET + "20250728140407-9.362um-1.2m-113keV-masked.zarr"
VOL24 = BUCKET + "20260102150214-2.399um-0.2m-78keV-masked.zarr"
TF_LOCAL = REPO / "data" / "eval" / "surface_labels_2026_07_sample" / "transform_0139_2p399_to_9p362.json"
CH = 128
VOL1667 = "vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr"
VOLP4 = "vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
SHAPES = {(VOL9, 0): (20974, 6621, 6621), (VOL24, 0): (76953, 26511, 26511), (VOL24, 1): (38477, 13256, 13256),
          (VOL1667, 0): (37076, 15229, 15229), (VOL1667, 1): (18538, 7615, 7615), (VOL1667, 2): (9269, 3808, 3808),
          (VOLP4, 0): (75784, 32693, 32693), (VOLP4, 1): (37892, 16347, 16347), (VOLP4, 2): (18946, 8174, 8174)}
RENDER_VERSION = 4
# Per-scroll sources. base = the fit frame the queue points live in (rendered at 0.5 vox); hi0 / hi1 = 2.399 um L0 / L1.
# hires_map: "affine" = published 2.399 -> 9.362 transform (PHerc0139); "pyramid" = base is level `base_level` of the
# same 2.399 volume (PHerc1667). All volumes are uint8 zarr v2, 128^3 chunks, uncompressed, C order.
SCROLLS = {
    "p0139": dict(base=(VOL9, 0), hi0=(VOL24, 0), hi1=(VOL24, 1), vox_um=9.362, cache=CACHE, hires_map="affine"),
    "p1667": dict(base=(VOL1667, 2), hi0=(VOL1667, 0), hi1=(VOL1667, 1), vox_um=9.596, cache=HITL / "cache_p1667",
                  hires_map="pyramid", base_level=2),
    # PHercParis4 (Scroll 1): fit frame = level 2 of the 2.400 um volume 20260411134726 (9.6 um), as PHerc1667
    "paris4": dict(base=(VOLP4, 2), hi0=(VOLP4, 0), hi1=(VOLP4, 1), vox_um=9.6,
                   cache=REPO / "data" / "scratch" / "hitl_session" / "cache_paris4", hires_map="pyramid", base_level=2),
}

# ----------------------------------------------------------------------------------------------------------------
# CT access: byte ranges of uncompressed zarr v2 chunks (C order, z-major, so a z sub-range of a chunk is contiguous)
# ----------------------------------------------------------------------------------------------------------------
_FS = None
_FS_LOCK = threading.Lock()


def fs():
    global _FS
    with _FS_LOCK:
        if _FS is None:
            import s3fs
            _FS = s3fs.S3FileSystem(anon=True)
    return _FS


def read_block(vol: str, level: int, lo, hi) -> np.ndarray:
    """uint8 block [lo, hi) (zyx, voxel indices at that level); outside the volume / missing chunks = 0."""
    shape = SHAPES[(vol, level)]
    lo = [int(v) for v in lo]
    hi = [int(v) for v in hi]
    out = np.zeros([h - l for l, h in zip(lo, hi)], np.uint8)
    paths, starts, ends, places = [], [], [], []
    for cz in range(max(lo[0], 0) // CH, (min(hi[0], shape[0]) - 1) // CH + 1):
        z0, z1 = max(lo[0], cz * CH, 0), min(hi[0], cz * CH + CH, shape[0])
        if z1 <= z0:
            continue
        for cy in range(max(lo[1], 0) // CH, (min(hi[1], shape[1]) - 1) // CH + 1):
            for cx in range(max(lo[2], 0) // CH, (min(hi[2], shape[2]) - 1) // CH + 1):
                paths.append(f"{vol}/{level}/{cz}/{cy}/{cx}")
                starts.append((z0 - cz * CH) * CH * CH)
                ends.append((z1 - cz * CH) * CH * CH)
                places.append((cz, cy, cx, z0, z1))
    if not paths:
        return out
    res = fs().cat_ranges(paths, starts, ends, on_error="return")
    for (cz, cy, cx, z0, z1), buf in zip(places, res):
        if isinstance(buf, BaseException) or len(buf) != (z1 - z0) * CH * CH:
            continue  # missing chunk = fill value 0
        a = np.frombuffer(buf, np.uint8).reshape(z1 - z0, CH, CH)
        y0, y1 = max(lo[1], cy * CH), min(hi[1], cy * CH + CH)
        x0, x1 = max(lo[2], cx * CH), min(hi[2], cx * CH + CH)
        out[z0 - lo[0]:z1 - lo[0], y0 - lo[1]:y1 - lo[1], x0 - lo[2]:x1 - lo[2]] = \
            a[:, y0 - cy * CH:y1 - cy * CH, x0 - cx * CH:x1 - cx * CH]
    return out


def sample(vol: str, level: int, pts: np.ndarray, order: int = 1) -> np.ndarray:
    """Trilinear samples at zyx points (level voxel coords, any leading shape); fetches only their bbox."""
    flat = pts.reshape(-1, 3)
    lo = np.floor(flat.min(0)).astype(int) - 1
    hi = np.ceil(flat.max(0)).astype(int) + 2
    blk = read_block(vol, level, lo, hi).astype(np.float32)
    v = ndi.map_coordinates(blk, (flat - lo).T, order=order, mode="nearest")
    return v.reshape(pts.shape[:-1])


_AFF = None


def affine():
    """p9_xyz = A p24_xyz + t (published transform)."""
    global _AFF
    if _AFF is None:
        M = np.array(json.loads(TF_LOCAL.read_text())["transformation_matrix"])
        _AFF = (M[:, :3], M[:, 3], np.linalg.inv(M[:, :3]))
    return _AFF


def to24(p9_zyx: np.ndarray, level: int = 0, scroll: str = "p0139") -> np.ndarray:
    S = SCROLLS[scroll]
    if S["hires_map"] == "pyramid":  # base = level b of the same 2.399 volume: L0 = f p + (f - 1) / 2 (voxel centres)
        f0 = 2 ** S["base_level"]
        p = f0 * np.asarray(p9_zyx, float) + (f0 - 1) / 2.0
    else:
        A, t, Ai = affine()
        p = (p9_zyx[..., ::-1] - t) @ Ai.T  # xyz at 2.399 L0
        p = p[..., ::-1]
    if level:
        f = 2 ** level
        p = (p - (f - 1) / 2.0) / f  # level-l voxel centres sit at f*i + (f-1)/2 in L0
    return p


# ----------------------------------------------------------------------------------------------------------------
# Item geometry and rendering
# ----------------------------------------------------------------------------------------------------------------
def item_key(A, B) -> str:
    s = "%.1f,%.1f,%.1f|%.1f,%.1f,%.1f" % (*A, *B)
    return hashlib.sha1(s.encode()).hexdigest()[:14]


def frame(A, B):
    """Rotated frame: centre c (zyx), unit u (A->B in yx), unit v (perpendicular in yx), crop widths."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d = B[1:] - A[1:]
    L = float(np.hypot(*d))
    u = d / max(L, 1e-6)
    v = np.array([-u[1], u[0]])  # (y, x): +90 deg in the image plane
    c = (A + B) / 2.0
    W = float(np.clip(3.0 * L + 24.0, 112, 320))  # ~3x the step: context to follow sheets into contacts
    W = math.ceil(W / 8) * 8
    H = math.ceil(0.7 * W / 8) * 8
    return c, u, v, L, W, H


def grid_axial(c, u, v, us, vs):
    """zyx points of the axial plane at c[0]; rows = v (top = +v), cols = u."""
    U, V = np.meshgrid(us, vs[::-1])
    y = c[1] + U * u[0] + V * v[0]
    x = c[2] + U * u[1] + V * v[1]
    return np.stack([np.full_like(y, c[0]), y, x], -1)


def grid_rayz(c, u, us, ws):
    """zyx points of the (u, z) plane through the line; rows = z (top = +z), cols = u."""
    U, Wz = np.meshgrid(us, ws[::-1])
    return np.stack([c[0] + Wz, c[1] + U * u[0], c[2] + U * u[1]], -1)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def best_shift(ref: np.ndarray, mov: np.ndarray, m: int):
    """Integer (dy, dx) in [-m, m]^2 maximising NCC of mov[m+dy : , m+dx : ] (same size as ref) against ref."""
    h, w = ref.shape
    best = (-2.0, 0, 0)
    for step, cand in ((2, [(dy, dx) for dy in range(-m, m + 1, 2) for dx in range(-m, m + 1, 2)]), (1, None)):
        if cand is None:
            _, by, bx = best
            cand = [(dy, dx) for dy in range(by - 1, by + 2) for dx in range(bx - 1, bx + 2)
                    if abs(dy) <= m and abs(dx) <= m]
        for dy, dx in cand:
            s = ncc(ref, mov[m + dy:m + dy + h, m + dx:m + dx + w])
            if s > best[0]:
                best = (s, dy, dx)
    return best


def to_png(img: np.ndarray, lo: float, hi: float) -> bytes:
    a = np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "L").save(b, "PNG", optimize=False, compress_level=6)
    return b.getvalue()


def window(img: np.ndarray):
    v = img[img > 0]
    if v.size < 100:
        return 0.0, 255.0
    lo, hi = np.percentile(v, [1.0, 99.7])
    return float(lo), float(hi)


def render_item(A, B, hires: bool = True, scroll: str = "p0139") -> tuple[dict, dict]:
    """Returns (meta, {name: png bytes}). meta has the frame, pixel spacings, profiles and registration shifts.
    A, B are zyx in the scroll's fit frame (SCROLLS[scroll]["base"])."""
    S = SCROLLS[scroll]
    t0 = time.time()
    c, u, v, L, W, H = frame(A, B)
    s9 = 0.5  # 9.362 render spacing (vox)
    us9 = np.arange(-W / 2, W / 2, s9) + s9 / 2
    vs9 = np.arange(-H / 2, H / 2, s9) + s9 / 2
    # one 9.362 fetch for axial (3 z planes for the profile) + ray-z plane
    pts_ax = np.stack([grid_axial(c + np.array([dz, 0, 0]), u, v, us9, vs9) for dz in (-1, 0, 1)])
    pts_rz = grid_rayz(c, u, us9, vs9)
    allp = np.concatenate([pts_ax.reshape(-1, 3), pts_rz.reshape(-1, 3)])
    lo = np.floor(allp.min(0)).astype(int) - 1
    hi = np.ceil(allp.max(0)).astype(int) + 2
    blk = read_block(*S["base"], lo, hi).astype(np.float32)
    t_fetch9 = time.time() - t0
    ax3 = ndi.map_coordinates(blk, (pts_ax.reshape(-1, 3) - lo).T, order=1).reshape(pts_ax.shape[:-1])
    ax9 = ax3[1]
    rz9 = ndi.map_coordinates(blk, (pts_rz.reshape(-1, 3) - lo).T, order=1).reshape(pts_rz.shape[:-1])
    # profile: 0.25-vox steps along u, mean over v in [-1.5, 1.5] and z-1..z+1
    up = np.arange(-W / 2, W / 2 + 1e-6, 0.25)
    vp = np.linspace(-1.5, 1.5, 7)
    pp = np.stack([grid_axial(c + np.array([dz, 0, 0]), u, v, up, vp) for dz in (-1, 0, 1)])
    prof9 = ndi.map_coordinates(blk, (pp.reshape(-1, 3) - lo).T, order=1).reshape(pp.shape[:-1]).mean((0, 1))
    wlo, whi = window(ax9)
    pngs = {"ax9": to_png(ax9, wlo, whi), "rz9": to_png(rz9, wlo, whi)}
    meta = dict(version=RENDER_VERSION, A=list(map(float, A)), B=list(map(float, B)), c=c.tolist(), u=u.tolist(),
                v=v.tolist(), L=L, W=W, H=H, s9=s9, prof_u0=float(up[0]), prof_du=0.25,
                prof9=[round(float(x), 2) for x in prof9], win9=[wlo, whi], hires=False)
    if scroll != "p0139":  # p0139 metas stay byte-identical to the existing band-1/2 caches
        meta.update(scroll=scroll, vox_um=S["vox_um"])
    t1 = time.time()
    if hires:
        try:
            uh = np.arange(-W / 2, W / 2, 0.25) + 0.125
            vh = np.arange(-H / 2, H / 2, 0.25) + 0.125
            ph = grid_axial(c, u, v, uh, vh)
            ref = ndi.map_coordinates(blk, (ph.reshape(-1, 3) - lo).T, order=1).reshape(ph.shape[:-1])
            meta.update(_render_hires(c, u, v, W, H, ref, rz9, up, pngs, scroll))
        except Exception as e:  # noqa: BLE001
            meta["hires_error"] = repr(e)
    meta["timing"] = dict(fetch9=round(t_fetch9, 3), total9=round(t1 - t0, 3), hires=round(time.time() - t1, 3))
    return meta, pngs


def _render_hires(c, u, v, W, H, ref, rz9, up, pngs, scroll: str = "p0139") -> dict:
    S = SCROLLS[scroll]
    m_vox = 3.0  # registration search radius (9.362 vox)
    # axial at 2.399 L0, 0.25-vox spacing, with a margin for the shift search
    s = 0.25
    mpx = int(m_vox / s)
    us = np.arange(-W / 2 - m_vox, W / 2 + m_vox, s) + s / 2
    vs = np.arange(-H / 2 - m_vox, H / 2 + m_vox, s) + s / 2
    p24 = to24(grid_axial(c, u, v, us, vs), 0, scroll)
    lo = np.floor(p24.reshape(-1, 3).min(0)).astype(int) - 1
    hi = np.ceil(p24.reshape(-1, 3).max(0)).astype(int) + 2
    blk = read_block(*S["hi0"], lo, hi).astype(np.float32)
    big = ndi.map_coordinates(blk, (p24.reshape(-1, 3) - lo).T, order=1).reshape(p24.shape[:-1])
    if (big > 0).mean() < 0.2:
        raise RuntimeError("2.4 crop mostly empty")
    # register: blur 2.4 to ~9.362 resolution, compare against the 9.362 render upsampled to 0.25 spacing
    blur = ndi.gaussian_filter(big, 1.6)
    sc, dy, dx = best_shift(ref, blur, mpx)
    ax24 = big[mpx + dy:mpx + dy + ref.shape[0], mpx + dx:mpx + dx + ref.shape[1]]
    # (dy, dx) in rows/cols: the 9.362 content at row r sits at row r + dy of the 2.4 render -> shift (v, u) in vox
    shift_ax = dict(ncc=round(sc, 3), dv=-dy * s, du=dx * s)
    # profile along u from the registered 2.4 axial, +-1.5 vox in v
    r0 = ax24.shape[0] // 2
    band = ax24[r0 - 6:r0 + 6].mean(0)
    uu = np.arange(-W / 2, W / 2, s) + s / 2
    prof24 = np.interp(up, uu, band)
    # ray-z plane from 2.4 L1 (4.8 um), 0.5-vox spacing
    s1 = 0.5
    m1 = int(m_vox / s1)
    us1 = np.arange(-W / 2 - m_vox, W / 2 + m_vox, s1) + s1 / 2
    ws1 = np.arange(-H / 2 - m_vox, H / 2 + m_vox, s1) + s1 / 2
    q = to24(grid_rayz(c, u, us1, ws1), 1, scroll)
    lo = np.floor(q.reshape(-1, 3).min(0)).astype(int) - 1
    hi = np.ceil(q.reshape(-1, 3).max(0)).astype(int) + 2
    blk1 = read_block(*S["hi1"], lo, hi).astype(np.float32)
    bigz = ndi.map_coordinates(blk1, (q.reshape(-1, 3) - lo).T, order=1).reshape(q.shape[:-1])
    scz, dzr, dxz = best_shift(rz9, ndi.gaussian_filter(bigz, 0.8), m1)
    rz24 = bigz[m1 + dzr:m1 + dzr + rz9.shape[0], m1 + dxz:m1 + dxz + rz9.shape[1]]
    shift_rz = dict(ncc=round(scz, 3), dw=-dzr * s1, du=dxz * s1)
    wlo, whi = window(ax24)
    pngs["ax24"] = to_png(ax24, wlo, whi)
    pngs["rz24"] = to_png(rz24, wlo, whi)
    return dict(hires=True, s24=s, s24z=s1, prof24=[round(float(x), 2) for x in prof24], win24=[wlo, whi],
                shift_ax=shift_ax, shift_rz=shift_rz)


def cache_dir(key: str, scroll: str = "p0139") -> Path:
    return SCROLLS[scroll]["cache"] / key


def get_item(A, B, hires: bool = True, scroll: str = "p0139") -> tuple[str, dict]:
    """Cached render (renders and stores on a miss)."""
    key = item_key(A, B)
    d = cache_dir(key, scroll)
    mf = d / "meta.json"
    if mf.exists():
        meta = json.loads(mf.read_text())
        if meta.get("version") == RENDER_VERSION and (meta.get("hires") or not hires or "hires_error" in meta):
            return key, meta
    meta, pngs = render_item(A, B, hires, scroll)
    d.mkdir(parents=True, exist_ok=True)
    for k, b in pngs.items():
        (d / f"{k}.png").write_bytes(b)
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta))
    tmp.replace(mf)
    return key, meta


# ----------------------------------------------------------------------------------------------------------------
# Queue loading (tolerant of field names; never exposes labels to the browser)
# ----------------------------------------------------------------------------------------------------------------
def _pt(x, order: str):
    if isinstance(x, dict):
        return [float(x["z"]), float(x["y"]), float(x["x"])]
    p = [float(t) for t in x]
    return p[::-1] if order == "xyz" else p


def queue_scroll(path: Path) -> str:
    """The queue file's "scroll" field (p0139 when absent: the band-1/2 and stand-in queues)."""
    q = json.loads(Path(path).read_text())
    return str(q.get("scroll", "p0139")) if isinstance(q, dict) else "p0139"


def resolve_scroll(queue: Path, scroll: str) -> str:
    s = queue_scroll(queue) if scroll == "auto" else scroll
    assert s in SCROLLS, f"unknown scroll {s!r}"
    return s


def load_queue(path: Path, order: str = "auto", top: int = 0, controls: bool = True) -> list[dict]:
    """Items from a queue json: a list, {"items": [...]}, or the HITL export {"queue": [...], "controls": [...]}
    (p_inner_zyx / p_outer_zyx, step_id, rank). top > 0 keeps the first `top` queue items (export order = rank);
    controls (answerable steps mixed in blind) are added unless controls=False."""
    q = json.loads(Path(path).read_text())
    if isinstance(q, dict):
        main = q.get("queue", q.get("items", q.get("steps")))
        ctl = q.get("controls", []) if controls else []
        order = q.get("coords") or order
    else:
        main, ctl = q, []
    if top > 0:
        main = main[:top]
    out = []
    for i, (it, is_ctl) in enumerate([(x, False) for x in main] + [(x, True) for x in ctl]):
        if "p_inner_zyx" in it:
            A, B = [float(t) for t in it["p_inner_zyx"]], [float(t) for t in it["p_outer_zyx"]]
        else:
            a = next(it[k] for k in ("A", "a", "p_a", "pa", "point_a", "endpoint_a") if k in it)
            b = next(it[k] for k in ("B", "b", "p_b", "pb", "point_b", "endpoint_b") if k in it)
            o = it.get("coords", order)
            if o == "auto" and not isinstance(a, dict):
                # radial steps share z: if the first components differ and the last agree, the points are xyz
                o = "xyz" if (abs(a[0] - b[0]) > 0.5 and abs(a[2] - b[2]) <= 0.5) else "zyx"
            A, B = _pt(a, o), _pt(b, o)
        iid = str(it.get("step_id", it.get("id", it.get("item_id", f"{it.get('chain_id', 'q')}_{i}"))))
        out.append(dict(id=iid, A=A, B=B, control=is_ctl, src=it))
    ids = [x["id"] for x in out]
    if len(set(ids)) != len(ids):
        for i, x in enumerate(out):
            x["id"] = f"{x['id']}#{i}"
    return out


def session_order(items: list[dict], seed: int) -> list[int]:
    idx = sorted(range(len(items)), key=lambda i: items[i]["id"])
    random.Random(seed).shuffle(idx)
    return idx


# ----------------------------------------------------------------------------------------------------------------
# Stand-in queue (gm-raised steps of band_m7_ray_gmxc) and tutorial picking
# ----------------------------------------------------------------------------------------------------------------
def _oracle_rays(path: Path) -> dict:
    return {(int(k.split("|")[0]), float(k.split("|")[1])): v for k, v in json.loads(path.read_text()).items()}


def _match(orc, r, tol=4.0):
    if not orc:
        return None
    rr = np.array([x[0] for x in orc])
    d = np.abs(rr - r)
    i = int(np.argmin(d))
    s = np.partition(d, 1)[1] if len(d) > 1 else 1e9
    return orc[i][1] if d[i] <= tol and s - d[i] >= 1 else None


def cmd_standin(a) -> None:
    aw = REPO / "data" / "scratch" / "auto_windings"
    g = json.loads((aw / a.src / "chains.json").read_text())
    base = json.loads((aw / a.base / "chains.json").read_text())
    rays = _oracle_rays(aw / a.oracle / "rays.json")
    items = []
    for ci, (cg, cb) in enumerate(zip(g, base)):
        assert cg["pts"] == cb["pts"]
        orc = rays.get((int(cg["z"]), float(cg["theta"])), [])
        for k in range(len(cg["pts"]) - 1):
            dg = cg["wind"][k + 1] - cg["wind"][k]
            db = cb["wind"][k + 1] - cb["wind"][k]
            if not (dg == 2 and db == 1):
                continue
            ta, tb = _match(orc, cg["r"][k]), _match(orc, cg["r"][k + 1])
            items.append(dict(id=f"b1c{ci}s{k}", chain_id=ci, step=k, z=cg["z"], theta=cg["theta"],
                              A=cg["pts"][k], B=cg["pts"][k + 1], chain_count=int(dg), base_count=int(db),
                              gm_prob=cg["gm_p"][k], oracle_count=(tb - ta) if ta is not None and tb is not None
                              else None, ct_corrected_count=None))
    random.Random(0).shuffle(items)
    items = items[:a.n]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(coords="zyx", frame="9.362um L0 (20250728140407)", source=f"{a.src} gm-raised "
                                   f"steps (chain 2 vs base 1), random {a.n}, seed 0; STAND-IN for check_queue_band1",
                                   items=items), indent=0))
    oc = [x["oracle_count"] for x in items if x["oracle_count"] is not None]
    print(f"{len(items)} items -> {out}; oracle matched {len(oc)}: " +
          ", ".join(f"d={d}: {oc.count(d)}" for d in sorted(set(oc))))


def cmd_tutorial_pick(a) -> None:
    """Candidates from z 4500-6000: oracle steps on one ray whose both ends match an auto-chain point (m7, cut 1.0)
    and whose winding difference the auto chain reproduces; both windings outside the blacklisted / gap windings."""
    aw = REPO / "data" / "scratch" / "auto_windings"
    rays = _oracle_rays(aw / "band_oracle_z4500" / "rays.json")
    auto = json.loads((aw / "band_m7_z4500_ray_c" / "chains.json").read_text())
    # windings whose human labels are geometrically consistent (label-audit subsets, notes/2026-10-21-fit-figures.md)
    # and not affected by the CT relabel (w041=w042, w045=w046, missing w050): w029-w032, w052-w054, w057-w059
    good = set(range(29, 33)) | {52, 53, 54, 57, 58, 59}
    cands = []
    for c in auto:
        if not (4600 <= c["z"] <= 5900):
            continue
        th = float(c["theta"])
        if 110 <= th <= 270:  # human seams + #2167 sector
            continue
        orc = rays.get((int(c["z"]), th), [])
        tw = [_match(orc, r) for r in c["r"]]
        for k in range(len(c["r"]) - 1):
            for k2 in range(k + 1, min(k + 4, len(c["r"]))):
                if tw[k] is None or tw[k2] is None:
                    continue
                dw = tw[k2] - tw[k]
                if dw != c["wind"][k2] - c["wind"][k] or not 1 <= dw <= 3:
                    continue
                ws = list(range(tw[k], tw[k2] + 1))
                if any(w not in good for w in ws):
                    continue
                # all intermediate windings present in the oracle on this ray (so the answer is fully determined)
                rmid = [x[0] for x in orc if tw[k] < x[1] < tw[k2]]
                if len(rmid) != dw - 1:
                    continue
                ra, rb = c["r"][k], c["r"][k2]
                gaps = np.diff([ra] + sorted(rmid) + [rb])
                cands.append(dict(z=c["z"], theta=th, A=c["pts"][k], B=c["pts"][k2], ra=ra, rb=rb, n_between=dw - 1,
                                  w=[tw[k], tw[k2]], mid_u=[float(r - (ra + rb) / 2) for r in sorted(rmid)],
                                  min_gap=float(gaps.min()), L=float(rb - ra)))
    rng = random.Random(1)
    rng.shuffle(cands)
    sel = []
    want = [("n0_wide", lambda x: x["n_between"] == 0 and 18 <= x["L"] <= 40),
            ("n0_close", lambda x: x["n_between"] == 0 and 7 <= x["L"] <= 12),
            ("n1", lambda x: x["n_between"] == 1 and x["min_gap"] >= 12),
            ("n1_touch", lambda x: x["n_between"] == 1 and x["min_gap"] <= 6),
            ("n2", lambda x: x["n_between"] == 2 and x["min_gap"] >= 9),
            ("n2_touch", lambda x: x["n_between"] == 2 and x["min_gap"] <= 6)]
    for name, f in want:
        got = [x for x in cands if f(x)][:a.per]
        for j, x in enumerate(got):
            x["cand"] = f"{name}_{j}"
        sel += got
    out = HITL / "tutorial_candidates.json"
    out.write_text(json.dumps(sel, indent=1))
    print(f"{len(cands)} candidates, {len(sel)} selected -> {out}")
    # contact sheet: axial + ray-z + profile per candidate with oracle sheet positions
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda x: get_item(x["A"], x["B"]), sel))
    n = len(sel)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows * 3, cols, figsize=(6 * cols, 3.1 * rows * 3),
                            gridspec_kw=dict(height_ratios=[1, 1, 0.5] * rows))
    for i, (x, (key, meta)) in enumerate(zip(sel, res)):
        r, cc = (i // cols) * 3, i % cols
        W, H, L = meta["W"], meta["H"], meta["L"]
        ext = [-W / 2, W / 2, -H / 2, H / 2]
        for j, nm in enumerate(("ax24" if meta.get("hires") else "ax9", "rz9")):
            ax = axs[r + j, cc]
            ax.imshow(np.array(Image.open(cache_dir(key) / f"{nm}.png")), cmap="gray", extent=ext)
            ax.plot([-L / 2, L / 2], [0, 0], "o", mfc="none", mec="c", ms=7)
            for mu in x["mid_u"]:
                ax.plot([mu], [H / 2 - 3], "v", color="r", ms=5)
            ax.set_title(f"{x['cand']} z{x['z']} th{x['theta']:.0f} w{x['w']} {nm}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axs[r + 2, cc]
        pr = np.array(meta["prof9"])
        uu = meta["prof_u0"] + np.arange(len(pr)) * meta["prof_du"]
        ax.plot(uu, pr, "k", lw=0.8)
        if meta.get("hires"):
            ax.plot(uu, meta["prof24"], "b", lw=0.6, alpha=0.6)
        for xx in (-L / 2, L / 2):
            ax.axvline(xx, color="c")
        for mu in x["mid_u"]:
            ax.axvline(mu, color="r", ls="--")
        ax.set_xlim(-W / 2, W / 2)
    for k in range(n, rows * cols):
        for j in range(3):
            axs[(k // cols) * 3 + j, k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(HITL / "tutorial_candidates.png", dpi=70)
    print("contact sheet ->", HITL / "tutorial_candidates.png")


# Tutorial: candidates picked by eye from tutorial_candidates.json (tutorial-pick, seed 1, --per 4). Each has its
# answer from the human meshes on label-consistent windings, the m7 auto chain agrees, and the CT agrees by eye.
# (n2_touch_0, a two-between case, was dropped: its profile shows a third peak that could be fibre texture.)
TUTORIAL_PICKS = [
    ("n0_wide_2", "Easy case: none",
     "A and B sit on two neighbouring sheets with a wide dark gap between them. Nothing crosses the corridor, "
     "and the brightness curve is flat between the two peaks."),
    ("n1_0", "One sheet between",
     "One sheet crosses the corridor about half-way between A and B (one clear bright peak in the curve). "
     "You can follow it from the top edge to the bottom edge of the slice."),
    ("n0_close_2", "Close sheets, nothing between",
     "A and B are close (about 90 micrometres) but they are still two separate neighbouring sheets: a thin dark line "
     "runs between them in both views, and no third band squeezes in. Close does not mean there is a sheet between."),
    ("n1_touch_3", "Touching sheets",
     "There is one sheet between, running close to A's sheet. Below the line the two converge and touch, and from "
     "there down they look like one thick band. Follow them from the touching point upwards: near the line and above "
     "it a dark gap separates them, so count the second one. This is the typical hard case: two touching sheets that "
     "look like one band in places. Always look along the sheets for a spot where they separate."),
    ("n1_2", "One thick sheet (fibres at 2.4 um)",
     "The thick bright band in the middle is ONE sheet. Press h: at 2.4 um it shows several thin stripes with dark "
     "lines between them. Those are fibre layers inside one sheet, not separate sheets, so it counts once. The 9.4 um "
     "view and the continuity of the band are more reliable than the 2.4 um stripes."),
]


def cmd_tutorial_build(a) -> None:
    cands = {x["cand"]: x for x in json.loads((HITL / "tutorial_candidates.json").read_text())}
    out = []
    for name, title, explain in TUTORIAL_PICKS:
        x = cands[name]
        key, meta = get_item(x["A"], x["B"])
        pr = np.array(meta["prof9"])
        uu = meta["prof_u0"] + np.arange(len(pr)) * meta["prof_du"]
        mids = []
        for mu in x["mid_u"]:  # mesh crossings are ~1-2 vox off: show the marker on the nearest profile peak (+-5 vox)
            pk = np.where((pr[1:-1] > pr[:-2]) & (pr[1:-1] >= pr[2:]))[0] + 1  # local maxima
            pk = pk[np.abs(uu[pk] - mu) <= 5]
            mids.append(round(float(uu[pk[np.argmin(np.abs(uu[pk] - mu))]]), 2) if len(pk) else mu)
        out.append(dict(cand=name, title=title, explain=explain, A=x["A"], B=x["B"], z=x["z"], theta=x["theta"],
                        windings=x["w"], n_between=x["n_between"], mid_u=mids, mid_u_mesh=x["mid_u"]))
    TUTORIAL_JSON.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} tutorial items -> {TUTORIAL_JSON}")


# ----------------------------------------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------------------------------------
class Session:
    def __init__(self, queue: Path, out: Path, seed: int, prefetch: int, order: str, hires: bool, top: int = 0,
                 controls: bool = True, scroll: str = "auto"):
        self.queue_path = queue
        self.scroll = resolve_scroll(queue, scroll)
        self.items = load_queue(queue, order, top, controls)
        qd = json.loads(Path(queue).read_text())
        qd = qd if isinstance(qd, dict) else {}
        # "session_order": "file" keeps the queue's own order (e.g. ray by ray); default = seeded shuffle
        self.order = list(range(len(self.items))) if qd.get("session_order") == "file" else session_order(self.items, seed)
        # optional per-queue practice set (with feedback) and item-time cap (seconds, summed over all answers)
        tut = qd.get("tutorial")
        self.tut_path = (REPO / tut) if tut else TUTORIAL_JSON
        self.tut_scroll = str(qd.get("tutorial_scroll", "p0139"))
        self.time_cap_s = float(qd["time_cap_s"]) if qd.get("time_cap_s") else None
        self.out = out
        self.tut_out = out.with_suffix(".tutorial.jsonl")
        self.prefetch = prefetch
        self.hires = hires
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(6)
        self.pending: dict[str, object] = {}
        self.meta: dict[int, tuple[str, dict]] = {}
        self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.tutorial = json.loads(self.tut_path.read_text()) if self.tut_path.exists() else []

    def answers(self) -> dict[str, dict]:
        last = {}
        if self.out.exists():
            for line in self.out.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    last[r["item_id"]] = r
        return last

    def used_s(self) -> float:
        """Item time used so far: sum of 'seconds' over every main answer line (revisits included)."""
        if not self.out.exists():
            return 0.0
        return sum(float(json.loads(x).get("seconds") or 0.0) for x in self.out.read_text().splitlines() if x.strip())

    def tutorial_done(self) -> bool:
        if not self.tut_out.exists():
            return False
        return any(json.loads(x).get("event") == "tutorial_done" for x in self.tut_out.read_text().splitlines() if x)

    def _render(self, pos: int):
        it = self.items[self.order[pos]]
        return get_item(it["A"], it["B"], self.hires, self.scroll)

    def item(self, pos: int) -> tuple[str, dict]:
        with self.lock:
            if pos in self.meta:
                res = self.meta[pos]
                fut = None
            else:
                fut = self.pending.get(pos) or self.pool.submit(self._render, pos)
                self.pending[pos] = fut
        if fut is not None:
            res = fut.result()
            with self.lock:
                self.meta[pos] = res
                self.pending.pop(pos, None)
        for p in range(pos + 1, min(pos + 1 + self.prefetch, len(self.order))):
            with self.lock:
                if p not in self.meta and p not in self.pending:
                    self.pending[p] = self.pool.submit(self._render, p)
        return res

    def tut_item(self, i: int) -> tuple[str, dict]:
        t = self.tutorial[i]
        return get_item(t["A"], t["B"], True, self.tut_scroll)  # default tutorial: PHerc0139 z 4500-6000

    def log(self, rec: dict, tutorial: bool = False) -> None:
        with self.lock:
            with open(self.tut_out if tutorial else self.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


def client_meta(key: str, meta: dict, scroll: str = "p0139") -> dict:
    keep = ("W", "H", "L", "s9", "prof_u0", "prof_du", "prof9", "prof24", "hires", "s24", "s24z")
    m = {k: meta[k] for k in keep if k in meta}
    m["key"] = key
    m["vox_um"] = SCROLLS[scroll]["vox_um"]
    m["imgs"] = {k: f"/img/{scroll}/{key}/{k}.png" for k in ("ax9", "rz9", "ax24", "rz24")
                 if (cache_dir(key, scroll) / f"{k}.png").exists()}
    return m


def make_handler(S: Session):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            p = urlparse(self.path)
            parts = [x for x in p.path.split("/") if x]
            try:
                if not parts:
                    return self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
                if parts[0] == "api" and parts[1] == "state":
                    ans = S.answers()
                    ids = [S.items[i]["id"] for i in S.order]
                    first = next((k for k, x in enumerate(ids) if x not in ans), len(ids))
                    return self._json(dict(n=len(ids), ids=ids, answers={k: v["answer"] for k, v in ans.items()},
                                           first=first, tutorial_done=S.tutorial_done(), n_tutorial=len(S.tutorial),
                                           out=str(S.out), time_cap_s=S.time_cap_s, used_s=round(S.used_s(), 2)))
                if parts[0] == "api" and parts[1] == "item":
                    pos = int(parts[2])
                    t = time.time()
                    key, meta = S.item(pos)
                    m = client_meta(key, meta, S.scroll)
                    m.update(pos=pos, id=S.items[S.order[pos]]["id"], server_ms=round(1000 * (time.time() - t)))
                    return self._json(m)
                if parts[0] == "api" and parts[1] == "tutorial":
                    i = int(parts[2])
                    t = S.tutorial[i]
                    key, meta = S.tut_item(i)
                    m = client_meta(key, meta, S.tut_scroll)
                    m.update(idx=i, answer=t["n_between"], explain=t["explain"], mid_u=t.get("mid_u", []),
                             title=t.get("title", ""))
                    return self._json(m)
                if parts[0] == "img":  # /img/<scroll>/<key>/<name>.png (old form /img/<key>/<name> = p0139)
                    sc, key, name = parts[1:4] if len(parts) == 4 else ("p0139", *parts[1:3])
                    if sc in SCROLLS:
                        f = cache_dir(key, sc) / name
                        if f.suffix == ".png" and f.exists() and f.parent.parent == SCROLLS[sc]["cache"]:
                            return self._send(200, f.read_bytes(), "image/png", cache=True)
                return self._send(404, b"not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                return self._json(dict(error=repr(e)), 500)

        def do_POST(self):  # noqa: N802
            p = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if p.path == "/api/answer":
                pos = int(body["pos"])
                it = S.items[S.order[pos]]
                # digit = sheets strictly between; "s" = same sheet; "u" = unsure; "a" = an endpoint is in open air
                ans = str(body["answer"])
                used = S.used_s()
                rec = dict(item_id=it["id"], answer=ans, n_between=int(ans) if ans.isdigit() else None,
                           winding_step=int(ans) + 1 if ans.isdigit() else (0 if ans == "s" else None),
                           is_control=it["control"],
                           seconds=round(float(body.get("seconds", 0)), 2), timestamp=now,
                           hires_opened=bool(body.get("hires_opened", False)), pos=pos,
                           revisit=bool(body.get("revisit", False)), A=it["A"], B=it["B"],
                           queue=str(S.queue_path), session=S.session)
                if S.scroll != "p0139":
                    rec["scroll"] = S.scroll
                if S.time_cap_s:
                    rec["used_s_before"] = round(used, 2)
                    rec["over_cap"] = used >= S.time_cap_s
                S.log(rec)
                return self._json(dict(ok=True, used_s=round(used + rec["seconds"], 2)))
            if p.path == "/api/event":  # session events (start / resume / time up), main log's sidecar
                S.log(dict(body, timestamp=now, session=S.session, event=str(body.get("event", "?"))), tutorial=True)
                return self._json(dict(ok=True))
            if p.path == "/api/tutorial_answer":
                rec = dict(body, timestamp=now, session=S.session)
                S.log(rec, tutorial=True)
                return self._json(dict(ok=True))
            return self._json(dict(error="unknown"), 404)

    return H


def cmd_serve(a) -> None:
    S = Session(Path(a.queue), Path(a.out), a.seed, a.prefetch, a.order, not a.no_hires, a.top, not a.no_controls,
                a.scroll)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ans = S.answers()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(S))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{len(S.items)} items ({len(ans)} answered) from {a.queue} [{S.scroll}, cache "
          f"{SCROLLS[S.scroll]['cache']}]\nlog: {a.out}\nopen {url}  (Ctrl-C to stop)")
    # warm the first few unanswered items in the background
    ids = [S.items[i]["id"] for i in S.order]
    first = next((k for k, x in enumerate(ids) if x not in ans), 0)
    for p in range(first, min(first + a.prefetch, len(ids))):
        S.pending[p] = S.pool.submit(S._render, p)
    for i in range(len(S.tutorial)):
        S.pool.submit(S.tut_item, i)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def cmd_precache(a) -> None:
    scroll = resolve_scroll(Path(a.queue), a.scroll)
    items = load_queue(Path(a.queue), a.order, a.top, not a.no_controls)
    order = session_order(items, a.seed)[:a.n]
    qd = json.loads(Path(a.queue).read_text())
    qd = qd if isinstance(qd, dict) else {}
    tpath = (REPO / qd["tutorial"]) if qd.get("tutorial") else TUTORIAL_JSON
    tsc = str(qd.get("tutorial_scroll", "p0139"))
    if qd.get("session_order") == "file":
        order = list(range(len(items)))[:a.n]
    tut = json.loads(tpath.read_text()) if tpath.exists() else []
    jobs = [(t["A"], t["B"], True, tsc) for t in tut] + [(items[i]["A"], items[i]["B"], True, scroll) for i in order]
    t0 = time.time()
    done = 0
    errs = 0
    times = []

    def one(ab):
        t = time.time()
        k, m = get_item(*ab)
        return k, m, time.time() - t

    with ThreadPoolExecutor(a.workers) as ex:
        for k, m, dt in ex.map(one, jobs):
            done += 1
            times.append(dt)
            errs += "hires_error" in m
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} {time.time() - t0:.0f}s (hires errors {errs})", flush=True)
    keys = {(item_key(*ab[:2]), ab[3]) for ab in jobs}
    size = sum(f.stat().st_size for k, sc in keys for f in cache_dir(k, sc).iterdir())
    root = SCROLLS[scroll]["cache"]
    allsize = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"cached {len(keys)} items: {size / 1e6:.1f} MB for this queue (+tutorial); {root} total "
          f"{allsize / 1e6:.1f} MB; wall {time.time() - t0:.0f}s; hires errors {errs}")


def main() -> None:
    argv = sys.argv[1:]
    modes = ("serve", "precache", "standin", "tutorial-pick", "tutorial-build", "shots")
    mode = argv.pop(0) if argv and argv[0] in modes else "serve"
    ap = argparse.ArgumentParser(prog=f"check_viewer.py {mode}")
    if mode in ("serve", "precache", "shots"):
        ap.add_argument("--queue", default=str(HITL / "check_queue_band1.json"))
        ap.add_argument("--seed", type=int, default=0, help="item order shuffle seed (fixed)")
        ap.add_argument("--order", default="auto", choices=["auto", "zyx", "xyz"], help="point coordinate order")
        ap.add_argument("--top", type=int, default=0, help="only the first N queue items (export rank order); 0 = all")
        ap.add_argument("--no-controls", action="store_true", help="leave out the file's blind control steps")
        ap.add_argument("--scroll", default="auto", choices=["auto", *SCROLLS],
                        help="CT source / frame of the queue points; auto = the queue file's \"scroll\" (else p0139)")
    if mode == "serve":
        ap.add_argument("--out", default=str(HITL / f"answers_{datetime.now().strftime('%Y-%m-%d')}.jsonl"))
        ap.add_argument("--port", type=int, default=8765)
        ap.add_argument("--prefetch", type=int, default=5)
        ap.add_argument("--no-hires", action="store_true", help="skip the 2.4 um renders")
        ap.add_argument("--no-browser", action="store_true")
    if mode == "precache":
        ap.add_argument("--n", type=int, default=300)
        ap.add_argument("--workers", type=int, default=8)
    if mode == "standin":
        ap.add_argument("--src", default="band_m7_ray_gmxc")
        ap.add_argument("--base", default="band_m7_ray_c")
        ap.add_argument("--oracle", default="band_oracle")
        ap.add_argument("--n", type=int, default=400)
        ap.add_argument("--out", default=str(HITL / "check_queue_standin_band1.json"))
    if mode == "tutorial-pick":
        ap.add_argument("--per", type=int, default=3)
    if mode == "shots":
        ap.add_argument("--port", type=int, default=8799)
        ap.add_argument("--screens", default=str(HITL / "screens"))
    a = ap.parse_args(argv)
    if mode == "shots":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from check_viewer_test import run_shots
        return run_shots(a)
    dict(serve=cmd_serve, precache=cmd_precache, standin=cmd_standin, **{"tutorial-pick": cmd_tutorial_pick, "tutorial-build": cmd_tutorial_build})[mode](a)


if __name__ == "__main__":
    main()
