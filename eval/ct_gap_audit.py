"""CT audit of questionable human-label gaps in PHerc0139 (notes/2026-10-21-second-band-plan.md 7b, fit-figures notes).

For rays from the umbilicus (n_z levels x n_angles, human seam sectors avoided), find where each human mesh crosses the
ray (vertices within 3 vox of the ray and 3 z of the level; a mesh crossing twice is dropped for that ray), then sample
along the radial segment between two meshes:
  - CT (L0, 9.362 um, streamed crop): mean of 3 z x 3 tangential parallel lines, step 0.5 vox;
  - m7 surface mask (L0, th0.2, streamed crop): max over the same 9 lines (> 0 = sheet);
  - lasagna grad_mag (local, 4x): integral of u8/1000 dr between the two mesh radii (a turn count up to a scale).
Sheet crossings between the two meshes are counted (count_* functions); control gaps whose labels are consistent
(dk = 1, 2, 3) calibrate the counters. Only human meshes, CT and masks are read; no fit output.

  uv run python eval/ct_gap_audit.py extract --band 1        # -> data/scratch/label_audit/ct_profiles_band1.npz
  uv run python eval/ct_gap_audit.py count --band 1          # -> ct_counts_band1.json (+ printed tables)

--scroll p1667 (confirmation region, L2 frame of 20251217075048, z 6500-8000; files *_bandp1667*): CT = 2.399 level 2,
m7 = our bucket-recipe mask (spiral/predict_band.py --level 2 --threshold 0.2), grad_mag = prep_inputs.py
--scroll PHerc1667, labels = merged human mesh (eval/p1667_labels.py), umbilicus derived from it. Every label comes from
one merged read, so there is no questioned list: all dk = 1 pairs, dk 2/3/5 and 10-turn spans are counted and the
dk = 1 pairs calibrate the counters (an internal consistency audit). The label seam (144 deg) is avoided.

--scroll paris4 (third confirmation region, L2 frame of 20260411134726, z 9000-10500; files *_bandparis4*): m7 = the
organisers' bucket m7-L2-th0.2 (band rows copied to data/paris4/masks), grad_mag = prep_inputs.py --scroll PHercParis4,
labels = the 20260701 group meshes (spiral_eval --scroll paris4), public umbilicus. Same design as p1667 (all dk = 1
pairs, dk 2/3/5, 10-turn spans; dk = 1 pairs calibrate) with the label seam (3-10 deg) avoided. CT is NOT sampled
("ct": None; the CT profile is zeros, so n_ct = 0 everywhere): over the 8174^2 slice the CT crops of whole radial rays
would stream ~100 GB, and the p1667 verdict rested on m7 runs and grad_mag turns (CT peaks over-counted, section 9 of
notes/2026-10-22-confirmation-region.md). Long rays are sampled in pieces of 128 samples ("piece") so each crop stays
small; the values are identical to one crop (same points, same interpolation, the crop margin covers the stencil).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import label_subsets as ls  # noqa: E402
import spiral_eval as se  # noqa: E402

OUT = REPO / "data" / "scratch" / "label_audit"
B = "s3://vesuvius-challenge-open-data/PHerc0139"
CT = f"{B}/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0"
M7 = f"{B}/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr/0"
GM = REPO / "data" / "p0139" / "inputs" / "PHerc0139_grad_mag.ome.zarr" / "2"
STEP, PAD = 0.5, 14.0

# pairs (inner label, outer label) per band. kind: c1/c2/c3 = controls with dk 1/2/3 whose windings are all
# consistent in subsets_band<b>.json; t = questioned.
PAIRS = {
    1: {"c1": [(29, 30), (30, 31), (31, 32), (32, 33), (47, 48), (52, 53), (53, 54), (57, 58), (58, 59)],
        "c2": [(29, 31), (30, 32), (31, 33), (52, 54), (57, 59)],
        "c3": [(29, 32), (30, 33)],
        "t": [(40, 43), (40, 41), (40, 42), (41, 43), (42, 43), (41, 42), (43, 44), (44, 45), (44, 46), (45, 46),
              (46, 47), (45, 47), (44, 47), (39, 40), (38, 39), (37, 38), (33, 34), (34, 35), (35, 36), (34, 36),
              (36, 37), (48, 49), (49, 50), (50, 51), (51, 52),
              # long spans (direct counts, no compounding)
              (40, 44), (43, 47), (40, 47), (39, 48), (44, 48), (47, 52), (37, 43), (33, 37), (33, 40), (49, 51),
              (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (54, 55), (55, 56), (56, 57),
              (33, 51), (34, 51), (33, 52), (29, 52), (40, 52), (33, 43)],
        "c4": [(29, 33)],
        "c7x": [(52, 59)],   # passes w055/w056 (spacing-inconsistent, CT one turn each); long-span calibration
        "c10x": [(23, 33)]},  # passes w024-w028 (same)
    2: {"c1": [(28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56),
               (56, 57), (57, 58), (58, 59)],
        "c2": [(28, 30), (29, 31), (30, 32), (31, 33), (51, 53), (52, 54), (53, 55), (54, 56), (55, 57), (56, 58),
               (57, 59)],
        "c3": [(28, 31), (29, 32), (30, 33), (51, 54), (52, 55), (53, 56), (54, 57), (55, 58), (56, 59)],
        "c4": [(28, 32), (29, 33), (51, 55), (53, 57), (55, 59)],
        "c8": [(51, 59)],
        "c10x": [(23, 33)],
        "t": [(23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (23, 28), (33, 34), (34, 35), (35, 36), (34, 36),
              (36, 37), (33, 37), (37, 38), (38, 39), (39, 40), (40, 41), (40, 42), (41, 42), (41, 43), (42, 43),
              (40, 43), (43, 44), (44, 45), (45, 46), (44, 46), (46, 47), (45, 47), (44, 47), (47, 48), (48, 49),
              (49, 50), (50, 51), (49, 51), (48, 51), (34, 39), (39, 51), (34, 51), (33, 51), (37, 43), (40, 47),
              (44, 51)]},
    # band 3 = z 6000-8250 (backup confirmation region, 2026-10-22); controls from subsets_band3.json (consistent:
    # w023, w026-w030, w033, w037, w038, w040, w043, w044, w047, w052-w059)
    3: {"c1": [(26, 27), (27, 28), (28, 29), (29, 30), (37, 38), (43, 44), (52, 53), (53, 54), (54, 55), (55, 56),
               (56, 57), (57, 58), (58, 59)],
        "c2": [(26, 28), (27, 29), (28, 30), (52, 54), (53, 55), (54, 56), (55, 57), (56, 58), (57, 59)],
        "c3": [(26, 29), (27, 30), (52, 55), (53, 56), (54, 57), (55, 58), (56, 59)],
        "c4": [(26, 30), (52, 56), (55, 59)],
        "c7": [(52, 59)],
        "c10x": [(23, 33)],
        "t": [(23, 24), (24, 25), (25, 26), (30, 31), (31, 32), (32, 33), (30, 33), (33, 34), (34, 35), (35, 36),
              (34, 36), (36, 37), (33, 37), (37, 38), (38, 39), (39, 40), (40, 41), (40, 42), (41, 42), (41, 43),
              (42, 43), (40, 43), (43, 44), (44, 45), (44, 46), (45, 46), (45, 47), (46, 47), (44, 47), (47, 48),
              (48, 49), (49, 50), (50, 51), (49, 51), (51, 52), (47, 52), (43, 47), (40, 47), (37, 43), (33, 40),
              (39, 48), (44, 48), (33, 43), (33, 51), (34, 51), (33, 52), (29, 52), (40, 52)]},
}
SEAMS = [(115.0, 145.0), (220.0, 265.0)]   # human mesh seams (w<=34 ~120-140, w>=36 ~225-260): avoided

# PHerc1667 (--scroll p1667): labels k 11..41 of the merged mesh; seam at 144 deg (atan2(dy, dx) about the centre)
P1667_K = range(11, 42)
PAIRS["p1667"] = {
    "c1": [(k, k + 1) for k in P1667_K if k + 1 <= 41],
    "c2": [(k, k + 2) for k in P1667_K if k + 2 <= 41 and k % 2 == 1],
    "c3": [(k, k + 3) for k in P1667_K if k + 3 <= 41 and k % 3 == 2],
    "c5": [(k, k + 5) for k in (11, 16, 21, 26, 31, 36)],
    "c10": [(11, 21), (21, 31), (31, 41)],
}
P4_K = range(10, 130)
PAIRS["paris4"] = {
    "c1": [(k, k + 1) for k in P4_K if k + 1 <= 129],
    "c2": [(k, k + 2) for k in P4_K if k + 2 <= 129 and k % 2 == 0],
    "c3": [(k, k + 3) for k in P4_K if k + 3 <= 129 and k % 3 == 1],
    "c5": [(k, k + 5) for k in range(10, 125, 5)],
    "c10": [(k, k + 10) for k in range(10, 120, 10)] + [(119, 129)],
}
SCROLL_CFG = {
    "p1667": {"band": (6500.0, 8000.0), "seams": [(129.0, 159.0)],
              "ct": "s3://vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/2",
              "m7": str(REPO / "data" / "p1667" / "masks" / "m7_L2_th0.2.zarr" / "0"),
              "gm": REPO / "data" / "p1667" / "inputs" / "PHerc1667_grad_mag.ome.zarr" / "2"},
    "paris4": {"band": (9000.0, 10500.0), "seams": [(0.0, 16.0), (352.0, 360.0)], "ct": None, "piece": 128,
               "m7": str(REPO / "data" / "paris4" / "masks" / "m7_L2_th0.2.zarr" / "0"),
               "gm": REPO / "data" / "paris4" / "inputs" / "PHercParis4_grad_mag.ome.zarr" / "2"},
}
# fourth test (notes/2026-10-24-fourth-test-prereg.md): PHercParis4 z 7000-8500, same labels / pairs, own m7 / grad_mag rows
PAIRS["paris4z7"] = PAIRS["paris4"]
SCROLL_CFG["paris4z7"] = dict(SCROLL_CFG["paris4"], band=(7000.0, 8500.0),
                              m7=str(REPO / "data" / "paris4_z7000" / "masks" / "m7_L2_th0.2.zarr" / "0"),
                              gm=REPO / "data" / "paris4_z7000" / "inputs" / "PHercParis4_grad_mag.ome.zarr" / "2")
# f7 test: PHercParis4 z 10800-11800, same labels / pairs, own m7 / grad_mag rows
PAIRS["paris4z11"] = PAIRS["paris4"]
SCROLL_CFG["paris4z11"] = dict(SCROLL_CFG["paris4"], band=(10800.0, 11800.0),
                               m7=str(REPO / "data" / "paris4_z10800" / "masks" / "m7_L2_th0.2.zarr" / "0"),
                               gm=REPO / "data" / "paris4_z10800" / "inputs" / "PHercParis4_grad_mag.ome.zarr" / "2")
PIECE = 0   # > 0: sample long rays in pieces of PIECE points (set from SCROLL_CFG; 0 = one crop, unchanged)


def angles(n: int, avoid=SEAMS) -> np.ndarray:
    grid = np.arange(0, 360, 0.25)
    ok = np.ones(len(grid), bool)
    for a, b in avoid:
        ok &= ~((grid >= a) & (grid < b))
    g = grid[ok]
    return g[np.linspace(0, len(g), n, endpoint=False).astype(int)]


def z_levels(band: int, n: int) -> np.ndarray:
    z0, z1 = ls.BANDS[band]
    # mid-chunk z (CT chunks are 128): z0 + 136 + 256 i for band 1 -> 3136 ... 4416
    zs = np.arange(np.ceil(z0 / 128) * 128 + 64, z1 - 20, (z1 - z0) / n)
    return np.round(zs[:n]).astype(int)


def crossings(H: dict, umb: se.Umbilicus, zl: float, angs: np.ndarray, labels: list[int]) -> np.ndarray:
    """R[angle, label]: radius where the mesh crosses the ray (nan: absent or crossing twice)."""
    m = np.abs(H["pts"][:, 0] - zl) <= 3.0
    p, k = H["pts"][m].astype(np.float64), H["k"][m]
    uy, ux = np.interp(zl, umb.z, umb.y), np.interp(zl, umb.z, umb.x)
    dy, dx = p[:, 1] - uy, p[:, 2] - ux
    r = np.hypot(dy, dx)
    th = np.degrees(np.arctan2(dy, dx)) % 360
    R = np.full((len(angs), len(labels)), np.nan)
    for ia, a in enumerate(angs):
        d = np.radians((th - a + 180) % 360 - 180)
        sel = (np.abs(r * np.sin(d)) <= 3.0) & (np.cos(d) > 0)
        for il, lab in enumerate(labels):
            rr = np.sort(r[sel & (k == lab)])
            if len(rr) == 0 or (len(rr) > 1 and np.diff(rr).max() > 5.0):
                continue
            R[ia, il] = np.median(rr)
    return R


def ray_points(zl, uy, ux, a, r0, r1):
    """(9, n, 3) zyx sample points: 3 z offsets x 3 tangential offsets along the ray, r in [r0, r1)."""
    t = np.arange(r0, r1, STEP)
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    pts = []
    for dz in (-2.0, 0.0, 2.0):
        for dt in (-2.0, 0.0, 2.0):
            y = uy + t * s + dt * c
            x = ux + t * c - dt * s
            pts.append(np.stack([np.full_like(t, zl + dz), y, x], -1))
    return t, np.stack(pts)


def sample(arr, pts: np.ndarray, scale: float = 1.0, order: int = 1) -> np.ndarray:
    if arr is None:
        return np.zeros(pts.shape[:-1], np.float32)
    if PIECE and pts.shape[1] > PIECE:
        return np.concatenate([sample(arr, pts[:, i:i + PIECE], scale, order) for i in range(0, pts.shape[1], PIECE)], 1)
    q = pts / scale
    lo = np.floor(q.reshape(-1, 3).min(0)).astype(int) - 2
    hi = np.ceil(q.reshape(-1, 3).max(0)).astype(int) + 3
    lo = np.maximum(lo, 0)
    crop = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
    return ndi.map_coordinates(crop, (q - lo).reshape(-1, 3).T, order=order, mode="nearest").reshape(q.shape[:-1])


def extract(a) -> None:
    import zarr
    so = {"anon": True}
    ct_url, m7_url, gm_path, seams = CT, M7, GM, SEAMS
    if a.band in SCROLL_CFG:
        c = SCROLL_CFG[a.band]
        ct_url, m7_url, gm_path, seams = c["ct"], c["m7"], c["gm"], c["seams"]
        global PIECE
        PIECE = c.get("piece", 0)
    op = lambda u: None if u is None else (zarr.open_array(u, mode="r", storage_options=so) if u.startswith("s3://")
                                           else zarr.open_array(u, mode="r"))
    ct, m7 = op(ct_url), op(m7_url)
    gm = zarr.open_array(str(gm_path), mode="r")
    umb = se.Umbilicus()
    H = ls.load_band(a.band)
    labels = sorted(set(int(x) for x in H["k"]))
    angs = np.arange(*a.angles) if a.angles else angles(a.n_angles, seams)
    zs = z_levels(a.band, a.n_z)
    pairs = sorted({p for v in PAIRS[a.band].values() for p in v})
    li = {l: i for i, l in enumerate(labels)}
    jobs = []
    Rall = {}
    for zl in zs:
        R = crossings(H, umb, zl, angs, labels)
        Rall[int(zl)] = R
        uy, ux = np.interp(zl, umb.z, umb.y), np.interp(zl, umb.z, umb.x)
        for ia, ang in enumerate(angs):
            # radial intervals needed on this ray, merged
            iv = []
            for p, q in pairs:
                rp, rq = R[ia, li[p]], R[ia, li[q]]
                if np.isfinite(rp) and np.isfinite(rq):
                    iv.append((min(rp, rq) - PAD, max(rp, rq) + PAD))
            iv.sort()
            merged = []
            for lo, hi in iv:
                if merged and lo <= merged[-1][1] + 20:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                jobs.append((int(zl), ia, float(ang), uy, ux, lo, hi))
    print(f"{len(zs)} z x {len(angs)} angles, {len(jobs)} ray segments", flush=True)

    def run(job):
        zl, ia, ang, uy, ux, lo, hi = job
        t, P = ray_points(zl, uy, ux, ang, lo, hi)
        c = sample(ct, P).mean(0)
        m = sample(m7, P, order=0).max(0)
        g = sample(gm, P[4:5], scale=4.0)[0]          # centre line only (4x grid)
        return zl, ia, t, c, m, g

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(run, jobs))
    segs = {"z": [], "ia": [], "off": [], "n": []}
    T, C, Mk, G = [], [], [], []
    off = 0
    for zl, ia, t, c, m, g in res:
        segs["z"].append(zl); segs["ia"].append(ia); segs["off"].append(off); segs["n"].append(len(t))
        T.append(t); C.append(c); Mk.append(m); G.append(g)
        off += len(t)
    np.savez_compressed(OUT / f"ct_profiles_band{a.band}{a.tag}.npz", labels=np.array(labels), angles=angs, zs=zs,
                        R=np.stack([Rall[int(z)] for z in zs]), seg_z=np.array(segs["z"]), seg_ia=np.array(segs["ia"]),
                        seg_off=np.array(segs["off"]), seg_n=np.array(segs["n"]), t=np.concatenate(T).astype(np.float32),
                        ct=np.concatenate(C).astype(np.float32), m7=np.concatenate(Mk).astype(np.uint8),
                        gm=np.concatenate(G).astype(np.float32))
    print(f"saved {OUT / f'ct_profiles_band{a.band}{a.tag}.npz'}")


# --------------------------------------------------------------------------------------------------------- counting
def profile(D: dict, iz: int, ia: int, r0: float, r1: float):
    """CT, m7, gm and t on [r0, r1] of the stored segment covering it (None if not covered)."""
    zl = D["zs"][iz]
    for s in np.nonzero((D["seg_z"] == zl) & (D["seg_ia"] == ia))[0]:
        o, n = D["seg_off"][s], D["seg_n"][s]
        t = D["t"][o:o + n]
        if t[0] <= r0 and t[-1] >= r1:
            m = (t >= r0) & (t <= r1)
            return t[m], D["ct"][o:o + n][m], D["m7"][o:o + n][m], D["gm"][o:o + n][m]
    return None


def count_ct(t, c, ra, rb, prom_rel, anchor=6.0, excl=4.0, sigma=1.0):
    """Peaks of the smoothed CT profile strictly between the sheets at ra and rb. The sheet of each mesh = the
    strongest peak within +-anchor vox of its radius (the mesh radius itself if none). Returns (count, peaks, anchors)."""
    cs = ndi.gaussian_filter1d(c.astype(float), sigma / STEP)
    rng = np.percentile(cs, 95) - np.percentile(cs, 5)
    pk, pr = find_peaks(cs, prominence=prom_rel * max(rng, 1e-6), distance=int(3.0 / STEP))
    rp = t[pk]
    anc = []
    for r in (ra, rb):
        near = np.nonzero(np.abs(rp - r) <= anchor)[0]
        anc.append(rp[near[np.argmax(cs[pk][near])]] if len(near) else r)
    lo, hi = anc[0] + excl, anc[1] - excl
    inside = (rp > lo) & (rp < hi)
    return int(inside.sum()), rp[inside], anc


def count_m7(t, m, ra, rb, anchor=4.0):
    """Mask runs (> 0) strictly between the runs that contain / touch the two mesh radii."""
    on = m > 0
    lab, n = ndi.label(on)
    if n == 0:
        return 0
    runs = [(t[lab == i].min(), t[lab == i].max()) for i in range(1, n + 1)]
    cnt = 0
    for lo, hi in runs:
        if lo <= ra + anchor or hi >= rb - anchor:
            continue
        cnt += 1
    return cnt


def gm_turns(t, g, ra, rb):
    m = (t >= ra) & (t <= rb)
    return float(np.trapezoid(g[m] / 1000.0, t[m])) if m.sum() > 2 else np.nan


def count(a) -> None:
    D = dict(np.load(OUT / f"ct_profiles_band{a.band}{a.tag}.npz"))
    labels = list(D["labels"])
    li = {l: i for i, l in enumerate(labels)}
    rows = []
    for kind, pl in PAIRS[a.band].items():
        for p, q in pl:
            for iz in range(len(D["zs"])):
                for ia in range(len(D["angles"])):
                    ra, rb = D["R"][iz, ia, li[p]], D["R"][iz, ia, li[q]]
                    if not (np.isfinite(ra) and np.isfinite(rb)):
                        continue
                    lo, hi = min(ra, rb), max(ra, rb)
                    pf = profile(D, iz, ia, lo - PAD + 1, hi + PAD - 1)
                    if pf is None:
                        continue
                    t, c, m, g = pf
                    n_ct, _, anc = count_ct(t, c, lo, hi, a.prom)
                    rows.append({"kind": kind, "pair": [p, q], "z": int(D["zs"][iz]), "ang": float(D["angles"][ia]),
                                 "ra": round(float(ra), 2), "rb": round(float(rb), 2), "gap": round(float(rb - ra), 2),
                                 "n_ct": n_ct, "n_m7": count_m7(t, m, lo, hi),
                                 "gm": gm_turns(t, g, lo, hi), "anchor_shift": [round(float(anc[0] - lo), 2),
                                                                               round(float(anc[1] - hi), 2)]})
    # calibration: grad_mag turns per labelled turn on dk=1 controls
    c1 = [r for r in rows if r["kind"] == "c1" and np.isfinite(r["gm"])]
    gscale = 1.0 / np.median([r["gm"] for r in c1])
    for r in rows:
        r["gm_turns"] = round(r["gm"] * gscale, 3) if np.isfinite(r["gm"]) else None
        r["gm"] = round(r["gm"], 4) if np.isfinite(r["gm"]) else None
    summ = summarise(rows)
    out = {"band": a.band, "prom_rel": a.prom, "gm_scale_per_turn": round(float(gscale), 4), "summary": summ,
           "rows": rows}
    (OUT / f"ct_counts_band{a.band}{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"gm scale {gscale:.3f} (1 labelled turn on dk=1 controls)")
    print(f"{'pair':9s} {'kind':4s} {'rays':>4s} {'gap':>6s} | CT between: mean, P(0/1/2/3+) | m7 between mean, P(0/1/2+) | gm turns med")
    for key, s in summ.items():
        print(f"{key:9s} {s['kind']:4s} {s['rays']:4d} {s['gap_med']:6.1f} | {s['ct_mean']:.2f} {s['ct_p']} | "
              f"{s['m7_mean']:.2f} {s['m7_p']} | {s['gm_turns_med']}")


def summarise(rows: list[dict]) -> dict:
    out = {}
    keys = []
    for r in rows:
        k = f"w{r['pair'][0]}-w{r['pair'][1]}"
        if k not in keys:
            keys.append(k)
    pooled = {}
    for r in rows:
        if r["kind"].startswith("c"):
            pooled.setdefault(r["kind"], []).append(r)
    groups = [(k, [r for r in rows if f"w{r['pair'][0]}-w{r['pair'][1]}" == k]) for k in keys]
    groups += [(f"ALL_{k}", v) for k, v in pooled.items()]
    for k, rr in groups:
        nct = np.array([r["n_ct"] for r in rr])
        nm = np.array([r["n_m7"] for r in rr])
        gt = np.array([r["gm_turns"] for r in rr if r["gm_turns"] is not None])
        out[k] = {"kind": rr[0]["kind"], "rays": len(rr), "gap_med": float(np.median([r["gap"] for r in rr])),
                  "ct_mean": float(nct.mean()), "ct_p": [round(float((nct == i).mean()), 2) for i in range(3)] +
                  [round(float((nct >= 3).mean()), 2)],
                  "m7_mean": float(nm.mean()), "m7_p": [round(float((nm == i).mean()), 2) for i in range(2)] +
                  [round(float((nm >= 2).mean()), 2)],
                  "gm_turns_med": round(float(np.median(gt)), 2) if len(gt) else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["extract", "count"])
    ap.add_argument("--band", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--n-angles", type=int, default=24)
    ap.add_argument("--n-z", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--angles", type=float, nargs=3, help="a0 a1 step (deg) instead of n-angles avoiding the seams")
    ap.add_argument("--tag", default="", help="suffix of the profile / count files")
    ap.add_argument("--prom", type=float, default=0.15, help="CT peak prominence, fraction of the profile p5-p95 range")
    ap.add_argument("--scroll", choices=["p0139", "p1667", "paris4", "paris4z7", "paris4z11"], default="p0139",
                    help="p1667: band = z 6500-8000 (L2 frame); paris4: z 9000-10500 (L2 frame); paris4z7: z 7000-8500; "
                         "paris4z11: z 10800-11800")
    a = ap.parse_args()
    if a.scroll != "p0139":
        se.set_scroll("paris4" if a.scroll in ("paris4z7", "paris4z11") else a.scroll)
        a.band = a.scroll
        ls.BANDS[a.band] = SCROLL_CFG[a.band]["band"]
    (extract if a.cmd == "extract" else count)(a)


if __name__ == "__main__":
    main()
