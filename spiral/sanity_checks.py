#!/usr/bin/env python3
"""sanity_checks.py - checks of the PHerc0139 9.362-frame inputs before any fit (prep_inputs.py outputs).

1. Umbilicus. On several CT slices (9.362, level 3 overview + level 0 inset) draw the transformed umbilicus and score
   it against the m7 sheets at level 1: from a point, cast 72 rays to the scroll edge and count m7 sheet crossings.
   At the centre of a spiral the counts on opposite rays agree (+-1 winding); off-centre they do not. Scores:
   asym(p) = median_theta |n(theta) - n(theta + pi)| / (n(theta) + n(theta + pi)); asym at the umbilicus, at the
   point mirrored through the slice centre (what a missing flip would give), and the argmin of asym over a +-400
   vox grid (an independent centre estimate) with its distance to the umbilicus. Also: CT-mask centroid offset / R.
2. Normals. m7 level-0 crops (96^3) around sheet voxels at three z: sheet normal from the structure tensor of the
   smoothed mask (sigma 1.5, window 3), compared with the resampled lasagna normal at round(p / 4). Median |cos| on
   sheet voxels, also for controls that a wrong transform would produce: unrotated source vector, nx<->ny swap,
   x-only sign flip, and source sampled without the 180 deg flip of positions.
3. grad_mag. Sheet pitch along the structure-tensor normal (m7 rising edges over +-40 vox) vs 1000 / u8 of the
   resampled grad_mag (windings per voxel in the fitter's decoding at lasagna_scale 4).

    uv run python spiral/sanity_checks.py --inputs data/p0139/inputs --dl data/dl_p0139 \
        --out data/scratch/p0139_spiral
"""
import argparse
import json
import os
import sys

import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_inputs import (ChunkReader, LAS_RUN, SRC_GROUP, SRC_SCALE, TGT_GROUP, TGT_SCALE,  # noqa: E402
                         load_affine)

B = "s3://vesuvius-challenge-open-data/PHerc0139"
CT = f"{B}/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
M7 = f"{B}/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
SO = dict(anon=True)


def arr(url, g):
    return zarr.open_array(f"{url}/{g}", mode="r", storage_options=SO)


def umb_at(umb, z):
    pts = sorted(umb["control_points"], key=lambda p: p["z"])
    zs = np.array([p["z"] for p in pts]); xs = np.array([p["x"] for p in pts]); ys = np.array([p["y"] for p in pts])
    return float(np.interp(z, zs, xs)), float(np.interp(z, zs, ys))


def ray_counts(mask, p_xy, n_rays=72, step=0.5, rmax=None):
    """Sheet crossings along rays from p (pixel coords of mask) to the mask's bounding edge."""
    h, w = mask.shape
    rmax = rmax or float(np.hypot(h, w))
    t = np.arange(0, rmax, step)
    out = []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        x = p_xy[0] + t * np.cos(th); y = p_xy[1] + t * np.sin(th)
        ok = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        v = mask[np.rint(y[ok]).astype(int), np.rint(x[ok]).astype(int)]
        out.append(int(((v[1:] > 0) & (v[:-1] == 0)).sum()))
    return np.array(out)


def first_hit(mask, p_xy, n_rays=72, step=0.5):
    """Median and min distance (mask pixels) from p to the first sheet pixel over n_rays rays (free radius)."""
    h, w = mask.shape
    t = np.arange(0, 600, step)
    d = []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        x = p_xy[0] + t * np.cos(th); y = p_xy[1] + t * np.sin(th)
        ok = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        v = mask[np.rint(y[ok]).astype(int), np.rint(x[ok]).astype(int)]
        hit = np.nonzero(v)[0]
        d.append(t[hit[0]] if len(hit) else 600.0)
    return float(np.median(d)), float(np.min(d))


def asym(mask, p_xy):
    n = ray_counts(mask, p_xy)
    h = len(n) // 2
    a, b = n[:h], n[h:]
    s = a + b
    ok = s > 0
    return float(np.median(np.abs(a - b)[ok] / s[ok])) if ok.any() else 1.0, n


def check_umbilicus(a, umb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ct3, ct0, m1 = arr(CT, 3), arr(CT, 0), arr(M7, 1)
    res = {}
    zs = a.umb_z
    fig, axes = plt.subplots(2, len(zs), figsize=(4.2 * len(zs), 8.6))
    for k, z in enumerate(zs):
        ux, uy = umb_at(umb, z)
        sl3 = np.asarray(ct3[z // 8])
        # CT mask centroid (level 3)
        thr = np.percentile(sl3[sl3 > 0], 20) if (sl3 > 0).any() else 1
        m = ndi.binary_fill_holes(ndi.binary_closing(sl3 > thr, iterations=3))
        lab, nl = ndi.label(m)
        if nl:
            m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        cy, cx = [float(c) * 8 for c in ndi.center_of_mass(m)]
        R = float(np.sqrt(m.sum() / np.pi)) * 8
        # m7 level 1 slice, ray asymmetry
        s1 = np.asarray(m1[z // 2]) > 0
        a_u, n_u = asym(s1, (ux / 2, uy / 2))
        mx, my = 6621 - ux, 6621 - uy               # mirrored through the slice centre (missing flip)
        a_m, _ = asym(s1, (mx / 2, my / 2))
        best = (9, None)
        for dy in range(-400, 401, 50):
            for dx in range(-400, 401, 50):
                v, _ = asym(s1, ((ux + dx) / 2, (uy + dy) / 2))
                if v < best[0]:
                    best = (v, (dx, dy))
        # refine around the best
        bx, by = best[1]
        for dy in range(by - 40, by + 41, 20):
            for dx in range(bx - 40, bx + 41, 20):
                v, _ = asym(s1, ((ux + dx) / 2, (uy + dy) / 2))
                if v < best[0]:
                    best = (v, (dx, dy))
        fh_u = first_hit(s1, (ux / 2, uy / 2)); fh_m = first_hit(s1, (mx / 2, my / 2))
        fh_a = first_hit(s1, ((ux + best[1][0]) / 2, (uy + best[1][1]) / 2))
        rng_ = np.random.default_rng(z)
        yy_, xx_ = np.nonzero(ndi.binary_erosion(m, iterations=10))
        ii_ = rng_.choice(len(yy_), 200, replace=False)
        fh_rand = [first_hit(s1, (xx_[i] * 4, yy_[i] * 4), n_rays=36)[0] for i in ii_]
        r = {"free_radius_umb_med_min_vox": [2 * fh_u[0], 2 * fh_u[1]],
             "free_radius_mirrored_med_vox": 2 * fh_m[0], "free_radius_asym_argmin_med_vox": 2 * fh_a[0],
             "free_radius_random_in_scroll_p50_p99_vox": [round(2 * float(np.median(fh_rand)), 1),
                                                          round(2 * float(np.percentile(fh_rand, 99)), 1)],
             "umb_xy": [round(ux, 1), round(uy, 1)], "ct_centroid_xy": [round(cx, 1), round(cy, 1)],
             "ct_radius": round(R, 1), "centroid_offset_over_R": round(float(np.hypot(ux - cx, uy - cy) / R), 3),
             "asym_umb": round(a_u, 3), "asym_mirrored": round(a_m, 3), "asym_min": round(best[0], 3),
             "asym_argmin_offset_vox": list(best[1]), "argmin_dist_vox": round(float(np.hypot(*best[1])), 1),
             "crossings_umb_median": float(np.median(n_u))}
        res[z] = r
        print(f"umbilicus z {z}: {json.dumps(r)}", flush=True)
        ax = axes[0, k]
        ax.imshow(sl3, cmap="gray", vmin=np.percentile(sl3, 1), vmax=np.percentile(sl3, 99.7))
        ax.plot(ux / 8, uy / 8, "r+", ms=16, mew=2, label="umbilicus (affine)")
        ax.plot(mx / 8, my / 8, "bx", ms=10, mew=2, label="mirrored (no flip)")
        ax.plot((ux + best[1][0]) / 8, (uy + best[1][1]) / 8, "yo", mfc="none", ms=10, mew=1.5,
                label="min ray asymmetry")
        ax.set_title(f"z {z}  asym umb {a_u:.2f} / mirr {a_m:.2f}", fontsize=9)
        ax.axis("off")
        if k == 0:
            ax.legend(loc="lower left", fontsize=7)
        # level-0 inset 512^2 around the umbilicus
        x0, y0 = int(ux) - 256, int(uy) - 256
        ins = np.asarray(ct0[z, max(0, y0):y0 + 512, max(0, x0):x0 + 512])
        ax = axes[1, k]
        ax.imshow(ins, cmap="gray", vmin=np.percentile(ins, 1), vmax=np.percentile(ins, 99.7))
        ax.plot(ux - max(0, x0), uy - max(0, y0), "r+", ms=20, mew=2)
        ax.set_title(f"L0 inset 512 vox ({x0},{y0})", fontsize=9)
        ax.axis("off")
    fig.suptitle("PHerc0139 20250728140407 (9.362 um): umbilicus from 2.399 via published affine; +x right, +y down")
    fig.tight_layout()
    p = os.path.join(a.out, "umbilicus_overlay.png")
    fig.savefig(p, dpi=90)
    print("wrote", p)
    return res


def structure_normals(crop, sigma=1.5, rho=3.0):
    f = ndi.gaussian_filter(crop.astype(np.float32), sigma)
    gz, gy, gx = np.gradient(f)
    comps = {}
    for n1, g1 in (("x", gx), ("y", gy), ("z", gz)):
        for n2, g2 in (("x", gx), ("y", gy), ("z", gz)):
            if n1 + n2 in comps or n2 + n1 in comps:
                continue
            comps[n1 + n2] = ndi.gaussian_filter(g1 * g2, rho)
    return comps


def tensor_at(comps, idx):
    z, y, x = idx
    J = np.empty((len(z), 3, 3))
    order = "xyz"
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            k = a + b if a + b in comps else b + a
            J[:, i, j] = comps[k][z, y, x]
    w, v = np.linalg.eigh(J)
    n = v[:, :, 2]                                  # largest eigenvalue -> normal (xyz)
    coh = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2] + w[:, 1], 1e-12)
    return n, coh


def check_normals(a, umb):
    L, t = load_affine()
    Li = np.linalg.inv(L)
    tgt = {k: ChunkReader(os.path.join(a.inputs, f"PHerc0139_{k}.ome.zarr", TGT_GROUP)) for k in ("nx", "ny", "grad_mag")}
    src = {k: ChunkReader(os.path.join(a.dl, "lasagna_2p399", f"{LAS_RUN}_{k}.ome.zarr", SRC_GROUP)) for k in ("nx", "ny")}
    m0, m2 = arr(M7, 0), arr(M7, 2)
    rng = np.random.default_rng(0)
    cos = {k: [] for k in ("correct", "unrotated_src", "swap_xy", "flip_x_only", "src_no_position_flip")}
    cos_tilted = {k: [] for k in cos}
    pitch_rows = []
    per_crop = []

    def read_point(reader, zyx):
        c = reader.chunks
        out = np.zeros(len(zyx), np.uint8)
        keys = {}
        for i, (z, y, x) in enumerate(zyx):
            if not (0 <= z < reader.shape[0] and 0 <= y < reader.shape[1] and 0 <= x < reader.shape[2]):
                continue
            keys.setdefault((z // c[0], y // c[1], x // c[2]), []).append(i)
        for key, ii in keys.items():
            ch = reader.chunk(key)
            if ch is None:
                continue
            for i in ii:
                z, y, x = zyx[i]
                out[i] = ch[z - key[0] * c[0], y - key[1] * c[1], x - key[2] * c[2]]
        return out

    def dec(nx, ny):
        n = np.stack([(nx - 128.0) / 127.0, (ny - 128.0) / 127.0], -1)
        nz = np.sqrt(np.clip(1 - (n ** 2).sum(-1), 0, None))
        v = np.concatenate([n, nz[:, None]], -1)
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

    H = 48
    for z in a.normal_z:
        ux, uy = umb_at(umb, z)
        s2 = np.asarray(m2[z // 4]) > 0
        yy, xx = np.nonzero(s2)
        ang = np.arctan2(yy * 4 - uy, xx * 4 - ux)
        rad = np.hypot(yy * 4 - uy, xx * 4 - ux)
        for sector in range(a.crops_per_z):
            lo = -np.pi + 2 * np.pi * sector / a.crops_per_z
            sel = np.nonzero((ang >= lo) & (ang < lo + 2 * np.pi / a.crops_per_z) & (rad > 150))[0]
            if not len(sel):
                continue
            i = rng.choice(sel)
            cy, cx = int(yy[i] * 4), int(xx[i] * 4)
            cy = min(max(cy, H), 6621 - H); cx = min(max(cx, H), 6621 - H)
            crop = np.asarray(m0[z - H:z + H, cy - H:cy + H, cx - H:cx + H])
            if (crop > 0).mean() < 0.02:
                continue
            comps = structure_normals(crop)
            inner = np.zeros_like(crop, bool); inner[12:-12, 12:-12, 12:-12] = True
            pz, py, px = np.nonzero((crop > 0) & inner)
            if len(pz) > 3000:
                k = rng.choice(len(pz), 3000, replace=False); pz, py, px = pz[k], py[k], px[k]
            n_st, coh = tensor_at(comps, (pz, py, px))
            keep = coh > 0.2
            pz, py, px, n_st = pz[keep], py[keep], px[keep], n_st[keep]
            if len(pz) < 50:
                continue
            gz, gy, gx = pz + z - H, py + cy - H, px + cx - H
            p9 = np.stack([gx, gy, gz], -1).astype(np.float64)
            ti = np.rint(p9 / TGT_SCALE).astype(int)[:, ::-1]             # zyx target index
            tnx, tny = read_point(tgt["nx"], ti), read_point(tgt["ny"], ti)
            tg = read_point(tgt["grad_mag"], ti)
            si = np.rint(((p9 - t) @ Li.T) / SRC_SCALE).astype(int)[:, ::-1]
            snx, sny = read_point(src["nx"], si), read_point(src["ny"], si)
            s_noflip = np.rint(p9 / L[2, 2] / SRC_SCALE).astype(int)[:, ::-1]  # positions scaled, not rotated
            fnx, fny = read_point(src["nx"], s_noflip), read_point(src["ny"], s_noflip)
            v = (tnx > 0) | (tny > 0)
            if v.sum() < 50:
                continue
            nc = dec(tnx[v].astype(float), tny[v].astype(float))
            ns = n_st[v]
            ctrl = {"correct": nc,
                    "unrotated_src": dec(snx[v].astype(float), sny[v].astype(float)),
                    "swap_xy": nc[:, [1, 0, 2]],
                    "flip_x_only": nc * np.array([-1, 1, 1]),
                    "src_no_position_flip": dec(fnx[v].astype(float), fny[v].astype(float))}
            tilt = np.abs(ns[:, 2]) > 0.25
            for k_, vv in ctrl.items():
                c = np.abs((vv * ns).sum(-1))
                cos[k_].append(c)
                cos_tilted[k_].append(c[tilt])
            per_crop.append({"z": z, "yx": [cy, cx], "n": int(v.sum()),
                             "med_abs_cos": round(float(np.median(np.abs((nc * ns).sum(-1)))), 3)})
            # pitch along the normal from up to 15 seeds
            mbin = crop > 0
            seeds = rng.choice(len(pz), min(15, len(pz)), replace=False)
            tt = np.arange(-40, 40, 0.5)
            for s in seeds:
                n = n_st[s]
                q = np.array([px[s], py[s], pz[s]], float)[None] + tt[:, None] * n[None]
                ok = np.all((q >= 0) & (q < 2 * H - 1), axis=1)
                if ok.sum() < 100:
                    continue
                qi = np.rint(q[ok]).astype(int)
                line = mbin[qi[:, 2], qi[:, 1], qi[:, 0]]
                edges = int(((line[1:]) & (~line[:-1])).sum())
                span = ok.sum() * 0.5
                if edges >= 3:
                    pitch_rows.append((span / edges, float(tg[s]) if s < len(tg) else 0.0))
            print(f"normals z {z} sector {sector}: {per_crop[-1]}", flush=True)
    summ = {}
    for k_ in cos:
        c = np.concatenate(cos[k_]) if cos[k_] else np.array([np.nan])
        ct = np.concatenate(cos_tilted[k_]) if cos_tilted[k_] else np.array([np.nan])
        summ[k_] = {"median_abs_cos": round(float(np.median(c)), 4), "frac_gt_0.9": round(float((c > 0.9).mean()), 3),
                    "n": int(c.size), "median_abs_cos_tilted(|nz|>0.25)": round(float(np.median(ct)), 4),
                    "n_tilted": int(ct.size)}
    pr = np.array([p for p in pitch_rows if p[1] > 0])
    g = {}
    if len(pr):
        g = {"pitch_m7_median_vox": round(float(np.median(pr[:, 0])), 2),
             "pitch_from_grad_mag_median_vox": round(float(np.median(1000.0 / pr[:, 1])), 2),
             "ratio_median": round(float(np.median(pr[:, 0] * pr[:, 1] / 1000.0)), 3), "n": int(len(pr))}
    return summ, g, per_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="data/p0139/inputs")
    ap.add_argument("--dl", default="data/dl_p0139")
    ap.add_argument("--out", default="data/scratch/p0139_spiral")
    ap.add_argument("--umb-z", type=int, nargs="+", default=[3000, 3500, 3750, 4000, 4496])
    ap.add_argument("--normal-z", type=int, nargs="+", default=[3150, 3750, 4350])
    ap.add_argument("--crops-per-z", type=int, default=8)
    ap.add_argument("--skip-umbilicus", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    umb = json.load(open(os.path.join(a.inputs, "umbilicus.json")))
    out = {}
    if not a.skip_umbilicus:
        out["umbilicus"] = check_umbilicus(a, umb)
    out["normals"], out["grad_mag"], out["normal_crops"] = check_normals(a, umb)
    print(json.dumps({k: out[k] for k in ("normals", "grad_mag")}, indent=1))
    json.dump(out, open(os.path.join(a.out, "sanity_checks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
