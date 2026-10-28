"""W1: statistics of surface-labels-2026-07 sample patches (data/eval/surface_labels_2026_07_sample/).

  uv run python scripts/w1_label_stats.py

Per patch: label values, foreground fraction (of all voxels and of non-ignore voxels), band thickness along the
normal (2 x EDT on the band ridge), which face of the sheet the band sits on (CT profile along the band normal,
oriented so that more material is +), and the band-to-band pitch between distinct sheets (label components).
"""

import glob
import json
from pathlib import Path

import cc3d
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data" / "eval" / "surface_labels_2026_07_sample"


def band_stats(lab: np.ndarray, ct: np.ndarray, vox_um: float, core: int | None = None) -> dict:
    if core:
        c = np.array(lab.shape) // 2
        sl = tuple(slice(x - core // 2, x + core // 2) for x in c)
        lab, ct = lab[sl], ct[sl]
    vals, cnts = np.unique(lab, return_counts=True)
    fg = lab == 1
    out = {"shape": list(lab.shape), "values": {int(v): int(n) for v, n in zip(vals, cnts)},
           "fg_frac_all": float(fg.mean()), "fg_frac_non_ignore": float(fg.sum() / max(1, (lab != 2).sum())),
           "ignore_frac": float((lab == 2).mean())}
    if fg.sum() < 1000:
        return out
    # thickness: EDT inside the band, at ridge voxels (local maxima)
    e = ndi.distance_transform_edt(fg)
    ridge = fg & (e >= ndi.maximum_filter(e, 3))
    th = 2 * e[ridge] - 1
    out |= {"band_thickness_vox_median": float(np.median(th)), "band_thickness_vox_p10_p90": np.percentile(th, [10, 90]).tolist(),
            "band_thickness_um_median": float(np.median(th) * vox_um)}
    # normal from the structure tensor of the smoothed band
    s = ndi.gaussian_filter(fg.astype(np.float32), 2.0)
    g = np.stack(np.gradient(s), -1)
    J = np.einsum("...i,...j->...ij", g, g)
    J = np.stack([np.stack([ndi.gaussian_filter(J[..., i, j], 3.0) for j in range(3)], -1) for i in range(3)], -2)
    pts = np.argwhere(ridge)
    rng = np.random.default_rng(0)
    pts = pts[rng.permutation(len(pts))[:4000]]
    w, V = np.linalg.eigh(J[tuple(pts.T)])
    n = V[..., -1]  # principal gradient direction = band normal
    ts = np.arange(-30, 31)
    from scipy.ndimage import map_coordinates
    ctf = ct.astype(np.float32)
    otsu = threshold_otsu(ctf[::3, ::3, ::3])
    profs, labp = [], []
    for t in ts:
        q = pts + t * n
        profs.append(map_coordinates(ctf, q.T, order=1, mode="nearest"))
        labp.append(map_coordinates(fg.astype(np.float32), q.T, order=0, mode="nearest"))
    P, L = np.array(profs).T, np.array(labp).T  # (pts, ts)
    # orient each profile so that the side with more material within +-12 vox is +
    side = (P[:, (ts > 0) & (ts <= 12)] > otsu).mean(1) - (P[:, (ts < 0) & (ts >= -12)] > otsu).mean(1)
    flip = side < 0
    P[flip], L[flip] = P[flip, ::-1], L[flip, ::-1]
    mp, ml = P.mean(0), L.mean(0)
    # half-max edge of the CT relative to the band ridge, and the band extent
    air, mat = np.median(P[:, ts <= -20]), np.median(P[:, (ts >= 4) & (ts <= 12)])
    half = (air + mat) / 2
    i0 = np.where(ts == 0)[0][0]
    below = np.where(mp[: i0 + 1] < half)[0]
    edge_t = float(ts[below[-1]] + (half - mp[below[-1]]) / (mp[below[-1] + 1] - mp[below[-1]])) if len(below) else float("nan")
    band = ts[ml > 0.5]
    out |= {"profile_t_vox": ts.tolist(), "profile_ct": mp.round(1).tolist(), "profile_label": ml.round(3).tolist(),
            "ct_halfmax_edge_t_vox": edge_t, "band_t_range_vox": [int(band.min()), int(band.max())] if len(band) else None,
            "frac_ambiguous_side": float((np.abs(side) < 0.2).mean())}
    # pitch between distinct label components (sheets): distance from each ridge voxel of comp i to nearest voxel of comp j != i
    cc = cc3d.connected_components(fg, connectivity=26)
    ids, sizes = np.unique(cc[cc > 0], return_counts=True)
    big = ids[sizes > 2000]
    out["n_components"] = int(len(big))
    if len(big) >= 2:
        dmin = np.full(fg.shape, np.inf, np.float32)
        dsec = np.full(fg.shape, np.inf, np.float32)
        owner = np.zeros(fg.shape, np.int32)
        for i in big:
            d = ndi.distance_transform_edt(cc != i).astype(np.float32)
            better = d < dmin
            dsec = np.where(better, dmin, np.minimum(dsec, d))
            owner = np.where(better, i, owner)
            dmin = np.where(better, d, dmin)
        r = ridge & np.isin(cc, big)
        pitch = dsec[r]  # distance from a band ridge voxel to the nearest *other* sheet's band
        pitch = pitch[np.isfinite(pitch)]
        pu = pitch * vox_um
        out |= {"pitch_um_median": float(np.median(pu)), "pitch_um_p10_p50_p90": np.percentile(pu, [10, 50, 90]).tolist(),
                "frac_pitch_lt_32um": float((pu < 32).mean()), "frac_pitch_lt_37um": float((pu < 37.4).mean()),
                "frac_pitch_lt_60um": float((pu < 60).mean()), "frac_pitch_lt_100um": float((pu < 100).mean())}
    return out


def main():
    res = {}
    for f in sorted(D.glob("p0139_*.npz")):
        z = np.load(f)
        s = band_stats(z["lab24"], z["ct24"], 2.399, core=320)
        res[f.stem] = s
        print(f.stem, {k: v for k, v in s.items() if not k.startswith("profile")}, flush=True)
    for f in sorted(glob.glob(str(D / "kaggle" / "labels" / "*.tif"))):
        lab = tifffile.imread(f)
        ct = tifffile.imread(f.replace("/labels/", "/images/"))
        s = band_stats(lab, ct, 7.91)  # voxel size unverified; um figures assume 7.91
        res["kaggle_" + Path(f).stem] = s
        print("kaggle", Path(f).stem, {k: v for k, v in s.items() if not k.startswith("profile")}, flush=True)
    (D / "label_stats.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
