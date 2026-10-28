"""Image statistics used to calibrate the CT forward model against real scans (same code for real and sim).

  uv run python forward/measure.py        # measure real PHerc1667 crops at 2.4 and 7.91 um -> forward/calib_real.json
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

REPO = Path(__file__).resolve().parents[1]
EDGE_BINS_UM = np.arange(-40.0, 40.01, 2.0)  # signed distance to the material boundary (+ inside)
AIR_MIN_UM = 30.0  # air voxels at least this far from material are used for noise statistics
LAGS = (1, 2, 3, 4)


def measure(vol: np.ndarray, vox_um: float, otsu: float | None = None) -> dict:
    v = vol.astype(np.float32)
    if otsu is None:
        otsu = float(threshold_otsu(v[::2, ::2, ::2]))
    mask = ndi.gaussian_filter(v, 1.0) > otsu
    sd = (ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)) * vox_um
    # edge profile: mean intensity by signed distance bin
    idx = np.digitize(sd.ravel(), EDGE_BINS_UM)
    sums = np.bincount(idx, v.ravel(), minlength=len(EDGE_BINS_UM) + 1)
    cnts = np.bincount(idx, minlength=len(EDGE_BINS_UM) + 1)
    prof = (sums / np.maximum(cnts, 1))[1:-1]  # bins between consecutive edges
    # noise in air far from material: residual after removing slow background
    air = sd < -AIR_MIN_UM
    res = v - ndi.gaussian_filter(v, 8.0)
    out = {"otsu": otsu, "vox_um": vox_um, "edge_um": ((EDGE_BINS_UM[:-1] + EDGE_BINS_UM[1:]) / 2).tolist(),
           "edge_mean": prof.tolist(), "edge_count": cnts[1:-1].tolist(), "air_frac": float(air.mean())}
    if air.sum() > 1000:
        r = np.where(air, res, 0.0)
        var = float((r[air] ** 2).mean())
        acf = []
        for lag in LAGS:
            vals = []
            for ax in range(3):
                a = np.moveaxis(r, ax, 0)
                m = np.moveaxis(air, ax, 0)
                both = m[lag:] & m[:-lag]
                vals.append(float((a[lag:][both] * a[:-lag][both]).mean() / var))
            acf.append(vals)
        out |= {"air_mean": float(v[air].mean()), "noise_std": var**0.5, "acf": acf}
    mat = sd > 10.0
    out |= {"mat_mean": float(v[mat].mean()), "mat_std": float(v[mat].std()), "mat_p95": float(np.percentile(v[mat], 95))}
    return out


def measure_masked(vol: np.ndarray, vox_um: float, air_min_um=(30.0, 60.0, 90.0)) -> dict:
    """measure() for ESRF ~9 um scans: masked voxels (0) excluded, air noise at several distances from material, and the
    edge profile in 1-voxel bins with fringe depth/position and 10-90% edge width (scripts/w1_real9.py, forward/fit9.py)."""
    v = vol.astype(np.float32)
    valid = ndi.binary_erosion(vol > 0, iterations=3)
    otsu = float(threshold_otsu(v[valid][::7]))
    mask = ndi.gaussian_filter(v, 1.0) > otsu
    sd = (ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)) * vox_um
    idx = np.digitize(sd[valid], EDGE_BINS_UM)
    sums = np.bincount(idx, v[valid], minlength=len(EDGE_BINS_UM) + 1)
    cnts = np.bincount(idx, minlength=len(EDGE_BINS_UM) + 1)
    prof = (sums / np.maximum(cnts, 1))[1:-1]
    centers = (EDGE_BINS_UM[:-1] + EDGE_BINS_UM[1:]) / 2
    res = v - ndi.gaussian_filter(v, 8.0)
    valid_res = ndi.binary_erosion(valid, iterations=16)  # sigma-8 background must not see masked zeros
    out = {"otsu": otsu, "vox_um": vox_um, "valid_frac": float(valid.mean()), "edge_um": centers.tolist(),
           "edge_mean": prof.tolist(), "edge_count": cnts[1:-1].tolist()}
    mat = valid & (sd > 10.0)
    if mat.sum() > 1000:
        out |= {"mat_mean": float(v[mat].mean()), "mat_std": float(v[mat].std()), "mat_p95": float(np.percentile(v[mat], 95))}
    for amin in air_min_um:
        air = valid_res & (sd < -amin)
        key = f"air{int(amin)}"
        out[f"{key}_n"] = int(air.sum())
        if air.sum() < 2000:
            continue
        r = np.where(air, res, 0.0)
        var = float((r[air] ** 2).mean())
        acf = []
        for lag in LAGS:
            vals = []
            for ax in range(3):
                aa, mm = np.moveaxis(r, ax, 0), np.moveaxis(air, ax, 0)
                both = mm[lag:] & mm[:-lag]
                vals.append(float((aa[lag:][both] * aa[:-lag][both]).mean() / var))
            acf.append(float(np.mean(vals)))
        out |= {f"{key}_mean": float(v[air].mean()), f"{key}_noise_std": var**0.5, f"{key}_acf": acf}
    # edge profile in 1-voxel bins (EDT distances are discrete at ~9 um; 2 um bins leave holes)
    kb = np.arange(-15, 7)  # bin centres in voxels; sd is never in (-1, 1)
    sdv = sd[valid] / vox_um
    ib = np.digitize(sdv, np.r_[kb - 0.5, kb[-1] + 0.5])
    vs = v[valid]
    pm = np.array([vs[ib == i + 1].mean() if np.any(ib == i + 1) else np.nan for i in range(len(kb))])
    pc = np.array([int((ib == i + 1).sum()) for i in range(len(kb))])
    keep = pc > 500
    out |= {"edge9_vox": kb[keep].tolist(), "edge9_um": (kb[keep] * vox_um).tolist(), "edge9_mean": pm[keep].tolist()}
    far = out.get("air90_mean", out.get("air60_mean"))
    airside = keep & (kb < 0)
    if far is not None and airside.sum() > 3:
        i_min = int(np.nanargmin(np.where(airside, pm, np.nan)))
        mat_pl = float(np.nanmean(pm[keep & (kb >= 3)]))
        f = (pm - pm[i_min]) / (mat_pl - pm[i_min])
        xs = np.linspace(kb[i_min], 6, 2000)
        ok = keep & (kb >= kb[i_min])
        fi = np.interp(xs, kb[ok], f[ok])
        x10, x90 = xs[np.argmax(fi >= 0.1)], xs[np.argmax(fi >= 0.9)]
        out |= {"fringe_min": float(pm[i_min]), "fringe_depth": float(far - pm[i_min]), "fringe_pos_um": float(kb[i_min] * vox_um),
                "edge_mat_plateau": mat_pl, "edge_10_90_um": float((x90 - x10) * vox_um)}
    # haze: grey level in thin gaps (air voxels 1-2 vox from material on both... approximated by -15..-5 um) vs far air
    near = valid & (sd < -5) & (sd > -15)
    if near.sum() > 1000:
        out["near_air_mean_5_15um"] = float(v[near].mean())
    return out


def combine(ms: list[dict]) -> dict:
    """Median over volumes of the scalar stats; count-weighted mean of the edge profile."""
    out = {}
    for key in ("otsu", "air_mean", "noise_std", "mat_mean", "mat_std", "mat_p95", "air_frac"):
        vals = [m[key] for m in ms if key in m]
        out[key] = float(np.median(vals))
    acfs = [m["acf"] for m in ms if "acf" in m]
    out["acf"] = np.median(np.array(acfs), 0).tolist()
    c = np.array([m["edge_count"] for m in ms], float)
    e = np.array([m["edge_mean"] for m in ms], float)
    out["edge_um"] = ms[0]["edge_um"]
    out["edge_mean"] = ((e * c).sum(0) / np.maximum(c.sum(0), 1)).tolist()
    out["edge_count"] = c.sum(0).tolist()
    out["n"] = len(ms)
    return out


def main() -> None:
    import tifffile

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    crops = sorted(glob.glob(str(REPO / "data" / "xscan_crops" / "*.npz")))[:n]
    hi = []
    for f in crops:
        h = np.load(f)["hi"]
        c = np.array(h.shape) // 2
        h = h[c[0] - 160 : c[0] + 160, c[1] - 160 : c[1] + 160, c[2] - 160 : c[2] + 160]
        hi.append(measure(h, 2.399))
        print(Path(f).stem, "2.4", {k: round(v, 2) for k, v in hi[-1].items() if isinstance(v, float)}, flush=True)
    lo = []
    for f in crops:
        stem = Path(f).stem
        ct = tifffile.imread(REPO / "data" / "eval" / "jeong200" / "imagesTr" / f"{stem}_0000.tif")
        c = ct.shape[0] // 2
        lo.append(measure(ct[c - 80 : c + 80, c - 80 : c + 80, c - 80 : c + 80], 7.91))
        print(stem, "7.91", {k: round(v, 2) for k, v in lo[-1].items() if isinstance(v, float)}, flush=True)
    path = REPO / "forward" / "calib_real.json"
    calib = json.loads(path.read_text()) if path.exists() else {}
    calib["stats_v2"] = {"2p4": combine(hi), "7p91": combine(lo), "stems": [Path(f).stem for f in crops],
                         "note": "forward/measure.py; 2p4 central 320^3 of xscan crops, 7p91 central 160^3 of jeong200 patches"}
    path.write_text(json.dumps(calib, indent=1))
    for k in ("2p4", "7p91"):
        s = calib["stats_v2"][k]
        print(k, {a: round(b, 2) for a, b in s.items() if isinstance(b, float)}, "acf", np.round(s["acf"], 3).tolist())


if __name__ == "__main__":
    main()
