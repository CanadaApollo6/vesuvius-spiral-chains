"""Fit CTParams (forward/ct.py) to real PHerc1667 statistics (forward/calib_real.json "stats_v2").

  uv run python forward/fit.py          # -> forward/ct_params.json, data/scratch/ct_fit.png
  uv run python forward/fit.py --warm   # warm start from ct_params.json, single pass (after texture tweaks)

1. lo_a, lo_b: least squares of real 7.91 on (2.4 blurred sigma 2.5 vox) over registered pairs.
2. 2.4 noise: std + autocorrelation (lags 1-4) of the air residual, matched on pure synthetic noise.
3. 2.4 imaging (air, gain, PSF, fringe): edge profile (mean grey vs signed distance to boundary) + material stats,
   matched on sim renders measured with the same code as the real data (forward/measure.py).
4. 7.91 noise on pure synthetic noise (as in 2), then 7.91 gain/offset/halo on sim scans: edge profile + air/material.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "forward"), str(REPO / "sim"), str(REPO / "eval")]
from ct import VOX_HI, VOX_LO, CTParams, corr_noise, gauss3, hi_clean, hi_scan, lo_scan  # noqa: E402
from measure import EDGE_BINS_UM, combine, measure  # noqa: E402

DEV = "cuda"
REAL = json.loads((REPO / "forward" / "calib_real.json").read_text())["stats_v2"]


def noise_stats(n: torch.Tensor) -> tuple[float, np.ndarray]:
    """Same as measure(): residual after removing a sigma-8 background, then std and mean-over-axes ACF."""
    r = n - gauss3(n, 8.0)
    var = (r**2).mean()
    acf = [float(np.mean([((r.narrow(ax, lag, r.shape[ax] - lag) * r.narrow(ax, 0, r.shape[ax] - lag)).mean() / var).item()
                          for ax in range(3)])) for lag in (1, 2, 3, 4)]
    return float(var.sqrt()), np.array(acf)


def fit_noise(target_std, target_acf, vox_um, shape=(128, 128, 128)):
    def loss(x):
        s1, s2, w2 = x
        if not (0.3 < s1 < s2 < 40 and 0 <= w2 <= 0.95):
            return 1e3
        g = torch.Generator(device=DEV).manual_seed(0)
        std, acf = noise_stats(corr_noise(shape, 1.0, s1, s2, w2, g, DEV))
        return float(((acf - target_acf) ** 2).sum())

    best = min((minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 300})
                for x0 in ([1.0, 3.0, 0.2], [0.8, 6.0, 0.4], [1.5, 10.0, 0.1])), key=lambda r: r.fun)
    s1, s2, w2 = best.x
    g = torch.Generator(device=DEV).manual_seed(0)
    unit_std, acf = noise_stats(corr_noise(shape, 1.0, s1, s2, w2, g, DEV))
    return {"std": target_std / unit_std, "s1_um": s1 * vox_um, "s2_um": s2 * vox_um, "w2": w2, "acf_fit": acf.tolist(),
            "loss": best.fun}


def fit_affine(n_stems=8):
    from register_patch import core_grid, register
    from xscan import Correspondences

    corr = Correspondences()
    stems = REAL["stems"][:n_stems]
    xs, ys = [], []
    for s in stems:
        r = register(s, corr)
        b = ndi.gaussian_filter(r.hi.astype(np.float32), 2.5)
        pts = core_grid(r.ct.shape[0], step=3).astype(float)
        xs.append(r.sample_hi(pts, b))
        ys.append(r.ct[tuple(pts.astype(int).T)].astype(np.float32))
        print(f"  {s}: ncc {r.ncc:.3f}", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    return float(a), float(b), float(resid.std())


def sim_blocks(n=4, size=320, seed=11):
    from render import render_grid
    from scroll import Scroll, ScrollParams

    S = Scroll(ScrollParams(seed=seed))
    rng = np.random.default_rng(seed)
    blocks = []
    while len(blocks) < n:
        rad, ang, z = rng.uniform(3000, 9000), rng.uniform(0, 2 * np.pi), rng.uniform(-2000, 2000)
        c = np.array([rad * np.cos(ang) * 1.0, rad * np.sin(ang) * 0.62, z])
        S.set_reference(torch.tensor(c, dtype=torch.float32, device=DEV))
        h = size * VOX_HI / 2
        o = render_grid(S, (c[0] - h, c[1] - h, c[2] - h), (size,) * 3, VOX_HI, keys=("density", "instance"))
        frac = (o["instance"] > 0).mean()
        if 0.3 <= frac <= 0.7:
            blocks.append(torch.from_numpy(o["density"]).to(DEV))
            print(f"  block {len(blocks)}: in-sheet {frac:.2f}", flush=True)
    return blocks


def edge_profile_fast(img: torch.Tensor, sd: torch.Tensor):
    idx = torch.bucketize(sd.flatten(), torch.tensor(EDGE_BINS_UM, device=DEV, dtype=torch.float32), right=True)
    n = len(EDGE_BINS_UM) + 1
    s = torch.zeros(n, device=DEV).index_add_(0, idx, img.flatten())
    c = torch.zeros(n, device=DEV).index_add_(0, idx, torch.ones_like(img.flatten()))
    return s[1:-1], c[1:-1]


def fit_hi(p: CTParams, r2: dict, log: dict, warm: bool = False) -> list:
    """Steps 2-3: 2.4 um noise (pure noise), then imaging + noise jointly on sim renders. Returns the sim blocks."""
    print("2) 2.4 noise", flush=True)
    n2 = fit_noise(r2["noise_std"], np.mean(r2["acf"], 1), VOX_HI)
    p.n2_std, p.n2_s1_um, p.n2_s2_um, p.n2_w2 = n2["std"], n2["s1_um"], n2["s2_um"], n2["w2"]
    log["n2"] = n2 | {"target_acf": np.mean(r2["acf"], 1).tolist()}
    print(" ", n2)

    print("3) 2.4 imaging on sim renders", flush=True)
    blocks = sim_blocks()
    real_e = np.array(r2["edge_mean"])
    um = np.array(r2["edge_um"])

    # signed distance from the current fitted image, then refit (warm start from ct_params.json: one pass, one start)
    for it in range(1 if warm else 2):
        sds = []
        for rho in blocks:
            g = torch.Generator(device=DEV).manual_seed(1)
            img = hi_scan(hi_clean(rho, p), p, g).cpu().numpy()
            v = img.astype(np.float32)
            from skimage.filters import threshold_otsu

            mask = ndi.gaussian_filter(v, 1.0) > threshold_otsu(v[::2, ::2, ::2])
            sd = (ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)) * VOX_HI
            sds.append(torch.from_numpy(sd.astype(np.float32)).to(DEV))
        noise = [corr_noise(b.shape, p.n2_std, p.n2_s1_um / VOX_HI, p.n2_s2_um / VOX_HI, p.n2_w2,
                            torch.Generator(device=DEV).manual_seed(1), DEV) for b in blocks]

        airs = [sd < -30.0 for sd in sds]
        tgt_acf = np.mean(r2["acf"], 1)

        def loss(x):
            air, gain, psf, alpha, fs, nstd, s1, s2, w2 = x
            if not (0.3 <= psf <= 4 and 1 <= fs <= 10 and alpha >= 0 and gain > 10 and nstd > 0
                    and 0.5 < s1 < s2 < 30 and 0 <= w2 <= 0.95):
                return 1e9
            q = CTParams(**{**p.__dict__, "air": air, "gain": gain, "psf_sigma_um": psf, "fringe_alpha": alpha,
                            "fringe_sigma_um": fs, "n2_std": nstd, "n2_s1_um": s1, "n2_s2_um": s2, "n2_w2": w2})
            S, C, mats, var, lag = 0, 0, [], 0.0, np.zeros(4)
            for i, (rho, sd, am) in enumerate(zip(blocks, sds, airs)):
                g = torch.Generator(device=DEV).manual_seed(1 + i)
                img = hi_clean(rho, q) + corr_noise(rho.shape, nstd, s1 / VOX_HI, s2 / VOX_HI, w2, g, DEV)
                s_, c_ = edge_profile_fast(img, sd)
                S, C = S + s_, C + c_
                mats.append(img[sd > 10])
                r = torch.where(am, img - gauss3(img, 8.0), 0.0)
                var += float((r[am] ** 2).mean())
                for li, l in enumerate((1, 2, 3, 4)):
                    lag[li] += np.mean([float((r.narrow(ax, l, r.shape[ax] - l) * r.narrow(ax, 0, r.shape[ax] - l)).sum()
                                              / (am.narrow(ax, l, am.shape[ax] - l) & am.narrow(ax, 0, am.shape[ax] - l)).sum())
                                        for ax in range(3)])
            var /= len(blocks)
            acf = lag / len(blocks) / var
            prof = (S / C.clamp(min=1)).cpu().numpy()
            ok = (C.cpu().numpy() > 100) & (real_e > 0) & (um > -32) & (um < 18)
            m = torch.cat(mats)
            return float(3 * ((prof - real_e)[ok] ** 2).mean() + (m.mean().item() - r2["mat_mean"]) ** 2
                         + (m.std().item() - r2["mat_std"]) ** 2 + 10 * (var**0.5 - r2["noise_std"]) ** 2
                         + 1000 * ((acf - tgt_acf) ** 2).sum())

        x0 = [p.air, p.gain, p.psf_sigma_um, p.fringe_alpha, p.fringe_sigma_um, p.n2_std, p.n2_s1_um, p.n2_s2_um, p.n2_w2]
        starts = (x0, [p.air, p.gain, 1.2, 150.0, 6.0] + x0[5:]) if it == 0 and not warm else (x0,)
        best = min((minimize(loss, x, method="Nelder-Mead", options={"maxiter": 500 if warm else 1500, "xatol": 1e-2, "fatol": 1e-3})
                    for x in starts), key=lambda r: r.fun)
        (p.air, p.gain, p.psf_sigma_um, p.fringe_alpha, p.fringe_sigma_um,
         p.n2_std, p.n2_s1_um, p.n2_s2_um, p.n2_w2) = (float(v) for v in best.x)
        print(f"  pass {it}: loss {best.fun:.2f} air {p.air:.1f} gain {p.gain:.1f} psf {p.psf_sigma_um:.2f} um "
              f"alpha {p.fringe_alpha:.1f} fringe sigma {p.fringe_sigma_um:.2f} um noise {p.n2_std:.2f} "
              f"{p.n2_s1_um:.2f}/{p.n2_s2_um:.2f}/{p.n2_w2:.2f}", flush=True)
    log["hi_loss"] = best.fun

    return blocks


def main() -> None:
    only_lo = "--only-lo" in sys.argv  # keep the fitted 2.4 stage from ct_params.json, refit the 7.91 stage only
    warm = "--warm" in sys.argv  # start every stage from ct_params.json (after small texture changes)
    p = CTParams.load() if only_lo or warm else CTParams()
    log = {}
    print("1) 7.91 affine from registered pairs", flush=True)
    cache = REPO / "data" / "scratch" / "ct_fit_affine.json"
    if cache.exists():
        p.lo_a, p.lo_b, res = json.loads(cache.read_text())
    else:
        p.lo_a, p.lo_b, res = fit_affine()
        cache.write_text(json.dumps([p.lo_a, p.lo_b, res]))
    log["affine_resid_std"] = res
    print(f"  a={p.lo_a:.3f} b={p.lo_b:.2f} resid std {res:.2f}")

    r2 = REAL["2p4"]
    if only_lo:
        blocks = sim_blocks()
    else:
        blocks = fit_hi(p, r2, log, warm)

    print("4) 7.91 noise (pure noise), then imaging on sim scans", flush=True)
    r7 = REAL["7p91"]
    # Noise is fitted on pure synthetic noise, like the 2.4 stage: sim inter-sheet gaps are narrow, so sim "air" voxels
    # >= 30 um from material sit inside the blurred halo and a joint fit absorbed halo structure into the noise
    # (sim 7.91 air sd 20 vs real 10.9, 2026-10-17).
    # target: air >= 90 um from material (calib_real.json "noise7_far"); at >= 30 um real air is halo-dominated too
    far = json.loads((REPO / "forward" / "calib_real.json").read_text())["noise7_far"]
    n7 = fit_noise(far["noise_std"], np.array(far["acf"]), VOX_LO)
    p.n7_std, p.n7_s1_um, p.n7_s2_um, p.n7_w2 = n7["std"], n7["s1_um"], n7["s2_um"], n7["w2"]
    log["n7"] = n7 | {"target_acf": far["acf"]}
    print(" ", n7)

    def lo_stats(q, noise=True):
        ms = []
        for i, rho in enumerate(blocks):
            g = torch.Generator(device=DEV).manual_seed(2 + i)
            ms.append(measure(lo_scan(hi_clean(rho, q), q, g, noise=noise).cpu().numpy(), VOX_LO))
        return combine(ms)

    real7_e, um7 = np.array(r7["edge_mean"]), np.array(r7["edge_um"])

    def loss7(x):
        la, lb, fa, fs = x
        # bounds keep the gain near the registration-measured 0.72 (an unbounded fit traded gain for halo contrast)
        if not (0.55 < la < 0.9 and 0 <= fa <= 800 and 6 < fs < 30):
            return 1e9
        q = CTParams(**{**p.__dict__, "lo_a": la, "lo_b": lb, "lo_fringe_alpha": fa, "lo_fringe_sigma_um": fs})
        st = lo_stats(q)
        e = np.array(st["edge_mean"])
        ok = (e > 0) & (real7_e > 0) & (um7 > -42) & (um7 < 24)
        return float(((e - real7_e)[ok] ** 2).mean() / 25 + ((st["air_mean"] - r7["air_mean"]) / 5) ** 2
                     + ((st["mat_mean"] - r7["mat_mean"]) / 5) ** 2 + ((st["mat_std"] - r7["mat_std"]) / 5) ** 2)

    best7 = min((minimize(loss7, x0, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-2, "fatol": 1e-4})
                 for x0 in (([p.lo_a, p.lo_b, p.lo_fringe_alpha, p.lo_fringe_sigma_um],) if warm else
                            ([0.72, 42.0, 300.0, 15.0], [0.65, 50.0, 100.0, 10.0]))), key=lambda r: r.fun)
    p.lo_a, p.lo_b, p.lo_fringe_alpha, p.lo_fringe_sigma_um = (float(v) for v in best7.x)
    st7 = lo_stats(p)
    log["lo_loss"] = best7.fun
    print(f"  lo_a {p.lo_a:.3f} lo_b {p.lo_b:.2f} fringe alpha {p.lo_fringe_alpha:.0f} sigma {p.lo_fringe_sigma_um:.1f} um;"
          f" loss {best7.fun:.4f}", flush=True)

    # final comparison
    st2 = combine([measure(hi_scan(hi_clean(b, p), p, torch.Generator(device=DEV).manual_seed(5)).cpu().numpy(), VOX_HI)
                   for b in blocks])
    for name, sim, real in (("2.4", st2, r2), ("7.91", st7, r7)):
        print(name, " ".join(f"{k} sim {sim[k]:.1f} / real {real[k]:.1f}" for k in
                             ("otsu", "air_mean", "noise_std", "mat_mean", "mat_std", "mat_p95")))
        print("   acf sim", np.round(np.mean(sim["acf"], 1), 3).tolist(), "real", np.round(np.mean(real["acf"], 1), 3).tolist())
        log[f"cmp_{name}"] = {"sim": sim, "real": real}
    p.save(extra={"fit_log": {k: v for k, v in log.items() if not k.startswith("cmp_")}})
    (REPO / "data" / "scratch" / "ct_fit_log.json").write_text(json.dumps(log, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, name, sim, real in ((ax[0], "2.4 um", st2, r2), (ax[1], "7.91 um", st7, r7)):
        for st, lab in ((real, "real"), (sim, "sim")):
            e, c = np.array(st["edge_mean"]), np.array(st.get("edge_count", np.ones(len(st["edge_mean"]))))
            ok = e > 0
            a.plot(np.array(st["edge_um"])[ok], e[ok], "o-", label=lab)
        a.set_title(f"edge profile {name}")
        a.set_xlabel("signed distance to boundary (um, + inside)")
        a.legend()
    for st, lab in ((r2, "real 2.4"), (st2, "sim 2.4"), (r7, "real 7.91"), (st7, "sim 7.91")):
        ax[2].plot([1, 2, 3, 4], np.mean(st["acf"], 1), "o-", label=lab)
    ax[2].set_title("air noise autocorrelation vs lag (vox)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(REPO / "data" / "scratch" / "ct_fit.png", dpi=70)


if __name__ == "__main__":
    main()
