"""Fit an ESRF ScanStage (forward/ct.py) to real ~9 um crops (data/real9/stats.json, scripts/w1_real9.py).

  uv run python forward/fit9.py 9p362     # PHerc0826 + PHerc0139 targets -> ct_params.json["stages"]["9p362"]
  uv run python forward/fit9.py 8p64      # PHerc1447 targets            -> ct_params.json["stages"]["8p64"]

1. Noise: std + ACF of air >= 90 um from material, matched on pure synthetic noise (as forward/fit.py).
2. Imaging (air, gain, PSF, fringe amplitude and width) on sim renders of the scroll's outer surface, measured with the
   same code as the real crops (measure_masked): 1-voxel edge profile, material mean/std, fringe depth, 10-90% width.
   Real targets are the "edge" crops (outer surface, far air); "interior" crops are held out and only compared.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "forward"), str(REPO / "sim")]
from ct import VOX_HI, ScanStage, stage_scan  # noqa: E402
from fit import fit_noise  # noqa: E402
from measure import measure_masked  # noqa: E402

DEV = "cuda"
STAGES = {  # key: (voxel um, target crop prefixes, held-out crop prefixes)
    "9p362": (9.362, ("PHerc0826_edge", "p0139_"), ("PHerc0826_interior",)),
    "8p64": (8.640, ("PHerc1447_edge",), ("PHerc1447_interior",)),
}
SCALARS = ("air90_mean", "air90_noise_std", "mat_mean", "mat_std", "fringe_depth", "fringe_pos_um", "edge_10_90_um",
           "edge_mat_plateau", "near_air_mean_5_15um")


def targets(prefixes) -> dict:
    rows = json.loads((REPO / "data" / "real9" / "stats.json").read_text())
    rs = [r for k, r in rows.items() if k.startswith(prefixes)]
    out = {k: float(np.median([r[k] for r in rs if k in r])) for k in SCALARS}
    out["acf"] = np.median([r["air90_acf"] for r in rs if "air90_acf" in r], 0).tolist()
    # edge profile on the common 1-voxel bins (-15..6), mean over crops
    vox = np.arange(-15, 7)
    prof = [np.interp(vox, r["edge9_vox"], r["edge9_mean"], left=np.nan, right=np.nan) for r in rs]
    out["edge9_vox"], out["edge9_mean"] = vox.tolist(), np.nanmean(prof, 0).tolist()
    out["n"] = len(rs)
    return out


def edge_blocks(n=4, size=448, seed=21, inward_um=250.0):
    """Density blocks on the 2.4 grid straddling the outermost wrap (air outside), like the real "edge" crops."""
    from render import render_grid
    from scroll import Scroll, ScrollParams

    S = Scroll(ScrollParams(seed=seed))
    p = S.p
    rng = np.random.default_rng(seed)
    ca, sa = math.cos(p.squash_angle), math.sin(p.squash_angle)
    rads = torch.arange(p.r0, 3 * (p.r0 + p.n_wraps * (p.t_mean + p.g_mean)), 10.0, device=DEV)

    def world(rad, ang, z):
        x, y = rad * math.cos(ang) * p.squash[0], rad * math.sin(ang) * p.squash[1]
        return np.array([ca * x - sa * y, sa * x + ca * y, z])

    blocks = []
    while len(blocks) < n:
        ang, z = rng.uniform(0, 2 * math.pi), rng.uniform(-1500, 1500)
        x, y = rads * math.cos(ang) * p.squash[0], rads * math.sin(ang) * p.squash[1]
        pts = torch.stack([ca * x - sa * y, sa * x + ca * y, torch.full_like(x, z)], -1)
        S.set_reference(pts[len(pts) // 3])
        inst = S.sample(pts, want_density=False)["instance"]
        on = torch.nonzero(inst > 0)
        if len(on) == 0:
            continue
        r_edge = float(rads[on[-1, 0]])
        c = world(r_edge - inward_um, ang, z)
        S.set_reference(torch.tensor(c, dtype=torch.float32, device=DEV))
        h = (size - 1) * VOX_HI / 2
        o = render_grid(S, (c[0] - h, c[1] - h, c[2] - h), (size,) * 3, VOX_HI, keys=("density", "instance"))
        frac = float((o["instance"] > 0).mean())
        if 0.2 <= frac <= 0.6:
            blocks.append(torch.from_numpy(o["density"]))
            print(f"  block {len(blocks)}: r_edge {r_edge:.0f} um, in-sheet {frac:.2f}", flush=True)
    return blocks


def sim_stats(blocks, st: ScanStage, seed=3) -> dict:
    rs = []
    for i, rho in enumerate(blocks):
        g = torch.Generator(device=DEV).manual_seed(seed + i)
        img = stage_scan(rho.to(DEV), st, g).cpu().numpy().astype(np.uint8)
        rs.append(measure_masked(img, st.vox_um, air_min_um=(90.0,)))
    out = {k: float(np.median([r[k] for r in rs if k in r])) if any(k in r for r in rs) else float("nan") for k in SCALARS}
    vox = np.arange(-15, 7)
    prof = [np.interp(vox, r["edge9_vox"], r["edge9_mean"], left=np.nan, right=np.nan) for r in rs]
    out["edge9_vox"], out["edge9_mean"] = vox.tolist(), np.nanmean(prof, 0).tolist()
    return out


def main() -> None:
    key = sys.argv[1]
    vox, tgt_prefixes, held_prefixes = STAGES[key]
    T, H = targets(tgt_prefixes), targets(held_prefixes)
    print(f"targets ({T['n']} crops):", {k: round(T[k], 2) for k in SCALARS}, "acf", np.round(T["acf"], 3).tolist())
    st = ScanStage(vox_um=vox, air=T["air90_mean"])

    print("1) noise", flush=True)
    n = fit_noise(T["air90_noise_std"], np.array(T["acf"]), vox)
    st.n_std, st.n_s1_um, st.n_s2_um, st.n_w2 = n["std"], n["s1_um"], n["s2_um"], n["w2"]
    print(" ", {k: np.round(v, 3).tolist() for k, v in n.items()}, flush=True)

    print("2) imaging on sim outer-surface blocks", flush=True)
    blocks = edge_blocks()
    te = np.array(T["edge9_mean"])

    def loss(x, verbose=False):
        air, gain, psf, alpha, fs = x
        if not (2 <= psf <= 20 and 2 <= fs <= 40 and gain > 10 and -2000 <= alpha <= 5000):
            return 1e9
        q = ScanStage(**{**st.__dict__, "air": air, "gain": gain, "psf_sigma_um": psf, "fringe_alpha": alpha,
                         "fringe_sigma_um": fs})
        s = sim_stats(blocks, q)
        e = np.array(s["edge9_mean"])
        ok = np.isfinite(e) & np.isfinite(te)
        terms = {"edge": float(((e - te)[ok] ** 2).mean() / 9), "mat_mean": ((s["mat_mean"] - T["mat_mean"]) / 3) ** 2,
                 "mat_std": ((s["mat_std"] - T["mat_std"]) / 5) ** 2,
                 "fringe": ((np.nan_to_num(s["fringe_depth"], nan=0) - T["fringe_depth"]) / 3) ** 2,
                 "width": ((np.nan_to_num(s["edge_10_90_um"], nan=99) - T["edge_10_90_um"]) / 4) ** 2}
        if verbose:
            print("   ", {k: round(v, 3) for k, v in terms.items()})
        return float(sum(terms.values()))

    starts = ([T["air90_mean"], 80.0, 6.0, 300.0, 10.0], [T["air90_mean"], 80.0, 10.0, 800.0, 16.0])
    best = min((minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 250, "xatol": 1e-2, "fatol": 1e-3})
                for x0 in starts), key=lambda r: r.fun)
    st.air, st.gain, st.psf_sigma_um, st.fringe_alpha, st.fringe_sigma_um = (float(v) for v in best.x)
    loss(best.x, verbose=True)
    print(f"  loss {best.fun:.3f} air {st.air:.1f} gain {st.gain:.1f} psf {st.psf_sigma_um:.2f} um "
          f"fringe alpha {st.fringe_alpha:.0f} sigma {st.fringe_sigma_um:.2f} um", flush=True)

    S = sim_stats(blocks, st, seed=101)
    print("        " + " ".join(f"{k[:14]:>14}" for k in SCALARS))
    for name, d in (("sim", S), ("real", T), ("held", H)):
        print(f"{name:>8}" + " ".join(f"{d[k]:14.2f}" for k in SCALARS))
    print("edge9 sim ", np.round(S["edge9_mean"], 0).tolist())
    print("edge9 real", np.round(T["edge9_mean"], 0).tolist())
    st.save(key, extra={"fit": {"loss": best.fun, "noise": n, "targets": T, "sim": S, "held_out_interior": H}})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for d, lab in ((T, "real edge crops"), (S, "sim")):
        ax[0].plot(np.array(d["edge9_vox"]) * vox, d["edge9_mean"], "o-", label=lab)
    ax[0].set_xlabel("signed distance to boundary (um, + inside)")
    ax[0].set_title(f"{key} edge profile")
    ax[0].legend()
    g = torch.Generator(device=DEV).manual_seed(7)
    img = stage_scan(blocks[0].to(DEV), st, g).cpu().numpy()
    ax[1].imshow(img[img.shape[0] // 2], cmap="gray", vmin=0, vmax=200)
    ax[1].set_title("sim slice")
    fig.tight_layout()
    fig.savefig(REPO / "data" / "scratch" / f"ct_fit_{key}.png", dpi=70)


if __name__ == "__main__":
    main()
