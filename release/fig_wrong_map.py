"""Top-down wrong-winding maps (angle around the umbilicus x z) of PHerc0139 z 3000-4500: baseline fit (arm0), automatic
annotations (arm19), human-oracle annotations (arm2), plus where arm19 fixes / breaks arm0.

Per human-mesh vertex (eval/spiral_eval.py reference sample) the fit is right if its nearest surface carries the human
winding after the seam offset (wrong_area_seam). Cells are 3 deg x 25 z; colour = share of matched vertices on the
wrong winding, pooled over all windings at that (angle, z).

  uv run python release/fit_vertex_dump.py --fit data/scratch/p0139_spiral/<arm>/meshes_step4 --name <arm>   # x3
  uv run python release/fig_wrong_map.py --out release/figures/fit_wrong_winding_map.png

--scroll p1667: PHerc1667 z 6500-8000 (L2 of 20251217075048, 9.596 um), baseline c1667_base vs method c1667_m2s1
(confirmation arms, notes/2026-10-22-confirmation-region.md sections 11-12), dumps from data/scratch/confirm_region/
p1667/vertex (fit_vertex_dump.py --scroll p1667). Panels: the arms given by --arms (default base, M2, oracle), then
the second arm vs the first. No #2167 gap sector on this scroll.
  uv run python release/fig_wrong_map.py --scroll p1667 --out release/figures/p1667_wrong_winding_map.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/scratch/spiral_eval/cache/ref_z3000-4500_s2_m0_v2.npz"
VDUMP = REPO / "data/scratch/fit_figures/vertex"
ARMS = [("arm0", "baseline fit (no annotations)"), ("arm19", "+ automatic winding annotations"),
        ("arm2", "+ human winding annotations (ceiling)")]
GAP = (145.0, 260.0)   # villa #2167 / w035 gap sector in this band (spiral_eval gap2167: start ~148, width ~112)
P1667 = {"cache": REPO / "data/scratch/spiral_eval/cache/ref_p1667_z6500-8000_s2_m0_v2.npz",
         "vdump": REPO / "data/scratch/confirm_region/p1667/vertex", "z": (6500, 8000), "vox": "9.596 um",
         "arms": [("c1667_base", "baseline fit (no annotations)"),
                  ("c1667_m2s1", "M2: + automatic winding chains"),
                  ("c1667_oracle", "+ human-label chains (ceiling)")]}


def grid(ang, z, val, w, da=3.0, dz=25.0, z0=3000, z1=4500):
    ia = np.clip((ang // da).astype(int), 0, int(360 / da) - 1)
    iz = np.clip(((z - z0) // dz).astype(int), 0, int((z1 - z0) / dz) - 1)
    shape = (int((z1 - z0) / dz), int(360 / da))
    num = np.zeros(shape)
    den = np.zeros(shape)
    np.add.at(num, (iz, ia), val * w)
    np.add.at(den, (iz, ia), w)
    with np.errstate(invalid="ignore"):
        return num / den, den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scroll", choices=["p0139", "p1667"], default="p0139")
    ap.add_argument("--arms", nargs="+", default=None, help="p1667: arm names (dumps in the p1667 vertex dir)")
    ap.add_argument("--titles", nargs="+", default=None)
    ap.add_argument("--vdump", type=Path, default=None, help="override the vertex-dump dir")
    a = ap.parse_args()
    if a.scroll == "p1667":
        return main_p1667(a)
    V = np.load(CACHE)
    ang, z = V["ang"], V["pts"][:, 0]  # pts are zyx
    D = {n: np.load(VDUMP / f"{n}.npz") for n, _ in ARMS}
    info = {n: json.loads((VDUMP / f"{n}.json").read_text()) for n, _ in ARMS}
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True, constrained_layout=True)
    cm = plt.get_cmap("magma_r")
    ext = [0, 360, 3000, 4500]
    for ax, (n, title) in zip(axs, ARMS):
        m = D[n]["matched"]
        wrong = (D[n]["res_s"] != 0).astype(float)
        frac, den = grid(ang[m], z[m], wrong[m], np.ones(m.sum()))
        frac[den < 20] = np.nan
        im = ax.imshow(frac, origin="lower", extent=ext, aspect="auto", cmap=cm, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"{n}: {title}\nwrong-winding area {info[n]['wrong_area_seam']:.1f} %", fontsize=9)
        ax.set_xlabel("angle around umbilicus (deg)")
    axs[0].set_ylabel("z (slice, 9.362 um)")
    fig.colorbar(im, ax=axs[:3], shrink=0.8, label="share of human-mesh area on the wrong winding", pad=0.01)
    # arm19 vs arm0, per vertex matched in both: fixed (arm0 wrong, arm19 right) minus broken (arm0 right, arm19 wrong)
    m = D["arm0"]["matched"] & D["arm19"]["matched"]
    w0 = D["arm0"]["res_s"] != 0
    w1 = D["arm19"]["res_s"] != 0
    fixed, broken = (w0 & ~w1)[m].astype(float), (~w0 & w1)[m].astype(float)
    nf, den = grid(ang[m], z[m], fixed - broken, np.ones(m.sum()))
    nf[den < 20] = np.nan
    im2 = axs[3].imshow(nf, origin="lower", extent=ext, aspect="auto", cmap="PiYG", vmin=-0.6, vmax=0.6,
                        interpolation="nearest")
    axs[3].set_title(f"arm19 vs arm0 (vertices matched in both)\nfixed {fixed.mean() * 100:.1f} %  "
                     f"broken {broken.mean() * 100:.1f} %  net {(fixed.mean() - broken.mean()) * 100:+.1f} pp",
                     fontsize=9)
    axs[3].set_xlabel("angle around umbilicus (deg)")
    fig.colorbar(im2, ax=axs[3], shrink=0.8, label="green: arm19 fixes arm0 / pink: arm19 breaks it", pad=0.01)
    for ax in axs:
        for g in GAP:
            ax.axvline(g, color="#2a9df4", lw=0.8, ls="--")
        ax.set_xticks(range(0, 361, 60))
    axs[0].text(GAP[0] + 3, 4470, "#2167 sector", color="#2a9df4", fontsize=7, va="top")
    for ax in axs[:3]:  # the fits' own seam (173-176 deg) falls inside the 170-180 deg offset bin: scorer artefact
        ax.annotate("fit seam\n(bin artefact)", xy=(175, 3020), xytext=(196, 3150), fontsize=6.5, color="#2a9df4",
                    arrowprops=dict(arrowstyle="-", color="#2a9df4", lw=0.6))
    fig.suptitle("PHerc0139 z 3000-4500: where each spiral fit puts the papyrus on the wrong winding "
                 "(scored against 37 human meshes; grey = no human mesh or fit not within 4 vox)", fontsize=10)
    for ax in axs:
        ax.set_facecolor("#d0d0d0")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(a.out, {n: round(info[n]["wrong_area_seam"], 2) for n, _ in ARMS},
          {"fixed": round(fixed.mean() * 100, 2), "broken": round(broken.mean() * 100, 2)})


def main_p1667(a) -> None:
    """Baseline vs method on PHerc1667: one map per arm + method-vs-baseline fixed/broken map."""
    C = P1667
    vd = a.vdump or C["vdump"]
    arms = list(zip(a.arms, a.titles or a.arms)) if a.arms else C["arms"]
    z0, z1 = C["z"]
    V = np.load(C["cache"])
    ang, z = V["ang"], V["pts"][:, 0]
    D = {n: np.load(vd / f"{n}.npz") for n, _ in arms}
    info = {n: json.loads((vd / f"{n}.json").read_text()) for n, _ in arms}
    nk = len(np.unique(V["k"]))
    fig, axs = plt.subplots(1, len(arms) + 1, figsize=(3.9 * (len(arms) + 1) + 0.6, 4.8), sharey=True,
                            constrained_layout=True)
    ext = [0, 360, z0, z1]
    for ax, (n, title) in zip(axs, arms):
        m = D[n]["matched"]
        wrong = (D[n]["res_s"] != 0).astype(float)
        frac, den = grid(ang[m], z[m], wrong[m], np.ones(m.sum()), z0=z0, z1=z1)
        frac[den < 20] = np.nan
        im = ax.imshow(frac, origin="lower", extent=ext, aspect="auto", cmap="magma_r", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"{n}: {title}\nwrong-winding area {info[n]['wrong_area_seam']:.1f} %", fontsize=9)
        ax.set_xlabel("angle around umbilicus (deg)")
    axs[0].set_ylabel(f"z (slice, {C['vox']})")
    fig.colorbar(im, ax=axs[:len(arms)], shrink=0.8, label="share of human-mesh area on the wrong winding", pad=0.01)
    (b, _), (t, _) = arms[0], arms[1]
    m = D[b]["matched"] & D[t]["matched"]
    w0, w1 = D[b]["res_s"] != 0, D[t]["res_s"] != 0
    fixed, broken = (w0 & ~w1)[m].astype(float), (~w0 & w1)[m].astype(float)
    nf, den = grid(ang[m], z[m], fixed - broken, np.ones(m.sum()), z0=z0, z1=z1)
    nf[den < 20] = np.nan
    im2 = axs[-1].imshow(nf, origin="lower", extent=ext, aspect="auto", cmap="PiYG", vmin=-0.6, vmax=0.6,
                         interpolation="nearest")
    axs[-1].set_title(f"{t} vs {b} (vertices matched in both)\nfixed {fixed.mean() * 100:.1f} %  "
                      f"broken {broken.mean() * 100:.1f} %  net {(fixed.mean() - broken.mean()) * 100:+.1f} pp",
                      fontsize=9)
    axs[-1].set_xlabel("angle around umbilicus (deg)")
    fig.colorbar(im2, ax=axs[-1], shrink=0.8, label=f"green: {t} fixes {b} / pink: breaks it", pad=0.01)
    for ax in axs:
        ax.set_xticks(range(0, 361, 60))
        ax.set_facecolor("#d0d0d0")
    fig.suptitle(f"PHerc1667 z {z0}-{z1}: where each spiral fit puts the papyrus on the wrong winding\n"
                 f"(scored against the labelled human mesh, {nk} windings; grey = no human mesh or fit not within 4 vox)",
                 fontsize=10)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(a.out, {n: round(info[n]["wrong_area_seam"], 2) for n, _ in arms},
          {"fixed": round(fixed.mean() * 100, 2), "broken": round(broken.mean() * 100, 2)})


if __name__ == "__main__":
    main()
