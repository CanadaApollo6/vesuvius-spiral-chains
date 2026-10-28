"""Contact check: do GT-compressed sites show a resolved valley at 2.4 um that 7.91 um lacks?

Design and decision rule: notes/2026-10-16-contact-check.md (incl. amendments).

  uv run python eval/contact_check.py            # all 36 Scroll 4 patches
  uv run python eval/contact_check.py --limit 4  # quick run
Writes data/contact_check/{sites.csv, summary.json, sigma.json, gallery.png}.
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.feature import structure_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_patch import CORE, POOL, SCALE, core_grid, ncc, register  # noqa: E402
from xscan import Correspondences  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "contact_check"
SIGMAS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
COMPRESSED_MAX, WIDE_MIN, WIDE_MAX = 10.5, 16.0, 24.0
SEED_MARGIN = 24  # keep profiles inside the registered core
MAX_SITES = 400
DEPTH_THR = 0.15


def valley_depth(prof: np.ndarray, bg: float) -> float:
    """Deepest normalized valley strictly inside the profile."""
    best = 0.0
    left = np.maximum.accumulate(prof)
    right = np.maximum.accumulate(prof[::-1])[::-1]
    for t in range(1, len(prof) - 1):
        flank = min(left[t - 1], right[t + 1])
        denom = flank - bg
        if denom <= 1e-6:
            continue
        best = max(best, (flank - prof[t]) / denom)
    return best


def find_sites(gt: np.ndarray, rng: np.random.Generator):
    """Return list of (centre_zyx, normal_zyx, d) from GT bands, both directions per seed."""
    sm = ndi.gaussian_filter(gt.astype(np.float32), 1.0)
    st = structure_tensor(ndi.gaussian_filter(gt.astype(np.float32), 1.5), sigma=3, order="rc")
    edt = ndi.distance_transform_edt(gt)
    n = gt.shape[0]
    lo, hi = (n - CORE) // 2 + SEED_MARGIN, (n + CORE) // 2 - SEED_MARGIN
    seeds = np.argwhere(edt >= 2.0)
    seeds = seeds[((seeds >= lo) & (seeds < hi)).all(1)]
    seeds = seeds[rng.permutation(len(seeds))[:6000]]
    t = np.arange(-3, WIDE_MAX + 0.01, 0.25)
    sites = []
    for p in seeds:
        tp = tuple(p)
        T = np.array([[st[0][tp], st[1][tp], st[2][tp]], [st[1][tp], st[3][tp], st[4][tp]], [st[2][tp], st[4][tp], st[5][tp]]])
        nv = np.linalg.eigh(T)[1][:, -1]
        for sgn in (1.0, -1.0):
            u = sgn * nv
            prof = map_coordinates(sm, p[:, None] + u[:, None] * t, order=1)
            i = int(np.argmin(np.abs(t)))
            while 0 < i < len(t) - 1 and (prof[i + 1] > prof[i] or prof[i - 1] > prof[i]):
                i = i + 1 if prof[i + 1] > prof[i] else i - 1
            j = i
            while j < len(t) - 1 and prof[j + 1] <= prof[j]:
                j += 1
            while j < len(t) - 1 and prof[j + 1] > prof[j]:
                j += 1
            if prof[j] > 0.25 and j < len(t) - 1:
                sites.append((p + u * t[i], u, float(t[j] - t[i])))
    return sites


def profiles(reg, centre, u, length, blurred, smooth_hi, bg_lo, bg_hi):
    a0, a1 = -1.0, length + 1.0
    t_lo = np.arange(a0, a1 + 1e-9, 1.0)
    t_hi = np.arange(a0, a1 + 1e-9, 1.0 / SCALE)
    p_lo = centre[None] + t_lo[:, None] * u[None]
    p_hi = centre[None] + t_hi[:, None] * u[None]
    A = map_coordinates(reg.ct.astype(np.float32), p_lo.T, order=1)
    C = reg.sample_hi(p_lo, blurred)
    B = reg.sample_hi(p_hi).astype(np.float32)
    Bs = reg.sample_hi(p_hi, smooth_hi)
    return {
        "A": valley_depth(A, bg_lo), "C": valley_depth(C, bg_hi),
        "B": valley_depth(B, bg_hi), "Bs": valley_depth(Bs, bg_hi),
    }, (t_lo, A, C, t_hi, B)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted(Path(f).stem for f in glob.glob(str(POOL / "labelsTr" / "s4_*.tif")))
    if args.limit:
        stems = stems[: args.limit]
    corr = Correspondences()

    # 1) register all, calibrate blur sigma for arm C (global, by mean NCC)
    regs, ncc_tab = {}, {}
    for s in stems:
        r = register(s, corr)
        hi_f = r.hi.astype(np.float32)
        pts = core_grid(r.ct.shape[0], step=4).astype(float)
        target = r.ct[tuple(pts.astype(int).T)].astype(np.float32)
        ncc_tab[s] = {sg: ncc(target, r.sample_hi(pts, ndi.gaussian_filter(hi_f, sg))) for sg in SIGMAS}
        regs[s] = r
        print(f"registered {s}: translation-NCC {r.ncc:.3f}; NCC by sigma " +
              " ".join(f"{sg}:{v:.3f}" for sg, v in ncc_tab[s].items()), flush=True)
    mean_ncc = {sg: float(np.mean([ncc_tab[s][sg] for s in stems])) for sg in SIGMAS}
    sigma = max(mean_ncc, key=mean_ncc.get)
    (OUT / "sigma.json").write_text(json.dumps({"mean_ncc": mean_ncc, "chosen": sigma, "per_patch": ncc_tab}, indent=1))
    print(f"calibrated sigma = {sigma} (mean NCC {mean_ncc[sigma]:.3f})", flush=True)

    # 2) sites and valley scores
    rows, gallery = [], []
    rng = np.random.default_rng(0)
    for s in stems:
        r = regs[s]
        best_ncc = ncc_tab[s][sigma]
        if best_ncc < 0.6:
            print(f"EXCLUDED {s}: NCC {best_ncc:.3f} < 0.6")
            continue
        hi_f = r.hi.astype(np.float32)
        blurred = ndi.gaussian_filter(hi_f, sigma)
        smooth_hi = ndi.gaussian_filter(hi_f, 0.7)
        bg_lo, bg_hi = float(np.percentile(r.ct, 5)), float(np.percentile(r.hi, 5))
        sites = find_sites(r.gt, rng)
        comp = [x for x in sites if x[2] < COMPRESSED_MAX]
        wide = [x for x in sites if WIDE_MIN <= x[2] < WIDE_MAX]
        comp = [comp[i] for i in rng.permutation(len(comp))[:MAX_SITES]]
        wide = [wide[i] for i in rng.permutation(len(wide))[:MAX_SITES]]
        comp_d = [x[2] for x in comp] or [9.5]
        for cls, items in (("compressed", comp), ("wide", wide), ("control", wide)):
            for centre, u, d in items:
                length = float(rng.choice(comp_d)) if cls == "control" else d
                sc, prof = profiles(r, centre, u, length, blurred, smooth_hi, bg_lo, bg_hi)
                rows.append({"stem": s, "cls": cls, "d": d, "len": length, **sc})
                if cls == "compressed" and len(gallery) < 12 and rng.random() < 0.05:
                    gallery.append((s, d, prof))
        print(f"{s}: ncc={best_ncc:.3f} sites compressed={len(comp)} wide={len(wide)}", flush=True)

    with (OUT / "sites.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 3) summary + decision
    def rate(cls, arm, by_patch=False):
        sel = [x for x in rows if x["cls"] == cls]
        return 100.0 * np.mean([x[arm] >= DEPTH_THR for x in sel]) if sel else float("nan")

    summ = {cls: {arm: rate(cls, arm) for arm in ("A", "C", "B", "Bs")} | {"n": sum(x["cls"] == cls for x in rows)}
            for cls in ("compressed", "wide", "control")}
    # patch-level bootstrap CI for the DiD
    pstems = sorted({x["stem"] for x in rows})
    by = {s: [x for x in rows if x["stem"] == s] for s in pstems}

    def did(sample, arm="B"):
        sel = [x for s in sample for x in by[s]]
        def rt(c, a):
            v = [x[a] >= DEPTH_THR for x in sel if x["cls"] == c]
            return 100.0 * np.mean(v) if v else np.nan
        return (rt("compressed", arm) - rt("compressed", "C")) - (rt("control", arm) - rt("control", "C"))

    boot = np.random.default_rng(1)
    dids = [did(list(boot.choice(pstems, len(pstems)))) for _ in range(1000)]
    summ["DiD_B"] = did(pstems)
    summ["DiD_B_ci95"] = [float(np.nanpercentile(dids, 2.5)), float(np.nanpercentile(dids, 97.5))]
    summ["DiD_Bs"] = did(pstems, "Bs")
    summ["B_minus_C_compressed"] = summ["compressed"]["B"] - summ["compressed"]["C"]
    summ["alive"] = bool(summ["DiD_B"] >= 10 and summ["B_minus_C_compressed"] >= 10)
    summ["sigma"] = sigma
    summ["patches"] = len(pstems)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    if gallery:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        for ax, (s, d, (tl, A, C, th, B)) in zip(axes.ravel(), gallery):
            ax.plot(tl * 7.91, A, "o-", label="A real 7.91")
            ax.plot(tl * 7.91, C, "s-", label="C 2.4->7.91")
            ax.plot(th * 7.91, B, "-", label="B native 2.4")
            ax.axvline(0, color="r", lw=0.8)
            ax.axvline(d * 7.91, color="r", lw=0.8)
            ax.set_title(f"{s} d={d:.1f} vox", fontsize=8)
            ax.set_xlabel("um along normal")
        axes.ravel()[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "gallery.png", dpi=70)


if __name__ == "__main__":
    main()
