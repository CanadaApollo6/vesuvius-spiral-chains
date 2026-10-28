"""Axial CT slice of PHerc0139 with the human meshes and two spiral fits traced on it (z-iso contours of the tifxyz
surfaces). Fit traces are labelled in the human winding frame (spiral_eval seam offsets) and coloured green where they
sit on the human sheet with that number, red where they sit on a different human sheet, grey where no human sheet is
within 4 vox. CT is a single L0 slice streamed from the open-data bucket (one 128-chunk row of the crop).

  uv run python release/fig_slices.py --z 3300 --center 2510 3950 --size 420 --out release/figures/fit_slice_z3300.png

--scroll p1667: PHerc1667 confirmation band (fit frame = L2 of 20251217075048, 9.596 um; CT = zarr level 2), human
meshes = the labelled merged mesh cut per winding (confirm_region/p1667/oracle_z6500), default arms = the figure refits
c1667_base_fig / c1667_m2s1_fig (vertex dumps in confirm_region/p1667/vertex_fig).
  uv run python release/fig_slices.py --scroll p1667 --z 7200 --center <y> <x> --size 600 --out release/figures/p1667_slice_z7200.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_traces as ft  # noqa: E402
from fit_traces import ARMS, HumanFrame, blacklisted_windings, ct_slice, save_png, se, slice_traces  # noqa: E402

PAL = plt.get_cmap("tab10")
OK, BAD, UNK = "#2ecc40", "#ff2d2d", "#9a9a9a"


def segments(yx, colors):
    """Polyline -> per-segment list (x, y) with colours; NaN breaks the line."""
    seg, col = [], []
    for i in range(len(yx) - 1):
        a, b = yx[i], yx[i + 1]
        if np.isfinite(a).all() and np.isfinite(b).all() and np.hypot(*(a - b)) < 15:
            seg.append([(a[1], a[0]), (b[1], b[0])])
            col.append(colors[i])
    return seg, col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--center", type=int, nargs=2, required=True, help="y x (fit-frame vox: p0139 9.362 L0, p1667 L2)")
    ap.add_argument("--scroll", choices=["p0139", "p1667"], default="p0139")
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--arms", nargs="+", default=None, help="default p0139 arm0 arm19; p1667 c1667_base_fig c1667_m2s1_fig")
    ap.add_argument("--titles", nargs="+", default=None)
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ft.use_scroll(a.scroll)
    sname, um = ft.SCROLL_INFO[a.scroll]
    if a.arms is None:
        a.arms = ["arm0", "arm19"] if a.scroll == "p0139" else ["c1667_base_fig", "c1667_m2s1_fig"]
    if a.titles is None:
        a.titles = (["arm0: baseline fit", "arm19: + automatic annotations"] if a.scroll == "p0139" else
                    ["baseline fit (no annotations)", "M2: + automatic winding chains"])
    cy, cx = a.center
    h = a.size // 2
    box = (cy - h, cy + h, cx - h, cx + h)
    img = ct_slice(a.z, box).astype(np.float32)
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 255)
    black = blacklisted_windings()
    hum = slice_traces(ft.ORACLE, a.z, box)
    for t in hum:
        t["black"] = t["w"] in black
    ref = [t for t in hum if not t["black"]]
    hp = np.concatenate([t["yx"] for t in ref])
    hk = np.concatenate([np.full(len(t["yx"]), t["w"]) for t in ref])
    okh = np.isfinite(hp).all(1)
    tree = cKDTree(hp[okh])
    hk = hk[okh]
    n = 1 + len(a.arms)
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 6.6), constrained_layout=True)
    ext = [box[2], box[3], box[1], box[0]]
    stats = {}
    for ax in axs:
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=ext, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
    # human panel: traces coloured by winding, labelled at their outermost visible point
    for t in hum:
        c = PAL(t["w"] % 10)
        s, _ = segments(t["yx"], [c] * len(t["yx"]))
        axs[0].add_collection(LineCollection(s, colors=c, linewidths=1.0 if t["black"] else 1.3,
                                             linestyles=(0, (2, 2)) if t["black"] else "solid"))
        v = t["yx"][np.isfinite(t["yx"]).all(1)]
        if len(v) > 20:
            p = v[np.argmin(v[:, 1])]
            axs[0].text(p[1] + 2, p[0], str(t["w"]) + ("x" if t["black"] else ""), color=c, fontsize=7.5, ha="left", va="center", weight="bold",
                        bbox=dict(boxstyle="square,pad=0.05", fc="k", ec="none", alpha=0.5))
    note0 = "(dashed, 'x': blacklisted, not used for scoring); " if black else "(labelled merged human mesh); "
    axs[0].set_title(f"human meshes (reference) with winding numbers\n{note0}crop {a.size} vox = {a.size * um / 1000:.1f} mm",
                     fontsize=10)
    for ax, arm, title in zip(axs[1:], a.arms, a.titles):
        hf = HumanFrame(arm)
        tr = slice_traces(ARMS / arm / "meshes_step4", a.z, box)
        nok = nbad = nunk = 0
        for t in ref:  # faint human reference under the fit
            s, _ = segments(t["yx"], ["w"] * len(t["yx"]))
            ax.add_collection(LineCollection(s, colors=(1, 1, 1, 0.25), linewidths=0.6))
        for t in tr:
            yx = t["yx"]
            fin = np.isfinite(yx).all(1)
            zyx = np.column_stack([np.full(len(yx), a.z), np.nan_to_num(yx)])
            kf = hf.k(np.full(len(yx), t["w"]), zyx)
            d, i = tree.query(np.nan_to_num(yx, nan=-1e6), distance_upper_bound=a.tol)
            near = np.isfinite(d) & fin
            kh = np.where(near, hk[np.minimum(i, len(hk) - 1)], -999)
            col = np.where(~near, UNK, np.where(kh == kf, OK, BAD))
            s, cc = segments(yx, list(col))
            ax.add_collection(LineCollection(s, colors=cc, linewidths=1.5))
            nok += int((near & (kh == kf)).sum()); nbad += int((near & (kh != kf)).sum()); nunk += int((~near & fin).sum())
            v = yx[fin]
            if len(v) > 20:
                p = v[np.argmin(v[:, 1])]
                ax.text(p[1] + 2, p[0], str(int(np.median(kf[fin]))), color="w", fontsize=7, ha="left", va="center",
                        bbox=dict(boxstyle="square,pad=0.05", fc="k", ec="none", alpha=0.5))
        tot = max(nok + nbad, 1)
        stats[arm] = {"right": nok / tot, "wrong": nbad / tot, "no_human_within_tol": nunk / max(nok + nbad + nunk, 1)}
        ax.set_title(f"{title}\nright winding {100 * nok / tot:.0f} % / wrong winding {100 * nbad / tot:.0f} % "
                     f"of traced length", fontsize=10)
    from matplotlib.lines import Line2D
    axs[-1].legend(handles=[Line2D([], [], color=OK, lw=2, label="fit on the human sheet it is numbered as"),
                            Line2D([], [], color=BAD, lw=2, label="fit on a different human sheet"),
                            Line2D([], [], color=UNK, lw=2, label="no human sheet within 4 vox"),
                            Line2D([], [], color="w", lw=1, alpha=0.5, label="human meshes")],
                   loc="lower right", fontsize=7, framealpha=0.7)
    ang = round(float(se.Umbilicus().angle(np.array([[a.z, cy, cx]], float))[0])) % 360
    fig.suptitle(f"{sname} CT slice z {a.z}, centre y {cy} x {cx}, {ang:.0f} deg around the umbilicus. "
                 f"Labels: winding number (fits mapped to the human numbering). {a.note}", fontsize=10.5)
    save_png(fig, a.out, 105)
    print(a.out, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in stats.items()})


if __name__ == "__main__":
    main()
