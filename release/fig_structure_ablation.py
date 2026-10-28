"""Figure for the structure ablation (notes/2026-10-28-structure-ablation-prereg.md).

Left: per-ray chain pieces on a few rays of the Paris4 dev band (z 7000-8500, slice z 7008) for each chain set.
Right: share of the base -> oracle gain each chain set recovers (G, original labels, seed-1 base).
Label-free on the left; the right panel uses the frozen scorer's output (sa_table_decision.json).

    uv run python release/fig_structure_ablation.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SA = ROOT / "data/scratch/structure_ablation"
F6 = ROOT / "data/scratch/f6_screen/confirm/windings"
OUT = ROOT / "release/figures/structure_ablation_p4z7.png"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DARK, LIGHT = "#2a78d6", "#86b6ef"  # blue 450 / 250: alternate pieces on one ray

SETS = [  # (label, path, G key)
    ("oracle (labels)", F6 / "band_oracle_p4z7", None),
    ("A2: linked, thinned to our sheets", SA / "sets/sa_A2", "A2"),
    ("A3: our breaks, correct steps", SA / "sets/sa_A3", "A3"),
    ("A1: every sheet, cut short", SA / "sets/sa_A1", "A1"),
    ("ours (f6 chains)", F6 / "p4z7_f6", "f6z_ms1"),
]
Z, THETAS = 7008, [0, 15, 30, 45]
NAME = re.compile(r"_z(\d+)_t(\d+)_\d+$")


def pieces(path):
    """{theta: [array of xy points per piece]} for slice Z."""
    cols = json.load(open(path / "relative_windings.json"))["collections"]
    out = {}
    for c in cols.values():
        m = NAME.search(c["name"])
        if not m or int(m.group(1)) != Z or int(m.group(2)) not in THETAS:
            continue
        pts = [c["points"][k]["p"] for k in sorted(c["points"], key=int)]
        out.setdefault(int(m.group(2)), []).append(np.array(pts)[:, :2])
    return out


def main():
    data = [(lab, pieces(p), g) for lab, p, g in SETS]
    G = json.load(open(SA / "sa_table_decision.json"))["G"]

    fig = plt.figure(figsize=(13, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(len(THETAS), 2, width_ratios=[2.1, 1], wspace=0.62, hspace=0.55)

    for i, th in enumerate(THETAS):
        ax = fig.add_subplot(gs[i, 0], facecolor=SURFACE)
        # ray origin/direction from the oracle's points on this ray
        o = np.concatenate(data[0][1][th])
        d = (o[-1] - o[0]) / np.linalg.norm(o[-1] - o[0])
        for row, (lab, P, _) in enumerate(data):
            y = len(data) - 1 - row
            segs = sorted(P.get(th, []), key=lambda a: (a[0] - o[0]) @ d)
            for j, a in enumerate(segs):
                r = (a - o[0]) @ d
                col = DARK if j % 2 == 0 else LIGHT
                ax.plot(r, np.full_like(r, y), "-", color=col, lw=2, solid_capstyle="round")
                ax.plot(r, np.full_like(r, y), "o", color=col, ms=2.2)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([lab for lab, _, _ in data][::-1], fontsize=8, color=INK2)
        ax.set_ylim(-0.6, len(data) - 0.4)
        ax.tick_params(axis="x", labelsize=7, colors=INK2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_title(f"ray z {Z}, θ {th}°", fontsize=8, color=INK2, loc="left")
        if i == len(THETAS) - 1:
            ax.set_xlabel("distance along ray from innermost labelled sheet (9.6 µm vox)", fontsize=8, color=INK2)

    ax = fig.add_subplot(gs[:, 1], facecolor=SURFACE)
    short = ["oracle", "A2 linked", "A3 our breaks,\ncorrect steps", "A1 every sheet,\ncut short", "ours (f6)"]
    rows = [(short[0], 1.0)] + [(sh, G[g]) for sh, (_, _, g) in zip(short[1:], SETS[1:])]
    labels, vals = zip(*rows[::-1])
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=DARK, height=0.55)
    for y, v in zip(yy, vals):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.axvline(1.0, color=INK2, lw=1, ls=":")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("share of the oracle's gain over the base fitter\n(inner wrong area, 1.0 = oracle)", fontsize=8, color=INK2)
    ax.tick_params(axis="x", labelsize=7, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    fig.suptitle("What makes oracle chains work? Linkage, not completeness (Paris4, z 7000-8500, one seed per arm)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Left: chain pieces on 4 rays; alternating shades mark piece breaks; dots are points. "
             "Right: Villa spiral fitter, same recipe, only the chain set changes.", fontsize=7.5, color=INK2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
