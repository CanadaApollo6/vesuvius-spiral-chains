"""Overview figure of the PHerc1667 whole-scroll run (notes/2026-10-26-p1667-wholescroll-prereg.md section 2d.5).

Reads data/scratch/wholescroll/eval/pool_{s1,s2}_all.json (eval/wholescroll_pool.py --bands all) and writes
release/figures/p1667_wholescroll_overview.png:
  top: inner (k < 34) own-offset wrong area per band along L2 z, base vs M2, seed pair 1 (filled) and 2 (hollow),
       95 % block-bootstrap CI; bands seen before (test 8, test 2) shaded and labelled "not fresh".
  bottom: correctly numbered labelled area per band (cm^2, inner k < 34), base vs M2 (seed pair 1), with the labelled
       inner area as an outline.
  uv run python release/fig_p1667_wholescroll.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "data/scratch/wholescroll/eval"
OUT = REPO / "release/figures/p1667_wholescroll_overview.png"
BASE, M2 = "#eb6834", "#2a78d6"          # reference palette slots 2 / 1
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    R = {p: json.loads((EV / f"pool_{p}_all.json").read_text()) for p in ("s1", "s2")
         if (EV / f"pool_{p}_all.json").exists()}   # s2 may be missing while its arms are still queued
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]},
                                 facecolor=SURF)
    for a in (ax, bx):
        a.set_facecolor(SURF)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK2)
        a.tick_params(colors=INK2)
    s1 = R["s1"]["per_band"]
    for b in s1:
        z0, z1 = b["z"]
        if not b["fresh"]:
            for a in (ax, bx):
                a.axvspan(z0, z1, color="#efeeea", zorder=0, lw=0)
            ax.text((z0 + z1) / 2, 98, "not fresh", ha="center", va="top", fontsize=8, color=INK2)
    for pair, filled, dz in (("s1", True, -0.12), ("s2", False, 0.12)):
        for b in R.get(pair, {}).get("per_band", []):
            z0, z1 = b["z"]
            w = z1 - z0
            xc = (z0 + z1) / 2 + dz * w
            for key, col in (("a", BASE), ("b", M2)):
                r = b["inner"][key]
                if r["est"] is None:
                    continue
                ax.hlines(r["est"], z0 + 0.06 * w, z1 - 0.06 * w, color=col, lw=2, alpha=1 if filled else 0.45)
                ax.errorbar(xc, r["est"], yerr=[[r["est"] - r["ci"][0]], [r["ci"][1] - r["est"]]], fmt="o", ms=8,
                            color=col, mfc=col if filled else SURF, mec=col, mew=2, capsize=0, lw=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("inner (k < 34) wrong-winding area, %", color=INK)
    ax.set_title("PHerc1667, frozen M2 recipe band by band over the scroll (L2 z 400-9000)", color=INK, loc="left",
                 fontsize=12)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BASE, marker="o", ms=8, lw=2, label="base (no chains)"),
         Line2D([], [], color=M2, marker="o", ms=8, lw=2, label="M2 (automatic chains)"),
         Line2D([], [], color=INK2, marker="o", ms=8, lw=0, mfc=INK2, label="seed pair 1"),
         Line2D([], [], color=INK2, marker="o", ms=8, lw=0, mfc=SURF, mew=2, label="seed pair 2")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0, -0.0), ncol=4, frameon=False, fontsize=9,
              labelcolor=INK2, borderaxespad=-2.2)
    for b in s1:
        z0, z1 = b["z"]
        w = z1 - z0
        la = b.get("labelled_cm2", {}).get("inner")
        if la is None:
            continue
        ca, cb = b["correct_cm2_a"]["inner"], b["correct_cm2_b"]["inner"]
        bw = 0.36 * w
        bx.bar(z0 + 0.1 * w, la, width=0.8 * w, align="edge", color="none", edgecolor=INK2, lw=1, ls="--")
        bx.bar(z0 + 0.12 * w, ca, width=bw, align="edge", color=BASE, edgecolor=SURF, lw=2)
        bx.bar(z0 + 0.12 * w + bw, cb, width=bw, align="edge", color=M2, edgecolor=SURF, lw=2)
        bx.text(z0 + 0.12 * w + 1.5 * bw, cb + 3, f"{cb:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        bx.text(z0 + 0.12 * w + 0.5 * bw, ca + 3, f"{ca:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    bx.set_ylabel("correctly numbered\nlabelled area, cm²", color=INK)
    bx.set_xlabel("L2 z (9.596 µm slices)", color=INK)
    bx.set_title("dashed outline: labelled inner area (k 11-33); bars: base / M2, seed pair 1", loc="left",
                 fontsize=8, color=INK2)
    bx.set_xlim(300, 9100)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURF)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
