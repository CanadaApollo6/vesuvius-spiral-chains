"""Side-by-side of surface masks on a compressed PHerc0139 site: CT, GT labels, m7 native grid, m7 at 1.18x, m7d300.

  uv run --with tifffile --with matplotlib python release/fig_compressed.py --stem p0139_c049_z4640_y3296_x4288 \
      --axis 0 --index 96 --out release/figures/compressed_c049.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "data/eval/p0139_9um"
PANELS = [("GT labels", EVAL / "labelsTr"),
          ("m7, native grid", REPO / "data/preds/xing_m7pub_s10_seg"),
          ("m7, 1.18x", REPO / "data/preds/xing_m7pub_s11836_seg"),
          ("m7d300, 1.18x", REPO / "data/preds/xing_m7d300_s11836_seg")]


def take(v: np.ndarray, axis: int, i: int) -> np.ndarray:
    return np.take(v, i, axis=axis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--index", type=int, default=96)
    ap.add_argument("--pair", default=None, help="za,ya,xa,zb,yb,xb: mark a scoring pair and zoom to it (axis 0)")
    ap.add_argument("--label", default="", help="caption for the pair row, e.g. true/predicted winding deltas")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    win = None
    if a.pair:
        za, ya, xa, zb, yb, xb = map(int, a.pair.split(","))
        a.axis, a.index = 0, round((za + zb) / 2)
        c, h = ((ya + yb) // 2, (xa + xb) // 2), max(abs(ya - yb), abs(xa - xb)) // 2 + 18
        win = (slice(max(c[0] - h, 0), c[0] + h), slice(max(c[1] - h, 0), c[1] + h))
        pts = np.array([[ya, xa], [yb, xb]]) - [win[0].start, win[1].start]
    ct = take(tifffile.imread(EVAL / "imagesTr" / f"{a.stem}_0000.tif"), a.axis, a.index)
    rows = 2 if win else 1
    fig, ax = plt.subplots(rows, 1 + len(PANELS), figsize=(3.2 * (1 + len(PANELS)), 3.4 * rows), squeeze=False)
    masks = [take(tifffile.imread(d / f"{a.stem}.tif"), a.axis, a.index) for _, d in PANELS]
    for r in range(rows):
        sl = win if r else (slice(None), slice(None))
        ax[r, 0].imshow(ct[sl], cmap="gray")
        ax[r, 0].set_title("CT (9.362 um)" if r == 0 else "zoom on pair A-B")
        for k, ((name, _), m) in enumerate(zip(PANELS, masks), 1):
            m = m[sl]
            ax[r, k].imshow(ct[sl], cmap="gray")
            ov = np.zeros((*m.shape, 4))
            ov[m == 1] = (1.0, 0.25, 0.1, 0.65)
            if name.startswith("GT"):
                ov[m == 2] = (0.2, 0.5, 1.0, 0.25)  # unlabelled (ignore)
            ax[r, k].imshow(ov)
            ax[r, k].set_title(name if r == 0 else "")
        if r:
            for x in ax[r]:
                x.plot(pts[:, 1], pts[:, 0], "-", color="cyan", lw=1)
                x.scatter(pts[:, 1], pts[:, 0], c="cyan", s=18)
                for p, t in zip(pts, "AB"):
                    x.text(p[1] + 1.5, p[0] - 1.5, t, color="cyan", fontsize=10, weight="bold")
            if a.label:
                fig.text(0.5, 0.01, a.label, ha="center", fontsize=10)
        elif win:
            ax[0, 0].add_patch(plt.Rectangle((win[1].start, win[0].start), win[1].stop - win[1].start,
                                             win[0].stop - win[0].start, fill=False, ec="cyan", lw=1))
    for x in ax.flat:
        x.set_axis_off()
    fig.suptitle(f"{a.stem}  axis {a.axis} slice {a.index}", fontsize=9)
    fig.tight_layout(rect=(0, 0.04 if win else 0, 1, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(a.out)


if __name__ == "__main__":
    main()
