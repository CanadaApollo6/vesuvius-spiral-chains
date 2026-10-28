"""Side-by-side real vs sim slices at matched physical scale (eye test for realism).

  uv run python scripts/compare_real_sim.py data/sim/v1/s001_c000_000.zarr [more cells] -o data/scratch/real_vs_sim.png
Rows: real 7.91 | sim 7.91 | real 2.4 | sim 2.4. Field of view: 7.91 rows 1.5 mm, 2.4 rows 0.6 mm.
"""

import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def centre_slice(v, n, axis=0):
    c = np.array(v.shape) // 2
    sl = [slice(ci - n // 2, ci - n // 2 + n) for ci in c]
    sl[axis] = c[axis]
    return v[tuple(sl)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("-o", "--out", default=str(REPO / "data" / "scratch" / "real_vs_sim.png"))
    args = ap.parse_args()
    n_lo, n_hi = int(1500 / 7.91), int(600 / 2.399)
    crops = sorted(glob.glob(str(REPO / "data" / "xscan_crops" / "*.npz")))
    k = max(len(args.cells), 3)
    rows = []
    for i in range(k):
        stem = Path(crops[i * 3 % len(crops)]).stem
        ct = tifffile.imread(REPO / "data" / "eval" / "jeong200" / "imagesTr" / f"{stem}_0000.tif")
        hi = np.load(crops[i * 3 % len(crops)])["hi"]
        rows.append(("real", stem, centre_slice(ct, n_lo), centre_slice(hi, n_hi)))
    for c in args.cells:
        g = zarr.open_group(c, mode="r")
        lo, hi = g["ct_7p91"][:], g["ct_2p4"][:]
        rows.append(("sim", Path(c).stem, centre_slice(lo, min(n_lo, lo.shape[1])), centre_slice(hi, min(n_hi, hi.shape[1]))))
    real = [r for r in rows if r[0] == "real"]
    sim = [r for r in rows if r[0] == "sim"]
    fig, ax = plt.subplots(4, k, figsize=(4.5 * k, 18), squeeze=False)
    for j in range(k):
        for row, (src, idx, vmin, vmax) in enumerate(((real, 2, 40, 200), (sim, 2, 40, 200), (real, 3, 15, 220), (sim, 3, 15, 220))):
            a = ax[row, j]
            a.axis("off")
            if j < len(src):
                a.imshow(src[j][idx], cmap="gray", vmin=vmin, vmax=vmax)
                a.set_title(f"{src[j][0]} {'7.91' if idx == 2 else '2.4'} {src[j][1]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=60)
    print(args.out)


if __name__ == "__main__":
    main()
