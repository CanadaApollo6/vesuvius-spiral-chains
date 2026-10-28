"""Preview a p0139 test-set site: CT slices with per-winding surface overlay, plus the registered 2.4 render.

  uv run python scripts/p0139_preview.py <stem> [--root data/eval/p0139_9um/_work/all] [--axis 0] [--at 96,64,128]
                                         [--tiles] [--out data/scratch/p0139_previews/<stem>.png]
--tiles marks raw mesh voxels > 2 vox (cyan) and > 4 vox (magenta) from material (removed from the labels).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--root", default=str(REPO / "data/eval/p0139_9um/_work/all"))
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--at", default="64,96,128")
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = Path(a.root)
    ct = tifffile.imread(r / "imagesTr" / f"{a.stem}_0000.tif")
    surf = tifffile.imread(r / "surfacesTr" / f"{a.stem}.tif")
    inst = tifffile.imread(r / "instancesTr" / f"{a.stem}.tif")
    rend = tifffile.imread(r / "render2p4" / f"{a.stem}.tif")
    site = json.loads((REPO / "data/eval/p0139_9um/_work/sites" / f"{a.stem}.json").read_text())
    thr = site["gates"]["otsu"]
    dmat = ndi.distance_transform_edt(ndi.gaussian_filter(ct.astype(np.float32), 0.7) < thr)
    raw = None
    if a.tiles and (r / "surfacesRaw" / f"{a.stem}.tif").exists():
        raw = tifffile.imread(r / "surfacesRaw" / f"{a.stem}.tif")
    ids = [int(x) for x in np.unique(surf) if x]
    cmap = plt.get_cmap("tab20")
    col = {i: cmap(k % 20) for k, i in enumerate(ids)}
    idx = [int(x) for x in a.at.split(",")]
    fig, axes = plt.subplots(len(idx), 3, figsize=(15, 5 * len(idx)))
    axes = np.atleast_2d(axes)
    lo, hi = np.percentile(ct[ct > 0], [1, 99.5])
    for row, k in zip(axes, idx):
        sl = [slice(None)] * 3
        sl[a.axis] = k
        sl = tuple(sl)
        row[0].imshow(ct[sl], cmap="gray", vmin=lo, vmax=hi)
        row[0].set_title(f"9.362 CT, axis {a.axis} = {k}")
        row[1].imshow(ct[sl], cmap="gray", vmin=lo, vmax=hi)
        s2 = surf[sl]
        for i in ids:
            yy, xx = np.nonzero(s2 == i)
            if len(yy):
                w = site["instances"][str(i)]["winding"]
                row[1].scatter(xx, yy, s=0.6, color=col[i], label=f"w{w}")
        if a.tiles:
            src = (raw if raw is not None else surf)[sl]
            for lo_d, colr in ((2.0, "cyan"), (4.0, "magenta")):
                yy, xx = np.nonzero((src > 0) & (dmat[sl] > lo_d))
                row[1].scatter(xx, yy, s=3, color=colr, marker="x")
        row[1].legend(fontsize=6, markerscale=8, loc="upper right")
        row[1].set_title("surfaces; x = raw mesh voxel > 2 vox (cyan) / > 4 vox (magenta) from material" if a.tiles else "1-vox mesh surfaces")
        b = inst[sl].astype(float)
        row[2].imshow(rend[sl], cmap="gray", vmin=lo, vmax=hi)
        row[2].contour(b > 0, levels=[0.5], colors="yellow", linewidths=0.4)
        row[2].set_title(f"2.399 L2 render, shifted (NCC {site['ncc']:.3f}); yellow = band")
        for ax in row:
            ax.axis("off")
    fig.suptitle(f"{a.stem}  cls={site.get('cls')}  passed={site.get('passed')}\n" + "; ".join(site.get("fail", []))[:300], fontsize=9)
    fig.tight_layout()
    out = Path(a.out) if a.out else REPO / "data/scratch/p0139_previews" / f"{a.stem}.png"
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    main()
