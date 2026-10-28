"""Dataset114_M7ContactSep: Dataset112 with contact separators marked as label 3 (trained as background with extra
weight by train/run.py --sep-weight). Separators = background voxels in a gap of <= --gap voxels between surface voxels
on opposite sides along any of 13 directions (scripts/contact_set.py's 1-vox separators and other thin gaps).

  uv run --with tifffile --with scipy python scripts/make_ds114.py
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/nnunet/nnUNet_raw"
DIRS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]


def shift(a: np.ndarray, d) -> np.ndarray:
    """out[x] = a[x + d], False outside."""
    out = np.zeros_like(a)
    src = tuple(slice(max(k, 0), a.shape[i] + min(k, 0)) for i, k in enumerate(d))
    dst = tuple(slice(max(-k, 0), a.shape[i] + min(-k, 0)) for i, k in enumerate(d))
    out[dst] = a[src]
    return out


def separators(lab: np.ndarray, gap: int) -> np.ndarray:
    surf = lab == 1
    sep = np.zeros_like(surf)
    for d in DIRS:
        d = np.array(d)
        for a in range(1, gap + 1):  # surface a steps forward, b steps back, a + b - 1 <= gap background voxels
            fa = shift(surf, a * d)
            for b in range(1, gap + 2 - a):
                sep |= fa & shift(surf, -b * d)
    return sep & (lab == 0)


def work(job):
    src, dst, gap = job
    lab = tifffile.imread(src)
    s = separators(lab, gap)
    lab = lab.copy()
    lab[s] = 3
    tifffile.imwrite(dst, lab, compression="zlib")
    return dst.name, int(s.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Dataset112_M7Contact")
    ap.add_argument("--name", default="Dataset114_M7ContactSep")
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S, D = RAW / a.src, RAW / a.name
    for sub in ("imagesTr", "labelsTr"):
        (D / sub).mkdir(parents=True, exist_ok=True)
    for f in (S / "imagesTr").iterdir():
        if not (D / "imagesTr" / f.name).exists():
            os.link(f.resolve(), D / "imagesTr" / f.name)
    jobs = []
    for f in sorted((S / "labelsTr").iterdir()):
        t = D / "labelsTr" / f.name
        if t.exists():
            continue
        if f.name.startswith("p0172") and f.suffix == ".tif":
            jobs.append((f.resolve(), t, a.gap))
        else:
            os.link(f.resolve(), t)
    with Pool(a.workers) as p:
        res = p.map(work, jobs)
    ds = json.loads((S / "dataset.json").read_text())
    ds["name"] = a.name
    ds["description"] = (ds.get("description", "") + f" | label 3 in the file = separator (gap <= {a.gap} vox), "
                         "trained as background with extra weight; dataset.json keeps 3 undeclared on purpose")
    (D / "dataset.json").write_text(json.dumps(ds, indent=1))
    n = np.array([r[1] for r in res])
    print(f"{D}: {len(res)} contact labels rewritten; separator vox per crop median {np.median(n):.0f}, "
          f"total {n.sum()}, crops with none {(n == 0).sum()}")


if __name__ == "__main__":
    main()
