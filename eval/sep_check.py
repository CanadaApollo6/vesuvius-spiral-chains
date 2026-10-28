"""Do models keep touching sheets apart where the contact labels say so? On PHerc0172 contact crops (Dataset114,
label 3 = 1-vox separator between surface bands), report the share of separator voxels each model calls surface,
the share of other near-surface background it calls surface, and surface recall. Native 7.91 um grid.

  uv run --with nnunetv2 --with tifffile --with scipy python eval/sep_check.py --n 12 \
      m7=data/models/surface_m7_nnunet m7ct150=<trainer dir> ...
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "data/nnunet/nnUNet_raw/Dataset114_M7ContactSep"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=nnU-Net trainer dir")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--work", type=Path, default=REPO / "data/scratch/sepcheck")
    a = ap.parse_args()
    img = a.work / "img"
    img.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name[:-4] for p in (DS / "labelsTr").glob("p0172*_r0.tif"))[:: max(1, 320 // a.n)][: a.n]
    for s in stems:
        if not (img / f"{s}_0000.tif").exists():
            (img / f"{s}_0000.tif").symlink_to((DS / "imagesTr" / f"{s}_0000.tif").resolve())
    for m in a.models:
        name, d = m.split("=", 1)
        if not (a.work / name).exists():
            subprocess.run([sys.executable, str(REPO / "eval/infer_nnunet.py"), "--model", d, "--fold", "0", "--argmax",
                            "--images", str(img), "--out", str(a.work / name)], check=True, stdout=subprocess.DEVNULL)
    print(f"{len(stems)} PHerc0172 contact crops")
    print(f"{'model':12s} {'surface on separators':>22s} {'surface on near-bg':>19s} {'surface recall':>15s}")
    for m in a.models:
        name = m.split("=", 1)[0]
        acc = np.zeros(6)
        for s in stems:
            lab = tifffile.imread(DS / "labelsTr" / f"{s}.tif")
            sep, surf = lab == 3, lab == 1
            near = (lab == 0) & (ndi.distance_transform_edt(~surf) <= 2.5)
            p = tifffile.imread(a.work / name / f"{s}.tif") > 0
            acc += [(p & sep).sum(), sep.sum(), (p & near).sum(), near.sum(), (p & surf).sum(), surf.sum()]
        print(f"{name:12s} {100 * acc[0] / acc[1]:21.1f}% {100 * acc[2] / acc[3]:18.1f}% {100 * acc[4] / acc[5]:14.1f}%")


if __name__ == "__main__":
    main()
