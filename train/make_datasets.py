"""Build nnU-Net v2 raw datasets (symlinks + spacing sidecars) under data/nnunet/nnUNet_raw.

  uv run python train/make_datasets.py real                    # Dataset101_DS059real: 662 buffered train patches
  uv run python train/make_datasets.py sim --cells data/sim/v1 # Dataset102_SimV1: sim cells (7.91 CT + surface)
  uv run python train/make_datasets.py eval                    # jeong200 images as imagesTs of Dataset101

Labels: 0 background, 1 surface. No spacing sidecars: data is isotropic, so nnU-Net's default spacing (1, 1, 1) is
equivalent (the image sidecar lookup by case name trips over the _0000 suffix anyway).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "nnunet" / "nnUNet_raw"


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def dataset_json(d: Path, n: int, name: str) -> None:
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "surface": 1}, "numTraining": n,
        "file_ending": ".tif", "overwrite_image_reader_writer": "Tiff3DIO", "name": name}, indent=1))


def real() -> None:
    src = REPO / "data" / "train" / "ds059"
    split = json.loads((src / "split.json").read_text())
    d = RAW / "Dataset101_DS059real"
    for sub in ("imagesTr", "labelsTr"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for s in split["train"]:
        link(src / "imagesTr" / f"{s}_0000.tif", d / "imagesTr" / f"{s}_0000.tif")
        link(src / "labelsTr" / f"{s}.tif", d / "labelsTr" / f"{s}.tif")
    dataset_json(d, len(split["train"]), "DS059 train pool minus jeong200 and touching patches")
    print(d, len(split["train"]))


def eval_() -> None:
    src = REPO / "data" / "eval" / "jeong200" / "imagesTr"
    d = RAW / "Dataset101_DS059real" / "imagesTs"
    d.mkdir(parents=True, exist_ok=True)
    fs = sorted(src.glob("*_0000.tif"))
    for f in fs:
        link(f, d / f.name)
    print(d, len(fs))


def sim(cells: Path, name: str) -> None:
    import zarr

    d = RAW / name
    for sub in ("imagesTr", "labelsTr"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for c in sorted(cells.glob("*.zarr")):
        g = zarr.open_group(str(c), mode="r")
        img = d / "imagesTr" / f"{c.stem}_0000.tif"
        lab = d / "labelsTr" / f"{c.stem}.tif"
        tifffile.imwrite(img, np.asarray(g["ct_7p91"][:], np.uint8), compression="zlib")
        tifffile.imwrite(lab, np.asarray(g["surface_7p91"][:], np.uint8), compression="zlib")
        n += 1
    dataset_json(d, n, f"synthetic cells from {cells}")
    print(d, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["real", "sim", "eval"])
    ap.add_argument("--cells", type=Path, default=REPO / "data" / "sim" / "v1")
    ap.add_argument("--name", default="Dataset102_SimV1")
    a = ap.parse_args()
    {"real": real, "eval": eval_, "sim": lambda: sim(a.cells, a.name)}[a.what]()


if __name__ == "__main__":
    main()
