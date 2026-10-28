"""Dataset116_M7Contact3c: Dataset114 relabelled as a 3-class problem for a separator output channel:
bg 0 / surface 1 / separator 2 / ignore 3 (file values 2 -> 3, 3 -> 2). The seg head is re-initialised anyway
(load_pretrained_weights never transfers seg_layers), so m7's encoder/decoder carry over unchanged.

  uv run --with tifffile python scripts/make_ds116.py
"""
import json
from multiprocessing import Pool
from pathlib import Path
import os

import numpy as np
import tifffile

RAW = Path(__file__).resolve().parents[1] / "data/nnunet/nnUNet_raw"
S, D = RAW / "Dataset114_M7ContactSep", RAW / "Dataset116_M7Contact3c"


def work(f: Path):
    lab = tifffile.imread(f)
    out = lab.copy()
    out[lab == 2], out[lab == 3] = 3, 2
    tifffile.imwrite(D / "labelsTr" / f.name, out, compression="zlib")
    return int((out == 2).sum())


def main() -> None:
    for sub in ("imagesTr", "labelsTr"):
        (D / sub).mkdir(parents=True, exist_ok=True)
    for f in (S / "imagesTr").iterdir():
        if not (D / "imagesTr" / f.name).exists():
            os.link(f.resolve(), D / "imagesTr" / f.name)
    for f in (S / "labelsTr").glob("*.json"):
        if not (D / "labelsTr" / f.name).exists():
            os.link(f.resolve(), D / "labelsTr" / f.name)
    with Pool(10) as p:
        n = p.map(work, sorted((S / "labelsTr").glob("*.tif")))
    ds = json.loads((S / "dataset.json").read_text())
    ds.update(name=D.name, labels={"background": 0, "surface": 1, "separator": 2, "ignore": 3},
              description="Dataset114 as bg/surface/separator/ignore for a separator output channel")
    (D / "dataset.json").write_text(json.dumps(ds, indent=1))
    print(D, len(n), "labels; separator vox total", sum(n))


if __name__ == "__main__":
    main()
