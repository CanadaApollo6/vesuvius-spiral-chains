"""Dataset112_M7Contact: fine-tuning set for the public m7 checkpoint (scrollprize/surface_m7_nnunet, 7.91 um grid).

  uv run python scripts/make_ds112.py [--n107 700] [--rep111 2]
  uv run python scripts/make_ds112.py --rep111 0 --name Dataset113_M7Ctrl   # control: same Dataset107 subset only

- Dataset111_Contact0172 (PHerc0172 traced windings at 7.91 um, separators at contacts, ignore = 2), each crop
  repeated --rep111 times (hardlinks, suffix _r<k>), spacing 1.0.
- A random subset (seed 0) of --n107 Dataset107_M7Distill crops (bucket m7 pseudo-labels on six ~9 um scrolls,
  labels 0/1), declared at spacing 9.362 / 7.91 so nnU-Net resamples them to m7's 7.91 um grid (the scale at which m7
  scores best on PHerc0139). 0500P2 crops are 9.362 um too; 1447 / 0800-family (8.64 um) use 8.64 / 7.91.
Tiff3DIO reads the spacing from <stem>.json next to the image and the label. No PHerc0139 / 0343P / 1667 / MANBp data.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/nnunet/nnUNet_raw"
UM = {"pherc1447": 8.64}  # others in Dataset107 are 9.362 um ESRF scans


def link(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.link(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n107", type=int, default=700)
    ap.add_argument("--rep111", type=int, default=2)
    ap.add_argument("--name", default="Dataset112_M7Contact")
    args = ap.parse_args()
    out = RAW / args.name
    for d in ("imagesTr", "labelsTr"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n = 0
    s111 = RAW / "Dataset111_Contact0172"
    for f in sorted((s111 / "labelsTr").glob("*.tif")):
        for k in range(args.rep111):
            stem = f"{f.stem}_r{k}"
            link(s111 / "imagesTr" / f"{f.stem}_0000.tif", out / "imagesTr" / f"{stem}_0000.tif")
            link(f, out / "labelsTr" / f"{stem}.tif")
            for d in ("imagesTr", "labelsTr"):
                (out / d / f"{stem}.json").write_text(json.dumps({"spacing": [1.0, 1.0, 1.0]}))
            n += 1
    n111 = n
    s107 = RAW / "Dataset107_M7Distill"
    stems = sorted(f.stem for f in (s107 / "labelsTr").glob("*.tif"))
    for stem in np.random.default_rng(0).choice(stems, min(args.n107, len(stems)), replace=False):
        sp = UM.get(stem.split("_")[0], 9.362) / 7.91
        link(s107 / "imagesTr" / f"{stem}_0000.tif", out / "imagesTr" / f"{stem}_0000.tif")
        link(s107 / "labelsTr" / f"{stem}.tif", out / "labelsTr" / f"{stem}.tif")
        for d in ("imagesTr", "labelsTr"):
            (out / d / f"{stem}.json").write_text(json.dumps({"spacing": [sp] * 3}))
        n += 1
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": n, "file_ending": ".tif", "overwrite_image_reader_writer": "Tiff3DIO",
        "name": f"m7 fine-tune: PHerc0172 contact crops x{args.rep111} ({n111}) + {n - n111} Dataset107 crops at 7.91 grid",
    }, indent=1))
    print(f"{out}: {n} cases ({n111} contact-set, {n - n111} Dataset107)")


if __name__ == "__main__":
    main()
