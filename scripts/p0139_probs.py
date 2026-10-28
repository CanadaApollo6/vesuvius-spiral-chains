"""Cache surface probabilities (float16, class 1) of an nnU-Net model on the PHerc0139 9.362 um crops.

  uv run --with nnunetv2 python scripts/p0139_probs.py --model <trainer dir> --spacing 1.1836 --out data/preds/prob_real350
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=9.362 / 7.91)
    ap.add_argument("--fold", default="0")
    args = ap.parse_args()
    from infer_nnunet import load_predictor
    from p0139_sites import load_sites

    pred = load_predictor(args.model, fold=args.fold)
    args.out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(load_sites()):
        f = args.out / f"{s['stem']}.npy"
        if f.exists():
            continue
        ct = tifffile.imread(s["image"]).astype(np.float32)
        _, prob = pred.predict_single_npy_array(ct[None], {"spacing": [args.spacing] * 3}, None, None, True)
        np.save(f, prob[1].astype(np.float16))
        print(f"[{i + 1}] {s['stem']} {prob.shape}", flush=True)


if __name__ == "__main__":
    main()
