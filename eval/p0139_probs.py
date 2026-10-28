"""Surface softmax (uint8, x255) on the 87 PHerc0139 sites, for probability ensembles.

  uv run --with nnunetv2 python eval/p0139_probs.py --model <dir> --spacing 1.1836 --out data/preds/prob_<tag>
Then: eval/p0139_probs.py --ens prob_a,prob_b --thr 0.5 --out data/preds/ens_<tag>_seg (mean prob > thr) and
eval/p0139_graph.py --seg data/preds/ens_<tag>_seg.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_sites import ROOT, load_sites  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path)
ap.add_argument("--fold", default="0")
ap.add_argument("--spacing", type=float, default=9.362 / 7.91)
ap.add_argument("--ens", default=None, help="comma-separated prob dirs to average")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
sites = load_sites(ROOT)
if args.ens:
    dirs = [Path(d) for d in args.ens.split(",")]
    for s in sites:
        p = np.mean([tifffile.imread(d / f"{s['stem']}.tif").astype(np.float32) for d in dirs], 0)
        tifffile.imwrite(args.out / f"{s['stem']}.tif", (p > 255 * args.thr).astype(np.uint8), compression="zlib")
else:
    from infer_nnunet import load_predictor
    pred = load_predictor(args.model, fold=args.fold)
    for s in sites:
        dst = args.out / f"{s['stem']}.tif"
        if dst.exists():
            continue
        ct = tifffile.imread(s["image"]).astype(np.float32)
        _, prob = pred.predict_single_npy_array(ct[None], {"spacing": [args.spacing] * 3}, None, None, True)
        tifffile.imwrite(dst, np.rint(255 * prob[1]).astype(np.uint8), compression="zlib")
        print(s["stem"], flush=True)
