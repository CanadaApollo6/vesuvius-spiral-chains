"""Score predicted surface masks against GT patches with the official Kaggle metric.

Runs inside the isolated metric env (Python 3.11, Kaggle-pinned wheels):
  data/metric/.venv/bin/python eval/score.py --gt-dir data/eval/jeong200/labelsTr \
      --pred-dir <dir of {stem}.tif masks> --out results.csv

Reports the official leaderboard blend and every term separately. The blend
barely responds to fused sheets (Jeong, villa #191), so merge-sensitive
columns are kept next to it: VOI split/merge and 26-connected component
counts for pred and GT.
"""

import argparse
import csv
import sys
from pathlib import Path

import cc3d
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data/metric/topological-metrics-kaggle/src"))
from topometrics.leaderboard import compute_leaderboard_score  # noqa: E402

# Parameters as used for the Kaggle surface-detection leaderboard
# (topometrics README; same values in other public harnesses).
OFFICIAL = dict(
    dims=(0, 1, 2),
    spacing=(1.0, 1.0, 1.0),
    surface_tolerance=2.0,
    voi_connectivity=26,
    voi_transform="one_over_one_plus",
    voi_alpha=0.3,
    combine_weights=(0.3, 0.35, 0.35),
    fg_threshold=None,
    ignore_label=2,
    ignore_mask=None,
)

FIELDS = ["stem", "score", "topo", "surface_dice", "voi_score", "voi_split", "voi_merge",
          "cc_pred", "cc_gt", "fg_frac_pred", "fg_frac_gt"]


def n_components(mask: np.ndarray) -> int:
    return int(cc3d.connected_components(mask.astype(np.uint8), connectivity=26).max())


def score_pair(pred: np.ndarray, gt: np.ndarray) -> dict:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch pred={pred.shape} gt={gt.shape}")
    rep = compute_leaderboard_score(predictions=pred.astype(np.uint8), labels=gt.astype(np.uint8), **OFFICIAL)
    pred_fg, gt_fg = pred != 0, gt == 1
    return {
        "score": rep.score,
        "topo": rep.topo.toposcore,
        "surface_dice": rep.surface_dice,
        "voi_score": rep.voi.voi_score,
        "voi_split": rep.voi.voi_split,
        "voi_merge": rep.voi.voi_merge,
        "cc_pred": n_components(pred_fg),
        "cc_gt": n_components(gt_fg),
        "fg_frac_pred": float(pred_fg.mean()),
        "fg_frac_gt": float(gt_fg.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--pred-dir", type=Path, required=True, help="masks named {stem}.tif, nonzero = surface")
    p.add_argument("--out", type=Path, required=True, help="per-patch CSV")
    p.add_argument("--limit", type=int, default=0, help="score only the first N patches (0 = all)")
    args = p.parse_args()

    stems = sorted(f.stem for f in args.pred_dir.glob("*.tif"))
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no .tif predictions in {args.pred_dir}")

    rows = []
    for i, stem in enumerate(stems, 1):
        gt = tifffile.imread(args.gt_dir / f"{stem}.tif")
        pred = tifffile.imread(args.pred_dir / f"{stem}.tif")
        row = {"stem": stem, **score_pair(pred, gt)}
        rows.append(row)
        print(f"[{i}/{len(stems)}] {stem} score={row['score']:.4f} topo={row['topo']:.4f} "
              f"sd={row['surface_dice']:.4f} voi={row['voi_score']:.4f} cc {row['cc_pred']}/{row['cc_gt']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in FIELDS[1:7]}
    print("mean " + " ".join(f"{k}={v:.4f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
