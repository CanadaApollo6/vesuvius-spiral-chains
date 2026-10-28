"""Paired site-bootstrap comparison of graph CSVs (eval/p0139_graph.py outputs, same rows in the same order).

  uv run --with pandas python eval/p0139_paired.py data/preds/graph_split.csv data/preds/graph_m7d300_s11836.csv [...]

The first CSV is the reference. Prints all-87 accuracy with a site-bootstrap CI, the paired difference vs the
reference with its CI, and contact/open accuracy (column delta_graph_b12_r40_k30_t0 unless --col).
"""

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--col", default="delta_graph_b12_r40_k30_t0")
args = ap.parse_args()
base = pd.read_csv(args.csvs[0])
stems = sorted(base.stem.unique())
si = base.stem.map({s: i for i, s in enumerate(stems)}).values
bt = np.random.default_rng(1).integers(0, len(stems), (1000, len(stems)))
num = lambda c: np.bincount(si, c, len(stems))[bt].sum(1) / np.bincount(si, None, len(stems))[bt].sum(1)
c0 = ((base[args.col] - base.expected).abs() < 0.25).values.astype(float)
ct = base.contact.values.astype(bool)
for f in args.csvs:
    d = pd.read_csv(f)
    assert len(d) == len(base) and (d.stem.values == base.stem.values).all() and (d.expected.values == base.expected.values).all()
    c1 = ((d[args.col] - d.expected).abs() < 0.25).values.astype(float)
    dd = 100 * (num(c1) - num(c0))
    print(f"{f}: all {100 * c1.mean():.2f} CI {np.percentile(100 * num(c1), [2.5, 97.5]).round(1).tolist()} "
          f"vs ref {100 * (c1.mean() - c0.mean()):+.2f} CI {np.percentile(dd, [2.5, 97.5]).round(2).tolist()} "
          f"contact {100 * c1[ct].mean():.1f} open {100 * c1[~ct].mean():.1f}")
