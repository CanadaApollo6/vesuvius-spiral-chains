"""Exact winding accuracy by PHerc0139 site class and at contacts, per mask source (graph solver, all 87 sites).

  uv run --with pandas --with matplotlib python release/fig_classes.py --out release/figures/winding_by_class.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = [("GT labels", "graph_gt.csv"), ("m7, native grid", "graph_m7pub_s10.csv"),
        ("m7, 1.18x", "graph_m7pub_s11836.csv"), ("m7d300, 1.18x", "graph_m7d300_s11836.csv")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=REPO / "data/preds")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    groups = ["all", "compressed", "mixed", "wide", "contact"]
    acc = {}
    for name, f in RUNS:
        d = pd.read_csv(a.preds / f)
        col = next(c for c in d.columns if c.startswith("delta_graph") and not c.endswith("_sp"))
        ok = (d[col] - d.expected).abs() < 0.25
        acc[name] = [100 * ok.mean()] + [100 * ok[d.site_cls == g].mean() for g in groups[1:4]] + \
                    [100 * ok[d.contact].mean()]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(RUNS)
    for k, (name, v) in enumerate(acc.items()):
        x = np.arange(len(groups)) + (k - (len(RUNS) - 1) / 2) * w
        ax.bar(x, v, w, label=name)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.6, f"{vi:.0f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel("exact winding delta (%)")
    ax.set_ylim(40, 109)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("PHerc0139 winding pairs (87 sites, 24,972 pairs): binary masks fall to ~50 % at contacts", fontsize=9)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(a.out, {k: [round(x, 1) for x in v] for k, v in acc.items()})


if __name__ == "__main__":
    main()
