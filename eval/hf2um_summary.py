"""Summarise hf2um runs: official score terms and separation per model and volume, paired patch bootstrap vs a ref.

  uv run --with pandas python eval/hf2um_summary.py real350 m7d300 ...   (first tag = reference)
Reads data/preds/hf2um_<tag>_score.csv and hf2um_<tag>_sepd.csv (separation on labelsSep).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "data/preds"
tags = sys.argv[1:]
rng = np.random.default_rng(1)


def per_patch(tag):
    s = pd.read_csv(P / f"hf2um_{tag}_score.csv").set_index("stem")
    q = pd.read_csv(P / f"hf2um_{tag}_sepd.csv")
    q = q[q.cls == "compressed"].groupby("stem").agg(sep_n=("cc", "size"), sep_k=("cc", "sum"))
    return s.join(q, how="left").fillna({"sep_n": 0, "sep_k": 0})


def boot_diff(a, b, stat):
    idx = np.arange(len(a))
    d = [stat(a.iloc[i]) - stat(b.iloc[i]) for i in (rng.choice(idx, len(idx)) for _ in range(1000))]
    return np.percentile(d, [2.5, 97.5])


sep = lambda d: 100 * d.sep_k.sum() / max(d.sep_n.sum(), 1)
ref = per_patch(tags[0])
for t in tags[1:]:  # common patches only (m7 covers manbp and p0343p)
    ref = ref.loc[ref.index.intersection(per_patch(t).index)]
rows = []
for t in tags:
    d = per_patch(t).loc[ref.index]
    for vol in ["all", "manbp", "p1667", "p0343p"]:
        m = d.index.str.startswith(vol) if vol != "all" else np.ones(len(d), bool)
        r = ref[m]
        x = d[m]
        lo, hi = boot_diff(x, r, lambda z: z.score.mean()) if t != tags[0] else (0, 0)
        slo, shi = boot_diff(x, r, sep) if t != tags[0] else (0, 0)
        rows.append(dict(model=t, vol=vol, n=int(m.sum()), score=x.score.mean(), d_score=x.score.mean() - r.score.mean(),
                         d_ci=f"[{lo:+.3f},{hi:+.3f}]", topo=x.topo.mean(), sdice=x.surface_dice.mean(),
                         voi=x.voi_score.mean(), voi_merge=x.voi_merge.mean(), cc=x.cc_pred.mean(), fg=x.fg_frac_pred.mean(),
                         sep_comp=sep(x), d_sep_ci=f"[{slo:+.1f},{shi:+.1f}]"))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
