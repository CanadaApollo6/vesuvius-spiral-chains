"""Certifier test on PHerc0139: do the graph's own signals flag its wrong pair deltas?

  uv run --with pandas --with scikit-learn python eval/p0139_certify.py data/preds/graphc_m7d300.csv
Features (eval/p0139_graph.py c_* columns, frozen grid point): c_res (max node mean L1 residual), c_frac (max
non-integrality of the node windings), c_wdiff_frac, c_deg (min degree), c_disagree (|graph - direct|), c_same,
c_ok. Error = |delta_graph - expected| >= 0.25. Logistic model fitted on the tune half (even site index), scored on
the report half: AUC and accuracy of the most confident fraction of pairs (coverage).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv(sys.argv[1])
g = [c for c in df.columns if c.startswith("delta_graph") and not c.endswith("_sp")][0]
df["err"] = (np.abs(df[g] - df.expected) >= 0.25).astype(int)
stems = sorted(df.stem.unique())
df["half"] = df.stem.map({s: k % 2 for k, s in enumerate(stems)})
F = ["c_res", "c_frac", "c_wdiff_frac", "c_deg", "c_disagree", "c_same", "c_ok"]
df["log_deg"] = np.log1p(df.c_deg)
X = F[:3] + ["log_deg"] + F[4:]
print(f"pairs {len(df)}, graph acc {100 * (1 - df.err.mean()):.2f}")
for f in X:
    print(f"  AUC {f:13s} {roc_auc_score(df.err, df[f]):.3f}")
tr, te = df[df.half == 0], df[df.half == 1]
m = LogisticRegression(max_iter=1000).fit(tr[X], tr.err)
p = m.predict_proba(te[X])[:, 1]
print(f"logistic (fit tune, report half): AUC {roc_auc_score(te.err, p):.3f}")
o = np.argsort(p)
for cov in (0.5, 0.7, 0.8, 0.9, 1.0):
    k = int(cov * len(o))
    print(f"  coverage {cov:.0%}: accuracy {100 * (1 - te.err.values[o[:k]].mean()):.2f}  "
          f"(errors caught in the flagged {1 - cov:.0%}: {100 * te.err.values[o[k:]].sum() / max(te.err.sum(), 1):.1f} %)")
