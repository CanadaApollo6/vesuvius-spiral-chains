"""Test 13 confirmation stage 1: S1 trusted-vertex mask on the fresh band z 10800-11800 (label-only: reference labels +
the band's own label audit; no fit, no chains). Writes the mask and its shares for the stage-1 freeze.
  uv run python data/scratch/anneal/conf_trust.py --out data/scratch/anneal/freeze/conf_trust_z10800.npz"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval"))
import paris4_label_audit as PLA  # noqa: E402
import spiral_eval as se  # noqa: E402
import subset_wrong_area as swa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
o = ap.parse_args()
se.set_scroll("paris4")
V = swa.load_ref(10800, 11800)
A = PLA.load_audit("conf_z10800")
t1, nk = PLA.vertex_trust(V, A)
prim = np.isin(V["k"], sorted(swa.parse_kset("21-47,49-60,63-99")))
a0, a1 = V["adj"][:, 0], V["adj"][:, 1]
ks = sorted(set(V["k"][prim].tolist()))
perk = {int(k): round(float(t1[V["k"] == k].mean()), 4) for k in ks}
res = {"n_vertices": int(len(t1)), "n_primary_k_vertices": int(prim.sum()),
       "trusted_share_all": round(float(t1.mean()), 4),
       "primary_set_vertices": int((prim & t1).sum()),
       "primary_kept_share": round(float((prim & t1).sum() / max(prim.sum(), 1)), 4),
       "adj_pairs_kept_share": round(float((t1[a0] & t1[a1]).mean()), 4),
       "trusted_share_per_primary_k": perk}
np.savez_compressed(o.out, trusted=t1, primary=prim & t1)
o.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "trusted_share_per_primary_k"}))
lo = {k: v for k, v in perk.items() if v < 0.5}
print("primary k with trusted share < 0.5:", lo)
