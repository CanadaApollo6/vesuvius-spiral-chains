"""Ray-pair read-out split by human winding k, from eval/spiral_eval.py --dump-pairs CSVs (spiral_eval stays unchanged).

  # one fit (A) or a pair (A, B = the "_b" CSV spiral_eval writes for --fit-b); k split as the pre-registration (34)
  uv run python eval/pairs_ksplit.py --pairs data/scratch/spiral_eval/arms/x_pairs.csv [--pairs-b ..._b.csv] \
      --split 34 --kind ray --out data/scratch/spiral_eval/arms/x_pairs_k34.json

Rows are the dumped pairs with skip == 0 of --kind (as spiral_eval's `sel`). Subsets: all, inner (max(wa, wb) < split,
both points on k < split), outer (the rest). Metrics and bootstrap are spiral_eval's own (pair_metrics semantics:
acc_matched, coverage, acc_all, acc_matched_seam, acc_all_seam; Boot over site stems, seed 1, 1000 resamples, built
on all selected rows), so "all" reproduces spiral_eval's pairs_<kind> all/all numbers; B - A diffs when two CSVs.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spiral_eval import NOLAB, Boot, summ  # noqa: E402


def load(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    col = lambda k, t=int: np.array([t(r[k]) for r in rows])
    return {"row": col("row"), "stem": col("stem", str), "kind": col("kind", str), "wa": col("wa"), "wb": col("wb"),
            "expected": col("expected"), "skip": col("skip"), "jA": col("jA"), "jB": col("jB"),
            "kA_seam": col("kA_seam"), "kB_seam": col("kB_seam")}


def mets(P: dict) -> dict:
    exp = P["expected"]
    both = (P["jA"] != NOLAB) & (P["jB"] != NOLAB)
    corr = both & (np.abs(P["jB"] - P["jA"]) == exp)
    corr_s = both & (np.abs(P["kB_seam"] - P["kA_seam"]) == exp)
    one = np.ones(len(exp))
    return {"acc_matched": (corr, both), "coverage": (both, one), "acc_all": (corr, one),
            "acc_matched_seam": (corr_s, both), "acc_all_seam": (corr_s, one)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True, help="spiral_eval --dump-pairs CSV (fit A)")
    ap.add_argument("--pairs-b", type=Path, default=None, help="the _b CSV of the same run (fit B)")
    ap.add_argument("--split", type=int, default=34)
    ap.add_argument("--kind", default="ray")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    fits = [load(args.pairs)] + ([load(args.pairs_b)] if args.pairs_b else [])
    A = fits[0]
    for F in fits[1:]:
        assert np.array_equal(F["row"], A["row"]) and np.array_equal(F["expected"], A["expected"]), "CSVs differ in rows"
    sel = (A["kind"] == args.kind) & (A["skip"] == 0)
    boot = Boot(A["stem"][sel])
    u = np.zeros(len(sel), np.int64)
    u[sel] = boot.u
    boot.u = u
    kmax = np.maximum(A["wa"], A["wb"])
    subsets = {"all": sel, "inner": sel & (kmax < args.split), "outer": sel & (kmax >= args.split)}
    per = [mets(F) for F in fits]
    out = {"pairs": str(args.pairs), "pairs_b": str(args.pairs_b) if args.pairs_b else None, "kind": args.kind,
           "split": args.split, "subsets": {}}
    for sn, m in subsets.items():
        row = {"n": int(m.sum())}
        res_by = []
        for i, mt in enumerate(per):
            tag = "" if i == 0 else "_b"
            rr = {}
            for name, (num, den) in mt.items():
                est, res = boot.ratio(num, den, m)
                row[name + tag] = summ(est, res)
                rr[name] = (est, res)
            res_by.append(rr)
        if len(per) == 2:
            for name in per[0]:
                (ea, ra), (eb, rb) = res_by[0][name], res_by[1][name]
                row[name + "_diff"] = summ(eb - ea, rb - ra)
        out["subsets"][sn] = row
    txt = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(txt)
    for sn, r in out["subsets"].items():
        s = f"{sn:6s} n={r['n']:7d} acc_matched {r['acc_matched']} coverage {r['coverage']} acc_all {r['acc_all']}"
        if "acc_all_diff" in r:
            s += f"\n{'':15s}B-A acc_matched {r['acc_matched_diff']} coverage {r['coverage_diff']} acc_all {r['acc_all_diff']}"
        print(s)


if __name__ == "__main__":
    main()
