#!/usr/bin/env python3
"""arm_table.py - markdown table of the annotation arms vs arm0 from eval/spiral_eval.py A/B outputs.

    python3 spiral/arm_table.py data/scratch/spiral_eval/arms/arm0_vs_arm{1..6}.json

Per arm: the arm's value (B) and the paired B - A difference with its 95 % CI, for primary-pair matched accuracy,
coverage, all-pairs accuracy, contact matched accuracy, seam-aware wrong-winding area %, seam-aware switches / 1000
and unmatched %. All in percent (switches per 1000 vertex edges). Lower is better for wrong-area / switches /
unmatched. A is arm0 in every file; its values are printed once from the first file.
"""
import json
import sys
from pathlib import Path

COLS = [  # (label, block, subset, key, higher_is_better)
    ("pair acc (matched)", "pairs_mix", "all", "acc_matched", True),
    ("coverage", "pairs_mix", "all", "coverage", True),
    ("acc_all", "pairs_mix", "all", "acc_all", True),
    ("contact acc", "pairs_mix", "contact", "acc_matched", True),
    ("wrong area % (seam)", "vertices", "all", "wrong_area_seam", False),
    ("switches/1000 (seam)", "vertices", "all", "switch_per1000_seam", False),
    ("unmatched %", "vertices", "all", "unmatched", False),
]


def get(d, block, subset, key):
    return d[block]["metrics"]["all"][subset][key]


def fmt_diff(x):
    lo, hi = x["ci"]
    sig = "" if (lo is None or (lo <= 0 <= hi)) else "*"
    return f"{x['est']:+.1f} [{lo:+.1f}, {hi:+.1f}]{sig}"


def main(paths):
    rows = []
    a_vals = None
    for p in paths:
        d = json.loads(Path(p).read_text())
        base, name = Path(p).stem.split("_vs_")[0], Path(p).stem.split("_vs_")[-1]
        if a_vals is None:
            a_name = base
            a_vals = [get(d, b, s, k)["est"] for _, b, s, k, _ in COLS]
        rows.append((name, [(get(d, b, s, k + "_b")["est"], get(d, b, s, k + "_diff")) for _, b, s, k, _ in COLS]))
    print("| arm | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    print(f"| {a_name} (A) | " + " | ".join(f"{v:.1f}" for v in a_vals) + " |")
    for name, vals in rows:
        print(f"| {name} | " + " | ".join(f"{b:.1f}<br>{fmt_diff(dd)}" for b, dd in vals) + " |")
    print(f"\nCells: B value, then B - {a_name} [95 % CI]; * = CI excludes 0.")


if __name__ == "__main__":
    main(sys.argv[1:])
