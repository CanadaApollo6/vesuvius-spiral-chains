"""PHerc1667 method-arm decision table (notes/2026-10-22-confirmation-region.md section 11), from per-vertex dumps
(release/fit_vertex_dump.py) and single-fit spiral_eval --dump-pairs CSVs. No meshes needed.

  uv run python eval/method_table.py --out data/scratch/confirm_region/p1667/method_table.json

Per (method arm seed, baseline seed):
- primary: inner (k < 34) own-offset seam wrong area, method - base (eval/subset_wrong_area.sub_scores, paired
  spiral_eval.BootV, 30 deg x 250 z, 1000 resamples, seed 1); two-sided bootstrap p = 2 min(P(d >= 0), P(d <= 0));
- switch margin: seam switches per 1000 adjacent pairs, all vertices, method - base. This is spiral_eval's
  vertices.metrics.all.all.switch_per1000_seam_diff computed from the dumps (same res_s, adjacency and BootV);
- combo passes if the primary CI upper < 0 and the switch diff point estimate <= +5.
Per method arm: p_arm = max p over its combos (intersection-union); Holm over the method arms at alpha 0.05.
An arm passes only if Holm rejects for it AND every combo passes.
Secondary: subset_wrong_area metrics (all subsets, own and shared offsets) and eval/pairs_ksplit read-outs per combo,
method - oracle, and the oracle's own pair read-out vs the seed-1 baseline.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402
import subset_wrong_area as swa  # noqa: E402

VD = REPO / "data/scratch/confirm_region/p1667/vertex"
EV = REPO / "data/scratch/spiral_eval/arms"
SUBSETS = REPO / "data/scratch/label_audit/subsets_bandp1667.json"
Z0, Z1, SPLIT, MARGIN, ALPHA = 6500, 8000, 34, 5.0, 0.05
ARMS = {"M1": ["c1667_m1s1", "c1667_m1s2"], "M2": ["c1667_m2s1", "c1667_m2s2"]}
BASES = ["c1667_base", "c1667_bases2", "c1667_bases3"]
ORACLE = "c1667_oracle"


def boot_p(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    return float(min(1.0, 2 * min((r >= 0).mean(), (r <= 0).mean())))


def switch(V, D, boot, ma):
    a, b = V["adj"][:, 0], V["adj"][:, 1]
    m = D["matched"]
    pm = m[a] & m[b]
    sw = pm & (D["res_s"][a] != D["res_s"][b])
    return boot.ratio_adj(sw * 1000.0, pm, ma)


def pairs(a: str, b: str) -> dict | None:
    pa, pb = EV / f"{a}_pairs.csv", EV / f"{b}_pairs.csv"
    if not (pa.exists() and pb.exists()):
        return None
    out = EV / f"{a}_vs_{b}_pairs_k{SPLIT}.json"
    subprocess.run([sys.executable, str(REPO / "eval/pairs_ksplit.py"), "--pairs", str(pa), "--pairs-b", str(pb),
                    "--split", str(SPLIT), "--kind", "ray", "--out", str(out)], check=True, capture_output=True)
    return json.loads(out.read_text())["subsets"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--validate", action="store_true", help="only check the dump-based switch diff vs a paired json")
    args = ap.parse_args()
    se.set_scroll("p1667")
    V = swa.load_ref(Z0, Z1)
    boot = se.BootV(V)
    ma = np.ones(len(V["adj"]), bool)
    inner = V["k"] < SPLIT
    D = {n: dict(np.load(VD / f"{n}.npz")) for n in sum(ARMS.values(), []) + BASES + [ORACLE] if (VD / f"{n}.npz").exists()}
    if args.validate:
        ea, ra = switch(V, D["c1667_base"], boot, ma)
        eb, rb = switch(V, D[ORACLE], boot, ma)
        ref = json.loads((EV / "c1667_base_vs_c1667_oracle.json").read_text())["vertices"]["metrics"]["all"]["all"]
        print("dumps:", se.summ(eb - ea, rb - ra, 1.0), "spiral_eval:", ref["switch_per1000_seam_diff"])
        print("A:", se.summ(ea, ra, 1.0), ref["switch_per1000_seam"], "B:", se.summ(eb, rb, 1.0), ref["switch_per1000_seam_b"])
        return
    own = {n: swa.sub_scores(V, d["j"], {}, inner) for n, d in D.items()}
    res = {"config": {"split": SPLIT, "margin_per1000": MARGIN, "alpha": ALPHA, "arms": ARMS, "bases": BASES,
                      "missing": [n for n in sum(ARMS.values(), []) + BASES if n not in D]}, "combos": {}, "arms": {}}

    def endpoint(mn, bn):
        A, B = own[bn], own[mn]
        ea, ra = boot.ratio(A["matched"] & (A["res_s"] != 0), A["matched"], inner)
        eb, rb = boot.ratio(B["matched"] & (B["res_s"] != 0), B["matched"], inner)
        sa, sra = switch(V, D[bn], boot, ma)
        sb, srb = switch(V, D[mn], boot, ma)
        prim = se.summ(eb - ea, rb - ra)
        sw = se.summ(sb - sa, srb - sra, 1.0)
        return {"primary_inner_own_wrong_area_diff": prim, "p_two_sided": boot_p(rb - ra), "switch_per1000_seam_diff": sw,
                "pass": bool(prim["ci"][1] is not None and prim["ci"][1] < 0 and sw["est"] <= MARGIN)}

    for arm, seeds in ARMS.items():
        ps, ok, n = [], True, 0
        for mn in seeds:
            for bn in BASES:
                key = f"{bn}_vs_{mn}"
                if mn not in D or bn not in D:
                    ok = False
                    continue
                row = endpoint(mn, bn)
                sub = swa.run(argparse.Namespace(scroll="p1667", band=None, z0=Z0, z1=Z1, dump_a=VD / f"{bn}.npz",
                                                 dump_b=VD / f"{mn}.npz", subsets=SUBSETS, split=SPLIT, relabel=None,
                                                 relabel_key=None, excl_flag=False, own_offsets=True))
                row["secondary"] = {"subset_wrong_area": {"metrics": sub["metrics"], "metrics_own_offsets": sub["metrics_own_offsets"]},
                                    "pairs_ksplit": pairs(bn, mn)}
                res["combos"][key] = row
                ps.append(row["p_two_sided"])
                ok &= row["pass"]
                n += 1
        res["arms"][arm] = {"combos": n, "all_combos_pass": bool(ok and n == len(seeds) * len(BASES)),
                            "p_arm": max(ps) if ps else None}
        for mn in seeds:  # method - oracle (secondary)
            if mn in D and ORACLE in D:
                sub = swa.run(argparse.Namespace(scroll="p1667", band=None, z0=Z0, z1=Z1, dump_a=VD / f"{ORACLE}.npz",
                                                 dump_b=VD / f"{mn}.npz", subsets=SUBSETS, split=SPLIT, relabel=None,
                                                 relabel_key=None, excl_flag=False, own_offsets=True))
                res["combos"][f"{ORACLE}_vs_{mn}"] = {"secondary": {"subset_wrong_area": {
                    "metrics_own_offsets": sub["metrics_own_offsets"]}, "pairs_ksplit": pairs(ORACLE, mn)}}
    # Holm over the method arms
    order = sorted([a for a in res["arms"] if res["arms"][a]["p_arm"] is not None], key=lambda a: res["arms"][a]["p_arm"])
    alive = True
    for i, a in enumerate(order):
        thr = ALPHA / (len(order) - i)
        rej = alive and res["arms"][a]["p_arm"] <= thr
        alive = rej
        res["arms"][a] |= {"holm_threshold": thr, "holm_reject": bool(rej),
                           "PASS": bool(rej and res["arms"][a]["all_combos_pass"])}
    res["oracle_pairs_vs_c1667_base"] = pairs("c1667_base", ORACLE)
    args.out.write_text(json.dumps(res, indent=1))
    for k, r in res["combos"].items():
        if "primary_inner_own_wrong_area_diff" in r:
            print(f"{k:28s} primary {r['primary_inner_own_wrong_area_diff']} p {r['p_two_sided']:.3f} "
                  f"switch {r['switch_per1000_seam_diff']['est']} pass {r['pass']}")
    for a, r in res["arms"].items():
        print(a, r)


if __name__ == "__main__":
    main()
