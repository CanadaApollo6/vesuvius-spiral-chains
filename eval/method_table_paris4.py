"""PHercParis4 decision table (notes/2026-10-23-third-region-prereg.md, final pre-registration), from per-vertex dumps
(release/fit_vertex_dump.py --scroll paris4) and single-fit spiral_eval --dump-pairs CSVs. No meshes needed.

  uv run python eval/method_table_paris4.py --out data/scratch/paris4_run/method_table.json

Order (as pre-registered):
1. Oracle informativeness, first: oracle - base on the primary endpoint for each base seed. If any upper CI >= 0 the
   region is uninformative and no claim is made (the method rows are still computed and reported).
2. Per (method seed, base seed), 2 x 3 = 6 combos:
   - primary: own-offset seam wrong area on the primary windings (k in PRIMARY_K, the CT-reliable and label-consistent
     set), method - base (eval/subset_wrong_area.sub_scores on that mask alone; paired spiral_eval.BootV, 30 deg x
     250 z, 1000 resamples, seed 1); two-sided bootstrap p = 2 min(P(d >= 0), P(d <= 0));
   - switch margin: seam switches per 1000 adjacent pairs, all vertices, method - base (as eval/method_table.py);
   - a combo passes if the primary CI upper < 0 and the switch diff point estimate <= +5.
3. One method arm, so there is no Holm step: PASS iff all 6 combos pass and p_arm = max p over the combos < 0.05.
Secondary (reported whatever the outcome): global (shared-offset) seam wrong area on all vertices, which may undercount
at tight packing; subset_wrong_area on every subset (consistent/inconsistent, primary/nonprimary), own and shared
offsets; ray pairs via eval/pairs_ksplit.py --primary-k; method - oracle; oracle pairs vs base seed 1.
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

VD = REPO / "data/scratch/paris4_run/vertex"
EV = REPO / "data/scratch/spiral_eval/arms"
SUBSETS = REPO / "data/scratch/label_audit/subsets_bandparis4.json"
Z0, Z1, MARGIN, ALPHA = 9000, 10500, 5.0, 0.05
PRIMARY_K = "21-32,34-60,62-87,90-101,106-108"  # frozen in the pre-registration (section 7.3)
METHOD = ["p4_ms1", "p4_ms2"]
BASES = ["p4_base", "p4_bases2", "p4_bases3"]
ORACLE = "p4_oracle"


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
    out = EV / f"{a}_vs_{b}_pairs_p4.json"
    subprocess.run([sys.executable, str(REPO / "eval/pairs_ksplit.py"), "--pairs", str(pa), "--pairs-b", str(pb),
                    "--split", "1000", "--primary-k", PRIMARY_K, "--kind", "ray", "--out", str(out)],
                   check=True, capture_output=True)
    s = json.loads(out.read_text())["subsets"]
    return {k: v for k, v in s.items() if k in ("all", "primary", "nonprimary")}


def subsets_ab(a: str, b: str) -> dict:
    sub = swa.run(argparse.Namespace(scroll="paris4", band=None, z0=Z0, z1=Z1, dump_a=VD / f"{a}.npz",
                                     dump_b=VD / f"{b}.npz", subsets=SUBSETS, split=None, primary_k=PRIMARY_K,
                                     relabel=None, relabel_key=None, excl_flag=False, own_offsets=True))
    return {"global_seam_wrong_area_all": sub["metrics"]["all"], "metrics": sub["metrics"],
            "metrics_own_offsets": sub["metrics_own_offsets"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    se.set_scroll("paris4")
    V = swa.load_ref(Z0, Z1)
    boot = se.BootV(V)
    ma = np.ones(len(V["adj"]), bool)
    prim = np.isin(V["k"], sorted(swa.parse_kset(PRIMARY_K)))
    names = METHOD + BASES + [ORACLE]
    D = {n: dict(np.load(VD / f"{n}.npz")) for n in names if (VD / f"{n}.npz").exists()}
    own = {n: swa.sub_scores(V, d["j"], {}, prim) for n, d in D.items()}
    res = {"config": {"primary_k": PRIMARY_K, "n_primary_vertices": int(prim.sum()), "margin_per1000": MARGIN,
                      "alpha": ALPHA, "method": METHOD, "bases": BASES, "oracle": ORACLE,
                      "holm": "none: one method arm", "missing": [n for n in names if n not in D]},
           "oracle_check": {}, "combos": {}, "secondary": {}}

    def endpoint(mn, bn):
        A, B = own[bn], own[mn]
        ea, ra = boot.ratio(A["matched"] & (A["res_s"] != 0), A["matched"], prim)
        eb, rb = boot.ratio(B["matched"] & (B["res_s"] != 0), B["matched"], prim)
        sa, sra = switch(V, D[bn], boot, ma)
        sb, srb = switch(V, D[mn], boot, ma)
        p = se.summ(eb - ea, rb - ra)
        sw = se.summ(sb - sa, srb - sra, 1.0)
        return {"primary_own_wrong_area_diff": p, "base": se.summ(ea, ra), "arm": se.summ(eb, rb),
                "p_two_sided": boot_p(rb - ra), "switch_per1000_seam_diff": sw,
                "pass": bool(p["ci"][1] is not None and p["ci"][1] < 0 and sw["est"] <= MARGIN)}

    # 1. oracle informativeness
    inf = True
    for bn in BASES:
        if bn in D and ORACLE in D:
            r = endpoint(ORACLE, bn)
            res["oracle_check"][f"{bn}_vs_{ORACLE}"] = r
            inf &= bool(r["primary_own_wrong_area_diff"]["ci"][1] is not None and r["primary_own_wrong_area_diff"]["ci"][1] < 0)
        else:
            inf = False
    res["informative"] = inf
    # 2. method combos
    ps, ok, n = [], True, 0
    for mn in METHOD:
        for bn in BASES:
            if mn not in D or bn not in D:
                ok = False
                continue
            row = endpoint(mn, bn)
            row["secondary"] = {**subsets_ab(bn, mn), "pairs": pairs(bn, mn)}
            res["combos"][f"{bn}_vs_{mn}"] = row
            ps.append(row["p_two_sided"])
            ok &= row["pass"]
            n += 1
    p_arm = max(ps) if ps else None
    allc = bool(ok and n == len(METHOD) * len(BASES))
    res["arm"] = {"combos": n, "all_combos_pass": allc, "p_arm": p_arm,
                  "PASS": bool(inf and allc and p_arm is not None and p_arm < ALPHA)}
    # secondaries: method - oracle, oracle pairs vs base seed 1
    for mn in METHOD:
        if mn in D and ORACLE in D:
            res["secondary"][f"{ORACLE}_vs_{mn}"] = {**subsets_ab(ORACLE, mn), "pairs": pairs(ORACLE, mn)}
    res["secondary"]["oracle_pairs_vs_p4_base"] = pairs("p4_base", ORACLE)
    args.out.write_text(json.dumps(res, indent=1))
    f = lambda r: f"{r['est']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if r["est"] is not None else "n/a"
    for k, r in res["oracle_check"].items():
        print(f"oracle {k:28s} primary {f(r['primary_own_wrong_area_diff'])}")
    print("informative:", inf)
    for k, r in res["combos"].items():
        g = r["secondary"]["global_seam_wrong_area_all"]["wrong_area_seam_diff"]
        print(f"{k:24s} base {f(r['base'])} arm {f(r['arm'])} primary {f(r['primary_own_wrong_area_diff'])} "
              f"p {r['p_two_sided']:.3f} switch {r['switch_per1000_seam_diff']['est']:.2f} pass {r['pass']} | global {f(g)}")
    print("arm:", res["arm"])


if __name__ == "__main__":
    main()
