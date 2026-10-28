"""Test 13 confirmation table (notes/2026-10-26-anneal-prereg.md section 3d), fresh Paris4 z 10800-11800.
eval/paris4_label_audit.py cmd_rescore (unchanged) with a runtime TESTS["t13c"] entry on the band's own label audit
(data/scratch/label_audit/paris4/conf_z10800/audit.json, frozen in stage 1).
Primary set = planned k set 21-47,49-60,63-99 AND S1-trusted vertices (variant S1_trusted, own offsets refitted on that
mask). Switches = original labels, all vertices (variant original). Bases f7z_base / f7z_bases2 / f7z_bases3, oracle
f7z_oracle, method an_e03_c11s1 / an_e03_c11s2; all dumps in data/scratch/anneal/vertex.
PASS iff every one of the 6 method x base combos has S1 primary upper 95 % CI < 0 and original switch delta <= +5.
  uv run python data/scratch/anneal/conf_table_an.py --out data/scratch/anneal/conf_table_an.json
  (--methods / --bases / --oracle / --vd only for the stage-1 smoke test on dev-band-free dummy names)"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval"))
import paris4_label_audit as PLA  # noqa: E402

PRIMARY = "21-47,49-60,63-99"
GATE = dict(s1_primary_upper_ci_lt=0.0, switch_le=5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["an_e03_c11s1", "an_e03_c11s2"])
    ap.add_argument("--bases", nargs="+", default=["f7z_base", "f7z_bases2", "f7z_bases3"])
    ap.add_argument("--oracle", default="f7z_oracle")
    ap.add_argument("--vd", default="data/scratch/anneal/vertex")
    o = ap.parse_args()
    PLA.TESTS["t13c"] = {"band": "conf_z10800", "z": (10800, 11800), "primary": PRIMARY, "vd": o.vd,
                         "bases": o.bases, "methods": {m: None for m in o.methods}, "oracle": o.oracle or None,
                         "desc": "test 13 confirmation, fresh Paris4 z 10800-11800"}
    raw = o.out.with_name(o.out.stem + "_rescore.json")
    PLA.cmd_rescore(argparse.Namespace(test="t13c", band=None, out=str(raw)))
    R = json.loads(raw.read_text())
    S1, OR = R["variants"]["S1_trusted"], R["variants"]["original"]
    res = {"gate": GATE, "primary_k": PRIMARY, "S1_kept_share_primary": S1["kept_share_primary"],
           "S1_adj_pairs_kept_share": S1["adj_pairs_kept_share"], "combos": {}, "oracle": {}}
    ok_all = True
    print("\n== test 13 confirmation (primary: planned k AND S1 trusted; switch: original labels, all vertices)")
    for key in list(S1["combos"]) + list(S1["oracle_check"]):
        s = S1["combos"].get(key) or S1["oracle_check"][key]
        r = OR["combos"].get(key) or OR["oracle_check"][key]
        p, sw = s["primary_own_diff"], r["switch_per1000_diff"]
        items = {"s1_primary_upper_ci": bool(p["ci"][1] is not None and p["ci"][1] < GATE["s1_primary_upper_ci_lt"]),
                 "switch": bool(sw["est"] <= GATE["switch_le"])}
        row = {"s1_primary_diff": p, "switch_diff": sw, "orig_primary_diff": r["primary_own_diff"],
               "s1_switch_diff": s["switch_per1000_diff"], "s1_nonprimary_diff": s["nonprimary_own_diff"],
               "orig_nonprimary_diff": r["nonprimary_own_diff"], "items": items, "pass": all(items.values())}
        if key in S1["combos"]:
            res["combos"][key] = row
            ok_all &= row["pass"]
        else:
            res["oracle"][key] = row
        f = lambda x: f"{x['est']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
        print(f"{key:30s} S1 primary {f(p)} switch {sw['est']:+.2f} (orig primary {f(r['primary_own_diff'])}) "
              f"items {items}", flush=True)
    res["pass"] = bool(ok_all)
    print(f"confirmation PASS = {ok_all}")
    o.out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
