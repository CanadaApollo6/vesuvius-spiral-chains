"""Anneal DEV table (test 13, notes/2026-10-26-anneal-prereg.md).
Primary and switches on the ORIGINAL labels: data/scratch/f7_switch/dev_table_f7.py unchanged (own-offset seam wrong
area on PRIMARY_K 13,16-32,34-44,47-61,63-108, paired BootV, switches per 1000 adjacent pairs, against each of
f6z_base / f6z_bases2 / f6z_bases3). Gate item 4 = nonprimary own-offset delta on the label audit's S1 trusted
vertices (eval/paris4_label_audit.py cmd_rescore, unchanged, audit data/scratch/label_audit/paris4/z7000/audit.json),
via a runtime TESTS["t13"] entry. The original-label nonprimary is reported next to it (not gated).
  uv run python data/scratch/anneal/dev_table_an.py an_f03z7 an_e03z7 --evidence cl_annz7 --refs \
      --out data/scratch/anneal/dev_table_an_z7.json
Gate (each base seed): primary upper CI < 0, primary est <= -5.0, switch delta <= +2.0 /1000 (original labels),
S1 nonprimary delta <= +3.0. Evidence rows (cl_annz7) and refs are scored the same way but cannot pass."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data/scratch/f7_switch"))
sys.path.insert(0, str(REPO / "eval"))
import dev_table_f7 as T  # noqa: E402
import paris4_label_audit as PLA  # noqa: E402

VDIRS = [REPO / "data/scratch/anneal/vertex", REPO / "data/scratch/chain_loss/vertex"]
GATE = dict(upper_ci_lt=0.0, est_le=-5.0, switch_le=2.0, nonprimary_s1_le=3.0)
T.GATE.clear()
T.GATE.update(dict(upper_ci_lt=0.0, est_le=-5.0, switch_le=2.0, nonprimary_le=3.0))   # its 'nonprimary' = original labels
_orig_dump_path = T.dump_path


def dump_path(n, B):
    for d in VDIRS:
        if (d / f"{n}.npz").exists():
            return d / f"{n}.npz"
    return _orig_dump_path(n, B)


T.dump_path = dump_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    o = ap.parse_args()
    rows = o.arms + o.evidence
    assert all(r.endswith("z7") for r in rows)
    # 1) original labels (primary, switches, original nonprimary)
    t_out = o.out.with_name(o.out.stem + "_orig.json")
    sys.argv = ["dev_table_f7.py", *rows, "--out", str(t_out)] + (["--refs"] if o.refs else [])
    T.main()
    orig = json.loads(t_out.read_text())
    # 2) S1 trusted (label audit), same bases and primary set
    B = T.BANDS["z7"]
    meth = {}
    for r in rows + ([x for x in B["refs"] if x != "f6z_oracle"] if o.refs else []):
        p = dump_path(r, B)
        meth[r] = str(p.parent.relative_to(REPO))
    PLA.TESTS["t13"] = {"band": "z7000", "z": (B["z0"], B["z1"]), "primary": B["prim"],
                        "vd": str(B["vd"].relative_to(REPO)), "bases": list(B["bases"]), "methods": meth,
                        "oracle": "f6z_oracle" if o.refs else None, "desc": "test 13 anneal dev (S1 gate item 4)"}
    s_out = o.out.with_name(o.out.stem + "_audit.json")
    PLA.cmd_rescore(argparse.Namespace(test="t13", band=None, out=str(s_out)))
    aud = json.loads(s_out.read_text())
    S1 = aud["variants"]["S1_trusted"]
    allc = {**S1["combos"], **S1["oracle_check"]}
    res = {"gate": GATE, "arms": o.arms, "evidence": o.evidence, "orig_json": str(t_out), "audit_json": str(s_out),
           "S1_kept_share_primary": S1["kept_share_primary"], "S1_adj_pairs_kept_share": S1["adj_pairs_kept_share"],
           "rows": {}}
    print("\n== test 13 dev table (primary / switch on original labels; nonprimary S1 trusted [gated] and original)")
    for r in [x for x in orig if x != "gate"]:
        ok_all = True
        res["rows"][r] = {"role": "arm" if r in o.arms else ("evidence" if r in o.evidence else "ref"), "combos": {}}
        for bn in B["bases"]:
            c = orig[r]["combos"][bn]
            s1 = allc[f"{r}_vs_{bn}"]
            p, sw, npo, nps = c["primary_diff"], c["switch_diff"], c["nonprimary_own_diff"], s1["nonprimary_own_diff"]
            items = {"upper_ci": bool(p["ci"][1] is not None and p["ci"][1] < GATE["upper_ci_lt"]),
                     "estimate": bool(p["est"] <= GATE["est_le"]),
                     "switch": bool(sw["est"] <= GATE["switch_le"]),
                     "nonprimary_s1": bool(nps["est"] is not None and nps["est"] <= GATE["nonprimary_s1_le"])}
            ok = all(items.values())
            ok_all &= ok
            res["rows"][r]["combos"][bn] = {"primary_diff": p, "switch_diff": sw, "nonprimary_s1_diff": nps,
                                            "nonprimary_original_diff": npo, "primary_s1_diff": s1["primary_own_diff"],
                                            "switch_s1_diff": s1["switch_per1000_diff"], "items": items, "pass": ok}
            f = lambda x: f"{x['est']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
            print(f"{r:12s} - {bn:10s} primary {f(p)} switch {sw['est']:+.2f} nonprim S1 {f(nps)} "
                  f"(orig {npo['est']:+.2f}) items {items} pass {ok}", flush=True)
        res["rows"][r]["band_pass"] = bool(ok_all) if r in o.arms else None
        res["rows"][r]["would_pass_new_item"] = bool(ok_all)
        print(f"{r}: {'GATE PASS' if (ok_all and r in o.arms) else ('gate fail' if r in o.arms else f'not gated (would pass: {ok_all})')}")
    o.out.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {o.out}")


if __name__ == "__main__":
    main()
