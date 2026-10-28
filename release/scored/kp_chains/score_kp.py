"""KP-chains scorer (notes/2026-10-29-kp-chains-prereg.md section 4). Runs the two FROZEN scorers unchanged, with
only their dump look-ups redirected (module attributes set from here; eval/kelly_h2h.py and eval/wholescroll_pool.py
are not edited):

1. eval/kelly_h2h.py (sha bdfb5641...): kh.ours_files gains the KP arms ("kp", "kpm" -> data/scratch/kp_chains/vertex/
   kp_z<z0>.npz, kpm_z<z0>.npz) next to base and m2 (m2s2 dropped); kh.PAIRS gains (arm, k_all_comp), (arm, m2) and
   (arm, k_good_comp); kh.BANDS is restricted to the stage's bands (all flagged "fresh", so both pools = the stage
   pool). Variant "primary" (+ "lenient", "tol2" as sensitivity).
2. eval/wholescroll_pool.py (sha dfdb1f57...; full shas checked at start): wp.arm_files(z0, "s1") returns (M2 s1, arm) so band_parts' a = M2 and
   b = the KP arm; pooled() over the stage's bands: inner own-offset seam wrong area, switches, outer, unmatched.

  flock /tmp/vss_eval.lock uv run python data/scratch/kp_chains/score_kp.py --stage dev  --arms kp,kpm --out ...
  flock /tmp/vss_eval.lock uv run python data/scratch/kp_chains/score_kp.py --stage conf --arms <chosen> --out ...
  ... --smoke-standin : arm "kp" := the band's M2 s1 dump, "kpm" := its base dump (reproduces published rows; no KP
                        information exists in this mode)
Decision rules (section 4 of the note) are evaluated at the end and written to <out>_decision.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval"))
import kelly_h2h as kh  # noqa: E402
import spiral_eval as se  # noqa: E402
import wholescroll_pool as wp  # noqa: E402

FROZEN = {"eval/kelly_h2h.py": "bdfb56413873c05c97181aad098a613095e48ef5f3d7e8f07002aa0c58a2fe56",
          "eval/wholescroll_pool.py": "dfdb1f571cd718cb82ec7d122f058ba5dc0849d2f5eb3f449abb60613f1a780a"}
DEV = [(3000, 4500), (6000, 6500)]
CONF = [(400, 1500), (1500, 3000), (4500, 6000), (6500, 8000), (8000, 9000)]
VD = REPO / "data/scratch/kp_chains/vertex"
_orig_arm_files = wp.arm_files


def check_frozen():
    for f, h in FROZEN.items():
        got = hashlib.sha256((REPO / f).read_bytes()).hexdigest()
        assert got.startswith(h), (f, got)


def dump_path(arm, z0, standin):
    if standin:
        f = _orig_arm_files(z0, "s1")["dumps"]
        return f[1] if arm == "kp" else f[0]
    return VD / f"{arm}_z{z0}.npz"


def mesh_path(arm, z0, standin):
    if standin:
        f = _orig_arm_files(z0, "s1")["meshes"]
        return f[1] if arm == "kp" else f[0]
    return wp.SC / f"{arm}_z{z0}" / "meshes_step4"


def run_h2h(bands, arms, standin, variants):
    def ours_files(z0):
        f = _orig_arm_files(z0, "s1")
        out = {"base": f["dumps"][0], "m2": f["dumps"][1]}
        for a in arms:
            p = dump_path(a, z0, standin)
            assert p.exists(), p
            out[a] = p
        return out
    kh.ours_files = ours_files
    kh.PAIRS = list(kh.PAIRS) + [(a, b) for a in arms for b in ("k_all_comp", "m2", "k_good_comp")]
    kh.BANDS = [(z0, z1, True) for z0, z1 in bands]
    se.set_scroll("p1667")
    KI, med = kh.kelly_tables()
    Cc = np.load(REPO / "data/eval/p1667/labels_merged_v4.npz")["centre"]
    areas = {z0: wp.labelled_area(z0, z1)["inner"] for z0, z1 in bands}
    res = {"areas_inner_cm2": areas}
    for v in variants:
        res[v] = kh.run(v, set(bands), KI, med, Cc, areas)
    return res


def run_pool(bands, arm, standin):
    def arm_files(z0, pair):
        assert pair == "s1"
        f = _orig_arm_files(z0, "s1")
        names = (f["names"][1], f"{arm}_z{z0}" if not standin else f"standin_{arm}_z{z0}")
        return {"names": names, "dumps": (f["dumps"][1], dump_path(arm, z0, standin)),
                "meshes": (f["meshes"][1], mesh_path(arm, z0, standin))}
    wp.arm_files = arm_files
    try:
        P = []
        for z0, z1 in bands:
            p = wp.band_parts(z0, z1, "s1", True)
            P.append(p)
        return {"per_band": [p["summary"] for p in P], "pooled": wp.pooled(P)}
    finally:
        wp.arm_files = _orig_arm_files


def decide(stage, h2h, pools, arms):
    q = h2h["primary"]["pooled"]["all7"]           # = the stage pool (every stage band flagged fresh)
    M = q["methods"]
    out = {"stage": stage, "arms": {}}
    for a in arms:
        pk, pm = q["pairs"][f"{a}_vs_k_all_comp"], q["pairs"][f"{a}_vs_m2"]
        wa = pools[a]["pooled"]
        out["arms"][a] = {
            "C_A_Q": [M[a]["C"]["est"], M[a]["A"]["est"], M[a]["Q"]["est"]],
            "Q_minus_kelly": pk["dQ"], "Q_minus_m2": pm["dQ"],
            "A_cocovered_minus_kelly": pk["dA_cocovered"], "A_cocovered_minus_m2": pm["dA_cocovered"],
            "inner_wrong_minus_m2": wa["inner"]["diff"], "switch_minus_m2": wa["sw"]["diff"],
            "outer_wrong_minus_m2": wa["outer"]["diff"], "unmatched_minus_m2": wa["unm"]["diff"],
            "union_Q_with_kelly": pk["union_Q"]}
    if stage == "dev":
        qs = {a: M[a]["Q"]["est"] for a in arms}
        best = max(arms, key=lambda a: (qs[a], a == "kp"))
        if "kp" in arms and "kpm" in arms and abs(qs["kp"] - qs["kpm"]) < 0.5:
            best = "kp"
        go = qs[best] > M["m2"]["Q"]["est"]
        out["dev_rule"] = {"chosen": best, "Q": qs, "Q_m2": M["m2"]["Q"]["est"], "Q_kelly": M["k_all_comp"]["Q"]["est"],
                           "go_confirmation": bool(go)}
    else:
        a = arms[0]
        r = out["arms"][a]
        lo = lambda d: d["ci"][0]
        hi = lambda d: d["ci"][1]
        out["C1_beats_kelly_Q"] = bool(lo(r["Q_minus_kelly"]) is not None and lo(r["Q_minus_kelly"]) > 0)
        out["C1_kelly_beats_arm_Q"] = bool(hi(r["Q_minus_kelly"]) is not None and hi(r["Q_minus_kelly"]) < 0)
        out["C2_improves_fitter"] = bool(lo(r["Q_minus_m2"]) is not None and lo(r["Q_minus_m2"]) > 0
                                         and hi(r["inner_wrong_minus_m2"]) is not None and hi(r["inner_wrong_minus_m2"]) < 0
                                         and r["switch_minus_m2"]["est"] <= 5.0)
        out["S1_cocovered_A_within_1pp_of_kelly"] = bool(lo(r["A_cocovered_minus_kelly"]) is not None
                                                         and lo(r["A_cocovered_minus_kelly"]) > -1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["dev", "conf", "all"], required=True)
    ap.add_argument("--arms", default="kp,kpm")
    ap.add_argument("--smoke-standin", action="store_true")
    ap.add_argument("--variants", default="primary,lenient,tol2")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    check_frozen()
    arms = a.arms.split(",")
    bands = {"dev": DEV, "conf": CONF, "all": sorted(DEV + CONF)}[a.stage]
    h2h = run_h2h(bands, arms, a.smoke_standin, a.variants.split(","))
    res = {"prereg": "notes/2026-10-29-kp-chains-prereg.md", "stage": a.stage, "bands": bands, "arms": arms,
           "smoke_standin": a.smoke_standin, "h2h": h2h}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    pools = {arm: run_pool(bands, arm, a.smoke_standin) for arm in arms}
    res["pool_vs_m2"] = pools
    a.out.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    dec = decide(a.stage if a.stage != "all" else "conf", h2h, pools, arms)
    a.out.with_name(a.out.stem + "_decision.json").write_text(json.dumps(dec, indent=1))
    q = h2h["primary"]["pooled"]["all7"]["methods"]
    print("stage pool C/A/Q:", {n: (v["C"]["est"], v["A"]["est"], v["Q"]["est"]) for n, v in q.items()})
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
