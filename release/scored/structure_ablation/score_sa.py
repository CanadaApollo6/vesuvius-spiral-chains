"""Structure ablation scorer (notes/2026-10-28-structure-ablation-prereg.md), Paris4 z 7000-8500 dev band.

  flock /tmp/vss_eval.lock uv run python data/scratch/structure_ablation/score_sa.py \
      --out data/scratch/structure_ablation/sa_table.json

1. Endpoints: the test-13 dev scorer data/scratch/anneal/dev_table_an.py (f2c785a6..., unchanged; it runs
   data/scratch/f7_switch/dev_table_f7.py cc7786b7... on the original labels and eval/paris4_label_audit.py cmd_rescore
   on the S1 trusted vertices), imported, with data/scratch/structure_ablation/vertex put first in its dump search
   path, called as `dev_table_an.py sa_a1z7 sa_a2z7 sa_a3z7 --refs --out <out>`. --refs adds f6z_ms1, f6z_ms2 and
   f6z_oracle; bases f6z_base / f6z_bases2 / f6z_bases3. Its gate columns (test 13's) are printed but not used here.
2. Gap share per base seed b: G(arm, b) = (arm - b) / (f6z_oracle - b), own-offset primary (PRIMARY_K 13,16-32,34-44,
   47-61,63-108), original labels; G = mean over the 3 seeds. Same on S1 trusted primary (secondary).
3. Arm-vs-arm contrasts (2x2 cells; same code as dev_table_f7's primary: subset_wrong_area.sub_scores own offsets,
   spiral_eval.BootV paired ratio, 1000 resamples, seed 1): Y - X primary wrong area, pp, with 95 % CI.
4. Decision rules (section 5 of the prereg) -> <out stem>_decision.json.
--smoke ARM1 ARM2 ARM3: stand-in dumps for the three arms (scorer test only; output marked SMOKE).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data/scratch/anneal"))
import dev_table_an as A  # noqa: E402  (also imports dev_table_f7 as A.T, paris4_label_audit as A.PLA)

T = A.T
se, swa = T.se, T.swa
VD = REPO / "data/scratch/structure_ablation/vertex"
A.VDIRS.insert(0, VD)
ARMS = ["sa_a1z7", "sa_a2z7", "sa_a3z7"]
DELTA = 0.15          # material share of the base->oracle gap (about 4.7 pp at the f6 confirmation's 31.4 pp gap)
G_NEITHER = 0.85      # A3 retains >= 85 % of the gap -> structure is not what the oracle adds
CONTRASTS = [  # (label, X, Y): Y - X
    ("linkage | complete   (A1 - oracle)", "f6z_oracle", "A1"),
    ("linkage | thinned    (A3 - A2)", "A2", "A3"),
    ("completeness | linked     (A2 - oracle)", "f6z_oracle", "A2"),
    ("completeness | fragmented (A3 - A1)", "A1", "A3"),
    ("structure total      (A3 - oracle)", "f6z_oracle", "A3"),
    ("step correctness | our structure (f6z_ms1 - A3)", "A3", "f6z_ms1"),
]


def contrasts(names):
    B = T.BANDS["z7"]
    V = swa.load_ref(B["z0"], B["z1"])
    boot = se.BootV(V)
    prim = np.isin(V["k"], sorted(swa.parse_kset(B["prim"])))
    own = {}
    for k, n in names.items():
        p = A.dump_path(n, B)
        assert p is not None, f"missing dump {n}"
        own[k] = swa.sub_scores(V, np.load(p)["j"], {}, prim)
    out = {}
    for lab, x, y in CONTRASTS:
        X, Y = own[x], own[y]
        ex, rx = boot.ratio(X["matched"] & (X["res_s"] != 0), X["matched"], prim)
        ey, ry = boot.ratio(Y["matched"] & (Y["res_s"] != 0), Y["matched"], prim)
        out[lab] = dict(x=names[x], y=names[y], diff=se.summ(ey - ex, ry - rx))
        d = out[lab]["diff"]
        print(f"{lab:48s} {d['est']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]", flush=True)
    return out


def decide(G):
    g1, g2, g3, gm = G["A1"], G["A2"], G["A3"], G["f6z_ms1"]
    link = 0.5 * ((1 - g1) + (g2 - g3))
    comp = 0.5 * ((1 - g2) + (g1 - g3))
    inter = (1 - g1) - (g2 - g3)            # linkage loss when complete minus when thinned
    corr = g3 - gm                           # oracle-correct steps on our structure
    if g3 >= G_NEITHER:
        rule, nxt = "R1 neither", "structure is not the bottleneck: step correctness (revisit step checking / step classifier)"
    elif link >= DELTA and link - comp >= DELTA:
        rule, nxt = "R2 linkage-dominant", "patch / segment-derived chains (long linked pieces)"
    elif comp >= DELTA and comp - link >= DELTA:
        rule, nxt = "R3 completeness-dominant", "per-ray sheet recovery at contacts"
    elif link >= DELTA and comp >= DELTA:
        rule, nxt = "R4 both", "a source that gives both (human sheet tracing; segment patches that span contacts)"
    else:
        rule, nxt = "R5 either suffices", "the cheaper single lever; default patch / segment-derived chains"
    return dict(G=G, linkage_loss=link, completeness_loss=comp, interaction=inter, correctness_gain=corr,
                corr_material=bool(corr >= DELTA), rule=rule, next=nxt, delta=DELTA, g_neither=G_NEITHER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", nargs=3, metavar=("A1", "A2", "A3"))
    o = ap.parse_args()
    arms = o.smoke or ARMS
    roles = dict(zip(["A1", "A2", "A3"], arms))
    rows = list(dict.fromkeys(arms))
    sys.argv = ["dev_table_an.py", *rows, "--refs", "--out", str(o.out)]
    A.main()
    orig = json.loads(o.out.with_name(o.out.stem + "_orig.json").read_text())
    aud = json.loads(o.out.with_name(o.out.stem + "_audit.json").read_text())["variants"]["S1_trusted"]
    s1 = {**aud["combos"], **aud["oracle_check"]}
    bases = T.BANDS["z7"]["bases"]
    G, G_s1, per = {}, {}, {}
    for role, n in {**roles, "f6z_ms1": "f6z_ms1", "f6z_ms2": "f6z_ms2"}.items():
        g = [orig[n]["combos"][b]["primary_diff"]["est"] / orig["f6z_oracle"]["combos"][b]["primary_diff"]["est"]
             for b in bases]
        gs = [s1[f"{n}_vs_{b}"]["primary_own_diff"]["est"] / s1[f"f6z_oracle_vs_{b}"]["primary_own_diff"]["est"]
              for b in bases]
        G[role], G_s1[role], per[role] = float(np.mean(g)), float(np.mean(gs)), g
    print("\n== gap share G (original labels; S1 trusted)")
    for r in G:
        print(f"{r:8s} G {G[r]:.3f} (per seed {', '.join(f'{x:.3f}' for x in per[r])})  G_S1 {G_s1[r]:.3f}")
    print("\n== contrasts (own-offset primary wrong area, Y - X, pp)")
    C = contrasts({**roles, "f6z_oracle": "f6z_oracle", "f6z_ms1": "f6z_ms1"})
    dec = decide(G)
    dec.update(smoke=bool(o.smoke), roles=roles, G_per_seed=per, G_S1=G_s1, decision_S1=decide(G_s1), contrasts=C)
    print(f"\n== decision{' (SMOKE: stand-in dumps, not a result)' if o.smoke else ''}: {dec['rule']} -> {dec['next']}")
    print(f"linkage loss {dec['linkage_loss']:.3f}, completeness loss {dec['completeness_loss']:.3f}, "
          f"interaction {dec['interaction']:+.3f}, correctness gain (G_A3 - G_ms1) {dec['correctness_gain']:+.3f}")
    print(f"S1 read: {dec['decision_S1']['rule']}")
    o.out.with_name(o.out.stem + "_decision.json").write_text(json.dumps(dec, indent=1, default=str))


if __name__ == "__main__":
    main()
