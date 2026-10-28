"""PHercParis4 label audit: merge, summary, label-vs-CT offsets and sensitivity re-scoring of our Paris4 tests
(notes/2026-10-24-paris4-label-audit.md). Per-band audits come from eval/winding_audit.py (extract / analyse /
report) in data/scratch/label_audit/paris4/<band>/.

  uv run python eval/paris4_label_audit.py merge     # -> data/scratch/label_audit/paris4_label_corrections.{json,csv}
  uv run python eval/paris4_label_audit.py summary --band z7000 [--band ...] --out X.json
  uv run python eval/paris4_label_audit.py offsets --band z7000
  uv run python eval/paris4_label_audit.py rescore --test t6c --out X.json      # t3 t4 t6d t6c t7

z 10800-11800 is a fresh confirmation region: no band here covers it (winding_audit enforces it at extract).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
AUD = REPO / "data/scratch/label_audit/paris4"
OUT_CORR = REPO / "data/scratch/label_audit/paris4_label_corrections.json"
MIN_CONF = 0.8
SEALED = ("z9000", "ext_8500_9000", "ext_10500_10752")   # HITL session band and its neighbours: scratch only


def load_audit(band):
    d = AUD / band
    A = json.loads((d / "audit.json").read_text())
    return A


def trusted_map(m):
    return m["conf"] >= MIN_CONF and m.get("x_gm_int") == m["new"] and m.get("x_ct_int", m["new"]) == m["new"]


# ----------------------------------------------------------------------------------------------------------- merge
def cmd_merge(a):
    bands = a.band or sorted(p.name for p in AUD.iterdir() if (p / "label_corrections.json").exists())
    res = {"title": "CT-consistent winding numbering proposal and pair verdicts for the PHercParis4 organiser "
                    "20260701 winding-group labels (labels_w0701)",
           "note": "notes/2026-10-24-paris4-label-audit.md", "bands": {}}
    rows = []
    for b in bands:
        C = json.loads((AUD / b / "label_corrections.json").read_text())
        res["bands"][b] = C
        with open(AUD / b / "label_corrections.csv") as f:
            r = list(csv.reader(f))
        hdr = r[0]
        rows += [[b] + x for x in r[1:]]
    OUT_CORR.write_text(json.dumps(res, indent=1))
    with open(OUT_CORR.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band"] + hdr)
        w.writerows(rows)
    print(f"wrote {OUT_CORR} and .csv ({len(rows)} rows, bands {bands})")


# --------------------------------------------------------------------------------------------------------- summary
CLASS = {"ok": "consistent", "duplicate": "overcounted", "overcount": "overcounted", "untraced": "undercounted",
         "order": "order", "ambiguous": "ambiguous"}


def cls_of(r, key="verdict", conf="conf"):
    if r.get(key) is None:
        return "ambiguous"
    if r[key] == "ambiguous" or r[conf] < MIN_CONF:
        return "ambiguous"
    return CLASS[r[key]]


def shares(rows, key="verdict", conf="conf"):
    out = {}
    tot_n, tot_w = len(rows), sum(r["rays"] for r in rows)
    for c in ("consistent", "overcounted", "undercounted", "order", "ambiguous"):
        sel = [r for r in rows if cls_of(r, key, conf) == c]
        out[c] = {"cells": round(len(sel) / max(tot_n, 1), 4), "rays": round(sum(r["rays"] for r in sel) / max(tot_w, 1), 4)}
    dec = [r for r in rows if cls_of(r, key, conf) != "ambiguous"]
    out["n_cells"], out["n_rays"] = tot_n, tot_w
    out["of_decided"] = {c: round(sum(r["rays"] for r in dec if cls_of(r, key, conf) == c) / max(sum(r["rays"] for r in dec), 1), 4)
                         for c in ("consistent", "overcounted", "undercounted", "order")}
    return out


def cmd_summary(a):
    res = {}
    allrows = {}
    for b in a.band:
        A = load_audit(b)
        allrows[b] = [r for r in A["pairs"] if r["sector"] != "all"]
    sp = np.array([r.get("ct_spacing_med") for b in a.band if b in ("z7000", "z9000") for r in allrows[b]
                   if r.get("ct_spacing_med") is not None])
    terc = np.percentile(sp, [100 / 3, 200 / 3]) if len(sp) else np.array([np.nan, np.nan])
    for b in a.band:
        A = load_audit(b)
        rows = allrows[b]
        sl = [r for r in A["pairs"] if r["sector"] == "all"]
        out = {"sector_cells": shares(rows), "slab_cells_all_angles": shares(sl),
               "m7_only_rule_sector_cells": shares(rows, "verdict_m7", "conf_m7") if rows and "verdict_m7" in rows[0] else None,
               "ct_calibration": {k: v for k, v in A.get("ct_calibration", {}).items() if k != "detector"},
               "gm_scale_per_slab": A["gm_scale_per_slab"], "validation": {k: v for k, v in A["validation"].items() if k != "rows"}}
        if rows and "verdict_m7" in rows[0]:
            ch = {}
            for r in rows:
                key = f"{cls_of(r, 'verdict_m7', 'conf_m7')} -> {cls_of(r)}"
                ch[key] = ch.get(key, 0) + 1
            out["m7_to_combined_transitions_cells"] = dict(sorted(ch.items(), key=lambda x: -x[1]))
            dis = [r for r in rows if r["T_m7"] is not None and r["T_ct"] is not None and r["T_m7"] != r["T_ct"]]
            out["m7_ct_conflicts_cells"] = {"n": len(dis), "share": round(len(dis) / max(len(rows), 1), 4)}
        # clustering
        def grp(fn):
            g = {}
            for r in rows:
                g.setdefault(fn(r), []).append(r)
            return {str(k): shares(v) for k, v in sorted(g.items(), key=lambda x: str(x[0]))}
        def pack(r):
            s = r.get("ct_spacing_med")
            if s is None:
                return "unknown"
            return "tight" if s < terc[0] else ("loose" if s >= terc[1] else "mid")
        out["by_packing_tercile"] = grp(pack)
        out["packing_terciles_vox"] = [round(float(x), 2) for x in terc]
        out["by_contact"] = grp(lambda r: "contact (>=25% rays gap<8)" if r.get("contact_share", 0) >= 0.25 else "no contact")
        out["by_seam_sector"] = grp(lambda r: "seam sector 0-30" if r["sector"][0] == 0 else "other sectors")
        out["by_sector"] = grp(lambda r: f"{int(r['sector'][0]):03d}")
        out["by_k_decade"] = grp(lambda r: f"k{(r['k'] // 10) * 10:03d}")
        out["by_slab"] = grp(lambda r: f"{int(r['z'][0])}")
        # winding hot spots: per k, number of confident non-ok cells touching k
        hk = {}
        for r in rows:
            c = cls_of(r)
            if c not in ("consistent", "ambiguous"):
                for k in {r["k"], r["k2"]}:
                    hk.setdefault(k, {"overcounted": 0, "undercounted": 0, "order": 0})[c] += 1
        out["hot_windings"] = dict(sorted(hk.items(), key=lambda x: -sum(x[1].values()))[:15])
        res[b] = out
    Path(a.out).write_text(json.dumps(res, indent=1))
    for b, o in res.items():
        s = o["sector_cells"]
        print(f"{b}: {s['n_cells']} sector cells, ray-weighted: " +
              ", ".join(f"{c} {s[c]['rays'] * 100:.1f}%" for c in ("consistent", "overcounted", "undercounted", "order", "ambiguous")))
    print(f"wrote {a.out}")


# --------------------------------------------------------------------------------------------------------- offsets
def cmd_offsets(a):
    """Signed radial offset of each label crossing to the nearest CT-run centre and m7-run centre (label - sheet,
    vox; positive = label outward of the sheet), on all rays, crossings with a sheet within 8 vox."""
    import winding_audit as W
    from scipy import ndimage as ndi
    res = {}
    for b in a.band:
        D = dict(np.load(AUD / b / "rays.npz"))
        A = load_audit(b)
        pr = A["ct_calibration"]["prominence"]
        oc, om, oce = [], [], []
        for k in range(0, len(D["lev"]), a.every):
            il, ia = int(D["lev"][k]), int(D["ia"][k])
            r = D["R"][il, ia]
            r = r[np.isfinite(r)]
            t, m7, _ = W.per_ray(D, il, ia, k)
            o, n = D["off"][k], D["n"][k]
            lo, hi = W.ct_runs(t, D["ct"][o:o + n], pr)
            cc = (lo + hi) / 2
            lab, nn = ndi.label(m7 > 0)
            ob = ndi.find_objects(lab)
            mc = np.array([(t[s[0].start] + t[s[0].stop - 1]) / 2 for s in ob]) if nn else np.zeros(0)
            for x in r:
                if len(cc):
                    d = x - cc
                    j = np.argmin(np.abs(d))
                    if abs(d[j]) <= 8:
                        oc.append(d[j])
                    # offset to the nearest CT-run edge inside: 0 if the crossing lies inside a run
                    inside = (x >= lo) & (x <= hi)
                    oce.append(0.0 if inside.any() else float(np.min(np.minimum(np.abs(x - lo), np.abs(x - hi)))))
                if len(mc):
                    d = x - mc
                    j = np.argmin(np.abs(d))
                    if abs(d[j]) <= 8:
                        om.append(d[j])
        oc, om, oce = map(np.array, (oc, om, oce))
        q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 25, 50, 75, 90])]
        res[b] = {"label_minus_ct_centre": {"n": len(oc), "mean": round(float(oc.mean()), 3), "p10_25_50_75_90": q(oc)},
                  "label_minus_m7_centre": {"n": len(om), "mean": round(float(om.mean()), 3), "p10_25_50_75_90": q(om)},
                  "label_inside_ct_run_share": round(float((oce == 0).mean()), 4),
                  "label_within_4vox_of_ct_run_share": round(float((oce <= 4).mean()), 4)}
        print(b, json.dumps(res[b]))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------------------------------------- rescore
TESTS = {
    "t3": {"band": "z9000", "z": (9000, 10500), "primary": "21-32,34-60,62-87,90-101,106-108",
           "vd": "data/scratch/paris4_run/vertex", "bases": ["p4_base", "p4_bases2", "p4_bases3"],
           "methods": {"p4_ms1": None, "p4_ms2": None}, "oracle": "p4_oracle",
           "desc": "test 3 (third region, no-raise chains, frozen prereg)"},
    "t4": {"band": "z9000", "z": (9000, 10500), "primary": "21-32,34-60,62-87,90-101,106-108",
           "vd": "data/scratch/paris4_run/vertex", "bases": ["p4_base", "p4_bases2", "p4_bases3"],
           "methods": {"d4_r263": "data/scratch/fourth_test/vertex"}, "oracle": None,
           "desc": "test 4 dev fit (raise auto, gm 0.263) on z 9000"},
    "t6d": {"band": "z9000", "z": (9000, 10500), "primary": "21-32,34-60,62-87,90-101,106-108",
            "vd": "data/scratch/paris4_run/vertex", "bases": ["p4_base", "p4_bases2", "p4_bases3"],
            "methods": {"f6_cut": "data/scratch/f6_screen/vertex", "f6_w1": "data/scratch/f6_screen/vertex",
                        "f6_w03": "data/scratch/f6_screen/vertex"}, "oracle": None,
            "desc": "test 6 dev fits on z 9000"},
    "t6c": {"band": "z7000", "z": (7000, 8500), "primary": "13,16-32,34-44,47-61,63-108",
            "vd": "data/scratch/f6_screen/confirm/vertex", "bases": ["f6z_base", "f6z_bases2", "f6z_bases3"],
            "methods": {"f6z_ms1": None, "f6z_ms2": None}, "oracle": "f6z_oracle",
            "desc": "test 6 confirmation (A4 cut) on z 7000"},
    "t7": {"band": "z7000", "z": (7000, 8500), "primary": "13,16-32,34-44,47-61,63-108",
           "vd": "data/scratch/f6_screen/confirm/vertex", "bases": ["f6z_base", "f6z_bases2", "f6z_bases3"],
           "methods": {"f7_c10z7": "data/scratch/f7_switch/vertex", "f7_c15z7": "data/scratch/f7_switch/vertex"},
           "oracle": None, "desc": "test 7 dev fits on z 7000"},
}


def vertex_trust(V, A):
    """S1: trusted[v] iff every label pair (verdict row) in the vertex's (slab, 30 deg sector) cell that involves
    its k is combined-ok with conf >= 0.8 (at least one row). Pairs with fewer than min-rays rays in the cell have
    no row and are not counted. S2: new label from the trusted per-cell mapping (-1: not trusted / absent)."""
    edges = np.array(A["slab_edges"])
    sec = A["sector_deg"]
    nsl, nsec = len(edges) - 1, int(round(360 / sec))
    kmax = 200
    ok_any = np.zeros((nsl, nsec, kmax), bool)
    bad = np.zeros((nsl, nsec, kmax), bool)
    for r in A["pairs"]:
        if r["sector"] == "all":
            continue
        s = int(np.searchsorted(edges, r["z"][0], side="right") - 1)
        c = int(round(r["sector"][0] / sec))
        good = r["verdict"] == "ok" and r["conf"] >= MIN_CONF
        for k in (r["k"], r["k2"]):
            if good:
                ok_any[s, c, k] = True
            else:
                bad[s, c, k] = True
    T1 = ok_any & ~bad
    newk = np.full((nsl, nsec, kmax), -1, np.int64)
    for cell in A["mapping"]:
        if cell["sector"] == "all":
            continue
        s = int(np.searchsorted(edges, cell["z"][0], side="right") - 1)
        c = int(round(cell["sector"][0] / sec))
        for k, m in cell["map"].items():
            if trusted_map(m):
                newk[s, c, int(k)] = m["new"]
    z = V["pts"][:, 0]
    si = np.searchsorted(edges, z, side="right") - 1
    ci = (V["ang"] // sec).astype(np.int64) % nsec
    inb = (si >= 0) & (si < nsl)
    si = np.clip(si, 0, nsl - 1)
    k = np.clip(V["k"], 0, kmax - 1)
    t1 = inb & T1[si, ci, k]
    nk = np.where(inb, newk[si, ci, k], -1)
    return t1, nk


def cmd_rescore(a):
    import spiral_eval as se
    import subset_wrong_area as swa
    cfg = TESTS[a.test]
    se.set_scroll("paris4")
    V = swa.load_ref(*cfg["z"])
    A = load_audit(a.band[0] if a.band else cfg["band"])
    boot = se.BootV(V)
    prim = np.isin(V["k"], sorted(swa.parse_kset(cfg["primary"])))
    t1, nk = vertex_trust(V, A)
    keep2 = nk >= 0
    V2 = dict(V)
    V2["k"] = np.where(keep2, nk, V["k"])
    a0, a1 = V["adj"][:, 0], V["adj"][:, 1]
    variants = {"original": (np.ones(len(prim), bool), V), "S1_trusted": (t1, V), "S2_audited_labels": (keep2, V2)}
    names = list(cfg["bases"]) + list(cfg["methods"]) + ([cfg["oracle"]] if cfg["oracle"] else [])
    D = {}
    for n in names:
        vd = cfg["methods"].get(n) or cfg["vd"]
        D[n] = dict(np.load(REPO / vd / f"{n}.npz"))
    res = {"test": a.test, "desc": cfg["desc"], "band": cfg["band"], "primary_k": cfg["primary"],
           "n_vertices": int(len(prim)), "n_primary": int(prim.sum()), "variants": {}}
    for vn, (keep, VV) in variants.items():
        pm, npm = prim & keep, ~prim & keep
        ma = keep[a0] & keep[a1]
        own, ownn, sw = {}, {}, {}
        for n, d in D.items():
            own[n] = swa.sub_scores(VV, d["j"], {}, pm)
            ownn[n] = swa.sub_scores(VV, d["j"], {}, npm)
            if vn == "S2_audited_labels":   # switches with offsets refitted under the audited labels (kept vertices)
                full = swa.sub_scores(VV, d["j"], {}, keep)
                m, rs = full["matched"], full["res_s"]
            else:
                m, rs = d["matched"], d["res_s"]
            pmm = m[a0] & m[a1]
            s = pmm & (rs[a0] != rs[a1])
            sw[n] = boot.ratio_adj(s * 1000.0, pmm, ma)
        out = {"kept_share_all": round(float(keep.mean()), 4), "kept_share_primary": round(float(pm.sum() / max(prim.sum(), 1)), 4),
               "adj_pairs_kept_share": round(float(ma.mean()), 4), "combos": {}, "oracle_check": {}}

        def ep(mn, bn):
            A_, B_ = own[bn], own[mn]
            ea, ra = boot.ratio(A_["matched"] & (A_["res_s"] != 0), A_["matched"], pm)
            eb, rb = boot.ratio(B_["matched"] & (B_["res_s"] != 0), B_["matched"], pm)
            An, Bn = ownn[bn], ownn[mn]
            na, rna = boot.ratio(An["matched"] & (An["res_s"] != 0), An["matched"], npm)
            nb, rnb = boot.ratio(Bn["matched"] & (Bn["res_s"] != 0), Bn["matched"], npm)
            sa, sra = sw[bn]
            sb, srb = sw[mn]
            r = rb - ra
            r = r[np.isfinite(r)]
            p = float(min(1.0, 2 * min((r >= 0).mean(), (r <= 0).mean()))) if len(r) else None
            pr = se.summ(eb - ea, rb - ra)
            s_ = se.summ(sb - sa, srb - sra, 1.0)
            return {"primary_own_diff": pr, "base": se.summ(ea, ra), "arm": se.summ(eb, rb), "p_two_sided": p,
                    "switch_per1000_diff": s_, "nonprimary_own_diff": se.summ(nb - na, rnb - rna),
                    "test3_6_pass": bool(pr["ci"][1] is not None and pr["ci"][1] < 0 and s_["est"] <= 5.0),
                    "test7_gate": bool(pr["ci"][1] is not None and pr["ci"][1] < 0 and pr["est"] <= -4.0 and
                                       s_["est"] <= 1.0 and se.summ(nb - na, rnb - rna)["est"] <= 3.0)}
        for mn in cfg["methods"]:
            for bn in cfg["bases"]:
                out["combos"][f"{mn}_vs_{bn}"] = ep(mn, bn)
        if cfg["oracle"]:
            for bn in cfg["bases"]:
                out["oracle_check"][f"{cfg['oracle']}_vs_{bn}"] = ep(cfg["oracle"], bn)
        res["variants"][vn] = out
        f = lambda r: f"{r['est']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]" if r["est"] is not None else "n/a"
        print(f"== {a.test} {vn}: kept primary {out['kept_share_primary']:.3f}, adj pairs {out['adj_pairs_kept_share']:.3f}")
        for k, r in list(out["combos"].items()) + list(out["oracle_check"].items()):
            print(f"  {k:26s} base {r['base']['est']:.2f} arm {r['arm']['est']:.2f} primary {f(r['primary_own_diff'])} "
                  f"switch {r['switch_per1000_diff']['est']:+.2f} nonprim {f(r['nonprimary_own_diff'])}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["merge", "summary", "offsets", "rescore"])
    ap.add_argument("--band", action="append")
    ap.add_argument("--test", choices=list(TESTS))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    {"merge": cmd_merge, "summary": cmd_summary, "offsets": cmd_offsets, "rescore": cmd_rescore}[a.cmd](a)


if __name__ == "__main__":
    main()
