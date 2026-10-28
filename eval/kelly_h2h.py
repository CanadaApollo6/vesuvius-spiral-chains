"""Head-to-head on one metric set: our whole-scroll winding numbers (spiral fits, M2 / base) vs Kelly's index on
Stevens' patches, PHerc1667, on the same human-label vertices. Definitions: notes/2026-10-27-kelly-headtohead-prereg.md.

Evaluation set E per band: spiral_eval reference vertices (merged_v4, stride 2) with k < 34 and ang outside [160, 200).
Per method: covered (nearest surface within 4 vox), turn coordinate T, residual R = T_H - sigma T, correct if R lies in
the best width-1 window of its offset group (band, or band x Kelly component, or 250 z x 10 deg bin for S-b).
Metrics: coverage C, accuracy A, correct share Q = C A, co-covered accuracy, overlap table, union ceiling; paired
block bootstrap (BootV blocks, rng 2700 + band index, 1000 resamples), pooled over bands by summed counts.

  uv run python -I eval/kelly_h2h.py --out data/scratch/kelly_h2h/h2h_s1.json      (--smoke: shapes only, no metric)
Inputs read: release/fit_vertex_dump.py dumps, data/scratch/kelly_h2h/kelly_z<z0>.npz (kelly_cover.py),
Kelly's community/wrap_index.csv, data/scratch/patch_numbering/patches_all.tsv (medoids), labels_merged_v4.npz.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402
import wholescroll_pool as wp  # noqa: E402

BANDS = [(400, 1500, True), (1500, 3000, True), (3000, 4500, True), (4500, 6000, False), (6000, 6500, True),
         (6500, 8000, False), (8000, 9000, True)]
SPLIT, WEDGE, SEAM_O, TOL2, N_BOOT, MINBIN = 34, (160.0, 200.0), 180.0, 2.0, 1000, 50
H2H = REPO / "data/scratch/kelly_h2h"
KELLY_CSV = REPO / "data/untrusted/kelly_scrolls/repo/community/wrap_index.csv"
wrap180 = lambda a: (np.asarray(a) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------------------------ offsets
def window(r: np.ndarray) -> tuple[float, int]:
    """Lower edge and count of the width-1 window [lo, lo + 1) holding the most values."""
    if not len(r):
        return 0.0, 0
    r = np.sort(r)
    hi = np.searchsorted(r, r + 1.0, side="left")
    i = int(np.argmax(hi - np.arange(len(r))))
    return float(r[i]), int(hi[i] - i)


def fit(TH, Tfun, cov, groups=None):
    """(sigma, s) over {+-1}^2 and one window per group (groups: int array, -1 = none) maximising correct vertices.
    Returns correct mask (over all vertices) and the parameters."""
    best = None
    for sig in (1, -1):
        for s in (1, -1):
            R = TH - sig * Tfun(s)
            cor = np.zeros(len(TH), bool)
            offs = {}
            gs = np.zeros(len(TH), np.int64) if groups is None else groups
            for g in np.unique(gs[cov]):
                m = cov & (gs == g)
                lo, _ = window(R[m])
                cor[m] = (R[m] >= lo) & (R[m] < lo + 1.0)
                offs[int(g)] = lo + 0.5
            n = int(cor.sum())
            if best is None or n > best[0]:
                best = (n, sig, s, cor, offs, R)
    n, sig, s, cor, offs, R = best
    return cor, {"sigma": sig, "s": s, "n_correct": n, "n_groups": len(offs),
                 "offset": (offs[0] if groups is None else None)}, R


def fit_bins(R, cov, cor_band, bins):
    """S-b: one window per bin with >= MINBIN covered vertices (sigma, s, R from the band fit); others keep the band."""
    cor = cor_band.copy()
    for b in np.unique(bins[cov]):
        m = cov & (bins == b)
        if m.sum() >= MINBIN:
            lo, _ = window(R[m])
            cor[m] = (R[m] >= lo) & (R[m] < lo + 1.0)
    return cor


# ------------------------------------------------------------------------------------------------ inputs
def kelly_tables():
    KI = {}
    for r in csv.DictReader(open(KELLY_CSV)):
        KI[int(r["patch"])] = (int(r["k_q3c"]), int(r["component"]), float(r["thN"]))
    med = {}
    for r in csv.DictReader(open(REPO / "data/scratch/patch_numbering/patches_all.tsv"), delimiter="\t"):
        med[int(r["pid"])] = (float(r["mx"]), float(r["my"]), float(r["mz"]))
    return KI, med


def theta_h(zyx, C):
    """Human-frame angle (deg) around the label centre (C rows: z, x, y), as score.py th_hum."""
    return np.degrees(np.arctan2(zyx[..., 1] - np.interp(zyx[..., 0], C[:, 0], C[:, 2]),
                                 zyx[..., 2] - np.interp(zyx[..., 0], C[:, 0], C[:, 1])))


def ours_files(z0):
    f = wp.arm_files(z0, "s1")
    vd = f["dumps"][0].parent
    s2n = f["names"][1].replace("m2s1", "m2s2")
    return {"base": f["dumps"][0], "m2": f["dumps"][1], "m2s2": vd / f"{s2n}.npz"}


def band_inputs(z0, z1, KI, med, C, smoke=False):
    V = dict(np.load(se.ref_path(z0, z1), allow_pickle=False))
    k, pts, ang = V["k"], V["pts"], V["ang"]
    thH = theta_h(pts, C)
    TH = k - ((thH - 144.0) % 360.0) / 360.0
    inner = k < SPLIT
    wedge = (ang >= WEDGE[0]) & (ang < WEDGE[1])
    blk = np.floor(pts[:, 0] / 250.0).astype(np.int64) * 100 + (ang // 30.0).astype(np.int64)
    bins = np.floor(pts[:, 0] / 250.0).astype(np.int64) * 1000 + (ang // 10.0).astype(np.int64)
    phiO = ((ang - SEAM_O) % 360.0) / 360.0
    M = {}
    for name, p in ours_files(z0).items():
        if not p.exists():
            continue
        D = np.load(p)
        assert len(D["j"]) == len(k), f"{p}: other reference sample"
        j = D["j"].astype(np.float64)
        M[name] = {"cov": D["matched"].astype(bool), "dist": D["dist"].astype(np.float64),
                   "T": (lambda s, j=j: j + s * phiO), "groups": None, "file": str(p.relative_to(REPO))}
    K = np.load(H2H / f"kelly_z{z0:g}.npz")
    assert int(K["n_ref"]) == len(k)
    for tag in ("all", "good"):
        pid, dist = K[f"pid_{tag}"], K[f"dist_{tag}"].astype(np.float64)
        cov = pid >= 0
        kp = np.zeros(len(k)); thN = np.zeros(len(k)); comp = np.full(len(k), -1, np.int64); dth = np.zeros(len(k))
        u = np.unique(pid[cov])
        lut_k = np.array([KI[p][0] for p in u], float); lut_c = np.array([KI[p][1] for p in u])
        lut_t = np.array([np.degrees(KI[p][2]) for p in u])
        lut_m = np.array([med[p] for p in u])                                   # xyz
        lut_th = theta_h(lut_m[:, ::-1], C)                                      # medoid zyx
        ii = np.searchsorted(u, pid[cov])
        kp[cov], comp[cov], thN[cov] = lut_k[ii], lut_c[ii], lut_t[ii]
        dth[cov] = wrap180(thH[cov] - lut_th[ii])
        Tk = (lambda s, kp=kp, thN=thN, dth=dth: kp + s * (thN - dth) / 360.0)  # Kelly angle runs opposite to theta_H
        for conv in ("comp", "band"):
            if tag == "good" and conv == "band":
                continue
            M[f"k_{tag}_{conv}"] = {"cov": cov, "dist": dist, "T": Tk, "groups": (comp if conv == "comp" else None),
                                    "file": f"kelly_z{z0:g}.npz:{tag}"}
    if smoke:
        print(z0, {"n": len(k), "inner": int(inner.sum()), "E": int((inner & ~wedge).sum()), "methods": sorted(M)})
    return {"k": k, "TH": TH, "inner": inner, "wedge": wedge, "blk": blk, "bins": bins, "M": M, "z": [z0, z1]}


# ------------------------------------------------------------------------------------------------ scoring
def score_band(B, E, variant):
    """Correct and covered masks per method on evaluation mask E under a variant: primary | tol2 | lenient."""
    out, params = {}, {}
    for name, m in B["M"].items():
        cov = m["cov"] & E
        if variant == "tol2":
            cov &= m["dist"] < TOL2
        g = m["groups"]
        cor, prm, R = fit(B["TH"], m["T"], cov, None if g is None else np.where(cov, g, -1))
        if variant == "lenient":
            cor = fit_bins(R, cov, cor, B["bins"])
            prm["n_correct"] = int(cor.sum())
        out[name] = (cov, cor & cov)
        params[name] = prm
    return out, params


PAIRS = [("m2", "k_all_comp"), ("m2", "k_all_band"), ("m2", "k_good_comp"), ("base", "k_all_comp"),
         ("m2s2", "k_all_comp")]
OVERLAP_PAIR = ("m2", "k_all_comp")


def counts(B, E, S, ub, nb):
    """Per-block counts of every numerator/denominator used below."""
    c = {"E": np.bincount(ub[E], minlength=nb).astype(float)}
    bc = lambda x: np.bincount(ub[x], minlength=nb).astype(float)
    for n, (cov, cor) in S.items():
        c[f"cov:{n}"], c[f"cor:{n}"] = bc(cov), bc(cor)
    for a, b in PAIRS:
        if a not in S or b not in S:
            continue
        (ca, ra), (cb, rb) = S[a], S[b]
        both = ca & cb
        c[f"both:{a}:{b}"] = bc(both)
        c[f"corboth:{a}|{a}:{b}"], c[f"corboth:{b}|{a}:{b}"] = bc(ra & both), bc(rb & both)
        c[f"union:{a}:{b}"] = bc(ra | rb)
    a, b = OVERLAP_PAIR
    (ca, ra), (cb, rb) = S[a], S[b]
    cat = {"both_cover:both_right": ca & cb & ra & rb, "both_cover:only_ours_right": ca & cb & ra & ~rb,
           "both_cover:only_kelly_right": ca & cb & ~ra & rb, "both_cover:both_wrong": ca & cb & ~ra & ~rb,
           "only_ours_covers:right": ca & ~cb & ra, "only_ours_covers:wrong": ca & ~cb & ~ra,
           "only_kelly_covers:right": cb & ~ca & rb, "only_kelly_covers:wrong": cb & ~ca & ~rb,
           "neither_covers": E & ~ca & ~cb}
    for kname, x in cat.items():
        c[f"ov:{kname}"] = bc(x & E)
    return c


def summ(est, res):
    lo, hi = np.nanpercentile(res, [2.5, 97.5])
    r = lambda x: round(float(x) * 100, 3)
    return {"est": r(est), "ci": [r(lo), r(hi)]}


def ratio(C, idx, num, den):
    """C: list of per-band count dicts; idx: list of resample index arrays (same order). Pooled ratio est + resamples."""
    nu = sum(c[num].sum() for c in C); de = sum(c[den].sum() for c in C)
    rn = sum(c[num][ix].sum(1) for c, ix in zip(C, idx)); rd = sum(c[den][ix].sum(1) for c, ix in zip(C, idx))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nu / de if de else np.nan), rn / rd


def report(C, idx, methods, area_inner):
    """Metrics for one pool of bands (C, idx lists) -> dict."""
    out = {"n_E": int(sum(c["E"].sum() for c in C)), "methods": {}, "pairs": {}}
    R = {}
    for n in methods:
        R[n] = {"C": ratio(C, idx, f"cov:{n}", "E"), "A": ratio(C, idx, f"cor:{n}", f"cov:{n}"),
                "Q": ratio(C, idx, f"cor:{n}", "E")}
        q_est = R[n]["Q"][0]
        out["methods"][n] = {k: summ(*v) for k, v in R[n].items()}
        out["methods"][n]["correct_cm2"] = round(float(sum(
            c[f"cor:{n}"].sum() / c["E"].sum() * a for c, a in zip(C, area_inner))), 2)
        out["methods"][n]["Q_x_area_pooled_cm2"] = round(float(q_est * sum(area_inner)), 2)
    for a, b in PAIRS:
        if a not in methods or b not in methods:
            continue
        p = {}
        for m in ("C", "A", "Q"):
            (ea, xa), (eb, xb) = R[a][m], R[b][m]
            p[f"d{m}"] = summ(ea - eb, xa - xb)
        (ea, xa) = ratio(C, idx, f"corboth:{a}|{a}:{b}", f"both:{a}:{b}")
        (eb, xb) = ratio(C, idx, f"corboth:{b}|{a}:{b}", f"both:{a}:{b}")
        p["co_covered_share_of_E"] = summ(*ratio(C, idx, f"both:{a}:{b}", "E"))
        p[f"A_cocovered_{a}"], p[f"A_cocovered_{b}"], p["dA_cocovered"] = summ(ea, xa), summ(eb, xb), summ(ea - eb, xa - xb)
        eu, xu = ratio(C, idx, f"union:{a}:{b}", "E")
        p["union_Q"] = summ(eu, xu)
        best = np.maximum(R[a]["Q"][0], R[b]["Q"][0])
        xbest = np.where(R[a]["Q"][0] >= R[b]["Q"][0], R[a]["Q"][1], R[b]["Q"][1])
        p["union_minus_best_Q"] = summ(eu - best, xu - xbest)
        out["pairs"][f"{a}_vs_{b}"] = p
    ov = {}
    for key in C[0]:
        if key.startswith("ov:"):
            e, x = ratio(C, idx, key, "E")
            ov[key[3:]] = {"share": summ(e, x), "cm2": round(float(sum(c[key].sum() / c["E"].sum() * a
                                                                         for c, a in zip(C, area_inner))), 2)}
    out["overlap_m2_vs_k_all_comp"] = ov
    return out


def run(variant, bands, KI, med, Cc, areas, smoke=False):
    per, Cs, idxs, prm = [], [], [], []
    for i, (z0, z1, fresh) in enumerate(BANDS):
        if (z0, z1) not in bands:
            continue
        B = band_inputs(z0, z1, KI, med, Cc, smoke)
        if smoke:
            continue
        E = B["inner"] & (~B["wedge"] if variant != "nowedge" else True)
        S, params = score_band(B, E, "primary" if variant == "nowedge" else variant)
        ub = np.unique(B["blk"], return_inverse=True)[1]
        nb = int(ub.max()) + 1
        c = counts(B, E, S, ub, nb)
        ix = np.random.default_rng(2700 + i).integers(0, nb, (N_BOOT, nb))
        methods = [m for m in S]
        r = report([c], [ix], methods, [areas[z0]])
        r |= {"z": [z0, z1], "fresh": fresh, "params": params, "inner_cm2": areas[z0],
              "wedge_share_of_inner": round(float(B["wedge"][B["inner"]].mean()), 4)}
        per.append(r); Cs.append(c); idxs.append(ix); prm.append((z0, fresh, methods, None))
        print(f"[{variant}] z{z0}-{z1} done", flush=True)
    if smoke:
        return None
    pools = {}
    common = sorted(set.intersection(*[set(m) for _, _, m, _ in prm]))
    pools["all7"] = report(Cs, idxs, common, [areas[z0] for z0, *_ in prm])
    fr = [i for i, (z0, f, *_) in enumerate(prm) if f]
    pools["fresh5"] = report([Cs[i] for i in fr], [idxs[i] for i in fr], common, [areas[prm[i][0]] for i in fr])
    return {"per_band": per, "pooled": pools, "pooled_methods": common}


def kelly_wholescroll(KI, med, Cc, areas):
    """S-d: one (sigma, s, o) for Kelly-all over all 7 bands together (per-component groups ignored)."""
    TH, T, cov, band_of = [], {1: [], -1: []}, [], []
    for i, (z0, z1, _) in enumerate(BANDS):
        B = band_inputs(z0, z1, KI, med, Cc)
        E = B["inner"] & ~B["wedge"]
        m = B["M"]["k_all_band"]
        TH.append(B["TH"][E]); cov.append(m["cov"][E]); band_of.append(np.full(int(E.sum()), i))
        for s in (1, -1):
            T[s].append(m["T"](s)[E])
    TH, cov, band_of = np.concatenate(TH), np.concatenate(cov), np.concatenate(band_of)
    Ts = {s: np.concatenate(v) for s, v in T.items()}
    cor, prm, _ = fit(TH, lambda s: Ts[s], cov)
    out = {"params": prm, "A": round(float(cor.sum() / cov.sum()) * 100, 3), "Q": round(float(cor.sum() / len(cov)) * 100, 3),
           "C": round(float(cov.mean()) * 100, 3), "per_band": {}}
    for i, (z0, z1, _) in enumerate(BANDS):
        m = band_of == i
        out["per_band"][str(z0)] = {"A": round(float(cor[m].sum() / cov[m].sum()) * 100, 3),
                                    "Q": round(float(cor[m].mean()) * 100, 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--smoke", action="store_true", help="load every input and print shapes only")
    ap.add_argument("--variants", default="primary,nowedge,lenient,tol2,wholescroll")
    a = ap.parse_args()
    se.set_scroll("p1667")
    KI, med = kelly_tables()
    Cc = np.load(REPO / "data/eval/p1667/labels_merged_v4.npz")["centre"]
    bands = {(z0, z1) for z0, z1, _ in BANDS}
    if a.smoke:
        run("primary", bands, KI, med, Cc, None, smoke=True)
        return
    areas = {z0: wp.labelled_area(z0, z1)["inner"] for z0, z1, _ in BANDS}
    res = {"prereg": "notes/2026-10-27-kelly-headtohead-prereg.md", "areas_inner_cm2": areas}
    for v in a.variants.split(","):
        res[v] = kelly_wholescroll(KI, med, Cc, areas) if v == "wholescroll" else run(v, bands, KI, med, Cc, areas)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    p = res.get("primary")
    if p:
        for pool in ("all7", "fresh5"):
            q = p["pooled"][pool]
            print(pool, {n: (v["C"]["est"], v["A"]["est"], v["Q"]["est"]) for n, v in q["methods"].items()})
            print("   m2 vs k_all_comp", {k: v for k, v in q["pairs"]["m2_vs_k_all_comp"].items()})


if __name__ == "__main__":
    main()
