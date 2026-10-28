"""2.399 um sheet counting on chain steps: features, calibration (z 4500-6000 only) and band-1 evaluation.

Stage `feats --band z3000|z4500`: for every chain step of data/scratch/auto_windings/stepconf/feats_<band>.json
  1. radial registration: lag L (9.362 vox) maximising the NCC between the 9.362 CT profile and the 2.399 profile blurred
     to ~9 um, over [ra - 40, rb + 40] (the published affine is locally off by ~1-2 vox);
  2. the 2.399 profile (tube mean), smoothed with sigma s, normalised per window ((v - bg) / (p95 - bg), bg = p5);
  3. valleys strictly inside (ra + L + delta, rb + L - delta) and their prominences (scipy find_peaks on -profile over the
     whole window, so a valley's depth is measured against the higher of its two bases).
  The same is done on the 9.362 profile (control: does resolution add anything over the 9 um CT itself?).
  -> data/scratch/hires_step/vfeats_<band>.pkl

Stage `calib`: grid over (source, tube, sigma, delta, tau, min valley distance) on z 4500-6000 labels only.
Stage `evaluate`: frozen configuration applied to z 3000-4500, scored against band_oracle (raw and CT-relabelled).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
OUT = REPO / "data" / "scratch" / "hires_step"
SC = REPO / "data" / "scratch" / "auto_windings" / "stepconf"
DR = 0.125
TUBES = (1, 3)            # half width (2.399 vox) of the averaging tube: 3x3 or 7x7 lines
SIGMAS = (1.0, 2.0, 4.0)  # smoothing (samples of 0.125 9.362 vox = 0.49 2.399 vox)
DELTAS = (1.5, 3.0)       # endpoint exclusion (9.362 vox)
SIG9 = (1.0, 3.0)         # smoothing of the 9.362 profile (samples)
LAGMAX = 5.0


def load_prof(band):
    P = {}
    for f in sorted((OUT / f"prof_{band}").glob("*.npz")):
        if ".tmp" in f.name:
            continue
        d = np.load(f)
        P[(int(d["z"]), float(d["theta"]))] = d
    return P


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def valley_prom(s, lo, hi):
    """Prominences of valleys of s with index in (lo, hi); s normalised. Returns (positions, prominences)."""
    pk, pr = find_peaks(-s, prominence=0.02)
    m = (pk > lo) & (pk < hi)
    return pk[m], pr["prominences"][m]


def norm(v):
    bg, top = np.percentile(v, 5), np.percentile(v, 95)
    return (v - bg) / max(top - bg, 1.0)


def step_feats(d, ra, rb):
    r = d["r"]
    i0 = int(np.searchsorted(r, ra - 40))
    i1 = int(np.searchsorted(r, rb + 40))
    ia, ib = np.searchsorted(r, ra), np.searchsorted(r, rb)
    if ia <= 0 or ib >= len(r) - 1 or i1 - i0 < 40:
        return None
    p24 = d["p24"][i0:i1].astype(np.float32)
    p9 = d["p9"][i0:i1].astype(np.float32).mean((1, 2))
    if (p24[:, 3, 3] == 0).mean() > 0.2 or (p9 == 0).mean() > 0.2:
        return None  # outside the scanned / masked volume
    full = p24.mean((1, 2))
    blur = gaussian_filter1d(full, 6.0)  # ~0.75 9.362 vox: 2.399 degraded to ~9 um
    nl = int(LAGMAX / DR)
    best, lag = -2, 0
    for k in range(-nl, nl + 1, 2):  # p24 at index j ~ p9 at index j + k  ->  9.362 radius x appears at x - k*DR
        n = len(blur)
        a, b = (blur[:n - k], p9[k:]) if k >= 0 else (blur[-k:], p9[:n + k])
        if len(a) < 40 or len(b) != len(a):
            continue
        c = ncc(a, b)
        if c > best:
            best, lag = c, k
    L = -lag * DR  # 2.399 profile radius = 9.362 radius + L
    out = dict(lag=L, ncc=best, len=rb - ra)
    rr = r[i0:i1]
    for t in TUBES:
        prof = p24[:, 3 - t:4 + t, 3 - t:4 + t].mean((1, 2))
        for s in SIGMAS:
            sv = norm(gaussian_filter1d(prof, s))
            for de in DELTAS:
                lo = np.searchsorted(rr, ra + L + de)
                hi = np.searchsorted(rr, rb + L - de)
                pk, pr = valley_prom(sv, lo, hi)
                out[f"v24_{t}_{s:g}_{de:g}"] = (rr[pk] - L - ra, pr)
    for s in SIG9:
        sv = norm(gaussian_filter1d(p9, s))
        for de in DELTAS:
            lo = np.searchsorted(rr, ra + de)
            hi = np.searchsorted(rr, rb - de)
            pk, pr = valley_prom(sv, lo, hi)
            out[f"v9_{s:g}_{de:g}"] = (rr[pk] - ra, pr)
    return out


def feats(args):
    rows = json.loads((SC / f"feats_{args.band}.json").read_text())
    P = load_prof(args.band)
    out = []
    for i, r in enumerate(rows):
        d = P.get((r["z"], r["theta"]))
        f = None if d is None else step_feats(d, r["ra"], r["rb"])
        out.append(f)
        if (i + 1) % 5000 == 0:
            print(i + 1, flush=True)
    (OUT / f"vfeats_{args.band}.pkl").write_bytes(pickle.dumps(out))
    print(f"{sum(f is not None for f in out)} / {len(out)} steps with profiles")


def count(f, key, tau, mind):
    """Number of valleys with prominence >= tau; valleys closer than mind (9.362 vox) count once."""
    pos, pr = f[key]
    p = np.sort(pos[pr >= tau])
    if len(p) == 0:
        return 0
    return 1 + int((np.diff(p) >= mind).sum())


# ------------------------------------------------------------------------------------------------- labels / features
FKEYS = ("v24_3_2_1.5", "v24_1_1_3", "v24_3_4_3", "v9_3_1.5")
RELABEL = REPO / "data" / "scratch" / "label_audit" / "relabel_band1.json"
SECTOR = (145.0, 260.0)    # villa #2167 sector: no global label map holds there (relabel_band1 verdict)
ORACLE = {"z3000": REPO / "data/scratch/auto_windings/band_oracle", "z4500": REPO / "data/scratch/auto_windings/band_oracle_z4500"}


def vec(f):
    """Per-step 2.399 / 9.362 profile features (fixed list, no fitting)."""
    out = [f["lag"], f["ncc"]]
    for k in FKEYS:
        pr = np.sort(f[k][1])[::-1]
        top = [pr[j] if len(pr) > j else 0.0 for j in range(3)]
        out += top + [float((pr >= t).sum()) for t in (0.1, 0.2, 0.3, 0.5)]
    return out


def vnames():
    n = ["lag", "ncc"]
    for k in FKEYS:
        n += [f"{k}_top{j}" for j in range(1, 4)] + [f"{k}_n{t}" for t in (0.1, 0.2, 0.3, 0.5)]
    return n


def load(band):
    import auto_windings_stepconf as sc
    rows, ch, tws = sc.load_band(band)
    F = pickle.loads((OUT / f"vfeats_{band}.pkl").read_bytes())
    rays = sc.oracle_rays(ORACLE[band])
    for r, f in zip(rows, F):
        r["has24"] = f is not None
        o = [x for x in rays.get((r["z"], r["theta"]), []) if r["ra"] + 2 < x[0] < r["rb"] - 2]
        r["n_between"] = len(o)
    return rows, ch, tws, F


def tws_map(tws, key):
    if key == "raw":
        return tws
    m = {int(k): v for k, v in json.loads(RELABEL.read_text())[key]["map"].items()}
    return [[None if w is None else m.get(int(w), w) for w in tw] for tw in tws]


def step_truth(rows, tws):
    t = []
    for r in rows:
        tw = tws[r["chain"]]
        a, b = tw[r["k"]], tw[r["k"] + 1]
        t.append(None if a is None or b is None else b - a)
    return t


def in_sector(r):
    return SECTOR[0] <= r["theta"] <= SECTOR[1]


def touches(r, tws, bad):
    tw = tws[r["chain"]]
    a, b = tw[r["k"]], tw[r["k"] + 1]
    if a is None or b is None:
        return False
    return any(lo <= w <= hi for w in range(min(a, b), max(a, b) + 1) for lo, hi in bad)


# ------------------------------------------------------------------------------------------------- calibration
BAD_CLEAN = [(33, 37), (40, 47)]   # label-suspect windings (notes 10-20 / 10-21): excluded from calibration
COUNT_KEYS = [f"v24_{t}_{s:g}_{d:g}" for t in TUBES for s in SIGMAS for d in DELTAS] + \
             [f"v9_{s:g}_{d:g}" for s in SIG9 for d in DELTAS]
TAUS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
MINDS = (0.0, 2.0, 4.0)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(rows, F, idx, kind):
    g = [[logit(rows[i]["gm_p"]), rows[i]["Ir"], rows[i]["I"]] for i in idx]
    if kind == "gm":
        X = g
    elif kind == "gm+24":
        X = [a + [v for n, v in zip(vnames(), vec(F[i])) if not n.startswith("v9")] for a, i in zip(g, idx)]
    elif kind == "gm+9":
        X = [a + [v for n, v in zip(vnames(), vec(F[i])) if n.startswith("v9") or n in ("lag", "ncc")] for a, i in zip(g, idx)]
    elif kind == "24":
        X = [[v for n, v in zip(vnames(), vec(F[i])) if not n.startswith("v9")] for i in idx]
    elif kind == "9":
        X = [[v for n, v in zip(vnames(), vec(F[i])) if n.startswith("v9")] for i in idx]
    return np.array(X, float)


def mk():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def auc(s, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    rk = rankdata(s)
    n1 = y.sum()
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def calib(args):
    rows, ch, tws, F = load("z4500")
    tr = step_truth(rows, tws)
    M = [i for i, r in enumerate(rows) if r["has24"] and tr[i] is not None]
    C = [i for i in M if not touches(rows[i], tws, BAD_CLEAN)]
    print(f"z4500: {len(rows)} steps, {len(M)} matched with profiles, {len(C)} clean", flush=True)
    res = dict(n_matched=len(M), n_clean=len(C))
    # (b) replacement counter: best accuracy on clean matched steps
    best = []
    T = np.array([tr[i] for i in C])
    for k in COUNT_KEYS:
        for tau in TAUS:
            for md in MINDS:
                c = np.array([count(F[i], k, tau, md) for i in C])
                best.append((float((c == T).mean()), float((c - T).mean()), k, tau, md))
    best.sort(key=lambda x: -x[0])
    res["replace_top"] = best[:10]
    res["replace_best_24"] = [b for b in best if b[2].startswith("v24")][0]
    res["replace_best_9"] = [b for b in best if b[2].startswith("v9")][0]
    print("replace best:", best[:3], flush=True)
    # (c) veto of gm raises: keep a raise only if count >= 2; choose config maximising clean accuracy on raised steps
    R = [i for i in C if rows[i]["d"] > rows[i]["d0"]]
    TR = np.array([tr[i] for i in R])
    vb = []
    for k in COUNT_KEYS:
        for tau in TAUS:
            for md in MINDS:
                keep = np.array([count(F[i], k, tau, md) >= 2 for i in R])
                dn = np.array([rows[i]["d"] if kp else rows[i]["d0"] for i, kp in zip(R, keep)])
                vb.append((float((dn == TR).mean()), float(keep.mean()), k, tau, md))
    vb.sort(key=lambda x: -x[0])
    res["veto_asis_acc"] = float((np.array([rows[i]["d"] for i in R]) == TR).mean())
    res["veto_revert_all_acc"] = float((np.array([rows[i]["d0"] for i in R]) == TR).mean())
    res["veto_top"] = vb[:5]
    res["veto_best_24"] = [b for b in vb if b[2].startswith("v24")][0]
    res["veto_best_9"] = [b for b in vb if b[2].startswith("v9")][0]
    print(f"raised steps {len(R)}: as is {res['veto_asis_acc']:.3f}, revert all {res['veto_revert_all_acc']:.3f}, "
          f"veto best {vb[:2]}", flush=True)
    # raise models among d0 == 1 steps (clean, truth 1 or 2)
    D = [i for i in C if rows[i]["d0"] == 1 and tr[i] in (1, 2)]
    y = np.array([tr[i] == 2 for i in D])
    slab = np.array([rows[i]["z"] // 500 for i in D])
    res["raise_n"], res["raise_pos"] = len(D), int(y.sum())
    models = {}
    for kind in ("gm", "gm+24", "gm+9", "24", "9"):
        X = design(rows, F, D, kind)
        oof = np.zeros(len(D))
        for s_ in np.unique(slab):
            m = mk().fit(X[slab != s_], y[slab != s_])
            oof[slab == s_] = m.predict_proba(X[slab == s_])[:, 1]
        m = mk().fit(X, y)
        # thresholds on OOF scores: (i) zero bias = as many raises as true doubles; (ii) max F1
        srt = np.sort(oof)[::-1]
        thr0 = float(srt[int(y.sum()) - 1])
        f1b = max(((2 * ((oof >= t) & y).sum() / ((oof >= t).sum() + y.sum()), t) for t in np.unique(np.round(oof, 3))))
        models[kind] = dict(auc_oof=auc(oof, y), thr_zero_bias=thr0, thr_f1=float(f1b[1]), f1_oof=float(f1b[0]),
                            model=m)
        pr = (oof >= thr0)
        print(f"raise model {kind:6s}: OOF AUC {auc(oof, y):.3f}; at zero-bias thr {thr0:.3f}: precision "
              f"{(pr & y).sum() / max(pr.sum(), 1):.3f} recall {(pr & y).sum() / y.sum():.3f}; F1-max {f1b[0]:.3f}", flush=True)
    res["raise_models"] = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in models.items()}
    (OUT / "calib_z4500.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "calib_z4500_models.pkl").write_bytes(pickle.dumps({k: v["model"] for k, v in models.items()}))


# ------------------------------------------------------------------------------------------------- evaluation (z3000)
def drift(ch, tws, wind, chains=None):
    e = []
    for ci, c in enumerate(ch):
        if chains is not None and ci not in chains:
            continue
        w, tw = wind[ci], tws[ci]
        m = [k for k in range(len(w)) if tw[k] is not None]
        for a in range(len(m)):
            for b in range(a + 1, len(m)):
                t = tw[m[b]] - tw[m[a]]
                if 10 <= t <= 20:
                    e.append(w[m[b]] - w[m[a]] - t)
    e = np.array(e)
    return dict(drift=round(float(e.mean()), 3), within1=round(100 * float((np.abs(e) <= 1).mean()), 1),
                sd=round(float(e.std()), 2), pairs=int(len(e)))


def evaluate(args):
    cal = json.loads((OUT / "calib_z4500.json").read_text())
    models = pickle.loads((OUT / "calib_z4500_models.pkl").read_bytes())
    rows, ch, tws_raw, F = load("z3000")
    n = len(rows)
    d_gm = np.array([r["d"] for r in rows])
    d0 = np.array([r["d0"] for r in rows])
    has = np.array([r["has24"] for r in rows])
    _, _, kb, taub, mdb = cal["replace_best_24"]
    cnt = np.array([count(F[i], kb, taub, mdb) if has[i] else -1 for i in range(n)])
    _, _, kv, tauv, mdv = cal["veto_best_24"]
    cntv = np.array([count(F[i], kv, tauv, mdv) if has[i] else -1 for i in range(n)])
    pol = {"(a0) raw chain d0": d0.copy(), "(a) gmx chain d": d_gm.copy()}
    b = d_gm.copy()
    b[has] = cnt[has]
    pol["(b) 2.4um count replaces d"] = b
    c = d_gm.copy()
    veto = has & (d_gm > d0) & (cntv < 2)
    c[veto] = d0[veto]
    pol["(c) 2.4um vetoes gm raises (count < 2)"] = c
    one = np.flatnonzero(d0 == 1)
    for kind in ("gm", "gm+24", "gm+9"):
        ok = [i for i in one if has[i]]
        X = design(rows, F, ok, kind)
        p = models[kind].predict_proba(X)[:, 1]
        for tk in ("thr_zero_bias", "thr_f1"):
            dd = d_gm.copy()
            dd[one] = 1
            dd[np.array(ok)[p >= cal["raise_models"][kind][tk]]] = 2
            pol[f"(c') raise model {kind}, {tk}"] = dd
    pol["(c'') no raises at all (d0)"] = d0.copy()
    out = {"config": dict(replace=cal["replace_best_24"], veto=cal["veto_best_24"]), "coverage": float(has.mean())}
    labs = {"raw": tws_raw, "supported": tws_map(tws_raw, "supported")}
    sector = np.array([in_sector(r) for r in rows])
    chains_out = {ci for ci, c_ in enumerate(ch) if not (SECTOR[0] <= c_["theta"] <= SECTOR[1])}
    table = {}
    for lk, tws in labs.items():
        tr = step_truth(rows, tws)
        mt = np.array([t is not None for t in tr])
        T = np.array([t if t is not None else -99 for t in tr])
        for sub, msk, chs in (("all", np.ones(n, bool), None), ("outside #2167", ~sector, chains_out)):
            key = f"{lk} / {sub}"
            table[key] = {}
            for pn, dd in pol.items():
                m = mt & msk
                wind = []
                for ci, c_ in enumerate(ch):
                    wind.append(None)
                ws = {}
                for i, r in enumerate(rows):
                    ws.setdefault(r["chain"], {})[r["k"]] = dd[i]
                wind = [np.r_[0, np.cumsum([ws[ci][k] for k in range(len(c_["r"]) - 1)])].tolist() for ci, c_ in enumerate(ch)]
                s1 = m & (d0 == 1)
                rz = s1 & (dd >= 2)
                t2 = s1 & (T == 2)
                t1 = s1 & (T == 1)
                table[key][pn] = dict(matched=int(m.sum()), precision=round(100 * float((dd[m] == T[m]).mean()), 2),
                                      under=round(100 * float((dd[m] < T[m]).mean()), 2),
                                      over=round(100 * float((dd[m] > T[m]).mean()), 2),
                                      raises=int(rz.sum()), raise_precision=round(100 * float((rz & t2).sum() / max(rz.sum(), 1)), 1),
                                      raise_recall=round(100 * float((rz & t2).sum() / max(t2.sum(), 1)), 1),
                                      false_raise_rate=round(100 * float((rz & t1).sum() / max(t1.sum(), 1)), 2),
                                      **drift(ch, tws, wind, chs))
    out["table"] = table
    # confusion: strata x 2.4 count (replace config), raw and supported labels
    conf = {}
    for lk, tws in labs.items():
        tr = step_truth(rows, tws)
        T = np.array([t if t is not None else -99 for t in tr])
        strata = {"d=1 (not raised), truth 1": (d0 == 1) & (d_gm == 1) & (T == 1),
                  "d=1 (not raised), truth 2 (undercount)": (d0 == 1) & (d_gm == 1) & (T == 2),
                  "d=1 (not raised), truth 2, mesh crossing between": (d0 == 1) & (d_gm == 1) & (T == 2) & np.array([r["n_between"] >= 1 for r in rows]),
                  "gm-raised (1->2), truth 2 (raise right)": (d0 == 1) & (d_gm == 2) & (T == 2),
                  "gm-raised (1->2), truth 1 (raise wrong)": (d0 == 1) & (d_gm == 2) & (T == 1),
                  "graph d0=2, truth 2": (d0 == 2) & (T == 2),
                  "graph d0=2, truth 1": (d0 == 2) & (T == 1),
                  "any d, truth >= 3": (T >= 3)}
        conf[lk] = {}
        for sn, m in strata.items():
            m = m & has
            cc = cnt[m]
            cv = cntv[m]
            conf[lk][sn] = dict(n=int(m.sum()), **{f"count{j}": int((np.minimum(cc, 3) == j).sum()) for j in range(4)},
                                veto_count_ge2=round(100 * float((cv >= 2).mean()), 1) if m.sum() else None)
    out["confusion"] = conf

    # head-to-head raise ranking among d0 == 1 steps with profiles: same number of raises as the gm rule
    ok = np.array([i for i in one if has[i]])
    scores = {"gm_p (stored rule score)": np.array([rows[i]["gm_p"] for i in ok])}
    for kind in ("gm", "gm+24", "gm+9", "24", "9"):
        scores[f"model {kind}"] = models[kind].predict_proba(design(rows, F, list(ok), kind))[:, 1]
    h2h = {}
    for lk, tws in labs.items():
        tr = step_truth(rows, tws)
        T = np.array([tr[i] if tr[i] is not None else -99 for i in ok])
        msk = np.isin(T, (1, 2))
        K = int(((d_gm[ok] >= 2) & msk).sum())
        y = T[msk] == 2
        h2h[lk] = {"K_matched_raises": K, "true_doubles": int(y.sum()), "singles": int((~y).sum())}
        for sn, sc_ in scores.items():
            v = sc_[msk]
            o = np.argsort(-v)[:K]
            h2h[lk][sn] = dict(auc=round(auc(v, y), 3), precision_at_K=round(100 * float(y[o].mean()), 1),
                               recall_at_K=round(100 * float(y[o].sum() / y.sum()), 1),
                               false_raise_rate_at_K=round(100 * float((~y[o]).sum() / (~y).sum()), 2))
    out["head_to_head"] = h2h
    print(json.dumps(h2h, indent=1))
    (OUT / "eval_z3000.json").write_text(json.dumps(out, indent=1, default=float))
    for key, tb in table.items():
        print(f"\n== {key}")
        for pn, v in tb.items():
            print(f"{pn:45s} {v}")
    for lk, cf in conf.items():
        print(f"\n== confusion ({lk} labels): rows = chain stratum, cols = 2.4um count {kb} tau {taub} mind {mdb}")
        for sn, v in cf.items():
            print(f"{sn:55s} {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feats")
    p.add_argument("--band", choices=["z3000", "z4500"], required=True)
    sub.add_parser("calib")
    sub.add_parser("evaluate")
    a = ap.parse_args()
    {"feats": feats, "calib": calib, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
