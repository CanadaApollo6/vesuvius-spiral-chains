"""Per-step confidence for gm-corrected ray chains (band_m7_ray_gmx) and precision-raising filters.

Stage `feats` (per band): re-assembles the ray chains from the crop checkpoints exactly as
`auto_windings.py band --reassemble --conf 1.0 --bridge 1 --max-step 2` does, but keeps every per-step graph feature
(residual, non-integrality, degree, run lengths, second-crop agreement, crossings merged into the step, ...), applies
the grad_mag rule (gm_p, its margin from the threshold, line integral) and adds neighbour-ray consistency features.
Labels (human-mesh oracle on the same rays) are attached for scoring only.

For the fitting band (z 4500-6000) the gm rule is applied out of fold (refitted with each 500-z slab left out), so its
probabilities behave like they do on z 3000-4500, where the rule never saw labels.

Stage `fit`: confidence model fitted ONLY on z 4500-6000 (out-of-fold by slab for threshold selection); thresholds
are picked on z 4500-6000 and applied unchanged to z 3000-4500.

No PHerc0139 z 3000-4500 label is read by `fit` except for the final scoring table.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_windings as aw  # noqa: E402

CACHE = aw.CACHE
OUT = CACHE / "stepconf"
BANDS = {
    "z3000": dict(crops=CACHE / "band_m7" / "crops", ref=CACHE / "band_m7_ray", ref_gm=CACHE / "band_m7_ray_gmx",
                  oracle=CACHE / "band_oracle", z=(3000, 4500)),
    "z4500": dict(crops=CACHE / "band_m7_z4500" / "crops", ref=CACHE / "band_m7_z4500_ray", ref_gm=None,
                  oracle=CACHE / "band_oracle_z4500", z=(4500, 6000)),
}
GM_MODEL = Path(aw.__file__).with_name("auto_windings_gm_z4500.json")
GM_THR = 0.19


# ------------------------------------------------------------------------------------------------- assembly replica
def assemble_feats(results, max_step=2, require_agree=1):
    """Replica of aw.assemble with conf 1.0 / bridge 1; returns chains with per-step feature dicts ("sf")."""
    conf = json.loads(aw.CONF_JSON.read_text())
    model = conf["model"]
    rays = {}
    for c, res in results:
        if not res:
            continue
        for z, seg in res.items():
            rays.setdefault((z, c["theta"]), []).append((seg["tc"], seg["cr"]))
    chains = []
    for (z, th), lst in sorted(rays.items()):
        lst.sort(key=lambda x: x[0])
        allx = sorted([(x["t"], abs(x["t"] - tc), x) for tc, cr in lst for x in cr], key=lambda v: v[0])
        own, cl = [], []
        for v in allx + [(1e18, 0, None)]:
            if cl and v[0] - cl[-1][0] > 2:
                own.append(min(cl, key=lambda q: q[1])[2])
                cl = []
            cl.append(v)

        def pair(a, b):
            hold = []
            for tc, cr in lst:
                ia = [x for x in cr if abs(x["t"] - a["t"]) <= 2 and "w" in x]
                ib = [x for x in cr if abs(x["t"] - b["t"]) <= 2 and "w" in x]
                if ia and ib:
                    hold.append((abs((a["t"] + b["t"]) / 2 - tc), ia[0], ib[0], cr))
            hold.sort(key=lambda h: h[0])
            return hold
        seq, merged = [], []
        for x in own:
            if seq:
                h = pair(seq[-1], x)
                if h and round(h[0][2]["w"] - h[0][1]["w"]) == 0:
                    merged[-1] += 1  # a mask run folded into the previous crossing (same solved winding)
                    continue
            seq.append(x)
            merged.append(0)
        st = []
        for k, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
            h = pair(a, b)
            if not h:
                st.append(None)
                continue
            f = aw.step_feats(h[0][1], h[0][2])
            f["agree"] = None if len(h) < 2 else int(round(h[1][2]["w"] - h[1][1]["w"]) == f["dr"])
            f["n_hold"] = len(h)
            f["d2"] = None if len(h) < 2 else h[1][2]["w"] - h[1][1]["w"]
            f["merged"] = merged[k]  # runs merged into the inner crossing
            # runs of the best crop strictly between the two crossings (merged / unsolved ones)
            f["between"] = sum(1 for x in h[0][3] if a["t"] + 2 < x["t"] < b["t"] - 2)
            f["rl_a"], f["rl_b"] = a.get("runlen", 0.0), b.get("runlen", 0.0)
            f["res_a"], f["res_b"] = h[0][1]["res"], h[0][2]["res"]
            f["centre_off"] = h[0][0]  # distance of step midpoint from the best crop's segment centre
            st.append(f)
        pe = [None if (s is None or s["dr"] < 1 or s["dr"] > max_step) else float(aw.p_err(model, [s])[0]) for s in st]
        br = [p is None for p in pe]
        good = []
        for s, p, b in zip(st, pe, br):
            g = p is not None
            if g and require_agree and s["agree"] == 0:
                g = False
            good.append(g or b)
        cur = None
        for k, g in enumerate(good + [False]):
            if g:
                if cur is None:
                    cur = dict(z=z, theta=th, pts=[seq[k]["gpos"]], wind=[0], r=[seq[k]["t"]], perr=[], sf=[])
                cur["pts"].append(seq[k + 1]["gpos"])
                cur["wind"].append(cur["wind"][-1] + (1 if br[k] else st[k]["dr"]))
                cur["r"].append(seq[k + 1]["t"])
                cur["perr"].append(1.0 if br[k] else pe[k])
                cur["sf"].append(None if br[k] else st[k])
            elif cur is not None:
                chains.append(cur)
                cur = None
    return chains


def load_results(crops: Path):
    return [pickle.loads(p.read_bytes()) for p in sorted(crops.glob("*.pkl"))]


# ------------------------------------------------------------------------------------------------- gm rule
def gm_rows_labels(ch, feats, tws):
    X, Y, zz = [], [], []
    for c, f, tw in zip(ch, feats, tws):
        for k, r in enumerate(f):
            if c["wind"][k + 1] - c["wind"][k] != 1 or tw[k] is None or tw[k + 1] is None:
                continue
            t = tw[k + 1] - tw[k]
            if t in (1, 2):
                X.append([r[g] for g in aw.GM_FEATURES])
                Y.append(float(t == 2))
                zz.append(c["z"])
    return np.array(X), np.array(Y), np.array(zz)


def oracle_rays(path: Path):
    return {(int(k.split("|")[0]), float(k.split("|")[1])): v
            for k, v in json.loads((path / "rays.json").read_text()).items()}


# ------------------------------------------------------------------------------------------------- neighbours
def ray_index(ch):
    idx = {}
    for i, c in enumerate(ch):
        idx.setdefault((int(c["z"]), float(c["theta"])), []).append(i)
    return idx


def w_delta(ch, ids, r1, r2):
    """Interpolated winding change between radii r1 < r2 on a ray (one chain must span both)."""
    for i in ids:
        r = ch[i]["r"]
        if r[0] <= r1 and r2 <= r[-1]:
            w = ch[i]["wind"]
            return float(np.interp(r2, r, w) - np.interp(r1, r, w))
    return None


def neighbour_feats(ch, W=48.0, dz=64, dth=15.0):
    idx = ray_index(ch)
    out = []
    for c in ch:
        z, th = int(c["z"]), float(c["theta"])
        r, w = np.array(c["r"]), np.array(c["wind"], float)
        nz = [k for k in [(z - dz, th), (z + dz, th)] if k in idx]
        nt = [k for k in [(z, (th - dth) % 360), (z, (th + dth) % 360)] if k in idx]
        rows = []
        for k in range(len(r) - 1):
            ra, rb = r[k], r[k + 1]
            d = w[k + 1] - w[k]
            r1, r2 = max(ra - W, r[0]), min(rb + W, r[-1])
            selfw = float(np.interp(r2, r, w) - np.interp(r1, r, w))
            fz = {}
            for tag, nbs in (("z", nz), ("t", nt)):
                st = [w_delta(ch, idx[q], ra, rb) for q in nbs]
                st = [x for x in st if x is not None]
                wn = [w_delta(ch, idx[q], r1, r2) for q in nbs]
                wn = [x for x in wn if x is not None]
                fz[f"nb{tag}_n"] = len(wn)
                fz[f"nb{tag}_step"] = d - float(np.mean(st)) if st else 0.0
                fz[f"nb{tag}_win"] = selfw - float(np.mean(wn)) if wn else 0.0
                fz[f"nb{tag}_has"] = float(bool(st))
            rows.append(fz)
        out.append(rows)
    return out


# ------------------------------------------------------------------------------------------------- feats stage
def feats(args):
    OUT.mkdir(parents=True, exist_ok=True)
    B = BANDS[args.band]
    print("loading crops", flush=True)
    ch = assemble_feats(load_results(B["crops"]))
    ref = json.loads((B["ref"] / "chains.json").read_text())
    same = len(ref) == len(ch) and all(a["pts"] == b["pts"] and a["wind"] == b["wind"] for a, b in zip(ref, ch))
    print(f"replica chains {len(ch)} vs ref {len(ref)}; identical: {same}", flush=True)
    assert same, "assembly replica differs from the reference ray chains"
    G, za = aw.gm_volume(*aw._zrange(ch))
    gmf = [aw.gm_chain_features(c, G, za) for c in ch]
    rays = oracle_rays(B["oracle"])
    tws = [aw._oracle_tw(c, rays) for c in ch]
    gm = json.loads(GM_MODEL.read_text())
    if args.band == "z4500":
        # out-of-fold gm rule: refit on the other two 500-z slabs (same rows as gmfit --half -1)
        X, Y, zz = gm_rows_labels(ch, gmf, tws)
        slab = lambda z: int(z) // 500
        models = {}
        for s in sorted({slab(c["z"]) for c in ch}):
            m = aw.fit_logistic(X[np.array([slab(v) != s for v in zz])], Y[np.array([slab(v) != s for v in zz])])
            m["features"] = aw.GM_FEATURES
            models[s] = m
        gm_ps = [aw.p_err(models[slab(c["z"])], f) if f else np.zeros(0) for c, f in zip(ch, gmf)]
        gm_in = [aw.p_err(gm["model"], f) if f else np.zeros(0) for f in gmf]
    else:
        gm_ps = [aw.p_err(gm["model"], f) if f else np.zeros(0) for f in gmf]
        gm_in = gm_ps
    for c, P in zip(ch, gm_ps):
        d = np.diff(c["wind"])
        c["wind_raw"] = list(c["wind"])
        c["wind"] = np.r_[0, np.cumsum(d + ((d == 1) & (P >= GM_THR)))].astype(int).tolist()
        c["gm_p"] = [float(x) for x in P]
    if B["ref_gm"] is not None:
        rg = json.loads((B["ref_gm"] / "chains.json").read_text())
        ok = all(a["wind"] == b["wind"] for a, b in zip(rg, ch))
        print(f"gm windings identical to {B['ref_gm'].name}: {ok}", flush=True)
        assert ok
    nbf = neighbour_feats(ch)
    rows = []
    lg = lambda v: float(np.log(v + 1e-3))
    logit = lambda p: float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
    for ci, c in enumerate(ch):
        tw = tws[ci]
        for k in range(len(c["r"]) - 1):
            s = c["sf"][k]
            g = gmf[ci][k]
            d = c["wind"][k + 1] - c["wind"][k]
            d0 = c["wind_raw"][k + 1] - c["wind_raw"][k]
            p = c["gm_p"][k]
            lab = None
            if tw[k] is not None and tw[k + 1] is not None:
                lab = int(d == tw[k + 1] - tw[k])
            row = dict(chain=ci, k=k, z=int(c["z"]), theta=float(c["theta"]), ra=c["r"][k], rb=c["r"][k + 1],
                       d=d, d0=d0, raised=int(d > d0), bridged=int(s is None), graph2=int(d0 == 2),
                       gm_p=p, gm_margin=logit(p) - logit(GM_THR), gm_absmargin=abs(logit(p) - logit(GM_THR)),
                       gm_p_in=float(gm_in[ci][k]),
                       I=g["I"], Ir=g["Ir"], Lr=g["Lr"], logL=g["logL"], Iloc=g["Iloc"], cov=g["cov"],
                       Idis=g["I"] / max(g["Iloc"], 1e-3) - d, logperr=g["logperr"],
                       label=lab, true_d=None if lab is None else tw[k + 1] - tw[k])
            if s is None:
                row.update(l_res=0.0, l_dfrac=0.0, logdeg=0.0, runlen=0.0, graze=0.0, multi=0, l_ray_dfrac=0.0,
                           l_ray_res=0.0, l_ray_resmax=0.0, ray_dr2=0.0, agree=-1, n_hold=0, d2dis=0.0, merged=0,
                           between=0, rl_a=0.0, rl_b=0.0, l_res_a=0.0, l_res_b=0.0, centre_off=0.0, has_graph=0)
            else:
                row.update({f: s[f] for f in ("l_res", "l_dfrac", "logdeg", "runlen", "graze", "multi", "l_ray_dfrac",
                                              "l_ray_res", "l_ray_resmax", "ray_dr2")},
                           agree=-1 if s["agree"] is None else s["agree"], n_hold=s["n_hold"],
                           d2dis=0.0 if s["d2"] is None else abs(s["d2"] - s["d"]), merged=s["merged"],
                           between=s["between"], rl_a=s["rl_a"], rl_b=s["rl_b"], l_res_a=lg(s["res_a"]),
                           l_res_b=lg(s["res_b"]), centre_off=s["centre_off"], has_graph=1)
            row.update(nbf[ci][k])
            rows.append(row)
    for c in ch:
        c.pop("sf")
    out = OUT / f"feats_{args.band}.json"
    out.write_text(json.dumps(rows))
    (OUT / f"chains_{args.band}.json").write_text(json.dumps(ch))
    lab = [r["label"] for r in rows if r["label"] is not None]
    print(f"{len(rows)} steps, {len(lab)} matched, precision {100 * np.mean(lab):.2f} %", flush=True)


# ------------------------------------------------------------------------------------------------- fit / filter stage
FEATS = ["d", "raised", "bridged", "graph2", "gm_p", "I", "Ir", "Lr", "logL", "Iloc", "cov", "Idis", "logperr",
         "l_res", "l_dfrac", "logdeg", "runlen", "graze", "multi", "l_ray_dfrac", "l_ray_res", "l_ray_resmax", "ray_dr2",
         "agree", "n_hold", "d2dis", "merged", "between", "rl_a", "rl_b", "l_res_a", "l_res_b", "centre_off",
         "nbz_n", "nbz_step", "nbz_win", "nbt_n", "nbt_step", "nbt_win", "ctx_raised3", "ctx_gm1", "ctx_lres3"]
NO_SITE = [f for f in FEATS if f != "logperr"]  # variant without the site-fitted certifier p_err


def add_context(rows):
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["chain"], []).append(i)
    for ids in by.values():
        ids.sort(key=lambda i: rows[i]["k"])
        rs = np.array([rows[i]["raised"] for i in ids], float)
        gp = np.array([rows[i]["gm_p"] for i in ids], float)
        lr = np.array([rows[i]["l_res"] for i in ids], float)
        n = len(ids)
        for j, i in enumerate(ids):
            a, b = max(0, j - 3), min(n, j + 4)
            rows[i]["ctx_raised3"] = float(rs[a:b].sum() - rs[j])
            nb = [gp[q] for q in (j - 1, j + 1) if 0 <= q < n]
            rows[i]["ctx_gm1"] = float(max(nb)) if nb else 0.0
            rows[i]["ctx_lres3"] = float(lr[a:b].mean())
    return rows


def load_band(band):
    rows = add_context(json.loads((OUT / f"feats_{band}.json").read_text()))
    ch = json.loads((OUT / f"chains_{band}.json").read_text())
    rays = oracle_rays(BANDS[band]["oracle"])
    tws = [aw._oracle_tw(c, rays) for c in ch]
    return rows, ch, tws


def make_model(kind):
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=60,
                                              l2_regularization=1.0, random_state=0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def Xof(rows, feats_):
    return np.array([[float(r[f]) for f in feats_] for r in rows])


def oof_scores(rows, feats_, kind):
    """P(step correct) on z 4500-6000: model fitted on matched steps of the other two 500-z slabs."""
    X = Xof(rows, feats_)
    lab = np.array([-1 if r["label"] is None else r["label"] for r in rows])
    slab = np.array([r["z"] // 500 for r in rows])
    s = np.zeros(len(rows))
    for k in np.unique(slab):
        tr = (slab != k) & (lab >= 0)
        m = make_model(kind).fit(X[tr], lab[tr])
        s[slab == k] = m.predict_proba(X[slab == k])[:, 1]
    return s


def auc(score, y):
    from scipy.stats import rankdata
    rk = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pieces_of(ch, rows, keep, min_pts=2):
    """Split every chain at dropped steps; returns [(chain index, first point, last point)] with >= min_pts points."""
    by = {}
    for r, kp in zip(rows, keep):
        by.setdefault(r["chain"], {})[r["k"]] = kp
    out = []
    for ci, c in enumerate(ch):
        n = len(c["r"])
        ks = by.get(ci, {})
        s0 = 0
        for k in range(n - 1 + 1):
            if k == n - 1 or not ks.get(k, False):
                if k - s0 + 1 >= min_pts:
                    out.append((ci, s0, k))
                s0 = k + 1
    return out


def evaluate(ch, tws, rows, keep, min_pts=2):
    pcs = pieces_of(ch, rows, keep, min_pts)
    kept_step = set()
    for ci, a, b in pcs:
        kept_step.update((ci, k) for k in range(a, b))
    ok = n = 0
    e, pa_ok, pa_n = [], 0, 0
    for ci, a, b in pcs:
        w, tw = ch[ci]["wind"], tws[ci]
        for k in range(a, b):
            if tw[k] is not None and tw[k + 1] is not None:
                n += 1
                ok += int(w[k + 1] - w[k] == tw[k + 1] - tw[k])
        m = [k for k in range(a, b + 1) if tw[k] is not None]
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                t = tw[m[j]] - tw[m[i]]
                if 10 <= t <= 20:
                    e.append(w[m[j]] - w[m[i]] - t)
    tot_steps = sum(len(c["r"]) - 1 for c in ch)
    tot_pts = sum(len(c["r"]) for c in ch)
    lens = [b - a + 1 for _, a, b in pcs]
    spans = [ch[ci]["wind"][b] - ch[ci]["wind"][a] for ci, a, b in pcs]
    e = np.array(e)
    return dict(kept_steps=round(100 * len(kept_step) / tot_steps, 1), kept_points=round(100 * sum(lens) / tot_pts, 1),
                chains=len(pcs), len_med=float(np.median(lens)) if lens else 0, len_p90=float(np.percentile(lens, 90)) if lens else 0,
                span_med=float(np.median(spans)) if spans else 0, matched=n, precision=round(100 * ok / max(n, 1), 2),
                drift=round(float(e.mean()), 3) if len(e) else None, within1=round(100 * float((np.abs(e) <= 1).mean()), 1) if len(e) else None,
                pairs10_20=int(len(e)), short_lt5=round(100 * float(np.mean(np.array(lens) < 5)), 1) if lens else 0)


def policy_keep(rows, s, t1, t2=None):
    t2 = t1 if t2 is None else t2
    return np.array([(sc >= (t1 if r["d"] == 1 else t2)) for r, sc in zip(rows, s)])


def write_set(ch, rows, keep, min_pts, out: Path, meta: dict):
    pcs = pieces_of(ch, rows, keep, min_pts)
    sub = []
    for ci, a, b in pcs:
        c = ch[ci]
        w0 = c["wind"][a]
        sub.append(dict(z=c["z"], theta=c["theta"], pts=c["pts"][a:b + 1], wind=[w - w0 for w in c["wind"][a:b + 1]],
                        r=c["r"][a:b + 1], perr=c["perr"][a:b], gm_p=c["gm_p"][a:b]))
    out.mkdir(parents=True, exist_ok=True)
    aw.write_json(sub, out / "relative_windings.json", prefix="autogmc")
    (out / "chains.json").write_text(json.dumps(sub))
    s = dict(meta, chains=len(sub), points=sum(len(c["pts"]) for c in sub),
             median_chain_points=float(np.median([len(c["pts"]) for c in sub])),
             median_winding_span=float(np.median([c["wind"][-1] for c in sub])))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    return s


# ------------------------------------------------------------------------------------------------- other masks
ALT_MASKS = {"m7x": CACHE / "band_m7x" / "crops", "m7sep03x": CACHE / "band_m7sep03x" / "crops"}  # z 3000-4500 only
DELTA = 3.0  # vox: an alternate-mask crossing counts as "between" two chain points if inside (r_a + 3, r_b - 3)


def ray_crossings_all(crops: Path):
    """Per ray: radii of all mask runs (solved or not), deduplicated across crops as in aw.assemble."""
    rays = {}
    for c, res in load_results(crops):
        if not res:
            continue
        for z, seg in res.items():
            rays.setdefault((int(z), float(c["theta"])), []).append((seg["tc"], seg["cr"]))
    out = {}
    for key, lst in rays.items():
        allx = sorted([(x["t"], abs(x["t"] - tc)) for tc, cr in lst for x in cr])
        own, cl = [], []
        for v in allx + [(1e18, 0)]:
            if cl and v[0] - cl[-1][0] > 2:
                own.append(min(cl, key=lambda q: q[1])[0])
                cl = []
            cl.append(v)
        out[key] = np.array(own)
    return out


def mask_consensus(rows):
    """For every step: crossings of each alternate mask strictly between the two chain points (None: no ray data).
    A step of d windings should have d - 1 crossings between its ends."""
    for name, crops in ALT_MASKS.items():
        rx = ray_crossings_all(crops)
        for r in rows:
            t = rx.get((r["z"], r["theta"]))
            if t is None or len(t) == 0 or t[0] > r["ra"] or t[-1] < r["rb"]:
                r[f"n_{name}"] = None
            else:
                r[f"n_{name}"] = int(((t > r["ra"] + DELTA) & (t < r["rb"] - DELTA)).sum())
    return rows


def mask_keep(rows, rule):
    keep = []
    for r in rows:
        dis = [r[f"n_{m}"] != r["d"] - 1 for m in ALT_MASKS if r.get(f"n_{m}") is not None]
        if rule == "any":
            keep.append(not any(dis))
        else:  # "both": drop only if every alternate mask with data disagrees (and at least two have data)
            keep.append(not (len(dis) == len(ALT_MASKS) and all(dis)))
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feats")
    p.add_argument("--band", choices=list(BANDS), required=True)
    a = ap.parse_args()
    {"feats": feats}[a.cmd](a)


if __name__ == "__main__":
    main()
