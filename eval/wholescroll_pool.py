"""PHerc1667 whole-scroll read-out (notes/2026-10-26-p1667-wholescroll-prereg.md): per band and pooled over bands.

Per band (each band is its own fit, so offsets are refitted per band, exactly as eval/subset_wrong_area.py
--split 34 --own-offsets and eval/method_table.py do):
- primary: inner (k < 34) own-offset seam wrong area, M2 - base (subset_wrong_area.sub_scores on the inner vertices;
  spiral_eval.BootV paired bootstrap, 30 deg x 250 z blocks, 1000 resamples, seed 1). Must reproduce
  subset_wrong_area's "own-offsets inner" row for that band (checked by --validate on test 8).
- switches: seam switches per 1000 adjacent pairs, all vertices, M2 - base (method_table.switch).
- secondary: outer (k >= 34) own-offset wrong area; unmatched share; labelled human area (cm^2, merged_v4 mesh quads
  with z in the band); correctly numbered labelled area = labelled area x (matched and right winding) / all, inner
  and all-k (all-k uses the inner/outer own-offset residuals); fit surface area (sum of meta.json area_cm2 over the
  fit's w*_spliced surfaces; every fitted surface carries a winding number).
Pooled (scroll-wide): vertex-pooled ratio sum(wrong) / sum(matched) over the bands; bootstrap = each band's blocks
resampled independently (rng seed 1000 + band index, 1000 resamples), the same resample for both arms (paired), and
the pooled ratio taken per resample.

  uv run python eval/wholescroll_pool.py --pair s1 --bands fresh --out data/scratch/wholescroll/eval/pool_s1_fresh.json
  uv run python eval/wholescroll_pool.py --validate      # test 8 band: must equal subset_s1.json / spiral_s1.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402
import subset_wrong_area as swa  # noqa: E402

SPLIT, N_BOOT = 34, 1000
CM2_PER_VOX2 = (9.596e-4) ** 2
BANDS_TSV = REPO / "data/scratch/wholescroll/bands.tsv"
SC = REPO / "data/scratch/p0139_spiral"


def arm_files(z0: int, pair: str) -> dict:
    """(base, method) dump and mesh paths for a band and seed pair (s1 = base/m2s1, s2 = bases2/m2s2)."""
    b, m = ("base", "m2s1") if pair == "s1" else ("bases2", "m2s2")
    if z0 == 4500:
        names, vd = (f"ink2_{b}", f"ink2_{m}"), REPO / "data/scratch/ink_lowband/vertex"
    elif z0 == 6500:
        names, vd = (f"c1667_{b}", f"c1667_{m}"), REPO / "data/scratch/confirm_region/p1667/vertex"
    else:
        names, vd = (f"wsf{z0}_{b}", f"wsf{z0}_{m}"), REPO / "data/scratch/wholescroll/vertex"
    return {"names": names, "dumps": tuple(vd / f"{n}.npz" for n in names),
            "meshes": tuple(SC / n / "meshes_step4" for n in names)}


def read_bands() -> list[dict]:
    out = []
    for ln in BANDS_TSV.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split("\t")
        out.append({"z0": int(f[0]), "z1": int(f[1]), "fresh": f[2] == "1"})
    return out


def labelled_area(z0: float, z1: float) -> dict:
    """cm^2 of merged_v4 labelled mesh quads (all 4 corners labelled with the same k) whose mean z is in [z0, z1)."""
    g, k = se._p1667_merged()          # zyx L2 vox, nan where unlabelled
    rows = np.nonzero((np.isfinite(g[..., 0]) & (g[..., 0] >= z0 - 20) & (g[..., 0] < z1 + 20)).any(1))[0]
    g, k = g[rows[0]:rows[-1] + 2], k[rows[0]:rows[-1] + 2]
    p00, p01, p10, p11 = g[:-1, :-1], g[:-1, 1:], g[1:, :-1], g[1:, 1:]
    kk = k[:-1, :-1]
    ok = (np.isfinite(p00).all(-1) & np.isfinite(p01).all(-1) & np.isfinite(p10).all(-1) & np.isfinite(p11).all(-1)
          & (kk == k[:-1, 1:]) & (kk == k[1:, :-1]) & (kk == k[1:, 1:]) & (kk >= 0))
    zc = (p00[..., 0] + p01[..., 0] + p10[..., 0] + p11[..., 0]) / 4
    ok &= (zc >= z0) & (zc < z1)
    area = 0.5 * np.linalg.norm(np.cross(p11 - p00, p10 - p01), axis=-1)
    a_all = float(area[ok].sum() * CM2_PER_VOX2)
    a_in = float(area[ok & (kk < SPLIT)].sum() * CM2_PER_VOX2)
    return {"all": a_all, "inner": a_in, "outer": a_all - a_in}


def fit_area(d: Path) -> float | None:
    if not d.is_dir():
        return None
    tot = 0.0
    for m in sorted(d.glob("w*_spliced/meta.json")):
        tot += float(json.loads(m.read_text())["area_cm2"])
    return tot if tot > 0 else None


def band_parts(z0: int, z1: int, pair: str, with_area: bool) -> dict:
    """Per-band per-block numerators/denominators for both arms, plus the per-band summary (BootV, seed 1)."""
    V = swa.load_ref(z0, z1)
    f = arm_files(z0, pair)
    A, B = np.load(f["dumps"][0]), np.load(f["dumps"][1])
    assert len(A["j"]) == len(V["k"]) == len(B["j"]), f"z{z0}: dumps made on a different reference sample"
    boot = se.BootV(V)
    nb = len(boot.names)
    k = V["k"]
    inner, outer = k < SPLIT, k >= SPLIT
    a, b = V["adj"][:, 0], V["adj"][:, 1]
    ma = np.ones(len(a), bool)
    parts, summ = {}, {"z": [z0, z1], "arms": list(f["names"]), "n_vertices": int(len(k)),
                       "n_inner": int(inner.sum()), "n_outer": int(outer.sum())}
    for tag, D in (("a", A), ("b", B)):
        si = swa.sub_scores(V, D["j"], {}, inner)
        so = swa.sub_scores(V, D["j"], {}, outer)
        for sub, s, m in (("inner", si, inner), ("outer", so, outer)):
            mt = s["matched"] & m
            wr = mt & (s["res_s"] != 0)
            parts[(tag, sub)] = (np.bincount(boot.u, wr.astype(float), nb), np.bincount(boot.u, mt.astype(float), nb))
            parts[(tag, sub, "n")] = np.bincount(boot.u, m.astype(float), nb)
            parts[(tag, sub, "right")] = np.bincount(boot.u, (mt & (s["res_s"] == 0)).astype(float), nb)
        pm = D["matched"][a] & D["matched"][b]
        sw = pm & (D["res_s"][a] != D["res_s"][b])
        parts[(tag, "sw")] = (np.bincount(boot.ua, sw * 1000.0, nb), np.bincount(boot.ua, pm.astype(float), nb))
        parts[(tag, "unm")] = (np.bincount(boot.u, (~D["matched"]).astype(float), nb), np.bincount(boot.u, np.ones(len(k)), nb))
    # per-band summary with the band's own BootV (identical to subset_wrong_area / method_table)
    for key, scale in (("inner", 100.0), ("outer", 100.0), ("sw", 1.0), ("unm", 100.0)):
        ra = [parts[("a", key)][0], parts[("a", key)][1]]
        rb = [parts[("b", key)][0], parts[("b", key)][1]]
        ea, eb = ra[0].sum() / ra[1].sum(), rb[0].sum() / rb[1].sum()
        xa = ra[0][boot.idx].sum(1) / ra[1][boot.idx].sum(1)
        xb = rb[0][boot.idx].sum(1) / rb[1][boot.idx].sum(1)
        summ[key] = {"a": se.summ(ea, xa, scale), "b": se.summ(eb, xb, scale), "diff": se.summ(eb - ea, xb - xa, scale),
                     "p_two_sided": float(min(1.0, 2 * min(((xb - xa) >= 0).mean(), ((xb - xa) <= 0).mean())))}
    summ["pair_pass"] = bool(summ["inner"]["diff"]["ci"][1] is not None and summ["inner"]["diff"]["ci"][1] < 0
                             and summ["sw"]["diff"]["est"] <= 5.0)
    if with_area:
        la = labelled_area(z0, z1)
        summ["labelled_cm2"] = la
        for tag in ("a", "b"):
            fr_in = parts[(tag, "inner", "right")].sum() / parts[(tag, "inner", "n")].sum()
            fr_out = parts[(tag, "outer", "right")].sum() / max(parts[(tag, "outer", "n")].sum(), 1)
            summ[f"correct_cm2_{tag}"] = {"inner": la["inner"] * fr_in, "all": la["inner"] * fr_in + la["outer"] * fr_out}
        summ["fit_cm2"] = {"a": fit_area(f["meshes"][0]), "b": fit_area(f["meshes"][1])}
    return {"parts": parts, "nb": nb, "summary": summ}


def pooled(P: list[dict]) -> dict:
    out = {}
    idx = [np.random.default_rng(1000 + i).integers(0, p["nb"], (N_BOOT, p["nb"])) for i, p in enumerate(P)]
    for key, scale in (("inner", 100.0), ("outer", 100.0), ("sw", 1.0), ("unm", 100.0)):
        res = {}
        for tag in ("a", "b"):
            nu = sum(p["parts"][(tag, key)][0].sum() for p in P)
            de = sum(p["parts"][(tag, key)][1].sum() for p in P)
            rn = sum(p["parts"][(tag, key)][0][ix].sum(1) for p, ix in zip(P, idx))
            rd = sum(p["parts"][(tag, key)][1][ix].sum(1) for p, ix in zip(P, idx))
            res[tag] = (nu / de, rn / rd)
        (ea, xa), (eb, xb) = res["a"], res["b"]
        out[key] = {"a": se.summ(ea, xa, scale), "b": se.summ(eb, xb, scale), "diff": se.summ(eb - ea, xb - xa, scale),
                    "p_two_sided": float(min(1.0, 2 * min(((xb - xa) >= 0).mean(), ((xb - xa) <= 0).mean())))}
    S = [p["summary"] for p in P]
    if all("labelled_cm2" in s for s in S):
        out["labelled_cm2"] = {k: sum(s["labelled_cm2"][k] for s in S) for k in ("all", "inner", "outer")}
        for tag in ("a", "b"):
            out[f"correct_cm2_{tag}"] = {k: sum(s[f"correct_cm2_{tag}"][k] for s in S) for k in ("inner", "all")}
        fa = [s["fit_cm2"] for s in S]
        out["fit_cm2"] = {t: (sum(x[t] for x in fa) if all(x[t] is not None for x in fa) else None) for t in ("a", "b")}
        out["fit_cm2_known_bands"] = {t: sum(x[t] for x in fa if x[t] is not None) for t in ("a", "b")}
    out["n_vertices"] = int(sum(s["n_vertices"] for s in S))
    out["n_inner"] = int(sum(s["n_inner"] for s in S))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=["s1", "s2"], default="s1")
    ap.add_argument("--bands", default="fresh", help="fresh | all | comma list of z0")
    ap.add_argument("--no-area", action="store_true")
    ap.add_argument("--validate", action="store_true", help="test 8 band only; compare with subset_s1 / spiral_s1")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    se.set_scroll("p1667")
    if a.validate:
        p = band_parts(4500, 6000, "s1", with_area=True)
        s = p["summary"]
        ref = json.loads((REPO / "data/scratch/ink_lowband/eval/subset_s1.json").read_text())["metrics_own_offsets"]
        sp = json.loads((REPO / "data/scratch/ink_lowband/eval/spiral_s1.json").read_text())["vertices"]["metrics"]["all"]["all"]
        print("inner own  :", s["inner"]["diff"], "ref", ref["inner"]["wrong_area_seam_diff"])
        print("inner a/b  :", s["inner"]["a"], s["inner"]["b"], "ref", ref["inner"]["wrong_area_seam_a"], ref["inner"]["wrong_area_seam_b"])
        print("outer own  :", s["outer"]["diff"], "ref", ref["outer"]["wrong_area_seam_diff"])
        print("switch diff:", s["sw"]["diff"], "ref", sp["switch_per1000_seam_diff"])
        print("areas      :", s["labelled_cm2"], s["correct_cm2_a"], s["correct_cm2_b"], s["fit_cm2"])
        pp = pooled([p])
        print("pooled(1 band) inner diff:", pp["inner"]["diff"], "(point must equal; CI may differ: other rng)")
        return
    B = read_bands()
    if a.bands == "fresh":
        B = [b for b in B if b["fresh"]]
    elif a.bands != "all":
        want = {int(x) for x in a.bands.split(",")}
        B = [b for b in B if b["z0"] in want]
    P = []
    for b in B:
        p = band_parts(b["z0"], b["z1"], a.pair, not a.no_area)
        p["summary"]["fresh"] = b["fresh"]
        P.append(p)
        s = p["summary"]
        print(f"z{b['z0']}-{b['z1']} {'fresh' if b['fresh'] else 'USED '} inner {s['inner']['a']['est']} -> "
              f"{s['inner']['b']['est']} diff {s['inner']['diff']} sw {s['sw']['diff']['est']} pass {s['pair_pass']}", flush=True)
    res = {"pair": a.pair, "bands": [b["z0"] for b in B], "per_band": [p["summary"] for p in P], "pooled": pooled(P)}
    q = res["pooled"]
    print("POOLED inner", q["inner"], "\n       switches", q["sw"]["diff"], "\n       outer", q["outer"]["diff"])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
