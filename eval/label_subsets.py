"""Label-consistency subsets of the PHerc0139 human winding meshes (notes/2026-10-21-second-band-plan.md, 7a/7b).

Uses the human meshes only (2.399 tifxyz -> published affine, as eval/spiral_eval.py loads them). No fit output.

Rays from the umbilicus = bins of (50 z slab x 2 deg). Per mesh and bin: the radii (yx distance to the umbilicus) of
its vertices; if the sorted radii have a gap > --multi-gap vox the mesh crosses that ray twice (its own >360 deg
overlap, or a fold) and the bin is not used for that mesh. Otherwise r = median radius.

Per winding k (fixed before any band-2 fit output exists):
- coincident partner: another mesh m (blacklisted included) with |r_k - r_m| < 0.25 x local median spacing on >= 20 %
  of the (slab, angle) bins where both are single-valued (>= 10 shared bins).
- per-angle-bin spacing: for each 2 deg angle bin, the median over z slabs (>= 3 shared slabs) of the per-turn radial
  spacing (r_out - r_in) / (k_out - k_in). (Per (slab, angle) values were tried first: adjacent-sheet spacing in this
  scroll varies 10-30 vox at a single ray, so every winding failed; the z-median per angle bin is the "angle bin" of 7a.)
- neighbours: per angle bin, the nearest label below / above k among the meshes the scorer uses (not blacklisted,
  not a coincident partner of k) that has a spacing in that bin. Signed, so a neighbour on the wrong side fails.
- local median spacing (per angle bin): median per-turn spacing of label-consecutive scorer meshes with |dk| <= 3 and
  both labels within k +- 6, pooled over the bin and +-2 angle bins.
- an angle bin passes if every existing neighbour's spacing is within [0.6, 1.6] x local median.
- consistent = no coincident partner, >= 20 tested angle bins, and >= 80 % of tested angle bins pass.
Also reported: missing-sector fraction (2 deg bins without any vertex of k, over the z slabs where k is present),
z coverage, self-overlap bins.

  uv run python eval/label_subsets.py --band 1   # z 3000-4500 -> data/scratch/label_audit/subsets_band1.json
  uv run python eval/label_subsets.py --band 2   # z 1750-3000
  uv run python eval/label_subsets.py --band 3   # z 6000-8250 (backup confirmation region, 2026-10-22)
  uv run python eval/label_subsets.py --scroll p1667   # PHerc1667 merged-mesh labels, L2 z 6500-8000 -> subsets_bandp1667.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402

OUT = REPO / "data" / "scratch" / "label_audit"
BANDS = {1: (3000.0, 4500.0), 2: (1750.0, 3000.0), 3: (6000.0, 8250.0)}
ZBIN, ABIN = 50.0, 2.0
NA = int(360 / ABIN)
LO, HI = 0.6, 1.6
PASS_FRAC, MIN_BINS = 0.80, 20
COINC_REL, COINC_FRAC, COINC_MIN = 0.25, 0.20, 10
MIN_SLABS, LOCAL_K = 3, 6


def load_band(band: int) -> dict:
    """All labelled human mesh vertices (blacklisted included) with z in the band +- 60 (cached)."""
    z0, z1 = BANDS[band]
    f = OUT / f"human_band{band}.npz"
    if not f.exists():
        P, L, S, B = [], [], [], []
        for s in se.human_segments(z0 - 60, z1 + 60):
            g = s["grid"]
            p = g[np.isfinite(g).all(-1)]
            P.append(p); L.append(np.full(len(p), s["label"])); S.append(np.full(len(p), s["sid"]))
            B.append(np.full(len(p), s["blacklisted"]))
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(f, pts=np.concatenate(P).astype(np.float32), k=np.concatenate(L), sid=np.concatenate(S),
                 bl=np.concatenate(B))
    return dict(np.load(f))


def ray_table(H: dict, z0: float, z1: float, umb: se.Umbilicus, multi_gap: float):
    """Per (mesh, z slab, angle bin): median radius, or nan if the mesh crosses the ray twice. Returns
    R[k_index, slab, abin], multi[...], present[...], the label list and blacklist flags."""
    m = (H["pts"][:, 0] >= z0) & (H["pts"][:, 0] < z1)
    p, k = H["pts"][m].astype(np.float64), H["k"][m]
    bl = {int(a): bool(b) for a, b in zip(H["k"], H["bl"])}
    labels = sorted(set(int(x) for x in k))
    ki = np.searchsorted(labels, k)
    z = p[:, 0]
    r = np.hypot(p[:, 1] - np.interp(z, umb.z, umb.y), p[:, 2] - np.interp(z, umb.z, umb.x))
    a = umb.angle(p)
    ns = int(np.ceil((z1 - z0) / ZBIN))
    zs = np.clip(((z - z0) // ZBIN).astype(int), 0, ns - 1)
    ab = (a // ABIN).astype(int) % NA
    key = (ki * ns + zs) * NA + ab
    o = np.lexsort((r, key))
    key, r = key[o], r[o]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    ends = np.r_[starts[1:], len(key)]
    shape = (len(labels), ns, NA)
    R = np.full(shape, np.nan)
    multi = np.zeros(shape, bool)
    present = np.zeros(shape, bool)
    gaps = np.diff(r)
    for s0, e0 in zip(starts, ends):
        idx = np.unravel_index(key[s0], shape)
        present[idx] = True
        if e0 - s0 >= 2 and gaps[s0:e0 - 1].max() > multi_gap:
            multi[idx] = True
        else:
            R[idx] = np.median(r[s0:e0])
    return R, multi, present, labels, [bl[x] for x in labels]


def zmed_spacing(R: np.ndarray, i: int, j: int, labels: list[int]) -> np.ndarray:
    """Per angle bin: median over z slabs (>= MIN_SLABS shared) of the per-turn radial spacing (r_j - r_i) / (k_j - k_i)."""
    d = (R[j] - R[i]) / (labels[j] - labels[i])
    ok = np.isfinite(d).sum(0) >= MIN_SLABS
    out = np.full(R.shape[2], np.nan)
    if ok.any():
        out[ok] = np.nanmedian(d[:, ok], axis=0)
    return out


def local_median(R: np.ndarray, labels: list[int], use: np.ndarray, kc: int) -> np.ndarray:
    """Per angle bin: median per-turn spacing (z-median) of label-consecutive scorer meshes with |dk| <= 3 and both
    labels within kc +- LOCAL_K, pooled over +-2 angle bins."""
    near = [j for j in range(len(labels)) if use[j] and abs(labels[j] - kc) <= LOCAL_K]
    pairs = [(a, b) for a, b in zip(near[:-1], near[1:]) if labels[b] - labels[a] <= 3]
    med = np.full(R.shape[2], np.nan)
    if not pairs:
        return med
    P = np.array([zmed_spacing(R, a, b, labels) for a, b in pairs])
    for b in range(R.shape[2]):
        pool = P[:, [(b + d) % NA for d in range(-2, 3)]].ravel()
        pool = pool[np.isfinite(pool) & (pool > 0)]
        if len(pool) >= 3:
            med[b] = np.median(pool)
    return med


def coincidence(R: np.ndarray, labels: list[int], spacing_ref: dict) -> dict:
    """Per (z slab, angle bin) radii: share of shared bins where two meshes are < COINC_REL x local median apart."""
    out = {}
    for i, ki in enumerate(labels):
        med = spacing_ref[ki][None, :]
        for j, kj in enumerate(labels):
            if i == j:
                continue
            both = np.isfinite(R[i]) & np.isfinite(R[j]) & np.isfinite(med)
            nb = int(both.sum())
            if nb < COINC_MIN:
                continue
            close = both & (np.abs(R[i] - R[j]) < COINC_REL * np.where(np.isfinite(med), med, 0.0))
            frac = close.sum() / nb
            if frac >= 0.05:
                out.setdefault(ki, {})[kj] = {"shared_bins": nb, "frac_close": round(float(frac), 3),
                                               "median_abs_dr": round(float(np.median(np.abs(R[i] - R[j])[both])), 2)}
    return out


def audit(band: int, multi_gap: float) -> dict:
    z0, z1 = BANDS[band]
    umb = se.Umbilicus()
    H = load_band(band)
    R, multi, present, labels, bl = ray_table(H, z0, z1, umb, multi_gap)
    use = ~np.array(bl)
    n = len(labels)
    meds = {k: local_median(R, labels, use, k) for k in labels}
    coinc = coincidence(R, labels, meds)
    out = {}
    for i, k in enumerate(labels):
        partners = sorted(m for m, v in coinc.get(k, {}).items() if v["frac_close"] >= COINC_FRAC)
        med = meds[k]
        cand = [j for j in range(n) if j != i and use[j] and labels[j] not in partners]
        inner = [j for j in cand if labels[j] < k][::-1]   # nearest label first
        outer = [j for j in cand if labels[j] > k]
        sp_in = {j: zmed_spacing(R, j, i, labels) for j in inner}
        sp_out = {j: zmed_spacing(R, i, j, labels) for j in outer}
        tested = passed = fails_in = fails_out = 0
        rat_in, rat_out, vox_in, vox_out, nbr_in, nbr_out = [], [], [], [], {}, {}
        for b in range(NA):
            if not np.isfinite(med[b]):
                continue
            ji = next((j for j in inner if np.isfinite(sp_in[j][b])), None)
            jo = next((j for j in outer if np.isfinite(sp_out[j][b])), None)
            if ji is None and jo is None:
                continue
            good = True
            for j, sp, rat, vox, nbr, side in ((ji, sp_in, rat_in, vox_in, nbr_in, "in"),
                                                (jo, sp_out, rat_out, vox_out, nbr_out, "out")):
                if j is None:
                    continue
                q = sp[j][b] / med[b]
                rat.append(q)
                vox.append(sp[j][b] * abs(labels[j] - k))
                nbr[labels[j]] = nbr.get(labels[j], 0) + 1
                if not (LO <= q <= HI):
                    good = False
                    if side == "in":
                        fails_in += 1
                    else:
                        fails_out += 1
            tested += 1
            passed += good
        slabs = present[i].any(1)
        missing = float(1 - present[i][slabs].mean()) if slabs.any() else 1.0
        frac = passed / tested if tested else 0.0
        reasons = []
        if partners:
            reasons.append(f"coincident with {partners}")
        if tested < MIN_BINS:
            reasons.append(f"only {tested} tested angle bins")
        elif frac < PASS_FRAC:
            reasons.append(f"spacing passes in {frac * 100:.0f} % of angle bins (inner fails {fails_in}, outer fails {fails_out})")
        q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])] if len(v) else None
        out[str(k)] = {
            "consistent": not reasons, "reasons": reasons, "blacklisted": bool(bl[i]),
            "coincident_partners": partners, "near_meshes": {str(a): v for a, v in coinc.get(k, {}).items()},
            "tested_angle_bins": int(tested), "pass_frac": round(float(frac), 4),
            "ratio_inner_p10_50_90": q(rat_in), "ratio_outer_p10_50_90": q(rat_out),
            "spacing_inner_vox_p10_50_90": q(vox_in), "spacing_outer_vox_p10_50_90": q(vox_out),
            "inner_neighbour_bins": {str(a): v for a, v in sorted(nbr_in.items())},
            "outer_neighbour_bins": {str(a): v for a, v in sorted(nbr_out.items())},
            "local_median_spacing_vox": round(float(np.nanmedian(med)), 2) if np.isfinite(med).any() else None,
            "missing_sector_frac": round(missing, 4),
            "z_slabs_present": int(slabs.sum()), "z_slabs_band": int(R.shape[1]),
            "single_valued_ray_bins": int(np.isfinite(R[i]).sum()), "self_overlap_ray_bins": int(multi[i].sum()),
        }
    return {"band": band, "z": [z0, z1],
            "rules": {"zbin": ZBIN, "abin_deg": ABIN, "multi_gap_vox": multi_gap, "min_slabs_per_angle_bin": MIN_SLABS,
                      "ratio_range": [LO, HI], "pass_frac": PASS_FRAC, "min_angle_bins": MIN_BINS,
                      "coincidence": {"rel": COINC_REL, "frac": COINC_FRAC, "min_shared_ray_bins": COINC_MIN},
                      "neighbours": "per angle bin, nearest label each side among non-blacklisted meshes (partners "
                                    "excluded) with a z-median spacing",
                      "local_median": f"label-consecutive non-blacklisted pairs, |dk|<=3, labels within k+-{LOCAL_K}, "
                                      "+-2 angle bins"},
            "consistent": [int(k) for k, v in out.items() if v["consistent"]],
            "inconsistent": [int(k) for k, v in out.items() if not v["consistent"]],
            "windings": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, choices=[1, 2, 3])
    ap.add_argument("--scroll", choices=["p0139", "p1667", "paris4", "paris4z7", "paris4z11"], default="p0139",
                    help="p1667: z 6500-8000 (L2 frame); paris4: z 9000-10500 (L2 frame of 20260411134726); "
                         "paris4z7: z 7000-8500, same frame (fourth test); paris4z11: z 10800-11800, same frame (f7 test)")
    ap.add_argument("--multi-gap", type=float, default=5.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.scroll == "p1667":
        se.set_scroll("p1667")
        a.band = "p1667"
        BANDS["p1667"] = (6500.0, 8000.0)
    elif a.scroll == "paris4":
        se.set_scroll("paris4")
        a.band = "paris4"
        BANDS["paris4"] = (9000.0, 10500.0)
    elif a.scroll == "paris4z7":   # fourth test (notes/2026-10-24-fourth-test-prereg.md): same labels, z 7000-8500
        se.set_scroll("paris4")
        a.band = "paris4z7"
        BANDS["paris4z7"] = (7000.0, 8500.0)
    elif a.scroll == "paris4z11":  # f7 test (notes/2026-10-24-f7-*): same labels, z 10800-11800
        se.set_scroll("paris4")
        a.band = "paris4z11"
        BANDS["paris4z11"] = (10800.0, 11800.0)
    elif a.band is None:
        ap.error("--band is required for p0139")
    res = audit(a.band, a.multi_gap)
    out = a.out or OUT / f"subsets_band{a.band}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"band {a.band}: consistent {res['consistent']}\n  inconsistent {res['inconsistent']}")
    for k, v in res["windings"].items():
        print(f"  w{int(k):03d} {'OK ' if v['consistent'] else 'BAD'} bins {v['tested_angle_bins']:4d} pass {v['pass_frac']:.2f} "
              f"in {v['ratio_inner_p10_50_90']} out {v['ratio_outer_p10_50_90']} med {v['local_median_spacing_vox']} "
              f"miss {v['missing_sector_frac']:.2f} self {v['self_overlap_ray_bins']} {'; '.join(v['reasons'])}")
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}  {out}")


if __name__ == "__main__":
    main()
