"""Structure ablation chain sets (notes/2026-10-28-structure-ablation-prereg.md), Paris4 z 7000-8500.

  uv run python data/scratch/structure_ablation/build_sets.py [--out data/scratch/structure_ablation/sets]

Inputs (read-only, sha-checked): the oracle chains band_oracle_p4z7 (relative_windings.json + rays.json for the radius
of each oracle point) and our cut chains p4z7_f6 (relative_windings.json + chains.json for the radius of each point).
Both sets sit on the same 576 rays (z, theta from the collection names). Output: three point-collection v1 documents
in the same shape as the two inputs (name / points / metadata {} / color), one collection per piece, point ids
file-global and ascending outward, wind_a ints re-based to 0 at each piece's first point (differences unchanged):

  A1 complete+fragmented  every oracle point; each oracle piece cut into consecutive sub-pieces whose lengths are drawn
                          i.i.d. from p4z7_f6's piece-length list (numpy default_rng(SEED_A1)); a draw that would leave
                          a 1-point remainder takes that point too; the last sub-piece is the remainder. No shared
                          points, so the oracle step across each cut is lost (as in the f6 cut).
  A2 thinned+linked       oracle pieces kept whole, minus every oracle point with no p4z7_f6 partner on its ray.
                          Partners: one-to-one greedy match by |dr| ascending, |dr| <= R_MATCH (radius along the ray,
                          chain-frame voxels). R_MATCH = 0.5 x median radial spacing of oracle single steps (wind_a +1)
                          over the band. Kept points keep their oracle wind_a, so a gap carries the known step count.
  A3 thinned+fragmented   A2's points, split wherever two consecutive kept points are matched to different p4z7_f6
                          pieces (= where our pieces break), so the pieces are ours and the steps the oracle's.
Pieces with < 2 points are dropped in every set (the fitter drops them too). Label-free structure stats -> manifest.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
W = REPO / "data/scratch/f6_screen/confirm/windings"
SRC = {"oracle_rw": W / "band_oracle_p4z7/relative_windings.json", "oracle_rays": W / "band_oracle_p4z7/rays.json",
       "f6_rw": W / "p4z7_f6/relative_windings.json", "f6_chains": W / "p4z7_f6/chains.json"}
SHA = {"oracle_rw": "fc8a00a5f5653e4055bd2b874420d2974690bc98395de35ac99288853e397900",
       "f6_rw": "03c014bdf5e28d0e383122f10d98e173c35d46f5795cf6beac14455618319370",
       "f6_chains": "b5d20c1da2056e76b974126fc85f0e83cb3ca9289a07feef1dc2c5f8760380ba"}
SEED_A1 = 20261028
R_FACTOR = 0.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ray_of(name):
    _, z, t, _ = name.split("_")
    return z[1:], t[1:]          # strings as written in the source names, e.g. ("7008", "15")


def rkey(z, t):
    return int(float(z)), float(t)


def load_pieces(rw_path, radius_of):
    """-> list of pieces in collection-id order: dict(ray=(zs, ts), pts=[(p, wind_a, r)])"""
    cols = json.loads(Path(rw_path).read_text())["collections"]
    out = []
    for cid in sorted(cols, key=int):
        c = cols[cid]
        ids = sorted(c["points"], key=int)
        pts = [(c["points"][i]["p"], int(c["points"][i]["wind_a"]), radius_of(cid, k, c["points"][i]["p"]))
               for k, i in enumerate(ids)]
        out.append(dict(ray=ray_of(c["name"]), pts=pts))
    return out


def stats(pieces, n_oracle=None, n_rays=576):
    L = np.array([len(p["pts"]) for p in pieces])
    per_ray = collections.Counter()
    steps = collections.Counter()
    for p in pieces:
        per_ray[rkey(*p["ray"])] += len(p["pts"])
        w = [q[1] for q in p["pts"]]
        steps.update(b - a for a, b in zip(w, w[1:]))
    ppr = np.array(list(per_ray.values()) + [0] * (n_rays - len(per_ray)))
    s = dict(pieces=int(len(L)), points=int(L.sum()), median_len=float(np.median(L)), mean_len=round(float(L.mean()), 2),
             len_q10_q25_q75_q90=[float(x) for x in np.percentile(L, [10, 25, 75, 90])], max_len=int(L.max()),
             rays_covered=len(per_ray), points_per_ray_mean=round(float(ppr.mean()), 2),
             points_per_ray_median=float(np.median(ppr)), pieces_per_ray_mean=round(len(L) / n_rays, 2),
             linked_steps=int(sum(steps.values())), steps_gt1=int(sum(v for k, v in steps.items() if k > 1)),
             steps_le0=int(sum(v for k, v in steps.items() if k <= 0)))
    if n_oracle:
        s["frac_oracle_points"] = round(int(L.sum()) / n_oracle, 4)
    return s


def to_doc(pieces, prefix):
    cols, pid = {}, 0
    for ci, p in enumerate(pieces, start=1):
        w0 = p["pts"][0][1]
        pts = {}
        for q in p["pts"]:
            pid += 1
            pts[str(pid)] = {"p": q[0], "wind_a": int(q[1] - w0), "creation_time": 0}
        z, t = p["ray"]
        cols[str(ci)] = {"name": f"{prefix}_z{z}_t{t}_{ci}", "points": pts, "metadata": {}, "color": [0.0, 0.0, 0.0]}
    return {"vc_pointcollections_json_version": "1", "collections": cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data/scratch/structure_ablation/sets")
    o = ap.parse_args()
    o.out = o.out.resolve()
    in_sha = {k: sha(v) for k, v in SRC.items()}
    for k, v in SHA.items():
        assert in_sha[k] == v, f"{k} sha {in_sha[k]} != frozen {v}"

    # radius of each oracle point: rays.json entries [r, k, [z, y, x]] (k, the label winding, is not read)
    lut = {}
    for lst in json.loads(SRC["oracle_rays"].read_text()).values():
        for e in lst:
            z, y, x = e[2]
            lut[(round(x, 2), round(y, 2), round(z, 2))] = float(e[0])
    oracle = load_pieces(SRC["oracle_rw"], lambda cid, k, p: lut[tuple(round(v, 2) for v in p)])
    chains = json.loads(SRC["f6_chains"].read_text())
    ours = load_pieces(SRC["f6_rw"], lambda cid, k, p: float(chains[int(cid) - 1]["r"][k]))
    for cid, (pc, ch) in enumerate(zip(ours, chains), start=1):   # collection i <-> chains.json entry i
        assert rkey(*pc["ray"]) == (int(ch["z"]), float(ch["theta"])) and len(pc["pts"]) == len(ch["pts"]), cid
        assert all(abs(q[0][0] - c[2]) < 0.01 and abs(q[0][1] - c[1]) < 0.01 for q, c in zip(pc["pts"], ch["pts"]))
    n_or = sum(len(p["pts"]) for p in oracle)
    assert all(len(p["pts"]) >= 2 for p in oracle)
    assert all(all(b[2] > a[2] for a, b in zip(p["pts"], p["pts"][1:])) for p in oracle + ours), "radius not increasing"

    # R_MATCH from the oracle's own single-step radial spacing
    sp = np.array([b[2] - a[2] for p in oracle for a, b in zip(p["pts"], p["pts"][1:]) if b[1] - a[1] == 1])
    r_match = R_FACTOR * float(np.median(sp))

    # one-to-one match per ray, greedy by |dr|
    O = collections.defaultdict(list)   # ray -> [(piece idx, point idx, r)]
    F = collections.defaultdict(list)   # ray -> [(piece idx, r)]
    for i, p in enumerate(oracle):
        O[rkey(*p["ray"])] += [(i, j, q[2]) for j, q in enumerate(p["pts"])]
    for i, p in enumerate(ours):
        F[rkey(*p["ray"])] += [(i, q[2]) for q in p["pts"]]
    assert set(O) == set(F), "oracle and p4z7_f6 rays differ"
    partner = {}                        # (oracle piece, point) -> p4z7_f6 piece of the matched point
    for k in O:
        ro = np.array([x[2] for x in O[k]])
        rf = np.array([x[1] for x in F[k]])
        d = np.abs(ro[:, None] - rf[None, :])
        ii, jj = np.nonzero(d <= r_match)
        used_o, used_f = set(), set()
        for q in np.argsort(d[ii, jj], kind="stable"):
            a, b = int(ii[q]), int(jj[q])
            if a in used_o or b in used_f:
                continue
            used_o.add(a)
            used_f.add(b)
            partner[O[k][a][:2]] = F[k][b][0]

    # A1
    lens_f6 = np.array([len(p["pts"]) for p in ours])
    rng = np.random.default_rng(SEED_A1)
    A1 = []
    for p in oracle:
        n, pos = len(p["pts"]), 0
        while pos < n:
            ln = int(rng.choice(lens_f6))
            if n - pos - ln == 1:
                ln += 1
            ln = min(ln, n - pos)
            A1.append(dict(ray=p["ray"], pts=p["pts"][pos:pos + ln]))
            pos += ln
    assert all(len(p["pts"]) >= 2 for p in A1) and sum(len(p["pts"]) for p in A1) == n_or

    # A2, A3
    A2, A3 = [], []
    for i, p in enumerate(oracle):
        kept = [(q, partner[(i, j)]) for j, q in enumerate(p["pts"]) if (i, j) in partner]
        if len(kept) >= 2:
            A2.append(dict(ray=p["ray"], pts=[q for q, _ in kept]))
        run = []
        for q, fp in kept:
            if run and fp != run[-1][1]:
                if len(run) >= 2:
                    A3.append(dict(ray=p["ray"], pts=[x for x, _ in run]))
                run = []
            run.append((q, fp))
        if len(run) >= 2:
            A3.append(dict(ray=p["ray"], pts=[x for x, _ in run]))

    o.out.mkdir(parents=True, exist_ok=True)
    man = {"inputs": {k: str(v.relative_to(REPO)) for k, v in SRC.items()}, "input_sha256": in_sha,
           "params": {"seed_A1": SEED_A1, "r_factor": R_FACTOR, "oracle_single_step_spacing_median": float(np.median(sp)),
                      "oracle_single_step_spacing_n": int(len(sp)), "r_match_vox": r_match,
                      "match": "one-to-one greedy by |dr| ascending (stable), |dr| <= r_match, per ray"},
           "matched_pairs": len(partner), "sets": {}, "stats": {}}
    man["stats"]["oracle (band_oracle_p4z7)"] = stats(oracle, n_or)
    man["stats"]["p4z7_f6"] = stats(ours)
    for name, pcs, pre in (("sa_A1", A1, "saA1"), ("sa_A2", A2, "saA2"), ("sa_A3", A3, "saA3")):
        dst = o.out / name / "relative_windings.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(to_doc(pcs, pre)))
        man["sets"][name] = {"path": str(dst.relative_to(REPO) if dst.is_relative_to(REPO) else dst), "sha256": sha(dst)}
        man["stats"][name] = stats(pcs, n_or)
    # A1 length distribution vs p4z7_f6 (two-sample KS distance)
    la = np.sort([len(p["pts"]) for p in A1])
    grid = np.arange(1, max(la.max(), lens_f6.max()) + 1)
    ks = float(np.max(np.abs(np.searchsorted(la, grid, "right") / len(la)
                             - np.searchsorted(np.sort(lens_f6), grid, "right") / len(lens_f6))))
    man["A1_vs_f6_length_ks"] = round(ks, 4)
    (o.out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
