"""Label-free descriptive stats for the KP / KPM sets (notes/2026-10-29-kp-chains-prereg.md section 5).
Reads the built sets, the M2 chain sets and the umbilicus only (no label, mesh or label centre).

  uv run python -I data/scratch/kp_chains/stats.py REPO_ROOT [--sets data/scratch/kp_chains/sets]

Per band:
- coverage: share of rays (and of z levels) with a KP piece; pieces, points, median piece length, wind span per ray;
- step mix: share of linked KP steps with step count 1 / 2 / >= 3;
- gap plausibility: for each KP step, dr / (dw x s_ray), s_ray = the ray's median dr over its single steps (rays with
  >= 3 single steps); share outside [0.5, 2];
- KP vs M2 (inter-method, no labels): KP points with a one-to-one M2 partner on the same ray (|dr| <= TAU / 2, greedy
  by |dr|); among consecutive KP points whose partners sit in one M2 piece, the share where the M2 wind difference
  equals the KP winding difference.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
SETS = Path(sys.argv[sys.argv.index("--sets") + 1]) if "--sets" in sys.argv else REPO / "data/scratch/kp_chains/sets"
sys.argv = [sys.argv[0], str(REPO)]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sets as B  # noqa: E402

R_MATCH = B.TAU / 2


def pieces(doc, prefix):
    out = []
    for c in doc["collections"].values():
        if not c["name"].startswith(prefix):
            continue
        _, zs, ts, _ = c["name"].split("_")
        z, th = int(zs[1:]), float(ts[1:])
        P = [(c["points"][k]["p"], c["points"][k]["wind_a"]) for k in sorted(c["points"], key=int)]
        o = np.array([np.interp(z, B._U[0], B._U[1]), np.interp(z, B._U[0], B._U[2])])
        out.append({"z": z, "th": th, "r": [float(np.hypot(p[0] - o[0], p[1] - o[1])) for p, _ in P],
                    "w": [int(w) for _, w in P]})
    return out


def main():
    man = json.loads((SETS / "manifest.json").read_text())
    res = {}
    for z0, z1 in B.BANDS:
        mb = man["bands"][str(z0)]
        kp = pieces(json.loads((SETS / f"kp_z{z0}/relative_windings.json").read_text()), "kp_")
        m2doc = json.loads((REPO / B.M2[z0][0] / "relative_windings.json").read_text())
        m2 = pieces(m2doc, "auto_")
        n_rays = mb["rays"]
        zl = {p["z"] for p in kp}
        st = collections.Counter()
        ratios = []
        for p in kp:
            dr, dw = np.diff(p["r"]), np.diff(p["w"])
            for s in dw:
                st["steps"] += 1
                st["dw1" if s == 1 else "dw2" if s == 2 else "dw3+"] += 1
            single = dr[dw == 1]
            if len(single) >= 3:
                s_ray = float(np.median(single))
                ratios += list(dr / (dw * s_ray))
        ratios = np.array(ratios)
        # KP vs M2
        m2_by_ray = collections.defaultdict(list)
        for j, q in enumerate(m2):
            m2_by_ray[(q["z"], q["th"])].append(j)
        agree = collections.Counter()
        for p in kp:
            cand = [(abs(rk - rm), i, j, t) for j in m2_by_ray.get((p["z"], p["th"]), [])
                    for i, rk in enumerate(p["r"]) for t, rm in enumerate(m2[j]["r"]) if abs(rk - rm) <= R_MATCH]
            cand.sort()
            used_k, used_m, part = set(), set(), {}
            for _, i, j, t in cand:
                if i in used_k or (j, t) in used_m:
                    continue
                used_k.add(i); used_m.add((j, t)); part[i] = (j, t)
            agree["kp_points"] += len(p["r"])
            agree["kp_points_with_m2_partner"] += len(part)
            for i in range(len(p["r"]) - 1):
                if i in part and i + 1 in part and part[i][0] == part[i + 1][0]:
                    j, ta = part[i]; tb = part[i + 1][1]
                    agree["pairs"] += 1
                    agree["pairs_agree"] += int(m2[j]["w"][tb] - m2[j]["w"][ta] == p["w"][i + 1] - p["w"][i])
        spans = [p["w"][-1] - p["w"][0] for p in kp]
        res[str(z0)] = {
            "z": [z0, z1], "rays": n_rays, "levels": len(mb["levels"]),
            "rays_any_crossing": mb["rays_with_any_crossing_share"],
            "ray_coverage_kp": round(len(kp) / n_rays, 4), "z_level_coverage_kp": round(len(zl) / len(mb["levels"]), 4),
            "kp_pieces": len(kp), "kp_points": int(sum(len(p["r"]) for p in kp)),
            "kp_median_len": float(np.median([len(p["r"]) for p in kp])),
            "kp_median_wind_span": float(np.median(spans)),
            **{f"rays_span_ge{n}": round(sum(s >= n for s in spans) / n_rays, 4) for n in (10, 20, 30)},
            "step_mix": {k: round(st[k] / st["steps"], 4) for k in ("dw1", "dw2", "dw3+")},
            "gap_ratio_outside_0.5_2": round(float(((ratios < 0.5) | (ratios > 2)).mean()), 4) if len(ratios) else None,
            "m2_pieces": len(m2), "m2_points": int(sum(len(q["r"]) for q in m2)),
            "m2_median_len": float(np.median([len(q["r"]) for q in m2])),
            "kp_points_with_m2_partner": round(agree["kp_points_with_m2_partner"] / agree["kp_points"], 4),
            "kp_m2_step_pairs": agree["pairs"],
            "kp_m2_step_agreement": round(agree["pairs_agree"] / max(agree["pairs"], 1), 4),
            "conflicts": {k: v for k, v in mb["counts"].items()},
            "kpm": mb["kpm"]}
        r = res[str(z0)]
        print(f"z{z0}: cov {r['ray_coverage_kp']} pieces {r['kp_pieces']} pts {r['kp_points']} medlen {r['kp_median_len']}"
              f" span {r['kp_median_wind_span']} mix {r['step_mix']} gap_out {r['gap_ratio_outside_0.5_2']}"
              f" m2partner {r['kp_points_with_m2_partner']} agree {r['kp_m2_step_agreement']} (n {r['kp_m2_step_pairs']})",
              flush=True)
    (SETS.parent / "stats.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
