"""KP / KPM chain sets from Kelly-numbered Stevens patch crossings (notes/2026-10-29-kp-chains-prereg.md section 2).
Label-free: reads crossings_z<z0>.npz (crossings.py), Kelly's axis, the patch affine, the umbilicus and the M2 chain
sets. No human label, mesh or label centre is read.

  uv run python -I data/scratch/kp_chains/build_sets.py REPO_ROOT [--out data/scratch/kp_chains/sets]

Per ray (z, theta), crossings (patch p, radius r) of Kelly component-0 patches (good + bad zips):
  R1 patch self-consistency  a patch whose hits on the ray spread > TAU is dropped on that ray.
  R2a transport              Kelly angle at the crossing theta_K(c) (Kelly's own axis, crossing mapped back to the
                             7.91 um frame); d = wrap_pi(theta_K(c) - thN_p); |d| > 90 deg -> crossing dropped.
  R2b parallax               phi = Kelly angle of the ray direction (far field, affine linear part);
                             |wrap_pi(theta_K(c) - phi)| > 45 deg -> crossing dropped (Kelly's axis is a patch-centroid
                             line 50-500 vox from the umbilicus; near it Kelly's seam rule is not tied to our ray).
                             Continuous Kelly turn t_c = k_p + (thN_p + d) / 2pi; ray winding w_c = round(t_c - phi/2pi).
  R3 same winding            crossings with equal w: radial spread <= TAU -> one point at their median radius;
                             spread > TAU -> every crossing of that w on the ray dropped.
  R4 collision               two remaining points with different w closer than TAU in radius -> both dropped.
  R5 monotone                radius-sorted points must have strictly increasing w; every adjacent pair that is not
                             is dropped (both points), repeated until none is left.
  R6 piece                   >= 2 points -> one collection kp_z<z>_t<theta>_1, point ids ascending outward,
                             wind_a = w - min w (gaps carry Kelly's step count).
TAU = 0.59 x 134 um / 9.596 um (the Stevens / Kelly same-wrap separation, in L2 voxels).
KPM = KP + the band's M2 chain set, where on a ray with a KP piece spanning [r_lo, r_hi] every M2 point with
r in [r_lo - TAU, r_hi + TAU] is removed and its piece split there; M2 sub-pieces with >= 2 points are kept as
separate collections m2_z<z>_t<theta>_<i> (wind_a re-based to 0). Rays without a KP piece keep their M2 pieces whole.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else REPO / "data/scratch/kp_chains/sets"
KP = REPO / "data/scratch/kp_chains"
TAU = 0.59 * 134.0 / 9.596
TRANSPORT_MAX = np.pi / 2
PARALLAX_MAX = np.pi / 4
BANDS = [(400, 1500), (1500, 3000), (3000, 4500), (4500, 6000), (6000, 6500), (6500, 8000), (8000, 9000)]
M2 = {400: ("data/scratch/wholescroll/windings/p1667z400_m7_ray_c", "867e7e0da6cbb1e012d3f73b636ffd2f45594a50597c51971989f0aa4659efd1"),
      1500: ("data/scratch/wholescroll/windings/p1667z1500_m7_ray_c", "de6b44e1a92ec75df17303545f060e3fb17dd7b04dc7233c9888f84ecb2ab9a0"),
      3000: ("data/scratch/wholescroll/windings/p1667z3000_m7_ray_c", "0bd2cc58407be28adcef67572532f012265dfda6304f0caff06db680c1f9d909"),
      4500: ("data/scratch/ink_lowband/windings/p1667z45_m7_ray_c", "0fc2b0cbf757c848ae1a04a64577b68951a2628087a70b80a8faf0bdb58a4736"),
      6000: ("data/scratch/wholescroll/windings/p1667z6000_m7_ray_c", "fca951d472c5b64706563276e40a4e456ef6bf627aaa962f51b55c3174baf27d"),
      6500: ("data/scratch/confirm_region/p1667/windings/p1667_m7_ray_c", "85dcbfb46a66b192927cfef1adac91a080b9b1171a7690f34c6493129e0804b5"),
      8000: ("data/scratch/wholescroll/windings/p1667z8000_m7_ray_c", "82dbf46f5ec4f1037bbbd28de05a7846d599788ee61eeb5e9d0964669149d435")}
AFF = REPO / "data/scratch/patch_scout/affine_s4_7p91_to_2p399.json"
AXIS = REPO / "data/untrusted/kelly_scrolls/repo/phase/tools/vc_sheet_check/validation/results/v1-12/run/axis_contract_allpatches_zyx.txt"
SHA_IN = {AFF: "0599873e3bf602b0bed06bd625b0dc6b57630d26e6b37cb8fec59bc26afb24af",
          AXIS: "8260079a910c2421760906c5628c535499ccba0032c1971c6330e17065cbfb06"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kelly_angle(xyz_l2):
    """theta_K in [0, 2pi): Kelly's angle about Kelly's axis, crossing mapped back to 20231117161658 L0."""
    Mf = np.vstack([np.array(json.loads(AFF.read_text())["transformation_matrix"]), [0, 0, 0, 1]])
    Mi = np.linalg.inv(Mf)
    q = (4.0 * xyz_l2) @ Mi[:3, :3].T + Mi[:3, 3]
    A = np.loadtxt(AXIS, delimiter=",")                                   # z, y, x (7.91 um)
    ay, ax = np.interp(q[:, 2], A[:, 0], A[:, 1]), np.interp(q[:, 2], A[:, 0], A[:, 2])
    return np.arctan2(q[:, 1] - ay, q[:, 0] - ax) % (2 * np.pi)


def ray_direction_kelly(th):
    """Kelly angle of the ray direction (cos th, sin th, 0) in L2, through the affine's linear part (far-field angle)."""
    Mf = np.vstack([np.array(json.loads(AFF.read_text())["transformation_matrix"]), [0, 0, 0, 1]])
    Mi = np.linalg.inv(Mf)
    v = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1) @ Mi[:3, :3].T
    return np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ray_pieces(C):
    """C: dict of arrays for one band. -> {(z, ti): [(r, w, n_patches, n_bad), ...] radius-sorted}, counters."""
    cnt = collections.Counter()
    th = np.radians(C["th"])
    xyz = np.stack([C["ox"] + C["r"] * np.cos(th), C["oy"] + C["r"] * np.sin(th), C["z"].astype(float)], 1)
    thK = kelly_angle(xyz)
    thInf = ray_direction_kelly(th)
    d = wrap_pi(thK - C["thN"])                       # transport along the patch, centroid -> crossing (Kelly frame)
    par = wrap_pi(thK - thInf)                        # parallax of Kelly's axis vs the ray direction
    cnt["crossings"] = len(d)
    cnt["crossings_bad_zip"] = int(C["bad"].sum())
    keep1 = C["spread"] <= TAU
    cnt["R1_self_inconsistent"] = int((~keep1).sum())
    keep2 = np.abs(d) <= TRANSPORT_MAX
    cnt["R2a_transport"] = int((keep1 & ~keep2).sum())
    keep3 = np.abs(par) <= PARALLAX_MAX
    cnt["R2b_parallax"] = int((keep1 & keep2 & ~keep3).sum())
    keep = keep1 & keep2 & keep3
    # continuous Kelly turn at the crossing, integer winding relative to the ray's Kelly direction
    t = C["k"] + (C["thN"] + d) / (2 * np.pi)
    W = np.rint(t - thInf / (2 * np.pi)).astype(np.int64)
    rays = collections.defaultdict(list)
    for i in np.nonzero(keep)[0]:
        rays[(int(C["z"][i]), int(C["ti"][i]))].append(i)
    out, drop_by = {}, collections.Counter()
    for key, ii in rays.items():
        ii = np.array(ii)
        w, r, bad = W[ii], C["r"][ii], C["bad"][ii]
        # R3 same-winding groups
        pts = []
        for wv in np.unique(w):
            m = w == wv
            if r[m].max() - r[m].min() <= TAU:
                pts.append((float(np.median(r[m])), int(wv), int(m.sum()), int(bad[m].sum())))
            else:
                drop_by["R3_groups"] += 1
                drop_by["R3_crossings"] += int(m.sum())
                drop_by["R3_crossings_bad_zip"] += int(bad[m].sum())
        pts.sort()
        drop_by["R3_points_kept"] += len(pts)
        # R4 collision
        rr = np.array([p[0] for p in pts])
        col = np.zeros(len(pts), bool)
        if len(pts) > 1:
            close = np.diff(rr) < TAU
            col[:-1] |= close
            col[1:] |= close
        drop_by["R4_points"] += int(col.sum())
        pts = [p for p, c in zip(pts, col) if not c]
        # orientation check (descriptive): adjacent pairs whose winding rises outward
        drop_by["adj_pairs"] += max(len(pts) - 1, 0)
        drop_by["adj_pairs_rising"] += sum(b[1] > a[1] for a, b in zip(pts, pts[1:]))
        # R5 monotone
        while True:
            bad_pair = [i for i in range(len(pts) - 1) if pts[i + 1][1] <= pts[i][1]]
            if not bad_pair:
                break
            rm = set(bad_pair) | {i + 1 for i in bad_pair}
            drop_by["R5_points"] += len(rm)
            pts = [p for i, p in enumerate(pts) if i not in rm]
        if len(pts) >= 2:
            out[key] = pts
            drop_by["points_kept"] += len(pts)
            drop_by["points_single_patch"] += sum(1 for p in pts if p[2] == 1)
            drop_by["points_bad_zip_only"] += sum(1 for p in pts if p[3] == p[2])
        else:
            drop_by["R6_rays_lt2"] += 1
            drop_by["R6_points"] += len(pts)
    cnt.update(drop_by)
    cnt["transport_abs_p50_deg"] = round(float(np.degrees(np.median(np.abs(d)))), 2)
    cnt["parallax_abs_p50_deg"] = round(float(np.degrees(np.median(np.abs(par)))), 2)
    return out, cnt


def collection(cid0, pid0, name, pts_xyzw):
    pts = {}
    for j, (x, y, z, w) in enumerate(pts_xyzw):
        pts[str(pid0 + j)] = {"p": [round(float(x), 2), round(float(y), 2), round(float(z), 2)], "wind_a": int(w),
                              "creation_time": 0}
    return str(cid0), {"name": name, "points": pts, "metadata": {}, "color": [0.0, 0.0, 0.0]}


def m2_pieces(z0, origin_of):
    d, want = M2[z0]
    f = REPO / d / "relative_windings.json"
    h = sha(f)
    if want:
        assert h == want, (f, h)
    cols = json.loads(f.read_text())["collections"]
    out = []
    for cid in sorted(cols, key=int):
        c = cols[cid]
        _, zs, ts, _ = c["name"].split("_")
        z, th = int(zs[1:]), float(ts[1:])
        o = origin_of(z)
        P = [(c["points"][k]["p"], c["points"][k]["wind_a"]) for k in sorted(c["points"], key=int)]
        r = [float(np.hypot(p[0] - o[0], p[1] - o[1])) for p, _ in P]
        out.append({"z": z, "th": th, "P": P, "r": r, "name": c["name"]})
    return out, h


def stats(pieces_pts, n_rays, wind_spans):
    L = np.array([len(p) for p in pieces_pts]) if pieces_pts else np.zeros(0, int)
    steps = [b - a for p in pieces_pts for a, b in zip([q[1] for q in p], [q[1] for q in p][1:])]
    return {"pieces": int(len(L)), "points": int(L.sum()), "median_len": float(np.median(L)) if len(L) else 0,
            "mean_len": round(float(L.mean()), 2) if len(L) else 0,
            "q10_q90_len": [float(np.percentile(L, 10)), float(np.percentile(L, 90))] if len(L) else [0, 0],
            "rays_with_piece": int(len(L)), "ray_share": round(len(L) / n_rays, 4), "pts_per_ray": round(float(L.sum()) / n_rays, 2),
            "linked_steps": int(len(steps)), "steps_gt1": int(sum(s > 1 for s in steps)),
            "median_wind_span": float(np.median(wind_spans)) if wind_spans else 0,
            **{f"rays_span_ge{n}": round(sum(s >= n for s in wind_spans) / n_rays, 4) for n in (10, 20, 30)}}


def main():
    for p, h in SHA_IN.items():
        assert sha(p) == h, p
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"tau_vox": TAU, "transport_max_deg": 90.0, "parallax_max_deg": 45.0, "inputs": {str(p.relative_to(REPO)): h for p, h in SHA_IN.items()},
           "bands": {}}
    for z0, z1 in BANDS:
        cf = KP / f"crossings_z{z0}.npz"
        D = dict(np.load(cf))
        lv = list(D["levels"])
        org = {int(z): D["origin"][i] for i, z in enumerate(lv)}
        li = np.array([lv.index(z) for z in D["z"]])
        C = {"z": D["z"], "ti": D["ti"], "th": D["thetas"][D["ti"]], "r": D["r"], "spread": D["spread"],
             "k": D["k"], "thN": D["thN"], "bad": D["bad"], "ox": D["origin"][li, 0], "oy": D["origin"][li, 1]}
        n_rays = len(lv) * len(D["thetas"])
        rays_with_crossing = len({(int(a), int(b)) for a, b in zip(D["z"], D["ti"])})
        kp, cnt = ray_pieces(C)
        # --- KP document
        cols, cid, pid = {}, 1, 1
        kp_pts, spans = [], []
        kp_span = {}
        for (z, ti) in sorted(kp):
            pts = kp[(z, ti)]
            th = float(D["thetas"][ti])
            o = org[z]
            wmin = pts[0][1]
            xyzw = [(o[0] + r * np.cos(np.radians(th)), o[1] + r * np.sin(np.radians(th)), z, w - wmin) for r, w, *_ in pts]
            k, v = collection(cid, pid, f"kp_z{z}_t{th:g}_1", xyzw)
            cols[k] = v
            cid += 1
            pid += len(xyzw)
            kp_pts.append([(r, w) for r, w, *_ in pts])
            spans.append(pts[-1][1] - pts[0][1])
            kp_span[(z, th)] = (pts[0][0], pts[-1][0])
        doc_kp = {"vc_pointcollections_json_version": "1", "collections": cols}
        # --- KPM document: KP + M2 outside the KP radial span on the same ray
        m2, m2_sha = m2_pieces(z0, lambda z: np.array([np.interp(z, *_U[:2]), np.interp(z, _U[0], _U[2])]))
        cols2 = dict(cols)
        cid2, pid2 = cid, pid
        m2_kept, m2_removed, m2_all_pts, m2_sub = [], 0, [], 0
        for c in m2:
            span = kp_span.get((c["z"], c["th"]))
            keep = [True] * len(c["P"]) if span is None else [not (span[0] - TAU <= r <= span[1] + TAU) for r in c["r"]]
            m2_removed += sum(not k for k in keep)
            m2_all_pts.append(len(c["P"]))
            runs, cur = [], []
            for k, q in zip(keep, c["P"]):
                if k:
                    cur.append(q)
                elif cur:
                    runs.append(cur); cur = []
            if cur:
                runs.append(cur)
            for run in runs:
                if len(run) < 2:
                    continue
                m2_sub += 1
                w0 = run[0][1]
                xyzw = [(p[0], p[1], p[2], w - w0) for p, w in run]
                k, v = collection(cid2, pid2, f"m2_z{c['z']}_t{c['th']:g}_{m2_sub}", xyzw)
                cols2[k] = v
                cid2 += 1
                pid2 += len(xyzw)
                m2_kept.append(len(run))
        doc_kpm = {"vc_pointcollections_json_version": "1", "collections": cols2}
        res = {}
        for name, doc in ((f"kp_z{z0}", doc_kp), (f"kpm_z{z0}", doc_kpm)):
            d = OUT / name
            d.mkdir(exist_ok=True)
            f = d / "relative_windings.json"
            f.write_text(json.dumps(doc, separators=(",", ":")))
            res[name] = sha(f)
        man["bands"][str(z0)] = {
            "z": [z0, z1], "crossings_file_sha": sha(cf), "rays": n_rays, "levels": [int(x) for x in lv],
            "patches_with_crossing": int(len(np.unique(D["pid"]))),
            "rays_with_any_crossing_share": round(rays_with_crossing / n_rays, 4),
            "counts": dict(cnt), "kp": stats(kp_pts, n_rays, spans),
            "m2_set_sha": m2_sha, "m2_pieces": len(m2), "m2_points": int(sum(m2_all_pts)),
            "kpm": {"kp_pieces": len(kp_pts), "m2_points_removed": int(m2_removed), "m2_subpieces": len(m2_kept),
                    "m2_points_kept": int(sum(m2_kept)), "collections": len(cols2),
                    "points": int(sum(len(p) for p in kp_pts) + sum(m2_kept))},
            "set_sha": res}
        b = man["bands"][str(z0)]
        print(f"z{z0}: rays {n_rays}, KP pieces {b['kp']['pieces']} pts {b['kp']['points']} median len {b['kp']['median_len']}"
              f" | KPM coll {b['kpm']['collections']} pts {b['kpm']['points']} | {json.dumps(dict(cnt))}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))


def _umb():
    cp = json.loads((REPO / "data/eval/p1667/umbilicus_p1667_L2.json").read_text())
    cp = cp["control_points"] if isinstance(cp, dict) else cp
    Z = np.array([c["z"] for c in cp], float); X = np.array([c["x"] for c in cp], float); Y = np.array([c["y"] for c in cp], float)
    o = np.argsort(Z)
    return Z[o], X[o], Y[o]


_U = _umb()

if __name__ == "__main__":
    main()
