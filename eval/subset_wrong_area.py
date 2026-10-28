"""Seam-aware wrong-winding area per label-consistency subset, B - A, with spiral_eval's vertex bootstrap.

Inputs are per-vertex dumps of two fits on the same reference sample (release/fit_vertex_dump.py: j = nearest fit
winding within tol, or NOLAB) and a subsets file (eval/label_subsets.py). The offsets are recomputed here from j with
spiral_eval.vertex_scores, so a human-label correction (--relabel) can be applied before scoring: the per-bin seam
offsets and the orientation are refitted under the corrected labels, exactly as spiral_eval would if the meshes carried
those labels. Subset membership always uses the original label.

Bootstrap: spiral_eval.BootV (30 deg x 250 z blocks, 1000 resamples, seed 1), paired (same resamples for A and B).

  uv run python eval/subset_wrong_area.py --band 1 --dump-a data/scratch/fit_figures/vertex/arm0.npz \
      --dump-b data/scratch/fit_figures/vertex/arm19.npz --subsets data/scratch/label_audit/subsets_band1.json \
      [--split 41] [--relabel data/scratch/label_audit/relabel_band1.json --relabel-key supported] [--out x.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402

BANDS = {1: (3000, 4500), 2: (1750, 3000)}


def load_ref(z0: float, z1: float) -> dict:
    return dict(np.load(se.ref_path(z0, z1), allow_pickle=False))


def relabel_map(path: Path | None, key: str | None) -> dict[int, int]:
    if path is None:
        return {}
    d = json.loads(path.read_text())
    m = d[key]["map"] if key else d["map"]
    return {int(a): int(b) for a, b in m.items()}


def scores(V: dict, j: np.ndarray, kmap: dict[int, int]) -> dict:
    Vk = dict(V)
    if kmap:
        Vk["k"] = np.array([kmap.get(int(x), int(x)) for x in V["k"]])
    return se.vertex_scores(Vk, j.astype(np.int64))


def sub_scores(V: dict, j: np.ndarray, kmap: dict[int, int], m: np.ndarray) -> dict:
    """Offsets (orientation, global, per-bin) refitted on the vertices of one subset only; residuals scattered back."""
    Vs = {"k": V["k"][m], "pts": V["pts"][m], "ang": V["ang"][m]}
    vs = scores(Vs, j[m], kmap)
    full = {"matched": np.zeros(len(V["k"]), bool), "res_s": np.zeros(len(V["k"]), np.int64)}
    full["matched"][m], full["res_s"][m] = vs["matched"], vs["res_s"]
    return full


def wrong_area(V: dict, vsA: dict, vsB: dict, masks: dict, boot: se.BootV, own=None) -> dict:
    out = {}
    for name, m in masks.items():
        row = {"n_vertices": int(m.sum())}
        res = []
        pair = (("a", vsA), ("b", vsB)) if own is None else (("a", own[0](m)), ("b", own[1](m)))
        for tag, vs in pair:
            mt = vs["matched"]
            est, r = boot.ratio(mt & (vs["res_s"] != 0), mt, m)
            row[f"wrong_area_seam_{tag}"] = se.summ(est, r)
            row[f"matched_{tag}"] = int((mt & m).sum())
            res.append((est, r))
        row["wrong_area_seam_diff"] = se.summ(res[1][0] - res[0][0], res[1][1] - res[0][1])
        out[name] = row
    return out


def parse_kset(spec: str) -> set[int]:
    """'10-44,63-116,120' -> {10..44, 63..116, 120} (inclusive ranges)."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def run(a) -> dict:
    se.set_scroll(a.scroll)
    z0, z1 = BANDS[a.band] if a.band else (a.z0, a.z1)
    V = load_ref(z0, z1)
    A, B = np.load(a.dump_a), np.load(a.dump_b)
    assert len(A["j"]) == len(V["k"]) == len(B["j"]), "dumps were made on a different reference sample"
    sub = json.loads(a.subsets.read_text())
    cons = {int(k) for k, v in sub["windings"].items() if v["consistent"]}
    kmap = relabel_map(a.relabel, a.relabel_key)
    vsA, vsB = scores(V, A["j"], kmap), scores(V, B["j"], kmap)
    if not kmap:  # identity: must reproduce the dump (and therefore spiral_eval)
        for nm, vs, D in (("a", vsA, A), ("b", vsB, B)):
            m = vs["matched"]
            assert np.array_equal(m, D["matched"]) and np.array_equal(np.where(m, vs["res_s"], 0), D["res_s"]), nm
    k = V["k"]
    c = np.isin(k, list(cons))
    masks = {"all": np.ones(len(k), bool), "consistent": c, "inconsistent": ~c}
    if a.split is not None:
        inner, outer = k < a.split, k >= a.split
        masks |= {"inner": inner, "outer": outer, "inner_consistent": inner & c, "inner_inconsistent": inner & ~c,
                  "outer_consistent": outer & c, "outer_inconsistent": outer & ~c}
    if getattr(a, "primary_k", None):
        pk = parse_kset(a.primary_k)
        pm = np.isin(k, sorted(pk))
        masks |= {"primary": pm, "nonprimary": ~pm, "primary_consistent": pm & c, "primary_inconsistent": pm & ~c}
    if a.excl_flag:
        masks = {n: m & ~V["flag"].astype(bool) for n, m in masks.items()}
    boot = se.BootV(V)
    res = {"band": [z0, z1], "dump_a": str(a.dump_a), "dump_b": str(a.dump_b), "subsets": str(a.subsets),
           "consistent_windings": sorted(cons), "split_inner_lt": a.split, "excl_flag": a.excl_flag,
           **({"primary_k": a.primary_k} if getattr(a, "primary_k", None) else {}),
           "relabel": {"file": str(a.relabel) if a.relabel else None, "key": a.relabel_key,
                       "map": {str(x): y for x, y in sorted(kmap.items())}},
           "offsets": {"a": {"sign": vsA["sign"], "global": vsA["global_offset"]},
                       "b": {"sign": vsB["sign"], "global": vsB["global_offset"]}},
           "metrics": wrong_area(V, vsA, vsB, masks, boot)}
    if a.own_offsets:
        own = (lambda m: sub_scores(V, A["j"], kmap, m), lambda m: sub_scores(V, B["j"], kmap, m))
        res["metrics_own_offsets"] = wrong_area(V, vsA, vsB, {n: m for n, m in masks.items() if n != "all"}, boot, own)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump-a", type=Path, required=True)
    ap.add_argument("--dump-b", type=Path, required=True)
    ap.add_argument("--subsets", type=Path, required=True)
    ap.add_argument("--band", type=int, choices=[1, 2])
    ap.add_argument("--scroll", choices=list(se.SCROLLS), default="p0139", help="p1667 / paris4: give --z0/--z1")
    ap.add_argument("--z0", type=float)
    ap.add_argument("--z1", type=float)
    ap.add_argument("--split", type=int, help="also report inner (k < split) / outer (k >= split)")
    ap.add_argument("--primary-k", help="also report 'primary' (k in the set) and 'nonprimary', e.g. 10-44,63-116 "
                    "(inclusive ranges; paris4 pre-registration)")
    ap.add_argument("--relabel", type=Path, help="json with a label map (old k -> new k)")
    ap.add_argument("--relabel-key", help="key of the map inside the relabel json (default: top-level 'map')")
    ap.add_argument("--excl-flag", action="store_true", help="drop vertices in the #2167 gap sector")
    ap.add_argument("--own-offsets", action="store_true",
                    help="also score each subset with offsets refitted on that subset alone (independent of other labels)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = run(a)
    f = lambda r: "n/a" if r["est"] is None else f"{r['est']:6.2f} [{r['ci'][0]}, {r['ci'][1]}]"
    print(f"relabel: {res['relabel']['key'] or res['relabel']['file']}  offsets {res['offsets']}")
    for n, r in res["metrics"].items():
        print(f"  {n:20s} n={r['n_vertices']:8d}  A {f(r['wrong_area_seam_a'])}  B {f(r['wrong_area_seam_b'])}  "
              f"B-A {f(r['wrong_area_seam_diff'])}")
    for n, r in res.get("metrics_own_offsets", {}).items():
        print(f"  own-offsets {n:16s} A {f(r['wrong_area_seam_a'])}  B {f(r['wrong_area_seam_b'])}  B-A {f(r['wrong_area_seam_diff'])}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
