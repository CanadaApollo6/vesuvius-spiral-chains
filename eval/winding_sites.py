"""Real-data test of a winding-field model: does s_hat step by one wrap between adjacent GT sheets?

  uv run --with nnunetv2 python eval/winding_sites.py --ckpt data/winding/w1/checkpoint.pth --out data/preds/wind_w1 \
      [--limit N]

Sites are the jeong200 GT band pairs of eval/separation.py (same find_sites, seed 0, same caps): band centre a, the
next band centre b = a + u d. A correct winding field has |s_hat(b) - s_hat(a)| ~ 1 (adjacent wraps); a merge gives
~0, a spurious extra sheet ~2. Sign does not matter. s_hat comes from 128^3 tiles at stride 64 (the field is defined
per tile up to a constant, so each site is read from the single tile in which its midpoint is most central).
  correct: 0.5 <= delta < 1.5;  merge: delta < 0.5;  split: delta >= 1.5
Summary per class (compressed d < 10.5 vox, wide 16 <= d < 24) with a patch-bootstrap 95% CI on the correct rate.

--dataset p0139: the PHerc0139 9.362 um instance set (data/eval/p0139_9um, notes/2026-10-18-p0139-testset.md).
The ratio rule above is gameable when the expected step is always 1 and the site class is defined by the distance d:
a constant field of pitch 16 vox and no image input scores 99.9% / 100% on jeong200. So p0139 is scored differently
(main_p0139): distance-balanced pairs with expected steps 0, 1, 2-4 (eval/p0139_sites.scoring_pairs), primary metric
|delta - expected| < 0.25 wrap, broken down by expected step and by local compression (contact / open), next to three
baselines scored by the same code (GT plumbing check = 100%, best constant pitch, per-crop oracle pitch). A model only
counts as resolving sheets if it beats the per-crop oracle pitch. The 0.5-1.5 ratio numbers are kept as a labelled
secondary column, and the old adjacent normal-ray sites as a legacy block.
  uv run --with nnunetv2 python eval/winding_sites.py --dataset p0139 --ckpt <ckpt> --out data/preds/wind_w1_p0139
  uv run python eval/winding_sites.py --dataset p0139 --oracle --out data/preds/p0139_baselines   # no network
The jeong200 path (site sampler, row order aligned with data/preds/sep_*.csv) is unchanged.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "train")]  # winding (train/) is imported lazily: needs torch
from contact_check import COMPRESSED_MAX, WIDE_MAX, WIDE_MIN, find_sites  # noqa: E402
from separation import MAX_SITES  # noqa: E402

TILE, STRIDE = 128, 64


def starts(n: int) -> list[int]:
    s = list(range(0, max(n - TILE, 0) + 1, STRIDE))
    if s[-1] != max(n - TILE, 0):
        s.append(max(n - TILE, 0))
    return s


def tile_preds(net, vol: np.ndarray, dev) -> dict:
    """{(z0, y0, x0): s_hat tile} over the volume (padded by edge replication if smaller than a tile)."""
    import torch
    from winding import decode_phase, norm_ct

    pad = [(0, max(0, TILE - n)) for n in vol.shape]
    v = np.pad(norm_ct(vol), pad, mode="edge")
    out = {}
    for z in starts(v.shape[0]):
        for y in starts(v.shape[1]):
            for x in starts(v.shape[2]):
                t = torch.from_numpy(v[z:z + TILE, y:y + TILE, x:x + TILE].copy())[None, None].to(dev)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    o = net(t)[0].float().cpu().numpy()
                out[(z, y, x)] = decode_phase(o) if o.shape[0] == 3 else o[0]  # phase head: unwrap per tile
    return out


def oracle_tiles(surf: np.ndarray, wind: dict) -> dict:
    """GT stand-in for s_hat: winding number of the nearest surface voxel, cut into the same tiles."""
    from scipy import ndimage as ndi

    ind = ndi.distance_transform_edt(surf == 0, return_indices=True)[1]
    lut = np.zeros(int(surf.max()) + 1, np.float32)
    for k, w in wind.items():
        lut[int(k)] = w
    field = lut[surf[tuple(ind)]]
    pad = [(0, max(0, TILE - n)) for n in field.shape]
    v = np.pad(field, pad, mode="edge")
    return {(z, y, x): v[z:z + TILE, y:y + TILE, x:x + TILE]
            for z in starts(v.shape[0]) for y in starts(v.shape[1]) for x in starts(v.shape[2])}


def site_delta(tiles: dict, a: np.ndarray, b: np.ndarray) -> float:
    m = (a + b) / 2
    best = min(tiles, key=lambda o: np.abs(m - (np.array(o) + (TILE - 1) / 2)).max())
    o = np.array(best)
    pts = np.stack([a - o, b - o], 1)
    if (pts < 0).any() or (pts > TILE - 1).any():
        return float("nan")
    va, vb = map_coordinates(tiles[best], pts, order=1, mode="nearest")
    return float(abs(vb - va))


def summarize(rows: list[dict], classes=("compressed", "wide")) -> dict:
    stems = sorted({r["stem"] for r in rows})
    out = {"patches": len(stems)}
    boot = np.random.default_rng(1)
    for cls in classes:
        sel = [r for r in rows if r["cls"] == cls and np.isfinite(r["delta"])]
        if not sel:
            continue
        by = {s: [r for r in sel if r["stem"] == s] for s in stems}

        def rate(sample, key):
            v = [r[key] for s in sample for r in by.get(s, [])]
            return 100.0 * np.mean(v) if v else float("nan")

        bs = [rate(list(boot.choice(stems, len(stems))), "correct") for _ in range(1000)]
        d = np.array([r["delta"] for r in sel])
        out[cls] = {"n": len(sel), "correct": rate(stems, "correct"),
                    "correct_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                    "merge": rate(stems, "merge"), "split": rate(stems, "split"),
                    "delta_median": float(np.median(d)), "delta_iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))]}
    return out


def load_net(ckpt: Path, dev):
    import torch
    from winding import build_net

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    init = ck["args"].get("init")
    init = Path(init) if init and init != "None" else None
    if init is not None and "data/nnunet" in str(init) and not str(init).startswith(str(REPO)):  # trained on a pod
        init = REPO / ("data/nnunet" + str(init).split("data/nnunet", 1)[1])
    net = build_net(init, out=3 if ck["args"].get("phase_w") else 1)
    net.load_state_dict(ck["net"])
    return net.to(dev).eval()


def score_row(stem, cls, d, dl, expected=1, **extra) -> dict:
    """Same definition for both datasets: correct 0.5 <= delta/expected < 1.5, merge < 0.5, split >= 1.5."""
    r = dl / expected
    return {"stem": stem, "cls": cls, "d": round(float(d), 2), **extra, "delta": dl,
            "correct": 0.5 <= r < 1.5, "merge": r < 0.5, "split": r >= 1.5}


TOL = 0.25  # p0139 primary metric: |delta - expected| < TOL wrap (absolute)


def p0139_correct(delta: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.abs(delta - expected) < TOL


def ratio_correct(delta: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Legacy rule (gameable): 0.5 <= delta/expected < 1.5; for expected 0, delta < 0.5."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = delta / expected
    return np.where(expected == 0, delta < 0.5, (r >= 0.5) & (r < 1.5))


def fit_pitch(x: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Single pitch P maximising the pooled primary metric of delta = x / P (fitted on the test set itself)."""
    grid = np.arange(2.0, 40.0, 0.05)
    acc = [p0139_correct(x / P, expected).mean() for P in grid]
    k = int(np.argmax(acc))
    return float(grid[k]), float(acc[k])


def p0139_summary(rows: list[dict], methods: list[str], by_site_boot: int = 1000) -> dict:
    """Primary metric per method (pooled, by expected step, by contact/open, by site class) with a site bootstrap,
    the legacy ratio rule as a secondary column, and paired site-bootstrap differences against the oracle pitch."""
    stems = sorted({r["stem"] for r in rows})
    sidx = {s: k for k, s in enumerate(stems)}
    site = np.array([sidx[r["stem"]] for r in rows])
    exp = np.array([r["expected"] for r in rows])
    contact = np.array([r["contact"] for r in rows])
    scls = np.array([r["site_cls"] for r in rows])
    boot = np.random.default_rng(1).integers(0, len(stems), (by_site_boot, len(stems)))

    def rate(c, m):
        return round(100.0 * float(c[m].mean()), 2) if m.any() else None

    def site_boot(c):
        num = np.bincount(site, c.astype(float), len(stems))
        den = np.bincount(site, None, len(stems))
        return 100.0 * num[boot].sum(1) / np.maximum(den[boot].sum(1), 1)

    out, corr = {}, {}
    for meth in methods:
        d = np.array([r[f"delta_{meth}"] for r in rows], float)
        fin = np.isfinite(d)
        c = p0139_correct(np.where(fin, d, np.inf), exp)  # unreadable = wrong
        corr[meth] = c
        bs = site_boot(c)
        rc = ratio_correct(np.where(fin, d, np.inf), exp)
        out[meth] = {
            "primary_abs_tol_0.25": {
                "pooled": rate(c, np.ones_like(c)), "pooled_ci95": [round(float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)],
                "by_expected_step": {str(e): rate(c, exp == e) for e in sorted(set(exp.tolist()))},
                "by_step_group": {"0": rate(c, exp == 0), "1": rate(c, exp == 1), ">=2": rate(c, exp >= 2)},
                "by_compression": {"contact": rate(c, contact), "open": rate(c, ~contact)},
                "by_site_cls": {k: rate(c, scls == k) for k in sorted(set(scls.tolist()))},
                "unreadable": int((~fin).sum())},
            "secondary_ratio_rule_0.5_1.5_GAMEABLE": {
                "pooled": rate(rc, np.ones_like(rc)),
                "by_step_group": {"0 (delta<0.5)": rate(rc, exp == 0), "1": rate(rc, exp == 1), ">=2": rate(rc, exp >= 2)}},
            "delta_median_by_step": {str(e): round(float(np.nanmedian(d[exp == e])), 3) for e in sorted(set(exp.tolist()))},
        }
    if "oracle_pitch" in corr:
        for meth in methods:
            if meth == "oracle_pitch":
                continue
            diff = site_boot(corr[meth]) - site_boot(corr["oracle_pitch"])
            out[meth]["vs_oracle_pitch"] = {
                "diff_pp": round(float(100.0 * (corr[meth].mean() - corr["oracle_pitch"].mean())), 2),
                "diff_ci95": [round(float(np.percentile(diff, q)), 2) for q in (2.5, 97.5)],
                "beats_oracle_pitch": bool(np.percentile(diff, 2.5) > 0) if len(stems) >= 5 else None}  # CI needs sites
    return out


def legacy_summary(rows: list[dict], methods: list[str]) -> dict:
    """Old adjacent normal-ray sites, ratio rule, by spacing class of d (continuity with jeong200; gameable)."""
    from p0139_sites import spacing_class

    out = {}
    for meth in methods:
        rr = [{"stem": r["stem"], "cls": spacing_class(r["dist"]), "delta": r[f"delta_{meth}"],
               **{k: v for k, v in zip(("correct", "merge", "split"),
                                       (0.5 <= r[f"delta_{meth}"] < 1.5, r[f"delta_{meth}"] < 0.5, r[f"delta_{meth}"] >= 1.5))}}
              for r in rows if np.isfinite(r[f"delta_{meth}"])]
        out[meth] = summarize(rr, ("compressed", "intermediate", "wide"))
    return out


def main_p0139(args) -> None:
    """PHerc0139 scoring: distance-balanced pairs with expected steps 0..4 (eval/p0139_sites.scoring_pairs, frozen in
    <root>/scoring_pairs.csv), primary metric |delta - expected| < 0.25 wrap, scored for the model and three baselines
    through the same code:
      gt            GT winding of the nearest surface voxel, read through the same 128^3 tiles (plumbing; must be 100%)
      const_pitch   delta = projected distance along the local sheet normal / P, one P fitted on the set
      oracle_pitch  delta = projected normal distance / (median GT adjacent spacing of that crop)
      const_euclid  delta = Euclidean distance / P (diagnostic: should be near chance, the pairs are distance-balanced)
    A model resolves sheets only if it beats oracle_pitch (paired site-bootstrap 95% CI of the difference above 0).
    The legacy adjacent-pair normal-ray block with the 0.5-1.5 ratio rule is reported separately (gameable)."""
    from p0139_sites import PAIRS_CSV, ROOT, load_sites, read_pairs, write_pairs

    root = args.p0139_root or ROOT
    if args.remake_pairs or not (root / PAIRS_CSV).exists():
        write_pairs(root)
    allrows = read_pairs(root)
    sites = load_sites(root)[: args.limit or None]
    keep = {s["stem"] for s in sites}
    allrows = [r for r in allrows if r["stem"] in keep]
    net = dev = None
    if args.ckpt is not None and not args.oracle:
        import torch

        dev = torch.device(args.device)
        net = load_net(args.ckpt, dev)
    by = {}
    for r in allrows:
        by.setdefault(r["stem"], []).append(r)
    for i, s in enumerate(sites):
        rows = by.get(s["stem"], [])
        surf = tifffile.imread(s["surfaces"])
        wind = {int(k): v["winding"] for k, v in s["instances"].items()}
        tgt = oracle_tiles(surf, wind)
        tm = tile_preds(net, tifffile.imread(s["image"]), dev) if net is not None else None
        for r in rows:
            A, B = np.array([r["za"], r["ya"], r["xa"]]), np.array([r["zb"], r["yb"], r["xb"]])
            r["delta_gt"] = site_delta(tgt, A, B)
            if tm is not None:
                r["delta_model"] = site_delta(tm, A, B)
            r["delta_oracle_pitch"] = r["proj"] / r["pitch_site"]
        print(f"[{i + 1}/{len(sites)}] {s['stem']} ({s['cls']}): {sum(r['kind'] == 'mix' for r in rows)} mix, "
              f"{sum(r['kind'] == 'legacy' for r in rows)} legacy pairs", flush=True)
    mix = [r for r in allrows if r["kind"] == "mix"]
    leg = [r for r in allrows if r["kind"] == "legacy"]
    e = np.array([r["expected"] for r in mix])
    P, _ = fit_pitch(np.array([r["proj"] for r in mix]), e)
    Pe, _ = fit_pitch(np.array([r["dist"] for r in mix]), e)
    Pl, _ = fit_pitch(np.array([r["proj"] for r in leg]), np.array([r["expected"] for r in leg]))
    for r in mix:
        r["delta_const_pitch"], r["delta_const_euclid"] = r["proj"] / P, r["dist"] / Pe
    for r in leg:
        r["delta_const_pitch"], r["delta_const_euclid"] = r["proj"] / Pl, r["dist"] / Pl
    methods = (["model"] if net is not None else []) + ["gt", "const_pitch", "oracle_pitch", "const_euclid"]
    summ = {
        "dataset": "p0139", "ckpt": str(args.ckpt) if net is not None else None, "sites": len(sites),
        "primary_metric": f"share of pairs with |delta - expected| < {TOL} wrap, pooled over all pairs (expected 0..4)",
        "rule": "a model resolves sheets only if it beats oracle_pitch (paired site-bootstrap 95% CI of the difference > 0)",
        "pairs": {"n": len(mix), "by_expected_step": {str(k): int((e == k).sum()) for k in sorted(set(e.tolist()))},
                  "contact": int(sum(r["contact"] for r in mix)), "open": int(sum(not r["contact"] for r in mix)),
                  "dist_median_by_step_group": {g: round(float(np.median([r["dist"] for r in mix if f(r["expected"])])), 2)
                                                for g, f in (("0", lambda x: x == 0), ("1", lambda x: x == 1),
                                                             (">=2", lambda x: x >= 2))}},
        "fitted": {"const_pitch_P_vox": P, "const_euclid_P_vox": Pe, "legacy_const_pitch_P_vox": Pl},
        "methods": p0139_summary(mix, methods),
        "legacy_adjacent_normal_rays_ratio_rule_GAMEABLE": legacy_summary(leg, methods),
    }
    if net is not None:
        summ["model_resolves_sheets"] = summ["methods"]["model"]["vs_oracle_pitch"]["beats_oracle_pitch"]
    cols = ["stem", "site_cls", "kind", "a", "b", "wa", "wb", "expected", "dist", "proj", "dnn_min", "contact", "pitch_site"]
    out_rows = [{k: r[k] for k in cols} | {f"delta_{m}": r[f"delta_{m}"] for m in methods}
                | {f"correct_{m}": bool(abs(r[f"delta_{m}"] - r["expected"]) < TOL) for m in methods} for r in allrows]
    write(args.out, out_rows, summ)


def write(out: Path, rows: list[dict], summ: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["jeong200", "p0139"], default="jeong200")
    ap.add_argument("--p0139-root", type=Path, default=None)
    ap.add_argument("--oracle", action="store_true", help="p0139: score only GT + baselines, no network (no GPU)")
    ap.add_argument("--remake-pairs", action="store_true", help="p0139: regenerate <root>/scoring_pairs.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="writes <out>.csv and <out>.json")
    ap.add_argument("--images", type=Path, default=REPO / "data" / "eval" / "jeong200" / "imagesTr")
    ap.add_argument("--gt-dir", type=Path, default=REPO / "data" / "eval" / "jeong200" / "labelsTr")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.dataset == "p0139":
        return main_p0139(args)
    if args.ckpt is None:
        ap.error("--ckpt is required")
    import torch

    dev = torch.device(args.device)
    net = load_net(args.ckpt, dev)
    stems = sorted(p.stem for p in args.gt_dir.glob("*.tif"))[: args.limit or None]
    rows = []
    for i, stem in enumerate(stems):
        gt = tifffile.imread(args.gt_dir / f"{stem}.tif") == 1
        vol = tifffile.imread(args.images / f"{stem}_0000.tif")
        rng = np.random.default_rng(0)
        sites = find_sites(gt, rng)
        comp = [x for x in sites if x[2] < COMPRESSED_MAX]
        wide = [x for x in sites if WIDE_MIN <= x[2] < WIDE_MAX]
        comp = [comp[j] for j in rng.permutation(len(comp))[:MAX_SITES]]
        wide = [wide[j] for j in rng.permutation(len(wide))[:MAX_SITES]]
        tiles = tile_preds(net, vol, dev)
        for cls, items in (("compressed", comp), ("wide", wide)):
            for centre, u, d in items:
                dl = site_delta(tiles, centre, centre + u * d)
                rows.append(score_row(stem, cls, d, dl))
        print(f"[{i + 1}/{len(stems)}] {stem}: {len(comp)} compressed, {len(wide)} wide sites", flush=True)
    write(args.out, rows, summarize(rows))


if __name__ == "__main__":
    main()
