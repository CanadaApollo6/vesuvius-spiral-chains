"""CT-flagged PHerc0139 scoring pairs (2026-10-21 CT label audit) and pair metrics recomputed without them or under a
relabelled truth.

  # 1. flag table (human labels, sites.json, umbilicus and the CT audit only; no fit is read)
  uv run python eval/pairs_exclude.py flags --set spiral_band1|spiral_band2|bench \
      [--out data/scratch/label_audit/flagged_pairs_<set>.csv]
  # 2. per-pair matched windings of one spiral fit (the pair half of spiral_eval, ~1-3 GB instead of 11-16 GB;
  #    per-point results are identical to spiral_eval --dump-pairs, columns row,stem,kind,wa,wb,expected,jA,jB)
  uv run python eval/pairs_exclude.py dump --fit <dir>/meshes_step4 --z0 3000 --z1 4500 --out <csv>
  # 3. spiral pair metrics B - A (spiral_eval.pair_metrics, same site bootstrap) on all / excl_ct / relabelled truth
  uv run python eval/pairs_exclude.py spiral --dump-a arm0.csv --dump-b armN.csv \
      --flags data/scratch/label_audit/flagged_pairs_spiral_band1.csv [--relabel-keys supported outer_minus2] [--out x.json]
  # 4. frozen 87-site benchmark (eval/p0139_graph.py CSVs; p0139_paired bootstrap)
  uv run --with pandas python eval/pairs_exclude.py bench --ref data/preds/graph_m7pub_s10.csv \
      --new data/preds/graph_m7pub_s11836.csv --gt data/preds/graph_gt.csv \
      --flags data/scratch/label_audit/flagged_pairs_bench.csv [--out x.json]

Dumps from `spiral_eval.py --dump-pairs` work as --dump-a/--dump-b too (extra columns are ignored); the "row" column
(index into scoring_pairs.csv) is the join key. For a band-2 run: `flags --set spiral_band2` once (labels only), dump
or --dump-pairs each arm on z 1750-3000, then `spiral` with --relabel-keys minus1 minus2.

Flags (a pair is a row of data/eval/p0139_9um/scoring_pairs.csv; lo/hi = min/max of its two winding labels):
  f_dup41_42   lo <= 41 < hi: crosses the w041/w042 duplicate (one sheet; truth is 1 lower). Includes every
               pair straddling w040|w043 (2 physical turns, labels 3), which the duplicate explains.
  f_gap40_43   lo <= 40 and hi >= 43 (subset of f_dup41_42; reported separately).
  f_dup45_46   lo <= 45 < hi: crosses the w045/w046 duplicate.
  f_gap46_47   lo <= 46 < hi: crosses w046|w047 (2 turns on ~60-90 % of rays: an untraced sheet).
  f_s2167      [lo, hi] meets [34, 36] and either end lies at 145-260 deg around the umbilicus (villa #2167 sector:
               w036 on w034, w035 missing; no global relabel fits there). All z.
  skip         the pair skips a winding absent from its site (spiral_eval.load_pairs; 172 pairs, dropped by spiral_eval).
  flag         any of the above (= the "CT-flagged" set).
Relabelled truth: exp_<key> = |map(wa) - map(wb)| for each map in relabel_band1.json (spiral_band1, bench) or
relabel_band2.json (spiral_band2). `supported` is the band-1 best CT fit; it keeps w045 and w046 apart (45->44,
46->45), so it is wrong for pairs with exactly one end on w046. exp_supported_m46 additionally maps 46 -> 44.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402

LA = REPO / "data" / "scratch" / "label_audit"
SETS = {"spiral_band1": (3000.0, 4500.0, LA / "relabel_band1.json"),
        "spiral_band2": (1750.0, 3000.0, LA / "relabel_band2.json"),
        "bench": (None, None, LA / "relabel_band1.json")}
SECTOR = (145.0, 260.0)
FLAG_COLS = ["f_dup41_42", "f_gap40_43", "f_dup45_46", "f_gap46_47", "f_s2167", "skip"]


def relabel_maps(path: Path) -> dict[str, dict[int, int]]:
    d = json.loads(path.read_text())
    out = {k: {int(a): int(b) for a, b in v["map"].items()} for k, v in d.items() if isinstance(v, dict) and "map" in v}
    if "supported" in out:
        out["supported_m46"] = dict(out["supported"]) | {46: out["supported"][45]}
    return out


def pair_flags(wa: np.ndarray, wb: np.ndarray, angA: np.ndarray, angB: np.ndarray, skip: np.ndarray) -> dict:
    lo, hi = np.minimum(wa, wb), np.maximum(wa, wb)
    sec = lambda a: (a >= SECTOR[0]) & (a <= SECTOR[1])
    f = {"f_dup41_42": (lo <= 41) & (hi > 41), "f_gap40_43": (lo <= 40) & (hi >= 43),
         "f_dup45_46": (lo <= 45) & (hi > 45), "f_gap46_47": (lo <= 46) & (hi > 46),
         "f_s2167": (lo <= 36) & (hi >= 34) & (sec(angA) | sec(angB)), "skip": skip.astype(bool)}
    f["flag"] = np.logical_or.reduce([f[c] for c in FLAG_COLS])
    return f


def cmd_flags(a) -> None:
    z0, z1, rl = SETS[a.set]
    P = se.load_pairs(z0 if z0 is not None else -1e9, z1 if z1 is not None else 1e9, "ct")
    umb = se.Umbilicus()
    angA, angB = umb.angle(P["A"]), umb.angle(P["B"])
    sel = P["inb"] if a.set != "bench" else (P["kind"] == "mix")
    if a.set != "bench" and not a.legacy:
        sel &= P["kind"] == "mix"
    F = pair_flags(P["wa"], P["wb"], angA, angB, P["skip"])
    maps = relabel_maps(rl)
    exp_r = {k: np.abs(np.array([m.get(int(x), int(x)) for x in P["wa"]]) - np.array([m.get(int(x), int(x)) for x in P["wb"]]))
             for k, m in maps.items()}
    bench_row = np.cumsum(P["kind"] == "mix") - 1
    out = Path(a.out or LA / f"flagged_pairs_{a.set}.csv")
    cols = ["row", "bench_row", "stem", "kind", "wa", "wb", "expected", "contact", "zA", "angA", "angB", *FLAG_COLS, "flag",
            *[f"exp_{k}" for k in maps]]
    idx = np.nonzero(sel)[0]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in idx:
            w.writerow([i, bench_row[i] if P["kind"][i] == "mix" else -1, P["stem"][i], P["kind"][i], P["wa"][i], P["wb"][i],
                        P["expected"][i], int(P["contact"][i]), round(float(P["A"][i, 0]), 1), round(float(angA[i]), 1),
                        round(float(angB[i]), 1), *[int(F[c][i]) for c in FLAG_COLS], int(F["flag"][i]),
                        *[int(exp_r[k][i]) for k in maps]])
    mix = sel & (P["kind"] == "mix")
    print(f"{out}: {len(idx)} pairs ({mix.sum()} mix, {len(np.unique(P['stem'][mix]))} sites); relabel maps {list(maps)}")
    for c in FLAG_COLS + ["flag"]:
        print(f"  {c:12s} mix {int((F[c] & mix).sum()):6d}  (non-skip {int((F[c] & mix & ~P['skip']).sum())})")
    for k in maps:
        print(f"  exp_{k:22s} changes {int(((exp_r[k] != P['expected']) & mix).sum())} mix pairs")


def cmd_dump(a) -> None:
    t0 = time.time()
    zlo, zhi = a.z0 - a.tol - 2, a.z1 + a.tol + 2
    P = se.load_pairs(a.z0, a.z1, "ct")
    q = P["inb"] & np.isin(P["kind"], ["mix", "legacy"] if a.legacy else ["mix"])
    qpts = np.concatenate([P["A"][q], P["B"][q]])
    S = se.Surfaces(se.discover_fit(a.fit, zlo, zhi), zlo, zhi, a.min_edge)
    lab, _ = S.query(qpts, a.tol)
    nq = int(q.sum())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "stem", "kind", "wa", "wb", "expected", "jA", "jB"])
        for n, i in enumerate(np.nonzero(q)[0]):
            w.writerow([i, P["stem"][i], P["kind"][i], P["wa"][i], P["wb"][i], P["expected"][i], lab[n], lab[nq + n]])
    print(f"{a.out}: {nq} pairs, {S.stats['surfaces']} surfaces, {S.stats['quads']} quads, "
          f"{time.time() - t0:.0f}s, peak {se.peak_gb()} GB")


def read_dump(p: Path) -> dict[int, tuple[int, int]]:
    with p.open() as f:
        return {int(r["row"]): (int(r["jA"]), int(r["jB"])) for r in csv.DictReader(f)}


def read_flags(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def spiral_metrics(flags: list[dict], dumps: list[dict], kind: str, keys: list[str], exclude: str = "flag") -> dict:
    """spiral_eval's pair block (skip dropped, site bootstrap over the same units) with extra variants/truths."""
    rows = [r for r in flags if r["kind"] == kind]
    row = np.array([int(r["row"]) for r in rows])
    keep = np.array([all(int(i) in d for d in dumps) for i in row])
    rows, row = [r for r, k in zip(rows, keep) if k], row[keep]
    stem = np.array([r["stem"] for r in rows])
    col = lambda c, t=int: np.array([t(r[c]) for r in rows])
    skip, contact = col("skip").astype(bool), col("contact").astype(bool)
    excl = np.zeros(len(rows), bool)
    for c in exclude.split(","):
        excl |= col(c).astype(bool)
    sel = ~skip
    boot = se.Boot(stem[sel])
    full_u = np.zeros(len(sel), np.int64)
    full_u[sel] = boot.u
    boot.u = full_u
    fits = []
    for d in dumps:
        jA = np.array([d[int(i)][0] for i in row])
        jB = np.array([d[int(i)][1] for i in row])
        fits.append({"jA": jA, "jB": jB, "kA_seam": jA, "kB_seam": jB})
    one = np.ones(len(rows), bool)
    strata = {"all": one, "contact": contact, "open": ~contact}
    variants = {"all": one, "excl_ct": ~excl}
    out = {"n_pairs": int(len(rows)), "skip_dropped": int(skip.sum()), "sites": int(len(boot.names)),
           "excluded_non_skip": int((excl & sel).sum()), "exclude_cols": exclude, "truth": {}}
    for key in ["original"] + keys:
        exp = col("expected") if key == "original" else col(f"exp_{key}")
        P = {"expected": exp}
        m = se.pair_metrics(P, sel, fits, strata, boot, variants)
        out["truth"][key] = {"changed_non_skip": int(((exp != col("expected")) & sel).sum()), "metrics": m}
    return out


def cmd_spiral(a) -> None:
    flags = read_flags(a.flags)
    dumps = [read_dump(a.dump_a)] + ([read_dump(a.dump_b)] if a.dump_b else [])
    res = {"flags": str(a.flags), "dump_a": str(a.dump_a), "dump_b": str(a.dump_b) if a.dump_b else None}
    res["pairs_mix"] = spiral_metrics(flags, dumps, "mix", a.relabel_keys, a.exclude)
    two = len(dumps) == 2
    f = lambda r: "n/a" if r["est"] is None else f"{r['est']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
    p = res["pairs_mix"]
    print(f"pairs_mix: {p['n_pairs']} in band, {p['skip_dropped']} skip dropped, {p['sites']} sites, "
          f"{p['excluded_non_skip']} more excluded ({p['exclude_cols']})")
    for key, t in p["truth"].items():
        for vn in ("all", "excl_ct"):
            for sn in ("all", "contact"):
                r = t["metrics"][vn][sn]
                s = f"  {key:22s} {vn:8s} {sn:8s} n={r['n']:6d} acc_matched A {r['acc_matched']['est']:.2f}"
                if two:
                    s += (f" B {r['acc_matched_b']['est']:.2f}  B-A acc_matched {f(r['acc_matched_diff'])}  "
                          f"acc_all {f(r['acc_all_diff'])}  coverage {f(r['coverage_diff'])}")
                print(s)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


def cmd_bench(a) -> None:
    import pandas as pd
    F = pd.read_csv(a.flags).sort_values("bench_row").reset_index(drop=True)
    ref, new = pd.read_csv(a.ref), pd.read_csv(a.new)
    gt = pd.read_csv(a.gt) if a.gt else None
    for d in [new] + ([gt] if gt is not None else []):
        assert len(d) == len(ref) and (d.stem.values == ref.stem.values).all()
    assert len(F) == len(ref) and (F.stem.values == ref.stem.values).all() and (F.expected.values == ref.expected.values).all()
    stems = sorted(ref.stem.unique())
    si = ref.stem.map({s: i for i, s in enumerate(stems)}).values
    bt = np.random.default_rng(1).integers(0, len(stems), (1000, len(stems)))

    def rate(c, m):
        nu = np.bincount(si[m], c[m], len(stems))[bt].sum(1)
        de = np.bincount(si[m], None, len(stems))[bt].sum(1)
        return nu / de

    lo, hi = np.minimum(F.wa, F.wb).values, np.maximum(F.wa, F.wb).values
    band = (lo <= 37) & (hi >= 33)
    flag, skip = F.flag.values.astype(bool), F.skip.values.astype(bool)
    subsets = {"all": np.ones(len(F), bool), "excl_ct": ~flag, "excl_skip": ~skip,
               "w033-37 (non-skip)": band & ~skip, "rest (non-skip)": ~band & ~skip,
               "w033-37 excl_ct": band & ~flag, "rest excl_ct": ~band & ~flag}
    truths = {"original": F.expected.values} | {k: F[f"exp_{k}"].values for k in a.relabel_keys}
    res = {"n": len(F), "flagged": int(flag.sum()), "skip": int(skip.sum()), "rows": []}
    print(f"bench: {len(F)} pairs, {len(stems)} sites; flagged {flag.sum()} (skip {skip.sum()}, s2167 "
          f"{int(F.f_s2167.sum())}, overlap {int((F.f_s2167.values.astype(bool) & skip).sum())})")
    for tk, exp in truths.items():
        changed = int((exp != F.expected.values).sum())
        for sn, m in subsets.items():
            row = {"truth": tk, "subset": sn, "n": int(m.sum()), "sites": int(len(np.unique(si[m]))), "changed": changed}
            for col, tag in ((a.col, "graph"), ("delta_direct", "direct")):
                c0 = (np.abs(ref[col].values - exp) < 0.25).astype(float)
                c1 = (np.abs(new[col].values - exp) < 0.25).astype(float)
                r0, r1 = rate(c0, m), rate(c1, m)
                row[f"{tag}_ref"] = round(100 * c0[m].mean(), 2)
                row[f"{tag}_new"] = round(100 * c1[m].mean(), 2)
                row[f"{tag}_new_ci"] = np.percentile(100 * r1, [2.5, 97.5]).round(1).tolist()
                row[f"{tag}_diff"] = round(100 * (c1[m].mean() - c0[m].mean()), 2)
                row[f"{tag}_diff_ci"] = np.percentile(100 * (r1 - r0), [2.5, 97.5]).round(2).tolist()
                if tag == "graph":
                    row["contact_new"] = round(100 * c1[m & F.contact.values.astype(bool)].mean(), 1)
            if gt is not None:
                cg = (np.abs(gt[a.col].values - exp) < 0.25).astype(float)
                row["gt"] = round(100 * cg[m].mean(), 2)
            res["rows"].append(row)
            print(f"  {tk:14s} {sn:20s} n={row['n']:6d} sites {row['sites']:2d}  graph new {row['graph_new']:.2f} "
                  f"{row['graph_new_ci']} ref {row['graph_ref']:.2f} diff {row['graph_diff']:+.2f} {row['graph_diff_ci']}  "
                  f"direct diff {row['direct_diff']:+.2f} {row['direct_diff_ci']}  contact {row['contact_new']}  "
                  f"GT {row.get('gt', 'n/a')}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flags")
    f.add_argument("--set", choices=list(SETS), required=True)
    f.add_argument("--legacy", action="store_true", help="also list legacy pairs (spiral sets)")
    f.add_argument("--out", type=Path)
    d = sub.add_parser("dump")
    d.add_argument("--fit", type=Path, required=True)
    d.add_argument("--z0", type=float, default=3000)
    d.add_argument("--z1", type=float, default=4500)
    d.add_argument("--tol", type=float, default=4.0)
    d.add_argument("--min-edge", type=float, default=2.5)
    d.add_argument("--legacy", action="store_true")
    d.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("spiral")
    s.add_argument("--dump-a", type=Path, required=True)
    s.add_argument("--dump-b", type=Path)
    s.add_argument("--flags", type=Path, required=True)
    s.add_argument("--relabel-keys", nargs="*", default=[])
    s.add_argument("--exclude", default="flag", help="comma-separated flag columns that define excl_ct (default: flag)")
    s.add_argument("--out", type=Path)
    b = sub.add_parser("bench")
    b.add_argument("--ref", type=Path, required=True)
    b.add_argument("--new", type=Path, required=True)
    b.add_argument("--gt", type=Path)
    b.add_argument("--flags", type=Path, required=True)
    b.add_argument("--col", default="delta_graph_b12_r40_k30_t0")
    b.add_argument("--relabel-keys", nargs="*", default=[])
    b.add_argument("--out", type=Path)
    a = ap.parse_args()
    {"flags": cmd_flags, "dump": cmd_dump, "spiral": cmd_spiral, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
