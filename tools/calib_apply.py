"""calib_apply.py - "human calibrates, machine applies": fit the gm raise threshold from a check-viewer answer log.

Rule (data/scratch/hitl/calib.py; notes/2026-10-22-hitl-budget.md section 3):
  1. answered checks = queue steps with count_before_gm == 1 whose last answer is a sheet count:
     winding_step = n_between + 1 (>= 1). Unanswered: 'u' (unsure), 's' (same sheet, winding_step 0) and 'a'
     (an endpoint in open air); blind controls (is_control) are left out;
  2. fit P(true step >= 2 | logit gm_p) on them (calib.py fit1: Newton logistic, ridge 1e-3);
  3. threshold = the gm_p at which the number of raised single steps equals the predicted number of true doubles
     among ALL single steps of the chain set (calib.py thr_balance, unchanged, incl. its k = round(target) indexing).
The CI is a nonparametric bootstrap over the answered checks (refit + rebalance per resample).

LABEL-FREE: reads only the answer log, the queue (gm_p, count_before_gm) and the auto chain sets (population gm_p).
It never opens oracle / CT / sealed files.

  uv run python tools/calib_apply.py --queue data/scratch/hitl/check_queue_p1667_calib.json \\
      --answers data/scratch/hitl/answers_p1667_calib.jsonl [--mid-only] [--boot 2000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


# --- verbatim from data/scratch/hitl/calib.py (+ hitl_lib.logit) -------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit1(x, y, iters=50):
    X = np.c_[np.ones_like(x), x]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        g = X.T @ (y - p) - 1e-3 * w
        Hm = (X * W[:, None]).T @ X + 1e-3 * np.eye(2)
        w += np.linalg.solve(Hm, g)
    return w


def thr_balance(w, gm_single):
    p = 1 / (1 + np.exp(-(w[0] + w[1] * logit(gm_single))))
    target = p.sum()
    g = np.sort(gm_single)[::-1]
    k = int(np.clip(round(target), 0, len(g) - 1))
    return float(g[k]) if k < len(g) else 1.0
# -------------------------------------------------------------------------------------------------------------------


def population_gm(chains: Path, pre: Path) -> np.ndarray:
    """gm_p of every single step (pre-gm count == 1) of the chain set."""
    ch = json.loads(chains.read_text())
    pr = json.loads(pre.read_text())
    assert len(ch) == len(pr), (chains, pre)
    out = []
    for c, p in zip(ch, pr):
        for k in range(len(c["pts"]) - 1):
            if p["wind"][k + 1] - p["wind"][k] == 1:
                out.append(float(c["gm_p"][k]))
    return np.array(out)


def chain_paths(q: dict, chains: str | None, pre: str | None) -> tuple[Path, Path]:
    if chains is None:
        chains = q.get("chains_json") or str(Path(q["chains_source"]).with_name("chains.json"))
    if pre is None:
        pre = q.get("pre_chains_json")
        if pre is None:  # band-1/2 exports: <set>_gmxc -> <set>_c
            d = Path(chains).parent
            assert d.name.endswith("_gmxc"), f"cannot derive the pre-gm chain set from {d}; pass --pre"
            pre = str(d.with_name(d.name[:-len("_gmxc")] + "_c") / "chains.json")
    f = lambda s: Path(s) if Path(s).is_absolute() else REPO / s  # noqa: E731
    return f(chains), f(pre)


def read_answers(path: Path) -> dict[str, dict]:
    last = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["item_id"]] = r  # last answer per item wins (as in the viewer)
    return last


def answer_class(r: dict) -> str:
    a = str(r.get("answer"))
    ws = r.get("winding_step")
    if a == "u":
        return "unsure"
    if a == "a":
        return "open_air"
    if a == "s" or ws == 0:
        return "same_sheet"
    if isinstance(ws, int) and ws >= 1:
        return "count"
    return "other"


def collect(q: dict, ans: dict[str, dict], mid_only: bool = False):
    """(gm_p, y) of the answered calibration checks + bookkeeping counts."""
    items = {str(it["step_id"]): it for it in q.get("queue", q.get("items", []))}
    stats = dict(log_items=len(ans), controls_skipped=0, not_in_queue=0, not_single=0, outside_mid=0,
                 unsure=0, same_sheet=0, open_air=0, other=0, used=0)
    gm, y = [], []
    for iid, r in ans.items():
        if r.get("is_control"):
            stats["controls_skipped"] += 1
            continue
        it = items.get(iid.split("#")[0])
        if it is None:
            stats["not_in_queue"] += 1
            continue
        if int(it["count_before_gm"]) != 1:
            stats["not_single"] += 1
            continue
        if mid_only and not (0.10 <= it["gm_p"] <= 0.70):
            stats["outside_mid"] += 1
            continue
        c = answer_class(r)
        if c != "count":
            stats[c] += 1
            continue
        gm.append(float(it["gm_p"]))
        y.append(1.0 if int(r["winding_step"]) >= 2 else 0.0)
        stats["used"] += 1
    return np.array(gm), np.array(y), stats


def calibrate(gm: np.ndarray, y: np.ndarray, pop: np.ndarray, boot: int = 2000, seed: int = 0) -> dict:
    w = fit1(logit(gm), y)
    thr = thr_balance(w, pop)
    p = 1 / (1 + np.exp(-(w[0] + w[1] * logit(pop))))
    rng = np.random.default_rng(seed)
    bt = []
    for _ in range(boot):
        s = rng.integers(0, len(gm), len(gm))
        bt.append(thr_balance(fit1(logit(gm[s]), y[s]), pop))
    bt = np.array(bt)
    q = lambda v: [round(float(x), 4) for x in np.percentile(bt, v)]  # noqa: E731
    return dict(threshold=round(thr, 4), ci95=q([2.5, 97.5]), p10_p50_p90=q([10, 50, 90]), boot=boot, seed=seed,
                w=[round(float(x), 4) for x in w], n_fit=int(len(gm)), n_pos=int(y.sum()),
                pop_single_steps=int(len(pop)), predicted_doubles=round(float(p.sum()), 1),
                raises_at_threshold=int((pop >= thr).sum()))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--chains", help="gm chain set chains.json (default: from the queue file)")
    ap.add_argument("--pre", help="pre-gm chain set chains.json (default: from the queue file)")
    ap.add_argument("--mid-only", action="store_true", help="fit only on checks with gm_p in [0.10, 0.70] (calib.py "
                    "pool); default: all answered single-step checks (the tail strata give the logistic support)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--current-thr", type=float, default=None, help="reference threshold (default: queue gm_thr_chains "
                    "or 0.18)")
    ap.add_argument("--json", help="also write the result here")
    a = ap.parse_args(argv)
    q = json.loads(Path(a.queue).read_text())
    chains, pre = chain_paths(q, a.chains, a.pre)
    pop = population_gm(chains, pre)
    gm, y, stats = collect(q, read_answers(Path(a.answers)), a.mid_only)
    print(f"answers: {stats}")
    if len(gm) < 2 or y.min() == y.max():
        print("too few answered checks, or all answers on one side: no threshold", file=sys.stderr)
        sys.exit(2)
    res = calibrate(gm, y, pop, a.boot, a.seed)
    cur = a.current_thr if a.current_thr is not None else float(q.get("gm_thr_chains", 0.18))
    res.update(stats=stats, chains=str(chains), pre=str(pre), mid_only=a.mid_only, current_thr=cur,
               raises_at_current=int((pop >= cur).sum()))
    if len(gm) < 50:
        print(f"WARNING: only {len(gm)} answered checks (section 3 wants 100-150 for +-0.03)")
    print(f"fit on {res['n_fit']} checks ({res['n_pos']} with step >= 2): logit P = {res['w'][0]:+.3f} "
          f"{res['w'][1]:+.3f} * logit(gm_p)")
    print(f"predicted true doubles among {res['pop_single_steps']} single steps: {res['predicted_doubles']}")
    print(f"THRESHOLD gm_p >= {res['threshold']:.3f}   95% bootstrap CI [{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}]"
          f"   (p10/p50/p90 {res['p10_p50_p90']}, {a.boot} resamples)")
    print(f"raises at threshold: {res['raises_at_threshold']}; at current {cur:g}: {res['raises_at_current']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
