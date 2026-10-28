"""Test calib_apply.py on the band-2 queue (PHerc0139, data/scratch/hitl/check_queue_band2.json) with synthetic answers.

Synthetic truth: P(step >= 2 | gm_p) = sigmoid(W_TRUE . [1, logit gm_p]); the "true" threshold is calib.py's
thr_balance(W_TRUE) on band 2's single steps. Checks:
  1. log parsing: last answer per item wins; controls, 'u', 's', 'a' are left out (exact counts);
  2. the CLI runs end to end and prints a threshold inside its own CI;
  3. CI coverage of the true threshold over repeated synthetic sessions (>= 80 % for a nominal 95 % CI);
  4. --mid-only restricts the fit to gm_p in [0.10, 0.70].
No labels are used (synthetic answers only).
  uv run python tools/calib_apply_test.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_apply as ca  # noqa: E402

QUEUE = ca.REPO / "data/scratch/hitl/check_queue_band2.json"
W_TRUE = np.array([-0.4, 1.3])


def synth_log(q, rng, path: Path, n_u=15, n_s=10, n_a=5, n_rev=5):
    items = [it for it in q["queue"] if it["count_before_gm"] == 1]
    rng.shuffle(items)
    recs, expect = [], dict(unsure=n_u, same_sheet=n_s, open_air=n_a)
    special = ["u"] * n_u + ["s"] * n_s + ["a"] * n_a
    for j, it in enumerate(items):
        p = 1 / (1 + np.exp(-(W_TRUE[0] + W_TRUE[1] * ca.logit(it["gm_p"]))))
        ws = 2 if rng.random() < p else 1
        if j < len(special):
            a = special[j]
            rec = dict(answer=a, n_between=None, winding_step=0 if a == "s" else None)
        else:
            if j < len(special) + n_rev:  # a wrong first answer, overwritten by a revisit
                recs.append(dict(item_id=it["step_id"], answer=str(2 - ws), n_between=2 - ws,
                                 winding_step=3 - ws, is_control=False))
            rec = dict(answer=str(ws - 1), n_between=ws - 1, winding_step=ws)
        recs.append(dict(item_id=it["step_id"], is_control=False, **rec))
    for it in q["controls"][:10]:
        recs.append(dict(item_id=it["step_id"], answer="1", n_between=1, winding_step=2, is_control=True))
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    expect.update(controls_skipped=min(10, len(q["controls"])), used=len(items) - len(special))
    return expect


def main():
    q = json.loads(QUEUE.read_text())
    chains, pre = ca.chain_paths(q, None, None)
    pop = ca.population_gm(chains, pre)
    thr_true = ca.thr_balance(W_TRUE, pop)
    print(f"band 2: {len(pop)} single steps; synthetic truth w={W_TRUE.tolist()} -> true threshold {thr_true:.4f}")
    tmp = Path(tempfile.mkdtemp(prefix="calib_apply_test_"))
    rng = np.random.default_rng(0)
    log = tmp / "answers.jsonl"
    expect = synth_log(q, rng, log)
    # 1 + 2: CLI end to end
    res = ca.main(["--queue", str(QUEUE), "--answers", str(log), "--boot", "1000", "--json", str(tmp / "res.json")])
    for k, v in expect.items():
        assert res["stats"][k] == v, (k, res["stats"][k], v)
    assert res["ci95"][0] <= res["threshold"] <= res["ci95"][1], res
    assert json.loads((tmp / "res.json").read_text())["threshold"] == res["threshold"]
    # 3: CI coverage over repeated synthetic sessions
    hits, R = 0, 40
    for r in range(R):
        synth_log(q, np.random.default_rng(100 + r), log)
        gm, y, _ = ca.collect(q, ca.read_answers(log))
        c = ca.calibrate(gm, y, pop, boot=300, seed=r)
        hits += c["ci95"][0] <= thr_true <= c["ci95"][1]
    print(f"CI95 coverage of the true threshold: {hits}/{R}")
    assert hits >= 0.8 * R, hits
    # 4: --mid-only
    # (every band-2 queue step has gm_p in [0.10, 0.70], so move 8 of them to the tails in a copy)
    q2 = json.loads(json.dumps(q))
    singles = [it for it in q2["queue"] if it["count_before_gm"] == 1]
    for j, it in enumerate(singles[:8]):
        it["gm_p"] = 0.05 if j % 2 else 0.85
    synth_log(q2, np.random.default_rng(0), log, n_u=0, n_s=0, n_a=0, n_rev=0)
    gm, y, st = ca.collect(q2, ca.read_answers(log), mid_only=True)
    assert gm.size and gm.min() >= 0.10 and gm.max() <= 0.70 and st["outside_mid"] == 8, st
    print(f"mid-only: {st['used']} used, {st['outside_mid']} outside [0.10, 0.70]")
    print("calib_apply TEST OK")


if __name__ == "__main__":
    main()
