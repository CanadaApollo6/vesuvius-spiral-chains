# Label-free merged-contact detector: pre-set gate

**Gate frozen: 2026-10-23T22:52:49Z (UTC)**, written as the first action of this
work, before reading the dev step tables in detail and before any detector is
fitted. Nothing below this "Gate" section may be changed after this timestamp;
later sections are append-only and timestamped.

## Goal

A label-free per-step detector: for each chain step (consecutive m7 mask
crossings along a ray) decide whether the step hides a merged sheet contact,
i.e. should count as >= 2 windings instead of 1. Inputs allowed: CT intensity,
the m7 mask, grad_mag, geometry (ray, z, radius, local pitch from neighbouring
rays / z levels). Never labels. Labels are used only to fit (on training
regions) and to score (on the held-out region).

Small models only: thresholds or a logistic on a handful of features.

## Evaluation protocol

Dev regions (labels exist):
- PHerc0139 band 1 (z 3000-4500), tight
- PHerc0139 band 2 (z 1750-3000), tight
- PHercParis4 z 9000-10500 (burned), tight
- PHerc1667 z 6500-8000, wide spacing (control; `--raise auto` already
  gives no raise there, must stay unaffected)

Leave-one-region-out (LORO) over the three tight regions: fit all free
parameters (feature weights, threshold, any scaling) on two tight regions,
report on the third. In-sample fits on all three do NOT count toward the gate.
Feature *selection* is done on LORO results; the final reported model is one
fixed feature set evaluated LORO on all three folds.

Metrics are the same definitions as in notes/2026-10-24-fourth-test-prereg.md
and data/scratch/fourth_test/dev_table.json (raise precision, drift over
10-20 windings, step precision), computed with the existing dev tooling in
data/scratch/fourth_test/ (dev_steps.py etc.). Baselines: no-raise and
gm 0.263.

## GATE (all must hold on EACH held-out tight region)

1. **Raise precision >= 70 %** (fraction of raised steps that are true merged
   contacts; current gm raise: 33-38 %).
2. **Drift over 10-20 windings within +/-0.5** (no-raise: -0.95 to -1.87;
   gm 0.263: -0.12 to -0.51).
3. **Step precision no worse than no-raise minus 1 pp.**

PHerc1667 (not part of the gate, but required for wiring): the detector must
leave it unaffected (no or negligible raises; report its numbers).

**Deadline:** decision by 2026-10-26 12:00 UTC at the latest. If the gate
fails: stop, write up, report. No dev fit, no confirmation.

If the gate passes: one dev fit on Paris4 z9000 vs stored fits with the fourth
test's dev go criterion (primary CI upper bound < 0 vs every base seed,
switches within +5 per 1000). Only if that goes: separate pre-registration and
two-stage-frozen confirmation on Paris4 z7000-8500.

**z7000-8500 stays untouched** (no chains, no CT features, no label metrics)
until gate and dev fit pass.

---

## Log (append-only)

### L1 (2026-10-23T22:57Z): operational definitions, fixed before any detector fit
- Steps: chain single steps (d0 = 1) of the no-raise chains in data/scratch/fourth_test/dev_cache (the same chains and
  label matching as dev_steps.py; primary label variant and kmax per region as there). A step is a "true merged
  contact" iff its label winding difference is >= 2.
- Raise precision = share of raised d0 = 1 steps with a label (both ends labelled, k < kmax) whose true difference is
  >= 2 (the post-hoc table's definition). Drift = dev_steps.evaluate drift10. Step precision = evaluate prec.
- Detector = logistic regression (L2, standardised features) on few label-free features, fitted on the pooled
  labelled single steps of the two training regions, each region weighted equally. **Threshold rule (fixed now):** the
  probability threshold at which the mean of the two training regions' drift10 is zero (the rule that produced
  gm 0.263). Raised steps count 2. The same rule is applied to every candidate feature set.
- Feature exploration (per-region AUCs, LORO tables of several candidate feature sets) is reported in full; the gate
  verdict uses one final feature set chosen by LORO behaviour and named before the verdict. All candidates are listed.
- Feature extraction: data/scratch/contact_det/feats.py (mask 25-line neighbourhood, grad_mag profile; no labels).
### L2 (2026-10-23T22:58Z): definition correction, before any detector fit
Reproducing the baseline showed the stored post-hoc table (33.1 / 34.3 / 35.6 %, 3,510 raised labelled steps at
Paris4) counts a raise as correct iff the true difference is **exactly 2** (the raise makes the step exact); with
">= 2" the same raises score 37.7 / 45.8 / 50.2 %. The gate's baseline numbers use "== 2", so **raise precision =
share of raised labelled single steps with true difference == 2** (L1 corrected). The ">= 2" share is reported as a
secondary. The detector's fitting target is the same event (true difference == 2). Recall is reported for the same
event. Only per-feature AUCs had been computed when this was found (data/scratch/contact_det/feats.py outputs, AUC vs
">= 2"); no detector had been fitted.
### L3 (2026-10-23T23:03Z): exploration so far; final feature set named before the verdict run
- Explored (LORO, threshold rule of L1): gm7 (the frozen raise's 7 features refitted), A (8: logI, logIr, L_pitch,
  logperr, rlmax, split, m_gapmin, g_max), B (16: gm7 + all mask / gm-profile features). Held-out raise precision
  29-43 % in every fold (data/scratch/contact_det/loro_explore1.json). Held-out precision at ANY raised fraction
  (top 0.5-12 %) peaks at ~50 % (prcurve.py).
- Neighbour-ray features (nbfeats.py; z +- 64 / 128, same theta): AUC 0.50-0.65.
- CT profile (ctfeats.py, streamed CT chunks, pilot on 2 z levels each of PHerc0139 b1 (z 3360, 3424; 403 chunks) and
  Paris4 z 9000 (z 9504, 9568; 563 chunks)): AUC 0.52-0.67, below grad_mag (0.73-0.80). Optimistic in-sample check
  (21 features, fitted and scored on the same pilot steps): precision 61 % at a 4 % raised fraction (b1) and 49 % at
  9 % (Paris4); cross-pilot 25-42 %. CT was therefore not extracted for the full regions (disclosed).
- Side-ambiguity diagnostic (tol_diag.py): counting a raise correct if true diff == 2 at j-1, j or j+1 gives only
  48-56 %, so the ceiling is not just "which side of a merged crossing".
- **Final set for the gate verdict: F = A + nb_d2 + nb_cnt (10 features)**, run LORO with the L1 threshold rule.

## Result (2026-10-23T23:04Z): GATE FAILED. Stopped; no dev fit, no confirmation, no wiring.

Final set F (10 label-free features: logI, logIr, L_pitch, logperr, rlmax, split, m_gapmin, g_max, nb_d2, nb_cnt),
logistic, LORO, threshold = zero mean training drift10 (data/scratch/contact_det/loro_F.json; baselines base.json).

| held-out | detector thr | raise prec (== 2) [>= 2] | d10 | step prec | within ±1 | raised | recall | gate |
|---|---|---|---|---|---|---|---|---|
| PHerc0139 b1 | 0.193 | **41.5** [54.9] | −0.38 | 88.7 | 79.2 | 7.8 % | 25.5 | fail (1) |
| PHerc0139 b2 | 0.219 | **31.6** [37.7] | **−0.67** | **86.4** | 68.2 | 12.5 % | 30.4 | fail (1, 2, 3) |
| Paris4 z 9000 | 0.161 | **35.9** [47.5] | **+0.94** | **74.2** | 53.2 | 34.1 % | 62.4 | fail (1, 2, 3) |

| reference | b1: prec / d10 / step prec | b2 | Paris4 |
|---|---|---|---|
| no raise | – / −0.95 / 88.8 | – / −1.54 / 88.6 | – / −1.87 / 78.4 |
| gm 0.263 | 34.3 / −0.16 / 87.7 | 33.1 / −0.51 / 86.2 | 35.6 / −0.12 / 76.2 |

Gate items: (1) raise precision >= 70 %: fails everywhere (31.6-41.5 %). (2) |d10| <= 0.5: only b1 passes.
(3) step precision >= no raise − 1 pp: only b1 passes (−0.1 pp); b2 −2.2, Paris4 −4.2.
PHerc1667 (not gated): F would raise 2.5-6.2 % of its steps, d10 +0.95 -> +1.22 … +1.74 (it would still need the
S_pack branch to stay at no raise).

**Why.** Every label-free signal tried is weaker than, or redundant with, the lasagna grad_mag integral (per-step AUC
0.73-0.80 vs true diff == 2). The mask sees nothing in the gap (the missing label sheet lies anywhere in the gap,
mostly central, and true-2 steps are only ~1.1-1.25x the local pitch, so geometry barely separates them: L_pitch AUC
0.56-0.74). CT profiles (pilot) and neighbour rays add little (AUC <= 0.67). Held-out precision tops out near 50 %
at any raised fraction, and even an over-fitted in-sample CT model stays below 70 % at the needed raise volume. With a
true-2 prevalence of 8-16 % per single step, 70 % precision at the recall needed for |d10| <= 0.5 would take a much
sharper per-step signal than these inputs carry.

**Spend:** $0 (no pods; pod list checked empty, no vss- pods). Local CPU only; CT streamed: 966 chunks of 128^3
(~2 GB, data/scratch/contact_det/ctcache/). z 7000-8500 untouched.
