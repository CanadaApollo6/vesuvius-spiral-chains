# 2026-10-22: human check budget on the auto winding chains (HITL), simulated with oracle labels

Goal: show that automatic chains plus a small human check budget on only the most uncertain steps recover most of
the human-oracle gain in the villa spiral fitter. The human is simulated with the human-mesh oracle now. A real timed
session (~20-30 min, viewer built by another agent) follows around Oct 25.

Dev data only: band 1 (z 3000-4500) and band 2 (z 1750-3000). z 6000-8500 and other regions were not read.

Code and outputs are in data/scratch/hitl/:

| file | what |
|---|---|
| hitl_lib.py | band loader, uncertainty features, check simulator, drift/precision evaluator, chain writer |
| fit_unc.py -> unc_model.pkl/json, perr_band{1,2}.npy | uncertainty model (z 4500-6000 only) and per-step p_err |
| recal.py -> recal.json | budget-0 recalibrated gm threshold |
| sim.py -> sim_curve.json (show.py prints it) | offline budget curve |
| calib.py -> calib.json | extra: in-band threshold calibration from N human checks |
| write_arms.py -> arm_sets_sha256.json | the fit-arm chain sets |
| export_queue.py -> check_queue_band{1,2}.json | viewer queue |
| where.py | where the uncertain steps lie |
| pods/ | grab, boot, collector (flock-wrapped copy) and post-processing scripts and logs |

All commands run with `uv run --with scikit-learn python data/scratch/hitl/<script>.py`.

## 1. Uncertainty ranking (label-free, fitted on z 4500-6000 only)
- Model: logistic P(step correct), fitted on the z 4500-6000 stepconf rows (`feats_z4500.json`; gm_p out of fold by
  500-z slab, as in notes/2026-10-20-auto-windings.md). p_err = 1 - P.
- Features are the ones any chain set provides from chains.json alone:
  - d, raised, graph2, bridged;
  - gm_p, |logit gm_p - logit thr|, gm_p x raised;
  - chain context (raised steps within +-3, max gm_p of the neighbours);
  - neighbour-ray consistency (z +-64, theta +-15).
  - The site-fitted logperr and the crop-graph features are left out.
- OOF AUC on z 4500-6000 is 0.777. The full 41-feature stepconf set gives 0.775 and gm-only features 0.774, so nothing is
  lost.
- Applied frozen to the gmxc chains (arm21 / arm23 inputs). Labels are used only to score.
  - band 1: AUC 0.832 (0.846 vs CT `supported`);
  - band 2: AUC 0.881 (0.880 vs CT `minus1`).
- **What it ranks.** The top 10 % of steps are 95-99 % gm-raised steps (p_err 0.72-0.86). After them come the graph's own
  d = 2 steps. In practice the ranking says "check the gm raises, the least confident first".

**Human check, as simulated.** The chain's step count is replaced by the oracle's count (the human-mesh label difference
at the two endpoints, 4 vox, unique by 1 vox). Where the oracle has no count, the step is dropped and the chain is split
there (a human saying "unsure").
- Truth variants: original labels, and CT-corrected counts (relabel maps; band 1 `supported` / `outer_minus2`, band 2
  `minus1` / `minus2`).
- Checks against CT truth use the same map that is used for scoring.
- A check that comes out as 0 turns (a duplicate label, e.g. w041 = w042) is dropped.

**Answerability is the bottleneck.** The oracle can answer only 2.6-6 % of the top-2 % steps (overall: band 1 24 %,
band 2 14 %).
- The uncertain steps sit where the labels are not:
  - 18-30 % have the inner end at r < 600, where meshes are rare;
  - the median radius is ~1,200-1,460;
  - 10-28 % are bridged (no graph solve).
- So under the specified rule, the simulated human drops ~95 % of the checked steps at budgets up to 2 %, and ~90 % at
  10 %.
- I added two controls to separate "human" from "dropping":
  - U-drop: drop the top-N steps, no human;
  - U-ans: top-N among answerable steps only. This bounds a human who can always answer, which a CT-viewing human may come
    close to. Its precision is partly circular: checked steps are scored against the same labels.

## 2. Offline curve (step precision and drift vs CT-corrected counts)
Drift is the mean (chain - label) winding difference over chain point pairs 10-20 labelled windings apart, per chain
piece. Precision is over verifiable kept steps.
- U = uncertainty-ranked checks, answered with CT truth (band 1 `supported`, band 2 `minus1`) or dropped.
- R = random order (mean of 10 seeds).

Band 1 (40,658 steps; auto = arm23's chains):

| checks N | % of steps | U answered | U prec / drift sup / drift om2 | U-drop prec / drift sup | R prec / drift sup | U-ans prec / drift sup / om2 |
|---|---|---|---|---|---|---|
| 0 (auto) | 0 | | 84.9 / +0.57 / +1.02 | | | |
| 0 (recal0, thr 0.263) | 0 | | **87.7 / -0.15 / +0.28** | | | |
| 100 | 0.25 | 3 | 84.9 / +0.57 / +1.03 | 84.9 / +0.57 | 84.9 / +0.57 | 85.6 / +0.54 / +0.99 |
| 200 | 0.49 | 8 | 85.0 / +0.57 / +1.02 | 85.0 / +0.57 | 85.0 / +0.57 | 86.5 / +0.49 / +0.94 |
| 407 | 1 | 23 | 85.1 / +0.55 / +1.00 | 85.0 / +0.54 | 85.1 / +0.56 | 88.2 / +0.37 / +0.82 |
| 813 | 2 | 47 | 85.3 / +0.52 / +0.97 | 85.2 / +0.53 | 85.2 / +0.55 | 91.1 / +0.21 / +0.65 |
| 2,033 | 5 | 219 | 86.6 / +0.30 / +0.73 | 86.3 / +0.29 | 85.7 / +0.53 | 95.0 / +0.20 / +0.65 |
| 4,066 | 10 | 583 | 89.6 / -0.11 / +0.31 | 88.9 / -0.14 | 86.4 / +0.47 | 97.0 / +0.38 / +0.83 |
| 8,132 | 20 | 1,286 | 93.4 / -0.29 / +0.11 | 92.4 / -0.25 | 87.9 / +0.38 | 98.9 / +0.58 / +1.04 |

Band 2 (30,626 steps; auto = arm21's chains):

| checks N | % of steps | U answered | U prec / drift m1 / drift m2 | U-drop prec / drift m1 | R prec / drift m1 | U-ans prec / drift m1 / m2 |
|---|---|---|---|---|---|---|
| 0 (auto) | 0 | | 80.3 / +0.47 / +1.21 | | | |
| 0 (recal0, thr 0.263) | 0 | | **86.2 / -0.51 / +0.16** | | | |
| 100 | 0.33 | 1 | 80.3 / +0.46 / +1.19 | 80.3 / +0.46 | 80.3 / +0.46 | 82.3 / +0.39 / +1.13 |
| 200 | 0.65 | 3 | 80.3 / +0.44 / +1.17 | 80.3 / +0.44 | 80.4 / +0.45 | 84.4 / +0.30 / +1.04 |
| 306 | 1 | 8 | 80.5 / +0.44 / +1.17 | 80.4 / +0.44 | 80.5 / +0.43 | 86.5 / +0.25 / +0.99 |
| 613 | 2 | 18 | 80.7 / +0.41 / +1.13 | 80.6 / +0.40 | 80.7 / +0.40 | 91.8 / +0.11 / +0.84 |
| 1,531 | 5 | 87 | 82.1 / +0.16 / +0.86 | 81.8 / +0.19 | 81.2 / +0.32 | 96.4 / +0.18 / +0.91 |
| 3,063 | 10 | 298 | 86.4 / -0.23 / +0.37 | 85.4 / -0.32 | 82.3 / +0.25 | 98.4 / +0.29 / +1.02 |
| 6,125 | 20 | 590 | 91.7 / -0.54 / +0.04 | 90.3 / -0.59 | 84.2 / +0.21 | 99.0 / +0.32 / +1.04 |

CT says the truth lies between the two band-2 maps (w034 -> w051 = 14.6-15.4 turns vs 16 under minus1 and 15 under
minus2). So the band-2 target is drift about halfway between the two columns.

- **Uncertainty ranking beats random** from ~2 % on. At 10 %: band 1 89.6 vs 86.4 %, band 2 86.4 vs 82.3 %, and the drift
  falls to about zero (random leaves it at +0.25 to +0.47).
- **Almost all of that is dropping, not the human's answers.** U-drop (no human) is within 0.3-1.4 pp and ~0.1 turns of
  U at every budget. The oracle answers too few of the uncertain steps to matter.
- **Session-sized budgets do nothing band-wide.** Up to 300 checks (~1 h at 12 s each), every metric moves by <= 0.03 turns
  and <= 0.2 pp, in both bands, for U, U-drop and R alike.
- **An always-answering human (U-ans) helps precision quickly but not drift.**
  - Band 2 at 200 checks: 80.3 -> 84.4 % and drift +0.47 -> +0.30.
  - Band 2 at 613 checks: 91.8 % and +0.11.
  - Past ~2-3 % the drift rises again: the answerable set is labelled-region steps whose labels themselves over-count,
    and the drift is carried by unlabelled stretches the checks never touch.
- **Budget 0, recalibrated (recal0) is the strongest point on the curve** for its cost.
  - It matches the 10 % uncertainty budget on drift and is within 2 pp on precision, with zero checks.
  - Band 1: 87.7 % / -0.15 (sup) / +0.28 (om2).
  - Band 2: 86.2 % / -0.51 (m1) / +0.16 (m2), i.e. about -0.18 at the CT midpoint.

## 3. Budget 0, recalibrated: gm threshold for zero drift against CT-corrected counts
- Fitted on band 1 (gmxc gm_p) plus z 4500-6000 (OOF gm_p) only, against CT `supported`. The w041 = w042, w043+ -1 and
  w045 = w046 numbering errors are mesh-level, so the band-1 map is applied to the z4500 oracle labels too. That is an
  assumption: the CT audit itself covers z 1750-4500.
- Zero-drift thresholds (data/scratch/hitl/recal.json):

| target | band 1 | z 4500-6000 | pooled (pairs-weighted) |
|---|---|---|---|
| original labels | 0.175 | 0.186 | (frozen rule: 0.18) |
| CT `supported` | 0.237 | 0.295 | **0.263 (used)** |
| CT `outer_minus2` | 0.333 | > 0.60 (never reaches 0) | 0.423 |

- **Frozen: 0.263**, picked before any band-2 drift at that value was computed.
  - Raises: band 1 18.4 -> 9.9 % of single steps; band 2 25.2 -> 14.9 %.
  - The coordinator's 0.46 (read on band 1 alone) was not used.
- Band 2 (scored only): precision 79.9 -> 85.8 % (orig labels), drift -0.51 (m1) / +0.16 (m2), and over 3-6 windings
  +0.45 -> +0.13.

**Extra: "human calibrates, machine applies"** (calib.py).
- N checks on random single steps with gm_p in [0.10, 0.70], answerable ones only. Fit P(true >= 2 | logit gm_p) on them,
  then set the in-band threshold where the number of raises equals the predicted number of true doubles. 200 draws per N.

| band | N | threshold p10 / p50 / p90 | drift vs CT (sup or m1) at p10 / p50 / p90 | drift m2 (band 2) |
|---|---|---|---|---|
| 1 | 100 | 0.19 / 0.23 / 0.27 | +0.40 / +0.05 / -0.20 | |
| 1 | 150 | 0.20 / 0.23 / 0.26 | +0.29 / +0.05 / -0.14 | |
| 1 | all (3,020) | 0.229 | +0.05 | |
| 2 | 100 | 0.24 / 0.29 / 0.34 | -0.37 / -0.65 / -0.84 | +0.32 / +0.02 / -0.20 |
| 2 | 150 | 0.26 / 0.29 / 0.33 | -0.45 / -0.63 / -0.81 | +0.22 / +0.03 / -0.17 |
| 2 | all (1,619) | 0.290 | -0.65 | +0.02 |

- 100-150 checks pin an in-band threshold to about +-0.03-0.04, i.e. +-0.2-0.3 turns of drift.
- The in-band band-2 optimum (0.29) is close to the label-free 0.263. So in these bands, a session spent on calibration
  would confirm recal0 rather than beat it. It would matter in a region where the transfer fails.

## 4. Fit arms
Villa fitter 0fb38c45, weight 10, spacing 0, 30k steps, 1x L40S per arm. Each arm reuses its base arm's env with only
ARM and REL_WINDINGS changed (spiral/arms/arm26-29.env). The windings sha256 was verified on each pod
(data/scratch/hitl/arm_sets_sha256.json).

- Scoring:
  - spiral_eval B - base, with the 95 % vertex/site bootstrap;
  - pairs with CT-flagged pairs excluded (eval/pairs_exclude.py);
  - own-offset inner/outer wrong area on consistent subsets (eval/subset_wrong_area.py --own-offsets, split 41). This is
    the label-robust metric.
- Pooled own = matched-weighted inner + outer (all subsets).
- Table: `python3 data/scratch/hitl/hitl_table.py` -> arm_table.json.
- Lower is better for wrong area and switches; higher is better for pair accuracy.

### Band 2 (z 1750-3000), B - arm20 (no windings)

| arm | chains | checks | pair acc | pair acc excl CT-flag | wrong area seam (label-dep.) | switches/1000 | own-off inner cons | own-off outer cons | own-off pooled |
|---|---|---|---|---|---|---|---|---|---|
| arm21 | auto (gmxc, thr 0.18) | 0 | -4.4 [-9.5, +1.9] | -5.0 [-8.9, -1.0]* | -9.0 [-13.2, -5.1]* | +30.8 [+26.2, +35.6]* | +4.2 [+2.1, +6.4]* | +3.3 [+0.7, +5.8]* | +1.2 |
| arm22 | oracle chains | all | -1.6 [-5.9, +3.8] | -2.3 [-5.7, +1.2] | -19.3 [-24.1, -14.9]* | +7.9 [+3.1, +12.6]* | -4.5 [-7.3, -1.8]* | -1.7 [-4.1, +0.7] | -9.3 |
| **arm26** | **recal0 (thr 0.263)** | **0** | -3.7 [-8.8, +2.4] | -4.3 [-8.1, -0.6]* | -0.7 [-2.2, +0.9] | +18.5 [+14.9, +22.0]* | **-5.3 [-7.7, -3.0]*** | -0.5 [-2.5, +1.5] | **-6.9** |
| arm27 | U 2 % | 613 (18 answered, 595 dropped) | -3.9 [-8.7, +2.0] | -4.8 [-8.8, -0.7]* | -11.0 [-16.3, -6.1]* | +30.6 [+26.0, +35.9]* | +4.3 [+2.1, +6.6]* | -1.6 [-4.6, +1.4] | -1.5 |
| arm28 | U 10 % | 3,063 (298 answered, 2,765 dropped) | -2.4 [-6.8, +3.0] | -3.4 [-6.8, +0.1] | -1.3 [-3.0, +0.1] | +21.8 [+17.8, +26.1]* | +3.6 [+1.5, +5.9]* | +1.0 [-1.2, +3.3] | -0.7 |

(* = CI excludes 0.) In band 2 the oracle chains do not beat the no-windings base on pair accuracy, so "fraction of the
oracle gain" is only meaningful for wrong area and switches.

### Band 1 (z 3000-4500), B - arm0 (no windings)

| arm | chains | checks | pair acc (= excl CT-flag) | wrong area seam (label-dep.) | switches/1000 | own-off inner cons | own-off outer cons | own-off pooled |
|---|---|---|---|---|---|---|---|---|
| arm23 | auto (gmxc, thr 0.18) | 0 | +15.0 [+9.5, +21.3]* | -11.9 [-14.7, -8.9]* | +15.2 [+11.5, +19.5]* | -19.1 [-22.1, -16.1]* | +9.5 [+7.5, +11.6]* | -9.6 |
| arm2 | oracle chains | all | +14.1 [+8.0, +21.2]* | -13.5 [-16.0, -10.8]* | +8.2 [+5.5, +11.4]* | -18.6 [-21.3, -15.9]* | -0.2 [-0.6, +0.2] | -12.2 |
| arm29 | recal0 (thr 0.263) | 0 | +3.1 [-0.9, +8.0] | -0.6 [-2.7, +1.6] | +8.2 [+4.8, +12.1]* | -0.8 [-2.2, +0.6] | +0.8 [+0.5, +1.2]* | -7.4 |

Per the task, band 1 was to get "the best one" of the band-2 arms. That was recal0 on own-offset pooled, the label-robust
metric. The U arms lost to it in band 2 by a wide margin, so they were not run in band 1.

### Fraction of the oracle gain vs number of human checks
Own-off pooled is the headline (label-robust). Fraction = (arm - base) / (oracle - base). Human time assumes 10-15 s
per check.

| band | arm | checks | human time | own-off pooled | % of oracle gain (pooled) | % of auto->oracle gap closed (pooled) | switches: % of auto->oracle gap closed |
|---|---|---|---|---|---|---|---|
| 2 | arm21 auto | 0 | 0 | +1.2 | -13 % | 0 % | 0 % |
| 2 | **arm26 recal0** | **0** | **0** | **-6.9** | **74 %** | **76 %** | **54 %** |
| 2 | arm27 U 2 % | 613 | 1.7-2.6 h | -1.5 | 16 % | 25 % | 1 % |
| 2 | arm28 U 10 % | 3,063 | 8.5-12.8 h | -0.7 | 8 % | 18 % | 39 % |
| 2 | arm22 oracle | all (~30k steps) | ~85-130 h | -9.3 | 100 % | 100 % | 100 % |
| 1 | arm23 auto | 0 | 0 | -9.6 | 79 % | 0 % | 0 % |
| 1 | arm29 recal0 | 0 | 0 | -7.4 | 61 % | -89 % | 100 % |
| 1 | arm2 oracle | all | | -12.2 | 100 % | 100 % | 100 % |

- **More checks did not buy more of the oracle gain.** 613 and 3,063 simulated checks got 16 % and 8 % of the band-2
  oracle gain. Zero checks with the recalibrated threshold got 74 %.
- **The check arms are mostly drop arms.** The oracle answers 3 % (613) and 10 % (3,063) of the checked steps. The rest are
  dropped, which splits the chains: 597 -> 1,053 -> 2,702 pieces. Splitting removes long-range constraints, and the fitter
  pays for it in switches (+30.6 at 613 checks, the same as auto). The offline curve predicted this (U ≈ U-drop, §2).
- **What the fits say about a real session (100-150 checks, ~20-40 min):** those checks touch < 0.5 % of steps. They will
  not move a band-wide fit measurably, whichever way the checks are answered.

### Band 1 follow-up: is arm29's loss vs arm23 real, or human-label bias? (coordinator check)
All of this is CPU-only and flock-wrapped. Scripts are in data/scratch/hitl/arm29_check/: run_swa.sh, perk.py,
table2.py; outputs are swa_band1_*_{original,supported,outer_minus2}[_exclflag].json in data/scratch/label_audit/, plus
perk.json and pairs_arm23_arm29.json.

**(1) Pair accuracy, CT-flagged pairs excluded (band-1 flag set), paired vs arm0, 95 % site bootstrap.**

| arm | acc_matched | B - arm0 | under `supported_plus_2167` truth |
|---|---|---|---|
| arm0 | 78.0 | - | - |
| arm23 (0.18) | 93.0 | +15.0 [+9.5, +21.3] | +10.8 [+4.5, +17.6] |
| arm2 (oracle) | 92.1 | +14.1 [+8.0, +21.2] | +10.7 [+3.9, +18.5] |
| arm29 (0.263) | 81.1 | +3.1 [-0.9, +8.0] | (not run) |
| arm29 - arm23, paired | | **-11.9 [-16.8, -7.4]** | -7.9 [-12.9, -3.2] |

- The band-1 flag set excludes nothing beyond the 77 skip pairs. The band-1 pairs sit at k 24-39 and 48-58 and never
  straddle w041/42 or w045/46, so `supported` and `outer_minus2` change no expected value.
- So band-1 pair accuracy was never exposed to the w043+ over-count, and the arm29 loss is not a label artefact here.

**(2) Own-offset wrong area under the band-1 CT relabel** (own offsets refitted per subset, split 41). Each cell is
B - A [95 % CI].

| B - A | relabel | inner (k < 41) | outer (k >= 41) | inner cons | outer cons | pooled |
|---|---|---|---|---|---|---|
| arm23 - arm0 | supported | -32.5 [-36.4, -28.3] | +9.3 [+7.5, +11.2] | -19.1 [-22.1, -16.1] | +9.5 [+7.5, +11.6] | -9.6 |
| arm29 - arm0 | supported | -17.2 [-20.0, -14.3] | +0.8 [+0.5, +1.1] | -0.8 [-2.2, +0.6] | +0.8 [+0.5, +1.2] | -7.4 |
| arm2 - arm0 | supported | -26.6 [-29.6, -23.4] | +0.0 [-0.3, +0.4] | -18.6 [-21.3, -15.9] | -0.2 [-0.6, +0.2] | -12.2 |
| **arm29 - arm23** | supported | **+15.3 [+13.4, +17.1]** | **-8.6 [-10.4, -6.7]** | +18.3 [+14.5, +22.0] | -8.7 [-10.7, -6.6] | **+2.2** |
| arm23 - arm0 | outer_minus2 | -32.5 [-36.4, -28.3] | +4.7 [+3.8, +5.8] | -19.1 [-22.1, -16.1] | +4.7 [+3.8, +5.7] | -12.2 |
| arm29 - arm0 | outer_minus2 | -17.2 [-20.0, -14.3] | +0.8 [+0.5, +1.1] | -0.8 [-2.2, +0.6] | +0.9 [+0.6, +1.2] | -7.4 |
| arm2 - arm0 | outer_minus2 | -26.6 [-29.6, -23.4] | +0.1 [-0.1, +0.3] | -18.6 [-21.3, -15.9] | +0.0 [-0.3, +0.4] | -12.1 |
| **arm29 - arm23** | outer_minus2 | **+15.3 [+13.4, +17.1]** | **-4.0 [-4.9, -3.0]** | +18.3 [+14.5, +22.0] | -3.8 [-4.6, -3.0] | **+4.7** |
| arm29 - arm23 | supported, #2167 excl. | +17.5 [+15.4, +19.5] | -8.8 [-11.2, -6.3] | +23.3 [+19.2, +26.9] | -8.4 [-11.0, -5.8] | +3.2 |
| arm29 - arm23 | outer_minus2, #2167 excl. | +17.5 [+15.4, +19.5] | -4.5 [-5.8, -3.2] | +23.3 [+19.2, +26.9] | -4.0 [-5.0, -2.9] | +5.5 |

- Own-offset numbers under `supported` are identical to the original labels. The reference has no w041/w042/w045/w046
  vertices (blacklisted), and the k >= 43 shift is uniform inside each subset, so own offsets absorb it.
  `outer_minus2` (w050 = w051) does change the outer subset: it halves arm23's outer penalty.

**(3) Where arm29 loses vs arm23** (perk.py). Own offsets are refitted on each k group alone, so the numbers are
invariant to any uniform relabel inside the group.

| k group | arm23 - arm0 | arm29 - arm0 | arm2 - arm0 | arm29 - arm23 | arm29 - arm23, #2167 excl. |
|---|---|---|---|---|---|
| k 23-40 (inner; labels CT-consistent outside #2167) | -32.5 [-36.4, -28.3] | -17.2 [-20.0, -14.3] | -26.6 [-29.6, -23.4] | **+15.3 [+13.4, +17.1]** | +17.5 [+15.4, +19.5] |
| k 43-50 (the "w043+ -1" zone) | +21.6 [+19.0, +24.5] | +1.1 [+0.7, +1.4] | +0.5 [+0.3, +0.7] | **-20.5 [-23.4, -17.8]** | -21.8 [-25.2, -18.3] |
| k 51-59 (outer_minus2 zone) | +1.9 [+1.5, +2.2] | +1.0 [+0.6, +1.3] | -0.2 [-0.6, +0.1] | -0.9 [-1.1, -0.7] | -0.7 [-1.0, -0.5] |

Pair accuracy by range (arm29 - arm23): k 24-39 **-14.7 [-19.6, -9.9]** (n 6,393); k 48-58 -0.1 [-3.7, +3.6] (n 1,078).

- **Inner (k 23-40): arm29 loses on most windings.** Per k, with spiral_eval's global per-bin offsets, the loss is largest at k 28-34
  (+42 to +77 pp wrong at original labels). Labels there are CT-consistent, so this is a real fit loss: fewer raises means
  under-counted turns in the inner turns.
- **w043-w050: the opposite.** arm23 (0.18) is +21.6 pp worse than no windings at all. arm29 matches the oracle there
  (+1.1 vs +0.5). Inside k 43-50 the CT audit agrees with the labels' relative numbering (w044 -> w047 = 3 turns on most
  rays). So this too is a real fit effect, not label bias: 0.18 over-raises in the outer turns.
- **Interpretation (coordinator's rule).** arm23 still beats arm29 under (1): -11.9 [-16.8, -7.4], untouched by CT
  flags. It also beats it under (2) on pooled own-offset (+2.2 supported, +4.7 outer_minus2) and on inner by +15 to +18.
  The gap does not close; it widens under outer_minus2. **So 0.18 is genuinely better in band 1, and the arm29 loss is not
  a label-bias artefact.**
- The loss is concentrated in the inner turns (k 23-40), not in w043+. In w043-w050, 0.263 is genuinely better. The
  w043+ over-count does not drive the result; if anything it makes arm23's outer look slightly worse than it is
  (outer_minus2 halves arm23's outer penalty).
- **Implication.** The best gm threshold depends on radius in band 1 (lower inside, higher outside). The zero-drift-vs-CT
  calibration in §3 does not pick the fitter-optimal threshold here. A radius-dependent rule is a hypothesis for a new,
  pre-registered test. It must not be tuned on band 1 and then reported on band 1.

## 5. Cost
Four vss-hitl pods, all terminated by the collector right after the pull. The creates and terminations are logged in
data/scratch/p0139_spiral/pods_arms.txt, and `prime pods list` shows no vss-hitl pod afterwards.

| arm | provider, $/h | create -> terminate (UTC) | h | $ |
|---|---|---|---|---|
| arm26 | runpod 1x L40S, 1.11 | 10-21 22:49 -> 10-22 01:46 | 2.95 | 3.27 |
| arm27 | runpod 1x L40S, 1.11 | 23:03 -> 02:53 | 3.84 | 4.26 |
| arm28 | runpod 1x L40S, 1.11 | 23:10 -> 03:06 | 3.94 | 4.37 |
| arm29 | massedcompute 1x L40S, 0.82 | 23:11 -> 02:40 | 3.47 | 2.85 |
| total | | | 14.2 | **14.75** (cap 40) |

The scoring, vertex dumps, swa and the arm29 check ran locally on CPU under `flock /tmp/vss_eval.lock`.

## 6. Verdict
- **The hypothesis, as simulated, fails.** Auto chains plus a small human budget on the most uncertain steps do not
  recover most of the oracle gain.
  - Band 2: 613 checks (~2 h of human time) give 16 % of the oracle's own-offset gain, and 3,063 checks (~10 h) give
    8 %.
  - The uncertain steps are gm raises at small radius or in unlabelled stretches. The oracle cannot answer them, so a
    simulated check mostly drops the step.
  - Dropping breaks chains, and that costs about as much as it fixes.
- **Zero checks with a recalibrated threshold did better than any check budget in band 2.** recal0 (0.263) got 74 % of the
  oracle gain and closed 54 % of the switch gap.
- **But recal0 does not transfer to band 1.** There, 0.18 is genuinely better (§4 follow-up: not label bias). The loss is
  in the inner turns, while 0.263 wins in w043-w050.
  - So no single label-free threshold is right for both bands, nor even for both radii within band 1.
  - Recommended default: keep 0.18, the frozen rule, which matches the oracle in band 1. Treat 0.263 as a band-2 /
    outer-turn finding.
- **Where a human actually helps** is calibration, not step-by-step fixing. ~100-150 random checks on gm_p in
  [0.10, 0.70] pin an in-band threshold to ±0.03-0.04 (§3 extra).
  - Band 1 shows that a single in-band threshold can still be wrong by radius. A session should therefore stratify those
    checks by radius (inner/outer of the scroll).
  - That is a change to the queue: about 100 calibration items stratified by r, alongside the uncertainty queue. It is a
    recommendation and is not implemented here; check_queue_band{1,2}.json are unchanged.
- **For the real ~20-30 min session (band 2 queue):** treat it as a timing/answerability demo plus calibration data. It
  should not be expected to move band-wide fit metrics. The fit-level evidence says the valuable output of human time is
  a few hundred calibration labels, not corrections of the most uncertain steps.

## 7. Check queue for the viewer (data/scratch/hitl/check_queue_band2.json; band-1 alternative check_queue_band1.json)
**Band 2 is chosen for the real session.** Under the always-answering bound, band 2 gains more per check: +2.0 to +4.1 pp
precision and -0.08 to -0.17 turns of drift at 100-200 checks, vs +0.7 to +1.6 pp and -0.03 to -0.08 in band 1. Band 2
is also where the gm raise rule fails. The queue chains are arm21's input (band2_m7_ray_gmxc; sha256 in the file).

**Contents.**
- `queue`: the top 300 steps by p_err, ranks 1-300.
- `controls`: 40 oracle-answerable steps not in the queue: 20 gm-raised, 20 unraised with gm_p >= 0.10, shuffled. They
  measure human accuracy and time against `ct_count_minus1` / `ct_count_minus2`.
- Per item:
  - chain_id, step_index, chain_name (the collection name in relative_windings.json);
  - z and theta of the ray, r_inner / r_outer;
  - the two endpoints in 9.362 um coordinates, both as zyx and as xyz (xyz = relative_windings `p`);
  - chain_count (after gm), count_before_gm, gm_raised, gm_p, p_err, bridged;
  - oracle_count, ct_count_minus1, ct_count_minus2 (null when the oracle has no count);
  - oracle_bracket (the labelled windings just inside and outside on the ray);
  - label_error_zones.

**Queue composition (band 2).**
- 286 / 300 are gm raises; 87 are bridged (no graph solve; an endpoint can be in open air); 92 have r_inner < 600.
- Only 8 have an oracle / CT count.
- 253 have no labelled mesh on at least one side; 24 sit in the unlabelled w035-w050 gap; 5 in other unlabelled stretches.
- 114 lie in the #2167 sector (145-260 deg).
- **0 sit on a tight known label-error zone** (w034/w036, w040-w043 with w041 = w042, w045-w047, w049-w051 with w050
  missing). A zone is only counted when the step's own label bracket is <= 3 windings wide. The 40 controls include 1 such
  step.
- Band 1 alternative: 292 raises, 69 bridged, 13 with an oracle count, 4 on label-error zones, 132 in the #2167 sector.

**Expectations given to the viewer agent** (sent to the coordinator).
- Answer convention: the user counts sheets strictly between the two points. The log stores winding_step =
  n_between + 1, which replaces chain_count directly. 0 = same sheet, and is treated as a drop.
- 'u' (unsure), and an endpoint in open air, also mean drop: the chain is split at that step. The simulation used this
  same rule, so bridged steps stay in the queue and no re-precache is needed.
- With ~100-150 answers in random queue order (all top-300 items have p_err >= 0.83), the session will not move band-wide
  metrics measurably: ~0.1 turn at best under the always-answering bound.
- What the session can deliver:
  1. measured seconds per check (the time model below assumes 10-15 s);
  2. the human-measured precision of the least confident gm raises (the model expects ~14 % right: mean p_err 0.86, min 0.83; the oracle answers 8 of them, 1 right);
  3. a human answer rate on steps the oracle cannot answer (95 % of the queue);
  4. human accuracy on the 40 controls, scored against CT-corrected counts;
  5. enough checks for an in-band gm-threshold calibration (section 3: 100-150 checks pin it to +-0.03).

## 8. arm30: band 1 with the "human calibrates, machine applies" threshold (0.229)
Question: does the threshold that the simulated calibration picks on band 1 (§3, calib.py, N = all 3,020 labels -> 0.229;
p50 at N = 100-150 is 0.23) keep band 1's gain? It is the candidate method for a new scroll.

- **Chains** `hitl_b1_calib229`, written by data/scratch/hitl/write_arm30.py. That script runs the write_arms.py `recal`
  path, and at 0.263 it reproduces arm29's sha byte for byte.
  - sha256 9e065fd29a574cdeba361af03c1d287391de734efd5fb2b93556b4e09f0e0be9 (in arm_sets_sha256.json).
  - Diff vs hitl_b1_recal0: same 664 chains and 41,322 points. Only 876 steps change, each 1 -> 2: they all have
    dr0 = 1 and gm_p in [0.229, 0.263). Raises go 4,008 -> 4,884.
  - Offline drift vs CT `supported`: -0.15 -> +0.05.
- **Env** spiral/arms/arm30.env = arm29.env with only ARM and REL_WINDINGS changed.
- **Pod** vss-hitl-arm30: massedcompute 1x L40S, the same offer as arm29, at $0.82/h. The sha was verified on the pod.
  - 30k steps took 9,966 s.
  - On first boot, pod_setup died on the boot-time apt lock (unattended-upgrades). It was restarted once by hand.
- **Scoring** (all under the flock):
  - collect_and_score_hitl.sh on spiral/arms/pods_hitl30.tsv;
  - data/scratch/hitl/arm30_check/post_arm30.sh (vertex dump, swa, pairs);
  - perk30.py, a copy of arm29_check/perk.py.

**Band 1, B - arm0** (95 % CI; hitl_table.py)

| arm | thr | pair acc (= excl CT-flag) | wrong area seam | switches/1000 | own-off inner cons | own-off outer cons | own-off pooled |
|---|---|---|---|---|---|---|---|
| arm23 | 0.18 | +15.0 [+9.5, +21.3] | -11.9 [-14.7, -8.9] | +15.2 [+11.5, +19.5] | -19.1 [-22.1, -16.1] | +9.5 [+7.5, +11.6] | -9.6 |
| arm2 | oracle | +14.1 [+8.0, +21.2] | -13.5 [-16.0, -10.8] | +8.2 [+5.5, +11.4] | -18.6 [-21.3, -15.9] | -0.2 [-0.6, +0.2] | -12.2 |
| arm29 | 0.263 | +3.1 [-0.9, +8.0] | -0.6 [-2.7, +1.6] | +8.2 [+4.8, +12.1] | -0.8 [-2.2, +0.6] | +0.8 [+0.5, +1.2] | -7.4 |
| **arm30** | **0.229** | **+1.2 [-4.1, +7.0]** | **+6.4 [+4.0, +8.6]** | **+16.2 [+12.0, +20.9]** | **-1.4 [-2.6, -0.3]** | **+8.6 [+7.2, +10.0]** | **-1.1** |

Paired pair accuracy for arm30 against each arm:
- vs arm23: -13.7 [-18.6, -9.4];
- vs arm29: -1.9 [-4.7, +0.4];
- under `supported_plus_2167`: -9.0 vs arm23 and -1.1 vs arm29.

Own-offset pooled under outer_minus2: arm30 -5.1, compared with arm23 -12.2, arm29 -7.4 and arm2 -12.1.

**Per winding zone** (own offsets refitted per k group, as in §4 (3); original labels)

| k group | arm23 - arm0 | arm29 - arm0 | arm2 - arm0 | arm30 - arm0 | arm30 - arm23 | arm30 - arm29 |
|---|---|---|---|---|---|---|
| k 23-40 (inner) | -32.5 | -17.2 | -26.6 | **-15.4 [-18.0, -12.7]** | +17.1 [+15.0, +19.1] | +1.8 [+1.5, +2.2] |
| k 43-50 | +21.6 | +1.1 | +0.5 | **+21.4 [+18.7, +24.3]** | -0.3 [-0.6, +0.1] | +20.3 [+17.6, +23.1] |
| k 51-59 | +1.9 | +1.0 | -0.2 | +5.6 [+4.1, +7.0] | +3.7 [+2.2, +5.2] | +4.6 [+3.1, +6.1] |

Pairs by range, arm30 - arm0: k 24-39 +3.8 [-1.3, +10.3]; k 48-58 -12.0 [-24.2, +0.0].

- **No. The human-calibrated threshold loses band 1's gain.** arm30 gets the worst of both neighbours.
  - Inner turns (k 23-40): arm30 is at arm29's level (-15.4 vs -17.2), not arm23's (-32.5).
  - w043-w050: arm30 has arm23's over-raise penalty (+21.4 vs +21.6).
  - Pooled own-offset is -1.1, against -9.6 (arm23), -7.4 (arm29) and -12.2 (oracle), i.e. 9 % of the oracle gain.
- **The effect is not monotone in the threshold**, and the fit is very sensitive in w043-w050.
  - 0.229 adds only 876 raises to 0.263. Of those, 659 are on unlabelled stretches; 27 are labelled in k 43-50, of which
    14 are verifiable and 36 % truly 2.
  - Yet k 43-50 flips from +1 to +21 pp. That looks like a discrete fit outcome, such as a winding slip, not a
    proportional response.
- **Caveat: there is no replicate-fit estimate.** The CIs are vertex/site bootstraps on one fit each, so part of the
  non-monotonicity may be fit-to-fit variance.
- **Implication.** Zero-drift calibration of a single gm threshold, whether label-free (0.263) or human (0.229), does not
  reproduce the 0.18 inner gain in band 1. This holds even with all 3,020 labels; a 100-150-check session would only
  add noise (±0.03).
  - Inner gain needs a threshold near 0.18. w043-w050 needs one above ~0.229, or is unstable.
  - This is more evidence for the radius-dependent hypothesis in §4. It still needs a pre-registered test outside
    band 1.
- **Cost**: created 04:11:19Z, terminated 07:15:22Z by the collector right after the pull (logged in pods_arms.txt).
  That is 3.07 h at $0.82/h = **$2.52** (cap $8). `prime pods list` is empty afterwards.
