# 2026-10-24: screening chain-bias approaches for tight packing (f6), with pre-set gates

**Gates frozen: 2026-10-24T01:12:00Z (UTC).** This note was written as the first action of this work. At that time
no new screen, chain set, fit or metric existed. The reading behind it: notes 2026-10-21 annotation round 2,
2026-10-24 fourth test, 2026-10-24 contact detector, 2026-10-22 robust fitter, 2026-10-23 third region.
Sections 0-6 must not change after the timestamp. The log and results below them are append-only and timestamped.
Run directory: `data/scratch/f6_screen/`. Pods: `vss-f6-*`.

## 0. Reframe and what is already known
- Annotation round 2 (PHerc0139 b1) found three things:
  - Random step errors are tolerated: oracle chains with 10 % flips keep most of the oracle gain (−11.6 vs −13.5).
  - Biased errors are not: auto ray chains, which undercount, give +14.8 wrong area.
  - Linkage matters more than length: 4-point oracle pieces keep a good part of the gain.
- So the lever is bias, and how strongly the fitter trusts the chains, not per-step precision.
- **Prior evidence on chain weight (disclosed; the brief said it was never tested):** round 3 ran weight 2 on
  PHerc0139 b1 (arm14: m7 ray chains; arm15: m7x ray chains). It changed little against weight 10:
  - area +14.4 vs +14.8 (arm14 vs arm8);
  - area +4.7 vs +5.4 (arm15 vs arm13).
  - The strip radius term is L1-like (`relu(|normalised − mean| − 0.025 dr)`), so the fit either obeys a chain or
    ignores it. The crossover lies below weight 2, so weight 3 cannot be told apart from 10. **I therefore use
    weights 1 and 0.3, not the brief's example of 3 and 1** (deviation, justified here).
  - Paris4 has never been fit at any weight other than 10.

## 1. Data (all dev, all burned)
- **Step screens.** Inputs:
  - the cached no-raise ray chains and label matching of `data/scratch/fourth_test/dev_cache/*.pkl`
    (`dev_steps.build`), with primary label variant and kmax per region as there;
  - the frozen grad_mag logistic `gm_p` (`dev_steps.gm_p`, tools/auto_windings_gm_z4500_clean.json).
- **Metrics,** all as `dev_steps.evaluate`:
  - step precision;
  - drift d10 (chain minus label over 10-20 true windings);
  - within ±1;
  - raise precision, defined as in the contact-detector note L2: the share of raised labelled single steps whose
    true difference is exactly 2 (">= 2" reported as secondary);
  - raised fraction and recall.
- **Regions.** Tight: PHerc0139 b1 (`supported`), PHerc0139 b2 (`minus1`) and PHercParis4 z 9000-10500 (k < 109).
  Wide control: PHerc1667 (k < 34).
- **Baselines** (fourth-test dev table):

  | region | no raise: prec / d10 | gm 0.263: prec / d10 |
  |---|---|---|
  | b1 | 88.8 / −0.95 | 87.7 / −0.16 |
  | b2 | 88.6 / −1.54 | 86.2 / −0.51 |
  | Paris4 | 78.4 / −1.87 | 76.2 / −0.12 |

- **Base-fit lookups (chain-free judge).** The fit winding at a chain point is the winding of the nearest
  exported surface of the fit, within tol 4 vox (`spiral_eval.Surfaces.query`, as fit_vertex_dump). Otherwise it is
  NaN. Fits used:
  - b1: arm0;
  - b2: arm20;
  - Paris4: p4_base (seed 1);
  - PHerc1667: c1667_base_fig (report only).
- **The sign of the fit winding** along a ray is fixed label-free: the sign of the median fit difference over all
  chain steps of the region.

## 2. Candidates (all label-free; nothing below is fitted on a held-out region)
Every recipe is branched on the existing S_pack rule (`tools/autowind/packing.py`, threshold 21.0):
- **S_pack < 21:** apply the candidate.
- **Otherwise:** leave the chains and the weight exactly as the confirmed M2 recipe (no raise, weight 10).
- PHerc1667 (S_pack 24.0) therefore reproduces M2 chains byte for byte at weight 10. By the brief, its dev fit can
  then be skipped. PHerc1667 step numbers without the branch are reported for information only.

**A1, lower chain weight (dev fits only).**
- No-raise chains `paris4_m7_ray_c` (relative_windings sha 485029131145fcfe…, the p4_ms set).
- `loss_weight_unattached_pcl_radius` = **1.0** (arm `f6_w1`) and **0.3** (arm `f6_w03`). Spacing 0, seed 1, all
  else as p4_ms1.
- Recipe: weight w iff S_pack < 21, else 10.

**A2, per-ray (per-chain) deficit, with placement at the top gm_p steps.** For each chain, estimate a label-free
deficit D ≥ 0. Then raise exactly D single steps of that chain from 1 to 2: the D with the highest gm_p (ties by
order, capped at the number of single steps). Chains are the fitter's units (one ray segment each). Two estimators
are named now. Each is gated on its own (multiplicity disclosed: 2 estimators):
- **A2g, gm budget:** D = max(0, round(Σ over d0 = 1 steps of (Ir_j − 1))).
  - Ir = I / Iloc is the frozen gm feature: the step's grad_mag integral over the median of its ±5 neighbouring
    single steps.
  - A true single step gives Ir ≈ 1, a merged one ≈ 2.
- **A2z, z-neighbour consensus:**
  - Neighbour rays are the same θ at z ± 64 and z ± 128. From each, take the chain with the largest radial overlap
    with chain A.
  - The overlap is [lo, hi] = [max(r_A,0, r_B,0), min(r_A,end, r_B,end)]. A neighbour qualifies if
    hi − lo ≥ 0.5 × (r_A,end − r_A,0).
  - c_X = wind of X at its last point with r ≤ hi, minus wind at its first point with r ≥ lo.
  - D = max(0, round(median over qualifying neighbours of (c_B − c_A))). It needs ≥ 2 qualifying neighbours;
    otherwise D = 0.
- **Diagnostic, not eligible for the gate:** oracle D = the chain's true labelled deficit, with the same top-gm_p
  placement. It is a ceiling: it shows whether a perfect per-chain deficit with gm placement could pass at all.

**A3, two weak judges (gm_p × chain-free base fit).**
- bd_j = fit winding at point j+1 minus that at point j, from the region's no-chain base fit, sign-fixed. It is NaN
  if either point has no surface.
- Candidates, each gated on its own (2 candidates):
  - **A3and:** raise a d0 = 1 step iff gm_p ≥ 0.263 and bd ≥ 2. It has no free parameter, so every region counts
    as held out.
  - **A3lr:** logistic (L2, standardised) on [logit gm_p, clip(bd, 0, 3), isnan(bd)]. It is fitted
    leave-one-region-out on the two other tight regions, weighted equally, with target true difference == 2. The
    threshold is where the mean training drift10 is zero (the contact-detector L1 rule). It is reported on the
    held-out region.
- The independence check (correlation of gm_p and bd on single steps; joint 2×2 table) is reported.

**A4, cut instead of raise (dev fit).**
- Start from the no-raise chains. Hard-cut every d0 = 1 step with gm_p ≥ **0.263** (the existing zero-drift
  threshold; not tuned here). There is no shared point: the step simply disappears.
- Drop pieces with < 2 points and re-zero wind per piece.
- Weight 10, spacing 0, seed 1. Arm `f6_cut`.
- Step-level diagnostics are reported, not gated: share cut, kept-step precision, within-piece d10, share of
  true-2 steps cut.
- Implemented as a new module `tools/autowind/postproc.py`. It reads an `autowind chains --raise off` output, plus
  grad_mag, and writes a new set. Existing code paths are unchanged.

**A5 (own idea), base-fit-anchored deficit.**
- Per chain: D = max(0, round((fit winding at its outermost point with a surface − fit winding at its innermost such
  point) − (wind difference between the same two points))). Then the top-D gm_p placement, as A2.
- **Why:** the brief's per-ray deficit needs an independent global count. The strongest label-free consensus
  available is the chain-free base fit. It pools ~1 M tracks across all rays and angles. It exists in every
  protocol anyway (the base arms), and it is free to query.
- **Risk:** at Paris4 the base fit's absolute numbering is poor (69.5 % own-offset wrong area). Only its relative
  span along a ray is used here. In a confirmation, the method would be derived from base seed 1. That would be
  disclosed, and the method would still be compared against all 3 base seeds.

## 3. GATES (fixed now)
**Step screens.**
- **A3 (A3and, A3lr):** the contact-detector gate. On EACH tight region, held out (A3lr: leave-one-region-out):
  (1) raise precision ≥ 70 %; (2) |d10| ≤ 0.5; (3) step precision ≥ no-raise − 1 pp.
- **A2 (A2g, A2z) and A5: a drift-centred gate.** On EACH of the three tight regions: |d10| ≤ 0.3 and step
  precision ≥ no-raise − 1 pp. Raise precision is reported but not gated.
  - **Justification (written before any A2 or A5 number):** the reframe says bias, not per-step precision, decides
    the fit.
  - The gate is not weaker than what already failed in a fit. gm 0.263 (d4_r263, which tied base) fails it on step
    precision in all three regions (−1.1 / −2.4 / −2.2 pp).
  - The parameters (top gm_p placement from the frozen logistic, label-free D) are fixed, so all three regions act
    as held out.
- **A candidate passes its step screen iff every item holds on every tight region.** PHerc1667 is not gated, and the
  S_pack branch makes the recipe a no-op there.

**Dev fits (A1, A4, and every step-screen candidate that passes).**
- Paris4 z 9000-10500, seed 1, scored with `data/scratch/fourth_test/dev_table.py` against the stored dumps:
  primary own-offset seam wrong area on Paris4 PRIMARY_K, paired BootV 95 % CI; seam switches per 1000.
- **GO iff, against EACH of p4_base, p4_bases2 and p4_bases3: upper CI < 0 and switch difference ≤ +5.**
- **PHerc1667 condition.** Every recipe here is S_pack-branched, so at PHerc1667 it reproduces M2's chains byte for
  byte at weight 10. The brief's skip clause applies, and this is verified by sha before any confirmation.
  - If a recipe does not reproduce M2 exactly, it needs a PHerc1667 dev fit: CI upper < 0 against all 3 c1667 bases,
    keeping ≥ half of M2's −28 gain, i.e. an estimate ≤ −14.
- **Cap:** at most 6 dev fits. Planned: f6_w1, f6_w03, f6_cut, plus at most 3 for passing screen candidates. If more
  pass, take them in order of smallest worst-region |d10|.
- **If more than one recipe goes,** pick the one with the lowest worst-case upper CI over the 3 base seeds, then the
  fewest switches. Freeze that choice and its reason before any confirmation step.

## 4. Confirmation (only if a recipe goes)
- **Pre-registration.** A new pre-registration note (stage 1 freeze) for PHercParis4 z 7000-8500. It uses the stage 0
  of notes/2026-10-24-fourth-test-prereg.md, frozen 2026-10-23T18:26Z, with its primary set, labels, scorer, oracle
  and controls, and `eval/method_table_p4z7.py`.
- **Chains.** They are built blind from the resumed label-free crop solve. Their sha is frozen (stage 2) before any
  method pod.
- **Arms:** 2 method seeds and 3 base seeds, plus the oracle.
- **PASS** iff all 6 combos have upper CI < 0 and switches ≤ +5 per 1000, and the oracle is informative.
- **Blinding.** No label-based chain metric on z 7000 until all fits are scored.
- **Timing.** Results by Oct 28.

## 5. Stop rule
If no recipe goes by 2026-10-26 12:00Z, stop and write up. z 7000-8500 then stays untouched: no chain, CT, label
metric, feature or fit.

## 6. Budget and rules
- **Budget:** $70 for dev fits and the confirmation combined; check in before exceeding it.
- **Pods:** vss-f6-*, terminated right after each pull, logged in data/scratch/p0139_spiral/pods_arms.txt.
- **Eval lock.** Jobs over ~6 GB run under `flock /tmp/vss_eval.lock`, unless more than 20 GB is available; any
  exception is logged.
- **Frozen files:** tools/auto_windings.py and eval/p0139_graph.py are never modified. Every published set must
  still reproduce (release/AUTOWIND.md table).

---

## Log (append-only)
### L1 (2026-10-24T01:30Z): timestamp, dev fits launched, one implementation detail
- **Timestamp.** The note file was last written at 01:10:58Z (mtime). Its header says 01:12:00Z, written a minute ahead.
  The sha of sections 0-6 (`data/scratch/f6_screen/gate_freeze.sha256`, 6ad6f72b…) and the snapshot
  `gate_note_snapshot.md` were taken at 01:11:04Z, before any screen or fit.
- **Dev-fit chain sets.**
  - A1 (f6_w1 weight 1.0, f6_w03 weight 0.3) uses the p4_ms set, `paris4_m7_ray_c` (relative_windings sha
    485029131145fcfe…; identical to `fourth_test/dev/p4z9000_auto/noraise`).
  - A4 (f6_cut) uses `data/scratch/f6_screen/windings/p4z9000_cut263`: `python -m autowind.postproc --run
    data/scratch/fourth_test/dev/p4z9000_auto --mode cut`, relative_windings sha 0954c72a8ed7df88….
    - It cuts 10,904 of 44,405 single steps (24.6 %), leaving 6,209 pieces and 43,484 points (median 5 points);
      5,454 singleton points are dropped.
    - That is one step more than the gm raise's 10,903. The stored gm_p is rounded to 4 decimals, so a step with
      P = 0.26297 reads 0.2630.
- **PHerc1667 branch check.** `autowind.postproc --run data/scratch/fourth_test/repro/p1667_auto --mode cut` reads
  decision 'off' and copies the set unchanged: sha 85dcbfb46a66b192… = M2 (`band_m7_p1667_ray_c`), byte for byte.
  The A1 weight branch keeps weight 10 at S_pack 24.0. So the PHerc1667 dev fit is skipped for all three dev recipes,
  as section 3 allows.
- **Pods.** f6_w1 runs on vss-f6-f6_w1 (A6000, datacrunch, $0.79/h, created 01:19:26Z). f6_w03 and f6_cut wait in
  the grabber.

### R1 (2026-10-24T01:30Z): step screens (A2, A3, A5): ALL FAIL their gates
`data/scratch/f6_screen/screen.py` (sha ce18d812…) -> `screen.json` (sha a2f9dbf3…); base-fit lookups `fitlookup.py`
(sha 1923fde1…) -> `fitj_<region>.npz`. Base-fit surface coverage of chain points: b1 79.4, b2 75.4, Paris4 74.4,
PHerc1667 76.1 %; fit numbering sign +1 everywhere (label-free).

| region | variant | step prec | d10 | within ±1 | raise prec (==2) [>=2] | raised | recall | gate |
|---|---|---|---|---|---|---|---|---|
| b1 | no raise | 88.81 | −0.95 | 70.2 | – | – | – | |
| | gm 0.263 | 87.70 | −0.16 | 77.4 | 34.3 [45.8] | 10.4 % | 26.6 | |
| | **A2g** | 87.73 | **−0.25** | 74.4 | 33.1 [44.2] | 8.0 % | 21.7 | fail (prec −1.08 pp) |
| | **A2z** | 88.82 | −0.88 | 71.6 | 40.0 [62.0] | 1.4 % | 2.8 | fail (d10) |
| | oracle D (diag.) | 88.94 | −0.62 | 78.2 | 42.9 [61.9] | 3.9 % | 14.7 | (fail) |
| | **A5** | 78.95 | +1.57 | 57.0 | 21.2 [26.0] | 24.4 % | 55.2 | fail |
| | **A3and** | 89.19 | −0.63 | 77.0 | 48.3 [66.9] | 3.3 % | 15.9 | fail (1, 2) |
| | **A3lr** (LORO, thr 0.178) | 86.87 | +0.10 | 80.0 | 33.6 [42.2] | 13.5 % | 38.2 | fail (1, 3) |
| b2 | no raise | 88.62 | −1.54 | 58.1 | – | – | – | |
| | gm 0.263 | 86.23 | −0.51 | 67.3 | 33.1 [37.7] | 14.9 % | 36.3 | |
| | **A2g** | 86.09 | −0.71 | 62.9 | 28.5 [32.4] | 12.1 % | 25.2 | fail |
| | **A2z** | 88.62 | −1.44 | 59.7 | 44.4 [55.6] | 1.9 % | 5.0 | fail |
| | oracle D (diag.) | 87.79 | −1.11 | 65.5 | 35.2 [42.8] | 5.5 % | 17.4 | (fail) |
| | **A5** | 79.03 | +0.51 | 60.3 | 21.6 [24.3] | 26.0 % | 50.6 | fail |
| | **A3and** | 88.41 | −1.16 | 64.0 | 40.2 [49.5] | 4.2 % | 12.1 | fail |
| | **A3lr** (thr 0.195) | 85.52 | −0.35 | 71.7 | 31.0 [35.5] | 16.4 % | 38.5 | fail (1, 3) |
| Paris4 | no raise | 78.40 | −1.87 | 45.5 | – | – | – | |
| | gm 0.263 | 76.17 | −0.12 | 53.8 | 35.6 [50.2] | 24.6 % | 38.3 | |
| | **A2g** | 75.88 | −0.56 | 51.1 | 31.5 [47.2] | 19.3 % | 25.7 | fail |
| | **A2z** | 78.48 | −1.74 | 47.0 | 30.6 [75.5] | 2.4 % | 2.8 | fail |
| | oracle D (diag.) | 76.06 | −0.09 | 58.6 | 37.1 [50.3] | 23.4 % | 47.3 | (fail: prec −2.3 pp) |
| | **A5** | 74.88 | +0.14 | 56.3 | 35.3 [47.5] | 25.9 % | 49.7 | fail (prec) |
| | **A3and** | 79.61 | −1.29 | 53.7 | 46.3 [73.6] | 6.8 % | 19.4 | fail (1, 2) |
| | **A3lr** (thr 0.151) | 76.30 | +0.41 | 56.9 | 38.3 [51.9] | 29.9 % | 56.3 | fail (1, 3) |

Gate verdicts (section 3): A2g, A2z, A5 (drift-centred gate) and A3and, A3lr (contact-detector gate) **fail on
every tight region** (A2g is closest: b1 passes drift but misses precision by 0.08 pp; b2 and Paris4 fail both).
No step-screen candidate earns a dev fit.

Reading:
- **A2 is dead whatever the estimator.** Even the oracle deficit (the true per-chain shortfall, label-derived) with
  top-gm placement fails: at Paris4 it zeroes drift (−0.09) but loses 2.3 pp step precision, just like gm 0.263;
  at b1/b2 it leaves −0.6/−1.1 drift (per-chain net shortfalls hide over- and under-counts). The bottleneck is
  placement (gm ranks the true 2s poorly), not the count.
- **A2z sees almost nothing:** z ± 64/128 neighbours mostly share the merge (D ≈ 0.8-1.4 per chain vs oracle 2.2-13.7),
  consistent with the contact detector's neighbour-ray AUC 0.50-0.65. Merged contacts are coherent over >= 128 z.
- **A5 over-raises:** the base fit's span along a ray exceeds the true span by ~10 windings at b1/b2 (D 13-14 per chain
  vs oracle 2.2-2.6) and grossly at PHerc1667 (D 15 vs 0.08). The chain-free fit is not a usable count reference.
- **A3: the judges are not independent** (corr gm_p vs bd 0.31-0.34 on single steps). Where both fire (gm_p >= 0.263
  and bd >= 2), the true-2 share is 48 / 40 / 46 % (b1 / b2 / Paris4), the best cell, still far from 70 %.
- PHerc1667 (not gated; the S_pack branch makes all recipes a no-op): every raise variant hurts it (d10 +0.95 ->
  +1.3 … +7.2), as before.

**A4 cut, step diagnostics (reported, not gated):** cutting at gm_p >= 0.263 removes 26.6 / 36.3 / 38.3 % of true-2
steps while cutting 10.4 / 14.9 / 24.6 % of single steps. Kept-step precision rises 88.8 -> 90.9 (b1), 88.6 -> 91.1
(b2), 78.4 -> 83.7 (Paris4); within-piece d10 halves (−0.95 -> −0.45, −1.54 -> −0.98, −1.87 -> −0.69) and within ±1
rises 45.5 -> 73.2 at Paris4, with no false +1 introduced. This is the profile the reframe asks for (less bias, no
new error type), at the cost of linkage. Its dev fit (f6_cut) decides.
### L2 (2026-10-24T02:30Z): dev-fit pods
- The grabber found no L40S, A6000 or RTX6000Ada offer at <= $1.45/h for 30 min. Only a datacrunch L40S at $1.81 was
  offered. At 01:51Z the grabber was restarted with a $1.85 cap.
- Pods:
  - f6_w1: vss-f6-f6_w1, A6000, datacrunch, $0.79/h, created 01:19:26Z; fit start 02:05:48Z, overrides show
    `loss_weight_unattached_pcl_radius: 1.0`;
  - f6_cut: vss-f6-f6_cut, L40S, datacrunch, $1.81/h, created 01:52:20Z; fit start 02:24:16Z, fit log shows
    "pcls: 0 cross-patch, 6209 unattached" (= the cut set's 6,209 pieces);
  - f6_w03: vss-f6-f6_w03, A6000, massedcompute, $0.54/h, created 02:05:45Z.
- **Note on the cut arm.** The fitter samples 13 strips per step (32 points each), whatever their length. 6,209 short
  pieces are therefore each visited ~8× less often than the 759 long chains. The strip loss is a per-strip mean, so
  the total pull per step is similar, as in round 2's arm10 (4-point pieces). Recorded, not changed.
### R2 (2026-10-24T06:05Z): dev fit f6_cut: GO
- Pod vss-f6-f6_cut: fit finished ~05:44Z (30,000 iterations), pulled 05:50:14Z (240 windings), terminated 05:50:16Z
  (logged in pods_arms.txt). Vertex dump 663.7 s, under the eval lock. Cost ≈ 3.97 h × $1.81 ≈ $7.2.
- Scorer: `data/scratch/f6_screen/dev_table_f6.py f6_cut --out data/scratch/f6_screen/dev_table_cut.json` (run
  without the lock: 52 GB available, > 20 GB; exception logged in data/scratch/f6_screen/pods/collect_f6.exceptions.log).

| comparison | primary Δ (own-offset seam, PRIMARY_K) [95 % BootV] | arm / ref | switch Δ /1000 | global Δ |
|---|---|---|---|---|
| f6_cut − p4_base | −5.41 [−7.05, −3.94] | 64.1 / 69.5 | +3.8 | −3.93 [−5.42, −2.57] |
| f6_cut − p4_bases2 | −5.54 [−7.17, −4.06] | 64.1 / 69.6 | +2.0 | −3.99 [−5.48, −2.65] |
| f6_cut − p4_bases3 | −5.47 [−7.07, −4.06] | 64.1 / 69.5 | +3.1 | −4.04 [−5.53, −2.74] |
| f6_cut − p4_ms1 (no-raise, weight 10) | −15.95 [−17.68, −14.22] | 64.1 / 80.0 | −13.6 | −6.32 |
| f6_cut − p4_ms2 | −16.05 [−17.83, −14.29] | 64.1 / 80.1 | −14.6 | −6.79 |
| f6_cut − p4_oracle | +19.27 [+16.79, +21.88] | 64.1 / 44.8 | +21.7 | +21.74 |

- **Verdict (section 3): GO.** Upper CI < 0 against all three base seeds (worst −3.94) and switch Δ ≤ +5 (worst +3.8).
  PHerc1667: the S_pack branch reproduces M2 byte for byte at weight 10 (L1), so the PHerc1667 condition holds by the
  exception clause; no PHerc1667 fit.
- Under section 3 the choice of recipe waits for f6_w1 and f6_w03 (selection rule: lowest worst-case upper CI, then
  fewest switches). f6_cut's worst-case upper CI is −3.94, switches +3.8.
- **Confirmation infrastructure started now (recipe-independent, label-free):** a recipe has gone, so the z 7000-8500
  crop solve resumes (data/scratch/f6_screen/prep/q5_resume2.sh, same command and flags as the fourth test's
  q5_resume.sh, `--raise off`; 2,726 crops left; started 06:03:24Z). It is label-free; at the end it also writes the
  run's no-raise chain file, which is not used or inspected (the method chains are built after the stage 1 freeze
  with `--raise auto --no-solve` on these crops). No labels or label-based metrics on the band are touched.
### R3 (2026-10-24T07:31Z): dev fit f6_w1 (no-raise chains, weight 1.0): NO-GO
- Pod vss-f6-f6_w1 pulled 07:06:18Z (240 windings), terminated 07:06:23Z (logged). Dump 689.4 s (waited for the lock
  behind the crop solve). Cost ≈ 5.78 h × $0.79 ≈ $4.6. Scorer as R2 (`--out data/scratch/f6_screen/dev_table_w1.json`,
  no lock, 52 GB available, logged).

| comparison | primary Δ [95 % BootV] | arm / ref | switch Δ /1000 | global Δ |
|---|---|---|---|---|
| f6_w1 − p4_base | +1.56 [−1.43, +4.45] | 71.1 / 69.5 | +11.9 | −4.09 [−6.76, −1.61] |
| f6_w1 − p4_bases2 | +1.43 [−1.57, +4.35] | 71.1 / 69.6 | +10.1 | −4.15 [−6.80, −1.64] |
| f6_w1 − p4_bases3 | +1.50 [−1.50, +4.38] | 71.1 / 69.5 | +11.2 | −4.20 [−6.82, −1.69] |
| f6_w1 − p4_ms1 (weight 10) | −8.98 [−10.75, −7.41] | 71.1 / 80.0 | −5.5 | −6.48 |
| f6_w1 − p4_ms2 | −9.07 [−10.76, −7.49] | 71.1 / 80.1 | −6.5 | −6.95 |
| f6_w1 − p4_oracle | +26.24 [+24.75, +27.70] | 71.1 / 44.8 | +29.8 | +21.58 |

- **Verdict: NO-GO** (upper CI > 0 against every base seed; switches +10 to +12, over the +5 margin). Lowering the weight
  from 10 to 1 recovers 9 of the 10.5 points the no-raise chains cost, but does not get below the base fit, and it
  adds switches. Cutting (f6_cut) beats weight 1 by 7 points with fewer switches.
- f6_w03 is still running. Under section 3 the recipe choice waits for it.
### R4 (2026-10-24T08:40Z): dev fit f6_w03 (no-raise chains, weight 0.3): NO-GO. Final go/no-go and selection
- Pod vss-f6-f6_w03 pulled 08:28:03Z, terminated 08:28:06Z (logged). Dump 672.2 s. Cost ≈ 6.37 h × $0.54 ≈ $3.4.
  Combined table: `data/scratch/f6_screen/dev_table.json` (all three arms, no lock, 52 GB available, logged).

| comparison | primary Δ [95 % BootV] | arm / ref | switch Δ /1000 | global Δ |
|---|---|---|---|---|
| f6_w03 − p4_base | −2.24 [−4.47, −0.20] | 67.2 / 69.5 | +17.7 | −3.34 [−5.11, −1.66] |
| f6_w03 − p4_bases2 | −2.37 [−4.57, −0.33] | 67.2 / 69.6 | +15.9 | −3.40 [−5.18, −1.70] |
| f6_w03 − p4_bases3 | −2.30 [−4.46, −0.34] | 67.2 / 69.5 | +17.0 | −3.45 [−5.17, −1.87] |
| f6_w03 − p4_ms1 (weight 10) | −12.78 [−14.04, −11.54] | 67.2 / 80.0 | +0.3 | −5.73 |
| f6_w03 − p4_oracle | +22.43 [+20.76, +24.02] | 67.2 / 44.8 | +35.6 | +22.32 |

- **Verdict: NO-GO.** The primary CI clears 0 (upper −0.20 … −0.34), but switches are +16 to +18 per 1000, over the +5
  margin. A weaker pull leaves the fit free to jump between neighbouring sheets.

**Go/no-go per approach (final):**

| approach | stage reached | result |
|---|---|---|
| A1 lower weight, w = 1.0 (f6_w1) | dev fit | NO-GO (primary +1.4 … +1.6, CI spans 0; switches +10 … +12) |
| A1 lower weight, w = 0.3 (f6_w03) | dev fit | NO-GO (primary −2.2 … −2.4, CI < 0; switches +16 … +18) |
| A2 per-ray deficit, raises at top-gm steps (A2g, A2z) | step screen | FAIL (R1); no dev fit |
| A3 two weak judges, gm_p × base-fit difference (A3and, A3lr) | step screen | FAIL (raise precision 31-48 % vs 70 %); no dev fit |
| A4 cut instead of raise (f6_cut) | dev fit | **GO** (primary −5.4 … −5.5, worst upper CI −3.94; switches +2.0 … +3.8) |
| A5 base-fit-anchored deficit (own idea) | step screen | FAIL (over-raises); no dev fit |

- **Exactly one recipe went: A4 cut.** Under the confirmation pre-registration's section 0 clause the method is A4.
  Dev fits used: 3 of the 6 allowed. Confirmation: notes/2026-10-24-f6-confirm-prereg.md.
### R5 (2026-10-24T17:20Z): confirmation outcome
- A4 cut on PHercParis4 z 7000-8500 (notes/2026-10-24-f6-confirm-prereg.md, Results): **FAIL on the switch margin.**
  Primary Δ −7.85 … −8.48 (all CIs < 0, p < 0.001), but switch Δ +6.5 … +8.2 > +5 in all 6 combos. Oracle informative.
- Spend: dev $15.19 + confirmation $26.94 = $42.13 of $70. No pods left.
