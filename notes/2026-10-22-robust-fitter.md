# 2026-10-22: robust ("self-cleaning") chain annotations in the spiral fitter

Goal: make the villa spiral fitter tolerate wrong relative-winding steps in automatic chains, so band 2 (z 1750-3000)
stops getting worse while band 1 (z 3000-4500) keeps most of arm23's gain. Bands 1 and 2 are both dev data now. No
other region was touched (z 6000-8500 stays reserved).

## 1. How strips enter the loss (pinned villa 0fb38c45, volume-cartographer/scripts/spiral)
- Every chain in relative_windings.json becomes an "unattached strip" (input_disable_patches): id-sorted points
  `zyxs` plus `windings` (= `wind_a`) (fit_spiral.py strip loop, ~l.1422).
- Each step the loss draws `sample_count_unattached_pcls_per_step` strips (84 x 0.158 = 13 for a 1500-z band) and 32
  points from each (`get_unattached_pcl_strip_losses`, losses.py:1467).
  - The points go through the scroll->spiral transform. The shifted radius is unwrapped along the strip, and the
    annotation is subtracted: `normalised = shifted - wind_a * dr`.
  - **Radius term** (weight `loss_weight_unattached_pcl_radius`, 10 in our arms): `relu(|normalised - row mean| -
    0.025 dr)`, averaged. This is an L1-like pull of every point to the row mean, from step 0.
  - **DT term** (`loss_weight_unattached_pcl_dt`, 4): it pulls each point to the snapped sheet `round(row median) +
    wind_a * dr`. It is on only after `loss_start_patch_dt` = 25000.
- So `wind_a` enters only through cumulative offsets. A wrong step k shifts the target of every later point in the
  chain by one turn.
  - A per-point weight cannot remove a wrong step from a long ray chain. Down-weighting the points around it leaves the
    offset between the two sides in place.
  - Only cutting the chain at the step (or relabelling the step) removes it.

## 2. Offline check before any spend: which judge can tell good steps from bad? ($0)
Script: data/scratch/robust_fitter/judge_offline.py, judge_bins.py, cut_rules.py. They use the band2_diag lookups
(the fit winding at every chain point) and the verified steps (labelled crossings on both ends).

| judge | band | disagrees with chain | chain right when judge disagrees | judge right when it disagrees | chain right when they agree |
|---|---|---|---|---|---|
| chain fit B (arm23 / arm21) = "classic IRLS" | 1 | 8.5 % | 82.4 % | 10.5 % | 83.9 % |
| | 2 | 8.7 % | 74.6 % | 16.3 % | 79.9 % |
| chain-free fit A (arm0 / arm20) | 1 | 27.2 % | 62.5 % | 30.6 % | 90.6 % |
| | 2 | 32.1 % | 55.7 % | 39.4 % | 89.3 % |

- **The chain fit cannot judge its own chains.**
  - It copies them, and in the few places where it departs, the chain is right 75-82 % of the time.
  - Re-weighting or relabelling by the residual of the fit with chains (textbook IRLS / trimmed loss) would mostly
    remove correct steps.
  - Windowed B-disagreement does not separate either: chain precision is 84.5 vs 81.8 % across its bins.
- **A chain-free fit is a weak but real judge.**
  - Where it agrees with the chain, chain precision is about 90 %; where it disagrees, 56-62 %.
  - Its windowed disagreement (+-5 steps) ranks chain precision from 90-93 % down to 55-70 %. The top bins hold the
    raised steps: 43 % raised in the top band-2 bin.
  - Per bin, A beats the chain only in band 2's top bins (d >= 0.4: A 63-69 % vs chain 55-66 %). In band 1, the chain
    beats A everywhere except the bottom bin.
  - So the best a chain-free judge can deliver is a trade-off, not a free gain. This is the same frontier as the step
    filters (dsafe / p93) and the cell gate in the band-2 diagnosis.

## 3. Patch: spiral/patches/robust_unattached.patch
sha256 049ebefe844ba6174db605c3360019c49d44ba3281543d98e5a755a82e8ce1a7. It applies on the pinned villa after
pherc343's fit_spiral_shell_optional.patch (as on the pods). run_arm.sh applies it when an arm sets
`VILLA_PATCH=robust_unattached.patch`. It touches config.py, losses.py, point_collection.py and fit_spiral.py (+211/-6).
Every new option is off by default.

| key | default | meaning |
|---|---|---|
| unattached_pcl_point_weights | false | read optional per-point `"w"` from the JSON; weighted row mean and weighted radius/DT means |
| unattached_pcl_robust_mode | "off" | "cut": cut strips at steps the fit distrusts |
| unattached_pcl_robust_start_step | 10000 | first decision |
| unattached_pcl_robust_update_interval | 0 | 0 = decide once, then freeze (decision saved as unattached_robust_cuts.json next to the checkpoints and reused on resume); N = re-decide every N steps from the original strips |
| unattached_pcl_robust_warmup_scale | 1.0 | strip-loss multiplier before start_step; 0 = chain-free warm-up (the first decision is made by a fit that never saw the chains) |
| unattached_pcl_robust_step_threshold | 0.5 | a step disagrees if \|fitted - annotated winding difference\| >= this (windings) |
| unattached_pcl_robust_window | 0 | 0 = cut each disagreeing step; W > 0 = cut steps whose disagreeing fraction over +-W steps >= window_threshold |
| unattached_pcl_robust_window_threshold | 0.4 | see above |
| unattached_pcl_robust_min_piece_points | 2 | pieces shorter than this are dropped |

- Mechanics:
  - `compute_unattached_robust_cuts` evaluates all original strip points under the current transform. It reuses
    satisfaction_metrics' batched unwrap.
  - `split_unattached_strips` builds the pieces. `_maybe_update_unattached_robust` swaps them into the strip pool in
    place (`.flat` reset, sampling strata and components rebuilt, DT cache reset), the same way the interactive append
    path does.
  - Strips with fiber-link junctions are never cut.
- Checks (CPU, local; data/scratch/robust_fitter/tests/):
  - loss_equiv.py runs the strip loss on synthetic ray chains with an injected wrong step. Original vs patched at
    defaults: radius loss, DT loss and both gradients are **bitwise identical** over 3 sampled steps.
  - With `w` = 1 the weighted path is identical as well; random `w` changes it, as it should.
  - cut_test.py: with an exact fit, every injected wrong step is found (30/30 strips) and every kept piece is
    consistent.
  - method_test.py drives the real `_rebuild_*` methods on a stub context: pool 25 -> 42 pieces, the loss runs on the
    pieces, and a resumed context reuses the saved decision.
  - No local fitter environment or band data exists, so a full local fit was not feasible. On the pods, the S2 arms
    run the patched code with the mode on but inactive until step 10000, and their logs can be compared with arm23's /
    arm21's.

## 4. Arms (1 pod each, vss-rob-*)
Settings, fixed before any pod:
- **S1, chain-free judge**:
  - warmup_scale 0 up to step 12000, i.e. 2000 steps after track DT starts at 10000;
  - then cut every chain step the warm-up fit disagrees with by >= 0.5 turn;
  - drop pieces < 3 points; frozen (interval 0).
- **S2, classic IRLS**:
  - chains on from step 0;
  - from step 10000, every 2000 steps, re-cut the original chains at steps the current fit disagrees with by
    >= 0.5 turn;
  - min piece 2.
  - Offline (section 2) predicts that S2 is neutral to harmful. It is run because it is the textbook method, and
    because on oracle + random flips it is the one case where fit consensus should work.

| arm | band | chains | setting | compare | base |
|---|---|---|---|---|---|
| rob1 | 1 | band_m7_ray_gmxc | S1 | arm23 | arm0 |
| rob2 | 2 | band2_m7_ray_gmxc | S1 | arm21 | arm20 |
| rob3 | 2 | band2_m7_ray_gmxc | S2 | arm21 | arm20 |
| rob4 | 1 | band_corrupt10 (oracle, 10 % flips) | S2 | arm6 | arm0 |
| rob5 | 1 | band_m7_ray_gmxc | S2 | arm23 | arm0 |

Pod checks:
- The windings sha was verified on every pod before the fit, and run_arm.sh stamped "applied villa patch
  robust_unattached.patch (049ebefe…)".
- The patched code with the mode inactive (rob5 before step 10000) reproduces arm23: step 0 matches to the printed
  digit (loss 11409.3, strip radius 833.2, track radius 10541.4). Steps 1000 and 2000 are within GPU noise (427.9 vs
  428.6; 368.0 vs 367.6).
- No crashes, NaNs or OOMs. Track and strip losses after the cut stayed sane: rob2's strip loss went 26 -> 14 -> 15
  over steps 12200-12600.

## 5. What the cuts actually removed
Each arm's decisions are saved in `run/unattached_robust_cuts.json`. They are scored against verified chain steps
(band2_diag step table) by data/scratch/robust_fitter/score_cuts_verified.py, and against the known flips for rob4 by
score_cuts_corrupt.py.

| arm | decision | steps cut | wrong among cut verifiable | recall of wrong steps | good steps lost | chain precision kept / all | raised steps cut |
|---|---|---|---|---|---|---|---|
| rob1 (S1, b1) | step 12000 | 33.3 % | 29.4 % | 48.7 % | 21.9 % | 89.0 / 84.2 % | 70.0 % |
| rob2 (S1, b2) | step 12000 | 37.5 % | 44.5 % | 55.4 % | 17.3 % | 88.1 / 80.0 % | 73.7 % |
| rob5 (S2, b1) | last, step 28000 | 6.1 % | 19.8 % | 6.3 % | 4.8 % | 84.4 / 84.2 % | 10.6 % |
| rob3 (S2, b2) | last, step 28000 | 4.9 % | 37.7 % | 7.4 % | 3.1 % | 80.7 / 80.0 % | 8.8 % |
| rob4 (S2, oracle + 10 % flips) | last, step 28000 | 796 of 16588 | 18.0 % (143 true flips) | 8.3 % of 1732 flips | 653 good steps | — | — |

- **Classic IRLS (S2)** cuts close to the base rate of wrong steps. Even on random flips of perfect chains, the fit
  absorbs about 92 % of the flips before the first decision. This is the offline "B copies its chains" result,
  confirmed in-fit.
- **The chain-free judge (S1)** does what the offline table said:
  - it removes about half of the wrong steps and about 70 % of the "raised" steps;
  - but it also removes 17-22 % of the good ones, including every good step that would have moved the fit away from
    the chain-free solution.

## 6. Results (spiral_eval vs base; own-offset subset wrong area with original labels, split 41)
Values are B - base, with * = 95 % CI excludes 0:
- pairs = primary-pair matched accuracy (higher is better);
- seam WA = seam-aware wrong-winding area %;
- sw = seam switches / 1000;
- unm = unmatched %;
- own in_c / out_c = own-offset inner_consistent / outer_consistent wrong area (lower is better for all but pairs).

| arm | band | setting | pairs | coverage | seam WA | sw | unm | own in_c | own out_c |
|---|---|---|---|---|---|---|---|---|---|
| arm21 (ref) | 2 | chains | -4.4 | +10.8* | -9.0* | +30.8* | -1.6* | +4.21* | +3.25* |
| **rob2** | 2 | S1 | -3.1* | -1.5 | -5.4* | **-0.5** | +0.7 | **-0.45** | **-0.68** |
| rob3 | 2 | S2 | -4.0 | +9.2* | -9.3* | +24.3* | -1.7* | +4.06* | +2.79* |
| arm23 (ref) | 1 | chains | +15.0* | -1.0 | -11.9* | +15.2* | +0.2 | -19.1* | +9.5* |
| **rob1** | 1 | S1 | -0.4 | -4.5* | **+5.5*** | +4.3* | +3.2* | **-0.54*** | +4.83* |
| rob5 | 1 | S2 | +16.4* | +1.5 | -12.4* | +10.1* | +0.1 | -19.93* | +9.27* |
| arm6 (ref) | 1 | oracle + 10 % flips | +13.2* | +3.2 | -11.6* | +15.2* | -0.4* | -15.97 | +0.39 |
| rob4 | 1 | S2 | +13.2* | +3.7 | -12.2* | +11.7* | -0.4* | -16.16* | +0.30 |

Direct comparisons with the matching chain arm (own-offset, B - ref):

| comparison | in_c | out_c | inner | outer |
|---|---|---|---|---|
| rob3 vs arm21 | -0.15 | -0.46* | -0.18* | -0.45* |
| rob5 vs arm23 | -0.82* | -0.26* | -0.71* | -0.32* |
| rob4 vs arm6 | -0.19 | -0.09 | -0.92* | -0.17* |
| rob1 vs arm23 | +18.57* | -4.70* | +31.90* | -1.04 |
| rob2 vs arm21 | -4.67* | -3.93* | -1.82 | -3.66* |

Files:
- data/scratch/spiral_eval/arms/{arm20,arm0}_vs_rob*.json
- data/scratch/robust_fitter/readout/swa_*_vs_rob*.txt
- vertex dumps in data/scratch/fit_figures/vertex/rob*.npz, plus arm6.npz (new)

## 7. Verdict
- **The primary criterion is not met by any single setting.**
- **S1 (chain-free judge, cut and freeze) solves band 2 but destroys band 1:**
  - In band 2, rob2 is no longer worse than the baseline. Own-offset in_c and out_c are -0.45 and -0.68 (n.s.),
    switches -0.5 (vs +30.8 for arm21), seam wrong area -5.4*.
  - In band 1, rob1 loses all of arm23's gain and ends slightly worse than arm0: seam WA +5.5*, unmatched +3.2*,
    out_c +4.8*.
  - The reason is structural. Pieces that agree with the chain-free fit within half a turn cannot move it. The
    chain's value is exactly the steps where it disagrees with the chain-free fit and is right (62 % of those in
    band 1).
  - A frozen S1 cut therefore reduces the chain fit to the chain-free fit, plus a perturbation at the switch-on. That
    is a wash where the chains hurt (band 2) and a loss where they help (band 1).
- **S2 (classic IRLS, re-cut every 2000 steps) is harmless but does not clean:**
  - rob3, rob5 and rob4 track their chain arms on every metric. They have a small but consistent edge: own-offset
    wrong area -0.1 to -0.9 vs the matching chain arm (significant in most cells), and 3.5-6.5 fewer switches per
    1000 in all three arms (63.7 -> 58.6, 63.7 -> 60.2, 91.9 -> 85.4).
  - The switch differences are within the fit-to-fit spread seen before (arm23 vs arm19: 7 pp WA from 4 % different
    steps). Read them as "no harm, maybe slightly fewer switches", not as a fix. Band 2 stays exactly as bad as arm21.
- **Bottom line:** the fitter cannot clean its own annotations, because the information that separates good steps
  from bad is not in any fit:
  - the chain fit copies the chains;
  - the chain-free fit is a weak judge that trades band-1 gain for band-2 safety.
  - This matches the band-2 diagnosis and the dsafe/p93 frontier. Self-cleaning cannot replace better step labels
    (human checks / HITL) or a better automatic step signal.
- **What to use:**
  - keep the patch, off by default (byte-identical path verified);
  - S2 may be turned on for free (no measured harm, slightly fewer switches);
  - do not use S1 for the submission.
- **Untested next step**, if anyone wants to push the frontier: windowed S1 (`unattached_pcl_robust_window=5`,
  `..._window_threshold=0.4`, already in the patch). It cuts only where A's disagreement is dense, which is where A
  beats the chain in band 2 (section 2). Offline it predicts a trade-off between rob1 and arm23, not a free win.
  About $9 for the two bands.

## 8. Cost and pods
5 pods, vss-rob-rob1..5, created 22:58-23:02Z and terminated by the collector right after each pull (01:43-03:12Z):

| pod | GPU | hours | cost |
|---|---|---|---|
| rob1 | L40S, runpod, $1.11/h | 3.92 | $4.35 |
| rob2 | L40S, runpod, $1.11/h | 2.74 | $3.04 |
| rob3 | RTX6000Ada, runpod, $1.01/h | 4.22 | $4.26 |
| rob4 | RTX6000Ada, datacrunch | 3.33 | ~$4.6 (price not logged because the create "failed" and the pod was adopted; <= $6.0 at the $1.8/h grab cap) |
| rob5 | RTX6000Ada, runpod, $1.01/h | 4.17 | $4.21 |

- **Total about $20.5** (at most $21.9), against the $45 cap.
- `prime pods list` after the last termination shows no vss-rob pods. The create, port and terminate lines are in
  data/scratch/p0139_spiral/pods_arms.txt.
- All local spiral_eval and vertex-dump runs were wrapped in `flock /tmp/vss_eval.lock`.
