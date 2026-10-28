# Fit-seed variance on PHerc0139 (2026-10-22)

Question: are the gm-threshold arm differences real, or is the spiral fit flipping between basins from run to run? The
band-1 arms are non-monotone (arm23 0.18 / arm30 0.229 / arm29 0.263; zone k43-50 +21.6 / +21.4 / +1.1 vs arm0). To test
this, we refit the key arms with only the fit seed changed (villa fitter pinned at 0fb38c45, same inputs and chains) and
scored each replicate exactly as its original.

**Short answer:** chain arms reproduce to well under 1 pp on every metric. The no-chain baselines (arm0, arm20) are
bistable, mainly in zone k43-50. Seed noise does not explain the arm-to-arm k43-50 pattern; the chains set it. But every
"vs no-chain base" number that involves k43-50 (seam, outer cons, pooled, zone k43-50) depends on which basin the base
landed in.

## Method

- **Seed key.** villa's `optimizer_random_seed` (default 1) is already a config key. It seeds np.random and torch in
  `load_host_inputs` (fit_spiral.py:1093), the shell pool generator (:1778) and the influence stash (:1711). It is set via
  `FIT_OVERRIDES`, which run_arm.sh merges into `FIT_SPIRAL_CONFIG_OVERRIDES` (unknown keys are rejected). No patch was
  needed.
  - Every earlier arm ran with seed 1.
  - Replicate envs `spiral/arms/<arm>s<N>.env` are the original env plus `"optimizer_random_seed": N`. This is
    documented in the run_arm.sh header.
  - The seed takes effect: step-0 loss is 11409.3 for arm23 and 11502.0 for arm23s2.
- **Track extraction is not deterministic either.** Independently of the seed, every fit lands on one of two track-set
  variants: band 1 has 1159740 or 1159745 tracks, band 2 has 941744 or 941638. The originals already differed this way
  (arm0 1159740 vs arm2 1159745).

  | fit | tracks | replicate | tracks |
  |---|---|---|---|
  | arm0 | 1159740 | arm0s2 | 1159745 |
  | arm23 | 1159740 | arm23s3 | 1159740 |
  | arm29 | 1159740 | arm29s2 | 1159745 |
  | arm30 | 1159740 | arm30s2 | 1159745 |
  | arm2 | 1159745 | arm2s2 | 1159745 |
  | arm20 | 941744 | arm20s2 | 941638 |
  | arm26 | 941638 | arm26s2 | 941744 |

  So arm23s3-arm23 and arm2s2-arm2 are pure seed-only pairs. The other pairs change the seed and the track variant
  together. That is the run-to-run variation a new arm actually sees, so it is the right noise to measure.
- **Runs.**
  - Band 1: arm0s2, arm23s3, arm29s2, arm30s2, arm2s2. Band 2: arm20s2, arm26s2.
  - arm23s2 was fitted and pulled, then its meshes were deleted by mistake during a disk-full cleanup (see Caveats).
    arm23s3 (seed 3) stands in as arm23's replicate.
  - arm0s3 was not run: no offer at or below $0.90/h appeared in 75 min, and the dearer runpod would have pushed the
    projected spend to about the cap.
  - One 1x L40S per pod.
- **Scoring.** Everything ran locally under `flock /tmp/vss_eval.lock`, the same way as the originals:
  - spiral_eval `--tol 4` vs the base and vs the original;
  - eval/subset_wrong_area.py `--split 41 --own-offsets` (inner/outer consistent, pooled = matched-weighted as in
    hitl_table.py);
  - pairs_exclude (pair acc excl CT);
  - per-zone k23-40 / k43-50 with own offsets (perk30 logic).
- **Scripts:** data/scratch/seedvar/post_seed.sh, perk_seed.py, seed_table.py. Outputs: seed_table.md, seed_table.json,
  perk_seed.json.
- **Validation.** Every metric is a per-fit quantity, and B - A = b - a holds exactly (checked on arm0 vs arm23/29/30/2).
  So any two fits can be differenced, including replicate vs replicate. perk_seed.py run in `PERK_TEST=1` mode reproduces
  perk30.json exactly.

## Original vs replicate (B - arm0 / arm20 original; "- arm0s2" rows = replicate world)

All values are pp; switches are per 1000 vertices.

| arm | fit | pair acc | pair excl CT | seam wrong area | switches | inner cons | outer cons | pooled | k23-40 | k43-50 |
|---|---|---|---|---|---|---|---|---|---|---|
| arm23 (0.18) | arm23 | +15.0 | +15.0 | -11.9 | +15.2 | -19.1 | +9.5 | -9.6 | -32.5 | +21.6 |
| | arm23s3 | +15.0 | +15.0 | -11.8 | +15.8 | -19.0 | +9.7 | -9.6 | -32.4 | +21.6 |
| | arm23s3 - arm0s2 | +15.8 | +15.8 | -18.2 | +12.7 | -18.0 | +4.3 | -14.3 | -31.9 | +0.7 |
| arm29 (0.263) | arm29 | +3.1 | +3.1 | -0.6 | +8.2 | -0.8 | +0.8 | -7.4 | -17.2 | +1.1 |
| | arm29s2 | +2.2 | +2.2 | -0.6 | +8.6 | -1.1 | +0.9 | -7.5 | -17.5 | +1.0 |
| | arm29s2 - arm0s2 | +2.9 | +2.9 | -7.0 | +5.5 | -0.1 | -4.5 | -12.2 | -17.0 | -19.9 |
| arm30 (0.229) | arm30 | +1.2 | +1.2 | +6.4 | +16.2 | -1.4 | +8.6 | -1.1 | -15.4 | +21.4 |
| | arm30s2 | +1.5 | +1.5 | +6.4 | +16.6 | -1.5 | +8.7 | -1.1 | -15.6 | +21.3 |
| | arm30s2 - arm0s2 | +2.3 | +2.3 | -0.0 | +13.5 | -0.5 | +3.2 | -5.7 | -15.1 | +0.3 |
| arm2 (oracle) | arm2 | +14.1 | +14.1 | -13.5 | +8.2 | -18.6 | -0.2 | -12.2 | -26.6 | +0.5 |
| | arm2s2 | +14.9 | +14.9 | -13.7 | +9.3 | -18.8 | -0.2 | -12.3 | -26.8 | +0.5 |
| | arm2s2 - arm0s2 | +15.7 | +15.7 | -20.1 | +6.2 | -17.8 | -5.7 | -17.0 | -26.3 | -20.4 |
| arm26 (b2, 0.263) | arm26 | -3.7 | -4.3 | -0.7 | +18.5 | -5.3 | -0.5 | -6.9 | | |
| | arm26s2 | -1.6 | -2.6 | -0.6 | +18.1 | -4.8 | -0.1 | -6.8 | | |
| | arm26s2 - arm20s2 | -0.4 | -1.6 | +4.8 | +19.8 | -4.6 | +0.0 | -6.7 | | |

## Seed-only differences (replicate - original, same arm)

Bootstrap 95% CIs (site/vertex bootstrap, orig-vs-rep spiral_eval and perk_seed) are given where the shift matters.

| arm | pair acc | seam | switches | inner cons | outer cons | pooled | k23-40 | k43-50 |
|---|---|---|---|---|---|---|---|---|
| arm0s2 - arm0 | -0.8 [-2.3,+0.6] | **+6.4 [+5.5,+7.4]** | +3.1 [+0.7,+5.9] | -1.0 | **+5.5** | **+4.6** | -0.5 [-0.7,-0.3] | **+20.9 [+18.3,+23.8]** |
| arm23s3 - arm23 | +0.1 | +0.1 | +0.6 | +0.1 | +0.2 | +0.1 | +0.1 | +0.0 |
| arm29s2 - arm29 | -0.9 | -0.0 | +0.3 | -0.3 | +0.1 | -0.1 | -0.2 | -0.1 |
| arm30s2 - arm30 | +0.3 | +0.0 | +0.5 | -0.1 | +0.1 | +0.0 | -0.1 | -0.1 |
| arm2s2 - arm2 | +0.8 [-0.2,+1.9] | -0.1 | +1.1 | -0.2 | +0.0 | -0.1 | -0.2 | +0.0 |
| arm20s2 - arm20 | -1.2 [-3.5,+0.8] | **-5.4 [-7.3,-3.7]** | -1.7 | -0.2 | -0.1 | -0.1 | | |
| arm26s2 - arm26 | **+2.1 [+0.4,+4.2]** | +0.1 | -0.4 | +0.5 | +0.4 | +0.1 | | |

- **Baseline shifts are real.** The arm0 and arm20 seed shifts have bootstrap CIs that exclude 0. The site bootstrap
  therefore does not cover fit-to-fit noise for no-chain fits.
- **Chain-arm shifts are tiny.** Every one is at most ~1 pp, except arm26 pair acc +2.1, the only notable chain-arm
  shift. arm26s2 also has +4.9 pp coverage, so more pairs are matched.

## Fit-to-fit SD per metric (pooled within-arm SD)

| metric | no-chain baselines (dof 2; zones dof 1) | chain arms (dof 5; zones 4) |
|---|---|---|
| pair acc | 0.73 | 0.78 |
| pair acc excl CT | 0.65 | 0.67 |
| seam wrong area | **4.17** | 0.06 |
| switches/1000 | 1.78 | 0.46 |
| own inner cons | 0.51 | 0.20 |
| own outer cons | **2.73** | 0.14 |
| own pooled | **2.50** | 0.05 |
| zone k23-40 | 0.35 | 0.12 |
| zone k43-50 | **14.8** | 0.04 |

- **Chain arms.** Fit noise is negligible next to the site bootstrap CIs, which are typically ±1-3 pp. Pair accuracy has
  the largest chain-arm noise, about 0.8 pp.
- **Baselines.** The baseline SDs come from 2 dof, so they are very rough. "SD" is also the wrong model here: the baseline
  is two-state, not Gaussian.

## Zone k43-50 is two-state

The absolute own-offset k43-50 wrong area takes one of two values, about 27 or about 48:

| state | k43-50 value | fits |
|---|---|---|
| low | ~26.5-27.6 | arm0, arm2, arm2s2, arm29, arm29s2 |
| high | ~47.5-48.2 | arm0s2, arm23, arm23s3, arm30, arm30s2 |

- **Chain arms are pinned.** Each chain arm lands in the same state on every seed (arm23 high, arm30 high, arm29 low,
  arm2 low).
- **arm0 is not.** It flipped with the seed.
- **arm30 vs arm29 is real.** The k43-50 jump of +20.3 [+17.6, +23.1] reproduces identically on both seeds. It is
  deterministic given the chains, not seed noise. The non-monotone threshold behaviour is the chain sets themselves (the
  0.229 and 0.18 chains push k43-50 into the bad state; the 0.263 and oracle chains do not), not basin flipping.
- **arm23's over-raise penalty is a basin artifact.** The "+21.6 vs arm0" exists only because arm0 seed 1 landed in the
  low state. Against arm0s2 it is +0.7. Against the chain arms that sit in the low state (arm29, arm2), arm23 is +20.6 /
  +21.1 on both seeds.

## Key comparisons re-read

Values below are originals / replicate pair / all-fits mean. The SE comes from the group SDs above. The k43-50 zone SE for
anything vs a baseline is 10.5, but a 1-dof two-state SD is not really an SE.

| comparison | survives seed noise | does not survive |
|---|---|---|
| **arm23 vs arm0** | pair acc +15.0 / +15.8; inner cons -19.1 / -18.0; k23-40 -32.5 / -31.9; switches +15.2 / +12.7; seam -11.9 / -18.2 (sign robust, size baseline-dependent); pooled -9.6 / -14.3 (mean -12.0, SE 1.8) | k43-50 +21.6 / +0.7; outer cons +9.5 / +4.3 is directionally robust, but its size halves |
| **arm23 vs arm29** | every metric: pair +11.9 / +12.9; seam -11.3 / -11.2; inner -18.3 / -17.9; outer +8.7 / +8.8; pooled -2.2 / -2.1; k23-40 -15.3 / -14.9; k43-50 +20.5 / +20.6 | (none) |
| **arm2 vs arm0** | pair +14.1 / +15.7; inner -18.6 / -17.8; k23-40 -26.6 / -26.3; pooled -12.2 / -17.0 (mean -14.6) | k43-50 +0.5 / -20.4; outer cons -0.2 / -5.7 |
| **arm26 vs arm20** (band 2) | switches +18.5 / +19.8; inner cons -5.3 / -4.6; pooled -6.9 / -6.7 (SE 1.8) | pair acc -3.7 / -0.4 (mean -2.0, SE 0.8: arm26 is probably slightly worse on pairs, but the original -3.7 overstates it); pair excl CT -4.3 / -1.6; seam -0.7 / +4.8 (the baseline flipped) |

Two further comparisons (arm30 vs arm29, arm23 vs arm2) involve only chain arms, so they need no table:

- **arm30 vs arm29** reproduces fully: k43-50 +20.3 / +20.3, pooled +6.6, seam +7.0, outer cons +7.7.
- **arm23 vs arm2** (auto vs oracle) reproduces: pair +0.9 / +0.2 (a tie), outer cons +9.7 / +9.9, pooled +2.4 / +2.6,
  k43-50 +21.1.

**Practical consequences:**

1. Comparisons between chain arms can be trusted at their bootstrap CIs. One fit per arm is enough.
2. Do not quote seam, outer cons, pooled or k43-50 deltas against a single no-chain base fit. That includes the
   "fraction of oracle gain" figures, for example pooled vs arm0: arm23 -9.6 or -14.3, arm2 -12.2 or -17.0, depending on
   the arm0 basin. Either quote both arm0 basins, or use a chain arm in the low state (arm2 or arm29) as the anchor for
   k43-50.
3. Pair acc and inner / k23-40 vs base are robust (baseline SD ≤0.7 pp).
4. Band-2 pair-accuracy deltas of 2-4 pp are at the edge: fit SD is about 0.8 pp and arm26 moved 2.1 pp between seeds.

## Cost and pods

8 pods (1x L40S):

| pod | provider | $/h | h | $ |
|---|---|---|---|---|
| vss-seed-arm23s2 | massedcompute | 0.82 | 3.77 | 3.09 |
| vss-seed-arm0s2 | massedcompute | 0.82 | 3.32 | 2.72 |
| vss-seed-arm29s2 | runpod | 1.11 | 4.00 | 4.44 |
| vss-seed-arm30s2 | runpod | 1.11 | 4.49 | 4.98 |
| vss-seed-arm2s2 | runpod | 1.11 | 4.16 | 4.61 |
| vss-seed-arm20s2 | runpod | 1.11 | 2.89 | 3.21 |
| vss-seed-arm26s2 | runpod | 1.11 | 2.71 | 3.00 |
| vss-seed-arm23s3 | massedcompute | 0.82 | 3.02 | 2.47 |
| **total** | | | | **~28.5 (cap 35)** |

- **Provider.** massedcompute ran out of stock after two pods, so five pods went to runpod at ≤$1.12/h.
- **Pod log.** Every create, port, launch and terminate is logged in data/scratch/p0139_spiral/pods_arms.txt.
- **Pod status.** All pods were terminated right after their pulls. `prime pods list` is empty.

## Caveats

- **arm23s2 lost (my error).** Its meshes were pulled and verified, the pod was terminated, and I then deleted the meshes
  during the disk-full cleanup, taking them for a partial pull. The small files are kept in
  data/scratch/p0139_spiral/arm23s2/ (see LOST.txt). The seed-2 step-0 loss check came from this run.
- **Disk.** /home is at 100% (about 2.5 GB free) because of other processes. The meshes, meshes_step4 and run directories
  for arm23s3, arm29s2, arm30s2 and arm2s2 live only in tmpfs at /tmp/vss_seed/ (about 8.3 GB), symlinked from
  data/scratch/p0139_spiral/<arm>/. **They are lost on reboot.** The scores and vertex dumps
  (data/scratch/fit_figures/vertex/*.npz) are on disk.
- **Few replicates.** n is 2 per arm, and only arm23 and arm2 are pure seed-only pairs. The baseline SDs have 1-2 dof. The
  two-state reading of arm0 rests on one flip; a third arm0 seed would tell how often each basin occurs.
