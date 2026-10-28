# 2026-10-22: context-dependent gm raise rule vs global thresholds (offline, $0)

**Verdict: NO-GO.** Do not spend the ~$15-20 on fit arms for these rules. No chain sets were written.

Question: does a raise rule whose threshold depends on local, label-free context beat every single global gm threshold
on both bands? The fits set the target per zone:
- band 1 inner (w023-w040): 0.18 wins (arm23);
- band 1 w043-w050: 0.263 wins (arm29);
- band 2: 0.263 wins (arm26).

Data used: dev only (z 4500-6000 for fitting, bands 1 and 2 for scoring). Nothing from z 6000-8500 or PHerc1667 was
read; the z4500 loader asserts z < 6000.

Code and outputs are in data/scratch/ctxraise/:

| file | what |
|---|---|
| build.py -> tab_{z4500,band1,band2}.npz | per-step tables: d0, gm_p, covariates, truth counts, zone label |
| ev.py | per-zone evaluator: step precision, raise precision/recall, drift 10-20 / 3-6 |
| rules.py, freeze.py -> frozen_rule.json | context models fitted on z4500 only; frozen cut |
| evaluate.py -> eval_band{1,2}.json | frozen rules applied to each band; equivalent thresholds; Jaccard vs the winner |
| rbin.py -> rbin_rule.json, rbin_eval.txt | post-hoc radius-bin rule (secondary) |
| summary.py -> summary.json, summary.md | all rules on both bands, including within-zone drift |

Run with `uv run --with scikit-learn python <script>` from data/scratch/ctxraise/.

## Order of operations (freeze discipline)
1. Built the tables. Looked at z4500 and band-1 global-threshold sweeps, plus label-free covariate distributions by
   zone in all three sets. No band-2 truth metric was computed at this point.
2. `freeze.py` fitted every rule on z4500 only and froze the cuts. frozen_rule.json was written at
   2026-10-22 00:04:39 (sha256 of its text: fe8975e6...1c984d).
3. Scored on band 1 (00:05:04), then on band 2 (00:05:16).
4. After both bands had been scored, I built a radius-bin rule (`rbin`) as an extra probe. It was fitted on z4500
   only, but it was conceived after seeing the band results, so it is secondary and post-hoc.

## Setup
- **gm_p on z4500, out of fold.** The stored stepconf OOF gm_p came from the non-clean conf model (threshold 0.19),
  but the band chains use the clean model (0.18). So I refit the gm logistic for each held-out 500-z slab with the
  clean `logperr` (from band_m7_z4500_ray_c). The full-data refit reproduces auto_windings_gm_z4500_clean.json's
  coefficients exactly. Correlation with the old OOF logit is 0.97.
- **Covariates.** All come from a pre-gm chain set alone, without labels:
  - step length L;
  - radius;
  - local pitch: median length of the single steps within ±5 steps on the chain (`pitch5`), and the median gap between
    all chain points on the ray within ±100 vox;
  - crossing density within ±32 / ±64 vox;
  - neighbour-ray consistency on pre-gm counts (z ±64, θ ±15);
  - relative step length.
- **Truth.**
  - z4500 and band 1: CT `supported` (the band-1 map applied to the z4500 labels, as in hitl/recal.py).
  - band 2: the mean of the `minus1` and `minus2` metrics, i.e. midway between them.
- **Zones.** By the original label winding of the step's inner end: inner 23-40, 41-50, other (mostly 51-59).
  - drift10: pairs 10-20 labelled windings apart, assigned to the zone of their midpoint.
  - within: both pair points inside the zone (23-40: T 8-17; 41-50 and 51-59: T 3-9). This is the cleaner per-zone
    number.
- **Rules.** Each rule is a logistic P(true step ≥ 2 | single step, x), fitted on z4500 labelled single steps. The
  steps are raised where P ≥ cut, and the cut gives zero pooled drift10 on z4500 (using OOF P by slab). Feature sets:

  | rule | features (on top of logit gm_p) |
  |---|---|
  | gm | none |
  | pitch | log pitch, plus the interaction |
  | pitchbin | pitch tertiles (16.5 / 20.75 vox), with interactions |
  | radius | r, r², g·r |
  | ctx | pitch, r, r², density, neighbour consistency, relative L, interactions |

**Frozen rules.**
- gm: cut 0.290, which equals a global gm threshold of 0.277.
- pitchbin: cut 0.296, giving gm thresholds of 0.251 for pitch < 16.5 vox, 0.272 for 16.5-20.75 and 0.398 above
  20.75. So tight packing does get more raises.
- Context adds little: OOF AUC is 0.733 for gm alone and 0.745 for ctx; log-loss goes from 0.2565 to 0.2534.

## Results

Each cell is: raised % of single steps / step precision / raise precision / raise recall / drift10 / within-zone drift.
Drift is the chain count minus the CT count, in turns. The winning setting in each zone is in bold.

### Band 1 (scored first)

| rule | all: raised / prec / drift10 | 23-40 | 41-50 | other (within = 51-59) |
|---|---|---|---|---|
| global 0.18 | 18.4 / 84.9 / +0.57 | **14.1 / 86.1 / 34 / 49 / +0.43 / +0.19** | 12.7 / 78.3 / 42 / 24 / +0.79 / +0.65 | 20.3 / 85.8 / – / – / +0.21 / +0.21 |
| global 0.229 | 12.7 / 86.9 / +0.05 | 9.5 / 88.2 / 39 / 38 / -0.11 / -0.20 | 8.1 / 80.1 / 54 / 19 / +0.26 / +0.47 | 14.1 / 87.9 / – / – / -0.23 / +0.05 |
| global 0.263 | 10.4 / 87.7 / -0.15 | 7.5 / 89.1 / 44 / 34 / -0.33 / -0.38 | **6.6 / 80.4 / 59 / 16 / +0.08 / +0.41** | 11.7 / 88.6 / – / – / -0.37 / +0.01 |
| global 0.29 | 8.8 / 88.1 / -0.29 | 6.4 / 89.5 / 47 / 30 / -0.46 / -0.48 | 5.4 / 80.8 / 65 / 15 / -0.06 / +0.34 | 9.9 / 89.1 / – / – / -0.50 / -0.05 |
| frozen gm | 9.5 / 87.8 / -0.22 | 6.9 / 89.3 / 45 / 32 / -0.39 / -0.43 | 5.9 / 80.4 / 59 / 15 / -0.00 / +0.37 | 10.7 / 88.8 / – / – / -0.44 / -0.02 |
| frozen pitchbin | 8.5 / 88.1 / -0.29 | 6.1 / 89.6 / 48 / 31 / -0.47 / -0.50 | 5.2 / 80.6 / 61 / 14 / -0.05 / +0.34 | 9.5 / 89.0 / – / – / -0.47 / -0.04 |
| frozen pitch | 8.2 / 88.3 / -0.29 | 6.1 / 89.8 / 50 / 31 / -0.47 / -0.50 | 5.2 / 80.7 / 62 / 15 / -0.05 / +0.35 | 9.2 / 89.2 / – / – / -0.48 / -0.04 |
| frozen radius | 9.5 / 88.1 / -0.25 | 5.7 / 89.8 / 50 / 28 / -0.50 / -0.55 | 6.4 / 80.3 / 59 / 16 / +0.05 / +0.39 | 11.0 / 88.5 / – / – / -0.35 / +0.01 |
| frozen ctx | 8.3 / 88.4 / -0.33 | 4.6 / 90.1 / 54 / 25 / -0.60 / -0.64 | 5.5 / 80.8 / 63 / 16 / -0.01 / +0.36 | 9.7 / 88.6 / – / – / -0.39 / -0.00 |
| post-hoc rbin | 14.7 / 86.4 / +0.14 | 8.7 / 88.3 / 39 / 36 / -0.15 / -0.26 | 9.9 / 79.4 / 48 / 22 / +0.48 / +0.54 | 16.9 / 86.0 / – / – / +0.02 / +0.14 |
| zone-winner reference (label-zoned, not deployable) | 11.7 / 85.9 / +0.23 | 14.1 / 86.1 / 34 / 49 / +0.27 / +0.15 | 6.6 / 80.4 / 59 / 16 / +0.24 / +0.41 | 11.7 / 88.6 / – / – / -0.37 / +0.01 |

### Band 2 (scored second; midway between minus1 and minus2)

| rule | all: raised / prec / drift10 | 23-40 | 41-50 (n tiny) | other (within = 51-59) |
|---|---|---|---|---|
| global 0.18 | 25.2 / 80.3 / +0.85 | 21.2 / 80.4 / 23 / 52 / +0.86 / +0.87 | 18.7 / 76.6 / – / – / +0.86 / +0.27 | 26.0 / 80.0 / – / – / +0.22 / +0.31 |
| global 0.229 | 18.2 / 84.5 / +0.12 | 14.2 / 85.0 / 29 / 43 / +0.12 / +0.27 | 12.3 / 83.0 / – / – / +0.13 / +0.20 | 19.0 / 83.5 / – / – / -0.15 / +0.12 |
| **global 0.263** | **14.9 / 86.2 / -0.17** | **11.3 / 86.7 / 33 / 39 / -0.16 / +0.06** | 10.4 / 91.5 / – / – / -0.18 / +0.15 | 15.7 / 85.0 / – / – / -0.32 / +0.03 |
| global 0.29 | 12.9 / 87.0 / -0.31 | 9.5 / 87.6 / 35 / 34 / -0.28 / -0.05 | 8.5 / 91.5 / – / – / -0.33 / +0.15 | 13.6 / 85.6 / – / – / -0.42 / -0.03 |
| frozen gm | 13.9 / 86.8 / -0.25 | 10.4 / 87.3 / 35 / 37 / -0.22 / +0.00 | 8.9 / 91.5 / – / – / -0.27 / +0.15 | 14.6 / 85.6 / – / – / -0.36 / -0.00 |
| frozen pitchbin | 12.2 / 86.9 / -0.32 | 9.2 / 87.5 / 36 / 36 / -0.27 / -0.04 | 8.5 / 87.2 / – / – / -0.36 / +0.15 | 12.8 / 85.4 / – / – / -0.35 / -0.01 |
| frozen pitch | 11.7 / 86.6 / -0.28 | 9.2 / 87.3 / 35 / 36 / -0.22 / -0.00 | 8.3 / 91.5 / – / – / -0.33 / +0.19 | 12.2 / 84.9 / – / – / -0.36 / -0.00 |
| frozen radius | 13.0 / 87.2 / -0.30 | 8.4 / 88.2 / 38 / 30 / -0.32 / -0.12 | 9.1 / 91.5 / – / – / -0.28 / +0.12 | 13.9 / 84.7 / – / – / -0.34 / +0.03 |
| frozen ctx | 10.4 / 87.4 / -0.40 | 6.6 / 88.7 / 42 / 28 / -0.39 / -0.19 | 7.9 / 91.5 / – / – / -0.40 / +0.24 | 11.1 / 84.4 / – / – / -0.34 / +0.03 |
| post-hoc rbin | 18.6 / 84.3 / +0.17 | 14.0 / 85.0 / 28 / 40 / +0.12 / +0.26 | 14.6 / 80.9 / – / – / +0.22 / +0.25 | 19.5 / 83.1 / – / – / +0.02 / +0.12 |

Band-2 zone 41-50 has 481 single steps, only 2 verified true doubles and 3-10 verified raises, so it cannot be scored.

### Step 4: do the rules reproduce the winning raise decisions?

Two measures per zone:
- equivalent threshold: the global gm threshold that raises the same number of single steps in the zone;
- Jaccard: overlap of the rule's raised steps with the winner's raised steps.

| zone (winner) | frozen gm | pitchbin | pitch | radius | ctx | post-hoc rbin |
|---|---|---|---|---|---|---|
| band 1, 23-40 (**0.18**) | 0.277 / J 0.49 | 0.296 / 0.43 | 0.297 / 0.43 | 0.305 / 0.41 | **0.346 / 0.33** | ~0.24 |
| band 1, 41-50 (**0.263**) | 0.277 / 0.88 | 0.295 / 0.76 | 0.295 / 0.74 | 0.266 / 0.85 | 0.286 / 0.68 | ~0.21 |
| band 2, 23-40 (**0.263**) | 0.277 / 0.93 | 0.297 / 0.75 | 0.295 / 0.73 | 0.319 / 0.73 | 0.358 / 0.57 | ~0.23 |
| band 2, other (**0.263**) | 0.277 / 0.93 | 0.301 / 0.78 | 0.312 / 0.72 | 0.286 / 0.78 | 0.332 / 0.57 | |

- **Band 1 inner is the decisive zone, and every rule fails there in the wrong direction.**
  - The rules act like global 0.28-0.35, so they raise about half as many inner steps as 0.18 (4.6-6.9 % vs 14.1 %).
  - Within-zone drift is -0.43 to -0.64, against +0.19 for 0.18 and -0.38 for 0.263.
  - That is the arm29 failure mode, or worse: arm29 cost -14.7 pp of inner pair accuracy.
  - The richest model (ctx) moves furthest the wrong way.
- In band 1 41-50 and in band 2, the rules match or slightly beat the winner: band 2 inner within-zone drift is +0.00
  for frozen gm vs +0.06 for 0.263. But they get there because they are close to a global ~0.28. That is not context.
- The best overall offline candidate is the plain frozen gm rule (global 0.277), and on band 1 inner it is
  essentially arm29.

## Why context does not help (label-free covariates cannot separate the zones)
1. **Pitch does not mark the inner turns.** Median pitch5 is 18.8 vox in band-1 23-40 and 18.0 in 41-50; band 2 and
   z4500 are similar, and density is the same. The pitch rule does raise more in tight spots (thr 0.25 below 16.5 vox),
   but tight spots are spread evenly across zones.
2. **The labels push the wrong way in 41-50.** At equal gm_p, 41-50 has as many or more true doubles than inner (z4500
   at gm < 0.05: 10 % vs 2 %; band 1: 9 % vs 1 %). Its raise precision is also higher (band 1 at 0.263: 59 vs 44 %). So
   a model of P(double) learns "raise more in 41-50", the opposite of the fit result. The over-count in 41-50 is
   insensitive to the threshold: within-zone drift stays at +0.34 to +0.65 for every threshold, on z4500 too. So it
   does not come from imprecise raises. Its likely sources are the label structure there (w041 = w042 duplicates,
   w046/47, the w050 gap) and d0 = 2 steps. Fewer raises only offset it.
3. **The inner optimum is not stable across z.**
   - Within-zone zero drift in 23-40 sits at about 0.20 on both z4500 and band 1.
   - Band 2's inner winner is 0.263, where it raises 11.3 % (band 1 at 0.18: 14.1 %). Band 2's gm_p runs hotter at
     every zone (21 % raised at 0.18 vs 14 % in band 1 and 10.5 % at z4500).
   - So a single zone-to-threshold map breaks band 2, and a single zone-to-raise-rate map breaks band 1.
   - The radius-bin rule shows both failures at once: band 2 over-raises (+0.26 within inner), while band 1 41-50
     under-corrects.
4. **Zones are only partly identifiable.** Radius alone separates inner from 41-50 with AUC 0.64-0.71; with θ and z
   added, 0.80-0.84 (cross-validated by θ sector). That is a winding-number map in disguise, and it still does not
   give the right threshold, by point 3.

**What does agree with the fits:** the within-zone drift proxy (both points inside the zone). It reproduces the fit
ordering in band 1:
- inner: 0.18 (+0.19) beats 0.263 (-0.38);
- 41-50: 0.263 (+0.41) beats 0.18 (+0.65).

The midpoint drift10 and step precision do not reproduce it. Step precision rises monotonically with the threshold
everywhere. So within-zone drift is the offline score to use for any future raise rule.

## Go / no-go
**NO-GO on fit arms for these rules.**
- No frozen context rule reproduces the band-1 inner decision. All of them behave like global 0.28-0.35 there, which is
  the arm29 regime that lost the gain.
- In band 2 they add nothing over 0.263 / 0.277.
- A fit arm would most likely re-measure arm29 (band 1) and arm26 (band 2).

**What would change this:** a covariate that is high in band-1 inner, low in band-2 inner and low in band-1 41-50.
- None of pitch, radius, density, neighbour consistency or relative length qualifies.
- The only signal seen is the band-level gm_p shift, which already goes against band 1 inner.
- A real candidate would need new evidence, e.g. why arm23 helps inner even though it over-counts there by +0.2 turns.
  That points at fitter-side behaviour, not at the chain-level proxies.
- If an arm is wanted anyway: frozen gm (global 0.277) is the safest single setting on both bands offline, but it is
  arm29 / arm26 in all but name, so it is not worth $15-20.

Step 5 was skipped (no-go): no chain sets were written and no fits were launched.
