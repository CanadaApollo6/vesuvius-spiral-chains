# 2026-10-22: why the band-2 confirmation failed (PHerc0139 z 1750-3000; arm21 vs arm20)

Local analysis only: no pods, no new fits, $0. Nothing here was fitted or thresholded on band-2 outcomes. Every
pattern below was found after looking at band 2, so any rule it suggests needs a third, untouched region to confirm it.

- Band 1 = z 3000-4500: arm0 (base), arm23 (clean auto chains), arm2 (oracle chains).
- Band 2 = z 1750-3000: arm20 (base), arm21 (auto chains, the claim arm), arm22 (oracle chains).
- "own" wrong area: offsets are refitted separately on inner (k < 41) and outer (k >= 41) vertices (the 7b rule), so it
  does not depend on how the two groups are numbered relative to each other. Pooled own wrong area:
  - band 2: arm20 43.95, arm21 45.13 (+1.2), arm22 34.65;
  - band 1: arm0 43.28, arm23 33.55 (-9.7), arm2 31.15.
- "Verified steps": chain steps whose two points both match a labelled human crossing on the same ray (4 vox, unique
  by 1 vox, as in `auto_windings.py compare`). A fit's step is right if its winding difference between the two points
  equals the label difference.

Scripts and outputs are in data/scratch/band2_diag/:

| script | what it does | output |
|---|---|---|
| chain_lookup.py | fit winding at every chain point and oracle crossing, per arm (~70 s per arm) | lookup_<arm>.npz |
| vertex_loc.py | wrong area / switches by angle, z, winding and class | vertex_loc_band{1,2}.{md,json} |
| chain_diag.py | step tables, chain geometry, gap count, ray pairs, vertex classes, cells | chain_diag_band{1,2}.json, cells_band{1,2}.npz |
| pairs_diag.py, site_moderator.py | Q4 pair breakdowns | pairs_diag.json |
| gm_thr.py | stored gm_p re-thresholded (no refit) | gm_thr.json |
| cells_x.py | cross-band predictors | cells_x.json |

## Verdict
- **The failure is mostly on the chain side, and it is the gm raise rule.**
  - Plain dr = 1 steps are as good in band 2 as in band 1: 93.3 vs 92.4 % precise, and net +57 vs +48 per 1000 verified
    steps for the fit.
  - The gm-raised steps cancel that gain. In band 2, 25 % of single steps are raised (band 1: 18 %), and only 21.5 % of
    the verified raises are right. Net effect: -55 per 1000 (band 1: -31).
  - The fit copies the chain at 91 % of steps, including 90 % of the steps where the baseline disagrees with the chain.
  - So arm21 inherits the raises' noise:
    - 2.2x the count spread across the long w034-w051 span (sd 2.43 vs 1.09);
    - +31 switches per 1000, spread evenly over all angles, z and windings;
    - extra over-counting at pair sites.
- **The baseline also differs, and band 2 offers less to gain.**
  - arm0's pair errors were mostly over-counts (17.1 % over, 4.9 % under). High-precision dr = 1 steps fix those.
  - arm20's errors are mostly under-counts at contacts (11.0 % over, 15.7 % under). Only raises can fix under-counts.
  - The oracle-arm headroom on inner own wrong area is 16.8 pp in band 2 vs 26.6 in band 1.
- **The gm threshold over-raises in both bands.** It was set for zero drift against label counts, and the CT audit
  says those counts are 1-2 turns too high.
  - At the frozen threshold 0.18, the chains over-count by +0.5 to +1.2 turns per 10-20 windings against CT-corrected
    labels, in both bands.
  - Over 3-6 windings, where the labels are mostly right, band 2 over-counts by +0.42 (band 1: +0.12).
- **No credible fix is ready.**
  - One testable hypothesis: raise the gm threshold so that drift is zero against CT-corrected counts (about 0.24-0.34
    on band 1). Check it with two dev fits on bands 1 and 2, which are both spent and therefore usable as dev data,
    before any confirmation run.
  - Its prior is modest. Round 2's arm8 (no raises at all) wrecked the global count, and band 2's oracle shows almost no
    pair headroom.
  - The label-free predictors found in Q3 rank cells consistently in both bands, but used as a gate they recover only
    ~1.7 pp in band 2 and cost band 1 a third of its gain.

## Q1. Where arm21's extra switches and wrong area come from

**Switches** (the scorer's seam residual, B - A per 1000 adjacent pairs). Band 2 is uniform; band 1 is concentrated.

| split | band 2 arm21 - arm20 | band 1 arm23 - arm0 |
|---|---|---|
| 30-deg sectors (min / max) | +15.7 / +48.7, all 12 positive | +0.4 / +56.6; 66 % of the excess in 150-300 deg |
| z 250 slabs (min / max) | +23.8 / +37.5 | +8.9 / +22.6 |
| inner k < 41 / outer k >= 41 | +32.0 / +29.7 | +5.7 / +22.8 |
| contact / open | +34.0 / +30.2 | +13.4 / +15.2 |
| #2167 / pseudo-gap flagged / unflagged | +34.1 / +30.2 | +29.9 / +10.0 |
| windings with a rise | 31 of 33 (w056 +3.0, w057 -5.1) | |
| arm22 (oracle) / arm2 (oracle) total | +7.9 | +8.2 |

- The chain-induced excess (chain arm minus oracle arm) is +23 per 1000 in band 2 and +7 in band 1.

**Own wrong area, B - A, pp (band 2).**
- By angle:
  - worse at 30-240 deg: +3.0 to +8.0 in every sector;
  - better at 240-30 deg: -0.4 to -7.9.
  - The angle profile has the same shape as in band 1 (cell-profile correlation 0.64). In both bands chain precision is
    lowest at 75-165 deg: 58-75 % in band 2.
- By z: +2.2 / +4.1 / +2.1 / -0.4 / -1.0 for the five 250-z slabs from 1750 up. z < 2600, where w035-w050 are
  unlabelled: +2.5; z >= 2600: -1.0.
- By winding:
  - worse: the edges of the labelled range, w023-w025 (+5.7 to +10.1) and w058/w059 (+13.7 / +31.1), plus w031
    (+8.9), w033 (+10.8) and w055 (+9.7);
  - better: w034 (-12.1), w038-w040 (-12.5 to -34.0), w051 (-18.7) and w057 (-8.8).
  - At w023-w025 the oracle gains -48 to -65 where arm21 loses, so arm20 has an inner-edge counting failure there that
    the auto chains do not repair.
- Contact / open: +2.1 / +1.0. Flagged / unflagged: -0.3 / +1.5. So neither contacts nor the CT-flagged zones carry the
  loss.

**Relation to the chains.** Each human vertex is assigned to its nearest ray (64 z, 15 deg) and to the nearest chain
point by radius. Columns are own wrong area A -> B and scorer switches per 1000 A -> B.

| vertex class | band 2: share | own wrong | switches | band 1: share | own wrong | switches |
|---|---|---|---|---|---|---|
| within 4 vox of a chain point | 37.7 | 43.1 -> 45.2 | 61.6 -> 97.7 | 38.6 | 42.0 -> 34.5 | 50.0 -> 74.1 |
| ... adjacent steps verified right | 13.7 | 36.0 -> 37.6 (+1.6) | 44.7 -> 85.2 | 18.6 | 36.9 -> 23.8 (-13.1) | 35.5 -> 60.9 |
| ... an adjacent step wrong | 5.9 | 43.0 -> 46.4 (+3.4) | 70.3 -> 120.5 | 7.1 | 47.2 -> 40.5 (-6.7) | 62.7 -> 90.9 |
| ... adjacent steps unverifiable (gap etc.) | 18.1 | 48.9 -> 50.8 (+1.9) | 72.6 -> 99.7 | 13.0 | 46.9 -> 47.0 (+0.1) | 65.4 -> 84.4 |
| ... next to a gm-raised step | 13.2 | 47.1 -> 50.5 (+3.4) | 85.9 -> 125.2 | 10.8 | 48.0 -> 41.9 (-6.1) | 77.7 -> 102.3 |
| ... no adjacent raised step | 24.5 | 41.0 -> 42.3 (+1.3) | 48.7 -> 82.4 | 27.8 | 39.8 -> 31.6 (-8.2) | 39.6 -> 63.2 |
| inside a chain's radial span, between points | 61.8 | 44.4 -> 45.0 | 60.3 -> 87.7 | 60.8 | 44.0 -> 32.8 | 46.9 -> 56.7 |
| outside every chain span | 0.5 | 60.5 -> 62.1 | | 0.5 | 61.9 -> 48.1 | |

- No region is "away from chains": 99.5 % of vertices lie inside some chain's radial span in both bands.
- Losses are larger next to wrong or raised steps (+3.4) than next to verified-right ones (+1.6), but they are
  positive everywhere.
- Band 1 improved even next to wrong steps (-6.7). So the outcome is a band-wide property of the chain set and the
  fit, not a local copy of individual bad steps.
- Switches rise by +34 to +50 per 1000 in every class, including next to verified-right steps.

**Step-level mechanism** (verified steps; A and B both evaluable).

| cell | band 2: share | B right | band 1: share | B right |
|---|---|---|---|---|
| chain right, A right | 66.7 | 93.7 | 70.8 | 93.0 |
| chain right, A wrong (useful) | 13.9 | 93.3 | 13.4 | 93.9 |
| chain wrong, A right (harmful) | 9.9 | 6.3 | 6.7 | 7.1 |
| chain wrong, A wrong | 9.5 | 4.5 | 9.1 | 2.6 |
| fit step accuracy B - A | 0.0 [-2.7, +2.7] | | +1.7 [-0.2, +3.7] | |

- B follows the chain 91.3 % of the time (band 1: 91.5 %).
- Where A disagrees with the chain, B takes the chain's value 89.8 % of the time and keeps A's 6.1 %.
- When chain and A disagree, the chain is right 55.7 % of the time and A 39.4 % (band 1: 62.5 % / 30.6 %).

## Q2. What differs from band 1

**Step precision is not the difference; raised steps are.** Effect of each chain step type on fit step accuracy,
per 1000 verified steps.

| step type | band 2: share | precision | B - A | band 1: share | precision | B - A |
|---|---|---|---|---|---|---|
| dr = 1 | 80.2 | 93.3 | **+57.2** | 85.1 | 92.4 | **+47.7** |
| gm-raised 1 -> 2 | 15.5 | 21.5 | **-54.9** | 10.9 | 28.7 | **-31.1** |
| dr = 2 before gm | 4.4 | 55.8 | -2.7 | 3.9 | 61.5 | +0.7 |
| total | | 80.0 | -0.4 | | 84.2 | +17.3 |

Counterfactual swaps (per 1000; B follows the chain ~90 % of the time, so per-cell responses transfer):

| scenario | band-2 total |
|---|---|
| observed | -0.4 |
| band-1 raised share (10.9 %), band-2 raised precision | +15.9 |
| band-2 share, band-1 raised precision | +10.5 |
| band-1 raised contribution (-31.1) | +23.4 (above band 1) |

- Share and precision explain about half each.
- The 80 vs 84 % headline precision gap is entirely the raised steps.
- The CT relabels change precision by at most 0.4 pp (minus1 80.3, minus2 80.3; band 1 supported 84.9), so label
  errors do not explain the 80 % either.

**By winding range** (verified steps; A ok / B ok are fit step accuracy):

| lower winding | band 2: chain precision | raised % | A ok | B ok | band 1: chain precision | raised % | A ok | B ok |
|---|---|---|---|---|---|---|---|---|
| w023-w024 | 93.4 | 5.8 | 76.8 | 90.3 | 88.7 | 9.5 | 92.4 | 85.9 |
| w025-w026 | 86.4 | 9.8 | 66.5 | 81.7 | 87.0 | 10.9 | 87.7 | 82.5 |
| **w027-w032** | **71.8** | **25.2** | **84.4** | **69.6** | 87.8 | 10.2 | 62.9 | 84.5 |
| w033-w039 | 80.5 | 12.6 | 80.5 | 79.9 | 82.7 | 12.8 | 65.9 | 79.3 |
| w051-w059 | 80.1 | 16.3 | 75.6 | 72.6 | 85.8 | 11.7 | 88.6 | 78.1 |

- Band 1's gain sits where arm0 over-counted: w027-w039, A over-count 22-34 %.
- In band 2 the same windings (w027-w032) get 2.5x the raises at 20.5 % precision, exactly where arm20 was already
  right. Band 2's grad_mag pitch ratio is also higher (1.51 vs 1.31, sanity check in plan section 7). A plausible reading
  is that the gm rule, fitted at z 4500-6000, does not transfer to the more compressed low-z inner turns.

**Baseline (arm20 vs arm0).** Matched-pair error direction, % of matched pairs:

| arm | over-count | under-count | accuracy |
|---|---|---|---|
| arm0 | 17.1 | 4.9 | 78.0 |
| arm23 | 3.5 | 3.4 | 93.0 |
| arm2 | 5.0 | 2.8 | 92.1 |
| arm20 | 11.0 | 15.7 | 73.3 |
| arm21 | 20.1 | 10.9 | 68.9 |
| arm22 | 16.7 | 11.6 | 71.7 |

- In band 1, the share of pairs a site's baseline over-counts predicts the chain arm's pair gain at that site:
  Spearman +0.84 over 28 sites (oracle arm +0.76).
- In band 2 it does not (-0.04, 21 sites): the 5 sites where arm20 over-counts more than it under-counts still lose
  -5.9 pp.
- So the baseline explains part of the gap. The rest is band 2's chains adding over-counts everywhere.
- At the chain steps themselves the baselines look alike: A wrong 23.5 vs 22.6 %, over 16.9 vs 15.6 %.

**Chain geometry: not the cause.**

| | band 2 | band 1 |
|---|---|---|
| chains / points | 597 / 31,223 | 664 / 41,322 |
| median length (p10-p90) | 59 (13-76) | 65 (24-91) |
| median radial span, vox | 1,390 | 1,621 |
| rays with a chain | 480 / 480 | 552 / 552 |
| points per ray (median) | 66 | 74 |
| vertices inside a chain span | 99.5 % | 99.5 % |

- Points per ray predicts the outcome in opposite directions in the two bands (Q3).

**Sparser labels/oracle: affects the measurement, not arm21's inputs.**
- Only 13.8 % of band-2 steps are verifiable (band 1: 24.1 %); oracle points are 9.3k vs 17.8k.
- Vertices next to unverifiable steps lose about the same as those next to verified-right ones (+1.9 vs +1.6), so there
  is no sign of a large hidden error mass in the gap.
- The oracle arm, though, is not an oracle across the gap (Q4).

**Long-span count across the CT-flagged gap** (turns each fit puts between the two bracketing labels on each ray):

| | band 2, w034 -> w051 | band 1, w033 -> w052 |
|---|---|---|
| labels / CT | 17 / 14.6-15.4 | 19 / 17.1-17.9 |
| base | 16.27, sd 1.09 (170 rays) | 18.01, sd 0.71 |
| chain arm | 16.23, **sd 2.43** (302 rays) | 18.37, sd 1.81 |
| oracle arm | 17.11, sd 0.56 (copies the labels) | 19.56, sd 0.99 |
| auto chains themselves | 16.18, sd 2.32 | 18.21, sd 1.78 |

**gm calibration** (stored gm_p re-thresholded, nothing refitted). Drift = mean chain count minus label count, per
chain pair.

| band | thr | raised % | precision (orig) | drift 10-20 orig | drift 10-20 CT-corrected | drift 3-6 orig |
|---|---|---|---|---|---|---|
| 1 | 0.18 (frozen) | 18.4 | 84.2 | -0.08 | +0.57 (supported) / +1.03 (outer_minus2) | +0.12 |
| 1 | 0.26 | 10.6 | 87.0 | -0.76 | -0.13 / +0.31 | -0.08 |
| 1 | 0.30 | 8.3 | 87.5 | -0.96 | -0.34 / +0.09 | -0.14 |
| 1 | none | 0 | 88.1 | -1.55 | -0.95 / -0.54 | -0.31 |
| 2 | 0.18 (frozen) | 25.2 | 79.9 | -0.18 | +0.48 (minus1) / +1.21 (minus2) | **+0.42** |
| 2 | 0.26 | 15.2 | 85.7 | -1.08 | -0.48 / +0.19 | +0.12 |
| 2 | 0.30 | 12.2 | 86.7 | -1.26 | -0.68 / -0.02 | +0.04 |
| 2 | none | 0 | 88.2 | -2.05 | -1.54 / -0.95 | -0.21 |

- The "zero drift" of 0.18 is zero only against the human numbering, which over-counts by 1-2 turns (w041 = w042, and
  w040 -> w043 is 2 turns).
- Against CT-corrected counts both bands over-raise. Band 2 over-counts even over 3-6 windings.
- Raised steps are rarely needed: only 4-6 % of verified dr = 1 steps truly span 2 turns, yet 18-25 % are raised.
- Keeping a raise only where the baseline also has 2 does not rescue them: those raises are 29 % right in band 2 and 43 %
  in band 1.

## Q3. Label-free, pre-fit predictors of help vs hurt

Cells are 15-deg rays x 250 z (band 2: 120 cells, band 1: 144). The outcome is the own wrong area B - A per cell.
Spearman rho, raw and within-theta (each cell minus its ray's mean over z).

| predictor (pre-fit, no labels) | band 1 raw | band 1 within-theta | band 2 raw | band 2 within-theta | consistent? |
|---|---|---|---|---|---|
| raised share of single steps | +0.31 | +0.21 | +0.26 | +0.30 | yes (more raises, more harm) |
| mean gm_p | +0.38 | +0.25 | +0.25 | +0.30 | yes (same signal) |
| chain-vs-baseline step disagreement | +0.32 | +0.14 | +0.30 | +0.25 | yes |
| chain points per ray (track density) | +0.48 | -0.24 | -0.04 | -0.03 | no |
| baseline-unmatched chain points | -0.09 | -0.04 | -0.34 | -0.19 | weak |
| chain count / baseline count | +0.24 | -0.15 | -0.21 | -0.10 | no |
| mean step p_err | -0.12 | +0.11 | -0.02 | +0.13 | no |
| chain precision (label-based, for reference) | -0.25 | -0.24 | -0.36 | -0.22 | yes |

- Switch rise vs disagreement / raised share: band 2 +0.42 / +0.39; band 1 +0.13 / +0.10. The within-theta values are
  +0.33 to +0.42 in both bands for disagreement.
- Angle explains 81 % (band 1) and 53 % (band 2) of the cell variance, so these cells are not independent units.
- Rule fixed on band 1 alone: cells with raised share below the band-1 median (0.167) count as "chains help". Applied
  unchanged (cell-level splice, which ignores that fits couple across cells):

| | band 1 | band 2 |
|---|---|---|
| vertices in "help" cells | 49 % | 17 % |
| B - A, help cells / other cells | -13.3 / -7.0 | -3.5 / +1.5 |
| pooled own B - A: chains everywhere -> rule | -9.7 -> -6.4 | +1.2 -> -0.5 |

- The disagreement and gm_p rules are about the same (band 2: +1.2 -> -0.5 / -0.3).
- So raised share and disagreement are consistent rankers, but as a gate they turn band 2 from a loss into a wash and
  give up a third of band 1's gain.
- **This rule was found after looking at band 2.** Confirming it needs a third untouched region, and a real gated fit
  rather than a cell splice.

## Q4. Can band 2's pair benchmark show gains? (oracle -1.6)

It can show changes (the arm21 loss is significant on the CT-clean set: -5.0 [-9.3, -0.7]). But this band offers
almost no gain of the kind chains deliver, and its "oracle" is not one.

| pair group | n | arm20 | arm21 - arm20 | arm22 - arm20 | band-1 arm2 - arm0 |
|---|---|---|---|---|---|
| all (21 sites; band 1 33) | 5,872 | 73.3 | -4.4 [-9.9, +1.3] | -1.6 [-5.9, +3.5] | +14.1 [+8.0, +21.2] |
| dk = 0 | 1,956 | 89.0 | -5.5 | -2.2 | +0.8 |
| dk = 1 | 1,960 | 77.5 | **-13.2** [-21.0, -5.9] | **-8.6** [-15.0, -2.9] | +17.5 |
| dk = 2 | 1,494 | 53.1 | +8.5 [-5.6, +23.9] | +9.2 [-4.5, +23.8] | +25.4 |
| dk >= 3 | 462 | 35.4 | +9.2 | +5.9 | +21.6 |
| not touching w033-w037 | 2,833 | 84.3 | -11.4* | -6.9* | +19.3* |
| CT-flagged excluded | 5,327 | 74.0 | -5.0* | -2.3 | (none flagged) |

Why there is little to gain:
1. **Composition.** 90 % of pairs have both ends at k < 41, and in z < 2600 that means w023-w034 / w039. 52 % touch
   w033-w037.
   - Off w033-w037, arm20 already scores 84.3 % (band 1 arm0: 76.6 %).
   - arm20's errors are under-counts at dk >= 2 (dk = 2: 34.7 % under; dk = 3: 56.5 %), i.e. merged sheets at
     contacts.
   - Neither arm fixes those without adding over-counts.
2. **The oracle chains are wrong across the gap.**
   - They hold 9.3k points and have no labels for w035-w050 below z ~2600.
   - They impose the labelled 17 turns across w034 -> w051 (arm22 gap count 17.11, sd 0.56), where CT counts 14.6-15.4.
   - arm22 has to fit about 2 extra turns somewhere, and its over-count rises from 11.0 to 16.7 %. It loses dk = 1
     pairs at w030-w033: 67 -> 51 % at lo = w030, 83 -> 70 % at lo = w033.
   - So -1.6 is the ceiling of a label-contaminated oracle, not of chains in general.
3. **Power.** 21 sites give a 95 % half-width of ~5-6 pp; the 12 sites at z 6000-8250 would give ~7-8 pp.
   - A dense ray benchmark (7,411 consecutive labelled dk = 1 crossings on all 480 rays) agrees for arm21: -4.8
     [-7.1, -2.6].
   - It cannot score the oracle arm: those crossings are the oracle's own inputs (99 %).

## What a fair third-region test would need

**Before any fit:**
- Freeze the fix on bands 1 and 2, both now dev. Candidate: gm threshold set for zero drift against CT-corrected counts.
  Band 1 alone gives 0.24-0.34; band 2 agrees descriptively (0.21-0.34).
- Dev-fit it on both bands, about $3-7 per arm. Continue only if:
  - band 1 keeps most of arm23's gain;
  - band 2 is no longer worse on own wrong area, switches and CT-clean pairs.
- CT-audit the new region's numbering (w041 = w042 and the w046/47 gap are mesh-level, so they exist at every z).
- Rebuild the leak-free models for that region, as for band 2:
  - the clean conf model was fitted on the 33 sites at z >= 4500, which include all 12 sites at z 6000-8250;
  - the gm rule saw z 4500-6000.

**Metrics:**
- Primary: own wrong area (inner and outer, consistent subset), plus a switch non-inferiority margin. Both are vertex
  based and well powered.
- Pairs secondary (CT-flagged excluded) unless the region has >= 20 sites.
- Pre-register an oracle-headroom condition: if the oracle arm fails the primary criterion, the region is declared
  uninformative, not a pass or a fail.
- Optionally pre-register the moderators as predictions: baseline over-count share (needs labels and the baseline only)
  and raised share (label-free).

**Region options:**

| option | sites / mix pairs | vertices | leaks | verdict |
|---|---|---|---|---|
| PHerc0139 z 6000-8250 | 12 / 3,435, in two clusters (z 6016-6528: 5; z 7968-8288: 7) | ~600k per 250 z, 33 windings at 360 deg to z 8000; #2167 gap ~65 deg | conf model saw the 12 sites (refit without them) | **best untouched in-scroll option**, vertex-primary |
| PHerc0139 z 4500-6000 | 21 / 6,027 | full | old gm rule and conf model fitted there (no spiral fit ever scored there) | usable only if gm/conf are refitted on bands 1+2; disclose the design-level leak |
| PHerc0139 z 7000-8500 (plan fallback) | 7 / 1,995 | coverage drops above z 8250 | as z 6000-8250 | subset of the first option |
| PHerc1667 / PHerc0172 (7.91 um human meshes) | new scorer needed | | none for these chains | cleanest; 1-2 days of engineering (no bucket m7 at our scan level, no tracks, no 9.362 frame) |

Recommendation:
1. Do not spend on a confirmation until the gm-threshold dev fits on bands 1 and 2 show both bands non-negative on the
   label-robust metrics.
2. If they do, pre-register z 6000-8250 (vertex-primary), with z 4500-6000 as a secondary region after the gm/conf
   refits.
3. Treat a cross-scroll test as the real confirmation for a prize claim.
