# 2026-10-22: counting sheets on chain steps in the 2.399 um scan (PHerc0139)

Hypothesis: the auto chains undercount where two sheets touch. At 2.399 um (4x finer than the 9.362 um masks) touching
sheets might separate, so counting sheets along each chain step in the 2.4 um CT could fix the undercounts and decide
gm raises better than grad_mag. Local CPU only, no GPU, no pods, $0. About 18 GB streamed as byte ranges; nothing
stored except profiles (1.2 GB in data/scratch/hires_step/).

## 0. Verdict: no-go
- **The 2.4 um count cannot tell a missed sheet from a correct single step.** Among plain one-turn steps on band 1
  (z 3000-4500), the count says ">= 2 sheets" on 17.1 % of correct steps and 17.0 % of oracle undercounts (16.7 % where a
  human mesh actually crosses between the two points). It has no discriminating power there.
- **It decides raise vs no-raise worse than gm.** At the same number of raises as the gm rule on band 1 (d0 = 1 steps, truth 1
  or 2):
  - gm_p: precision 32.1 %, recall 39.8 %, false raises 7.9 % of singles, AUC 0.755;
  - the 2.4 um model alone: 10.1 % / 12.5 % / 10.4 %, AUC 0.573. The base rate is 8.5 %.
  - gm + 2.4 um: 29.1 % / 36.1 % / 8.2 %, AUC 0.726, which is worse than gm alone on band 1.
  - Against CT-corrected labels: gm 31.9 / 43.5 % (AUC 0.779); 2.4 um alone 9.7 / 13.2 % (AUC 0.593).
- **No 2.4 um policy reaches > 92 % precision without drift.** (b) Replacing the count gives 62 %. (c) The 2.4 um veto of gm
  raises gives 87.8 %, but simply keeping the same number of top gm_p raises does better: 88.2 % with the same drift.
- **Calibration (z 4500-6000) says the same.** Adding 2.4 um features to gm moves slab-out-of-fold AUC from 0.810 to 0.816,
  the same gain as adding 9.362 um CT features (0.817).
- **Why.** At 2.4 um each papyrus sheet shows its fibre strands, as several bright bands with dark lines between them. Those
  intra-sheet valleys are as deep as the contact line between two touching sheets, so a profile counter overcounts open
  sheets as often as it finds hidden ones. Also, many of the oracle "undercounts" are label artefacts the CT cannot fix
  (section 5).
- **Side finding, relevant to the next fit round.** Against the CT-corrected labels (relabel_band1 `supported`, outside the
  #2167 sector), the gmx chains drift **+0.74** windings over 10-20 (they over-raise), and raw d0 drifts -0.73.
  - A stricter gm threshold (gm_p >= 0.46, about 1/4 of the raises) gives 89.8 % step precision with drift -0.46, against
    86.2 % / +0.74 for gmx.
  - That raise count was matched to the 2.4 um veto, not tuned, but it was read on band 1. Re-pick the gm threshold on
    z 4500-6000 against CT-corrected labels before using it (section 4c).

## 1. Access and transform (task 1)
- **Volume.** `s3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr`.
  - zarr v2 with levels 0-5. Level 0 is 76953 x 26511 x 26511 uint8 in 128^3 chunks, **uncompressed**, keys `0/cz/cy/cx`.
  - Anonymous access (no auth).
  - The 9.362 um volume (20250728140407) uses the same layout.
- **Byte-range reads.** Because the chunks are uncompressed, each z slice of a chunk is a contiguous 16 KB, so a horizontal ray
  tube needs only a few slices per chunk.
  - About 0.3 s latency per request and about 65 requests/s per process (s3fs `cat_ranges`).
  - A full 128^3 chunk (2.1 MB) also takes 0.3 s.
- **Transform 9.362 -> 2.399.** Inverse of the published affine `transform_0139_2p399_to_9p362.json`
  (p9_xyz = A p2399_xyz + t; scale 0.257, ~177 deg proper rotation, z nearly decoupled). A point at constant 9.362 z
  drifts ~1.1 vox in 2.399 z per 100 vox in x. Round trip error is 1e-13.
- **Local residual of the published affine.** It is fixed per step by a 1-D NCC lag between the 9.362 profile and the
  2.399 profile blurred to ~9 um, searched over +-5 vox.
  - Lag p5/p50/p95: -2.0 / -0.5 / +1.25 vox (9.362).
  - NCC median 0.95 (p10 0.85).
  - This matches the 0.8-2.3 vox offsets found earlier with 3-D NCC (notes/2026-10-18-p0139-testset.md).
- **Spot check** (data/scratch/hires_step/figs/spot_check.png, script spot_check.py). Human meshes w029 (z 3424, 60 deg),
  w052 (z 3744, 255 deg) and w037 (z 3936, 210 deg), with their neighbours, are drawn from the native 2.399 tifxyz onto 2.399
  L0 slices. All three lie on a papyrus sheet. Mapped with the published affine onto the 9.362 slice, they land on the
  same sheets with a 1-2 vox offset.
- **Coverage** (task 4; coverage.py, coverage.json). At z 1750, 2500, 3000, 4500, 6000, 7500 and 8500 (8 rays each), 99.1-100 %
  of in-scan 9.362 ray samples are also inside the 2.399 scan. **The 2.4 um volume covers z 1750-8500.** Steps with
  profiles: 40,328 / 40,658 on band 1 and 42,427 / 42,747 on z 4500-6000.
- **Reusable helper: `tools/hires.py`.** It holds `to24` / `to9`, `umb`, `ray_xyz9` and `SlabReader`, with `R24` = 2.399 L0
  and `R9` = 9.362 L0 trilinear `sample()`. It is meant for the viewer agent (tools/check_viewer.py) and anyone else
  streaming 2.399 crops.

## 2. Method (tools/hires_step_count.py, tools/hires_step_eval.py)
- **Profiles** (`hires_step_count.py profile --band z3000|z4500`).
  - Every probe ray of the band's chains is sampled every 0.125 vox (9.362) from the first chain point -24 vox to the last
    +24 vox.
  - 2.399 L0: a 7 x 7 tube of parallel lines, 1 vox (2.4 um) apart, along the 2.399 z axis and the in-plane tangent.
  - 9.362 L0: a 3 x 3 tube.
  - Output: data/scratch/hires_step/prof_<band>/*.npz (resumable).
- **Per step** (`hires_step_eval.py feats`):
  1. Registration lag (section 1).
  2. Tube mean (3x3 or 7x7), Gaussian smoothing (sigma 0.5 / 1 / 2 vox at 2.4 um), and normalisation per window
     (p5 -> 0, p95 -> 1).
  3. Valleys strictly between the two registered chain points, skipping delta = 1.5 or 3 vox at each end. A valley's
     prominence is measured against the higher of its two bases (scipy find_peaks).
  4. Count = number of valleys with prominence >= tau, where valleys closer than `mind` count once. A step crossing d
     windings should show d gaps.
  - The same counter is run on the 9.362 profile as a control.
- **Calibration, z 4500-6000 only** (`calib`).
  - Data: band_m7_z4500_ray chains + band_oracle_z4500, gm_p out of fold by slab as in stepconf.
  - "Clean" calibration steps exclude windings w033-w037 and w040-w047, which are label-suspect: 8,151 matched steps.
  - Grid: tube x sigma x delta x tau (0.05-0.7) x mind (0 / 2 / 4) for the replacement counter and the veto.
  - Logistic raise models on d0 = 1 steps (gm only; gm + 2.4 um; gm + 9.362 um; 2.4 um only; 9.362 um only), with 3-fold
    slab-out-of-fold AUC.
  - Thresholds: zero bias (raises = true doubles) or F1-max, both on the OOF scores.
- **Frozen picks.**
  - Replacement: 2.4 um 7x7 tube, sigma 2 vox, delta 1.5, tau 0.5, mind 4. Clean accuracy 63.1 %; the 9.362 counter's best
    is 63.9 %.
  - Veto: same tube, tau 0.7, mind 0.
- **Also tried and dropped: a sheet-aligned 2-D profile** (proto_shear.py).
  - Method: a radial x tangential slice of +-6 vox (9.362), 3 z slices, averaged along the best-focus shear (the local sheet
    direction).
  - Tested on 360 z4500 steps. 2nd-valley AUC for single vs double-with-crossing was 0.60-0.62, against 0.59-0.63 for the
    centre line, so no gain.
- **Leakage.**
  - No band 1 or band 2 label was used to fit or threshold anything. Band 1 labels were used only to score.
  - The gm-matched control in section 4c used band 1 unlabelled raise counts only.
  - The CT relabel map comes from the band-1 CT audit; it was used only for scoring.

## 3. Calibration results (z 4500-6000, data/scratch/hires_step/calib_z4500.json)
- Raised steps (623 clean): as is 31.6 % right; reverting all of them 64.4 %; best 2.4 um veto 62.3 %. The veto never beats
  reverting everything.
- Raise models (clean d0 = 1 steps, slab-OOF AUC):

| model | all doubles (742 of 11.4k / clean 397 of 7.6k*) | doubles with a mesh crossing between only (clean) | top-50 precision (crossing set) |
|---|---|---|---|
| gm | 0.710 | 0.810 | 0.60 |
| gm + 2.4 um | 0.744 | 0.816 | 0.68 |
| gm + 9.362 um CT | 0.748 | 0.817 | 0.64 |
| 2.4 um only | 0.658 | 0.654 | 0.18 |
| 9.362 um CT only | 0.573 | 0.578 | 0.08 |

\* The first column is the clean set with label-gap doubles kept (calib stage). The second drops doubles with no mesh crossing
between the two points, which the CT cannot see by construction (calib_xbetween.py).

## 4. Band 1 (z 3000-4500) against band_oracle (data/scratch/hires_step/eval_z3000.json, eval_z3000.txt)
All 40,658 steps were profiled (not a sample). 9,794 are matched to the oracle: 7,726 plain d = 1 correct, 471 plain d = 1
undercounts (324 with a mesh crossing between), 973 gm raises (312 right / 661 wrong), 340 graph d0 = 2 and 226 with
truth >= 3. "Supported" = relabel_band1.json `supported` map (CT-majority: w041 = w042, k >= 43 one lower).

### 4a. Confusion: chain stratum x 2.4 um count (raw labels; supported labels within 1-2 points)
| chain step | n | count 0 | count 1 | count 2 | count 3+ | share >= 2 |
|---|---|---|---|---|---|---|
| d = 1 (not raised), truth 1 | 7,726 | 1,088 | 5,315 | 1,189 | 134 | 17.1 % |
| d = 1 (not raised), truth 2 (undercount) | 471 | 67 | 324 | 71 | 9 | 17.0 % |
| ... of which a mesh crosses between | 324 | 42 | 228 | 48 | 6 | 16.7 % |
| gm-raised 1 -> 2, truth 2 (right) | 312 | 25 | 143 | 125 | 19 | 46.2 % |
| gm-raised 1 -> 2, truth 1 (wrong) | 661 | 77 | 364 | 175 | 45 | 33.3 % |
| graph d0 = 2, truth 2 | 243 | 14 | 83 | 106 | 40 | 60.1 % |
| graph d0 = 2, truth 1 | 97 | 4 | 52 | 33 | 8 | 42.3 % |
| any, truth >= 3 | 226 | 24 | 97 | 75 | 30 | 46.5 % |

- On plain steps the count is blind: an undercount and a correct single look the same.
- On gm raises it carries a weak signal (46 vs 33 %), already contained in gm_p.
- 14 % of correct single steps get count 0: no valley between two chain points that sit on different sheets. This is a
  measure of how noisy the profile counter is.

### 4b. Step precision and drift (all steps; steps without a profile keep the chain value)
| policy | raw labels: precision / under / over | raw: drift 10-20 (within +-1) | supported, outside #2167: precision | supported, outside #2167: drift (within +-1) |
|---|---|---|---|---|
| (a0) raw chain d0 (no raises) | 88.1 / 10.3 / 1.6 | -1.55 (56 %) | 89.6 | -0.73 (78 %) |
| (a) gmx chain d, as is | 84.6 / 7.1 / 8.3 | -0.09 (73 %) | 86.2 | **+0.74** (74 %) |
| (b) 2.4 um count replaces d | 61.9 / 20.7 / 17.4 | -0.98 (39 %) | 62.9 | +0.10 (43 %) |
| (c) 2.4 um veto: keep a gm raise only if count >= 2 | 87.8 / 9.4 / 2.9 | -1.23 (63 %) | 89.1 | -0.39 (81 %) |
| control: same number of raises, top gm_p | **88.2** | -1.25 (63 %) | **89.8** | -0.46 (82 %) |
| control: same number of raises, random | 87.2 | -1.17 (65 %) | 88.8 | -0.35 (82 %) |
| (c') gm + 2.4 um raise model, F1 threshold | 86.0 | -0.54 (71 %) | 87.1 | +0.38 (77 %) |
| (c') gm-only raise model, F1 threshold | 84.6 | -0.12 (74 %) | 86.2 | +0.71 (74 %) |
| (c') gm + 9.362 um raise model, F1 threshold | 85.1 | -0.29 (70 %) | 86.0 | +0.66 (72 %) |

Raw labels outside #2167 and supported labels on all angles are in eval_z3000.txt; they rank the policies the same way.

### 4c. Raise vs no-raise (key read-out). d0 = 1 matched steps, truth 1 or 2, top-K at K = number of gm raises
| score | raw: AUC | raw: precision / recall | raw: false raises (% of singles) | supported: AUC | supported: precision / recall |
|---|---|---|---|---|---|
| gm_p (stored rule) | 0.755 | 32.1 / 39.8 | 7.9 | 0.779 | 31.9 / 43.5 |
| gm + 2.4 um model | 0.726 | 29.1 / 36.1 | 8.2 | 0.756 | 29.0 / 39.6 |
| gm + 9.362 um model | 0.732 | 29.8 / 37.0 | 8.1 | 0.761 | 29.5 / 40.3 |
| 2.4 um only | 0.573 | 10.1 / 12.5 | 10.4 | 0.593 | 9.7 / 13.2 |
| 9.362 um CT only | 0.602 | 15.7 / 19.5 | 9.8 | 0.613 | 15.2 / 20.7 |

K = 973 (raw) / 982 (supported). The pool holds 783 / 719 true doubles and 8,387 / 8,450 singles.

**Stricter gm threshold** (gm_matched.py, gm_matched_z3000.json). gm_p >= 0.46 keeps 1,796 of the gm raises over all steps
(250 matched). Supported labels outside #2167: 89.8 % precision, drift -0.46, within +-1 82 %; on all angles 88.9 % / -0.64.
gmx's +0.74 drift against the CT-corrected labels confirms the coordinator's diagnosis: the 0.19 threshold was tuned against
oracle labels that overcount, so it over-raises.

## 5. Why the undercounts don't respond: visual check (figs/under_z4500*.png, look_steps.py)
I looked at 12 z 4500-6000 one-turn "undercounts" on 2.399 and 9.362 slices: 6 random, and 6 short ones that gm missed.
- **At least 7 are label artefacts:**
  - the w041/w042 duplicate (2);
  - the w034/w036 duplicate in the #2167 sector (2);
  - w048 -> w050 or w049 -> w051 with nothing between (2);
  - a w054 crossing in an air gap (1).
- **About 2 look like real touching sheets**, but the 2.399 slice shows a fibre-textured bundle, not two clean bands.
- In the rest, the 2.399 sheet itself is several strands, so any "gap" counter sees extra valleys inside single sheets.
- The CT-supported relabel cuts the band-1 plain undercounts from 471 to 406, and it cannot fix the #2167 or w048-w051 cases.

## 6. Throughput and cost (task 4)
- **One band** (552-576 rays, all ~40k steps): 13-14 min wall with 6 threads.
  - Streaming: 7.6-8.1 GB of 2.399 byte ranges (47-50k requests) plus 1.1 GB of 9.362 (11-12k requests).
  - CPU is negligible. Step features take ~1 min.
  - A sample is not needed: the whole band is cheaper than building a stratified sample.
- **Total this session:** ~18 GB streamed, 1.2 GB on disk, $0, no GPU.
- At ~5.8 GB per 1,000 z (9.362) of band, z 1750-8500 would be ~40 GB streamed and ~1 h. Not recommended (section 0).

## 7. Files and commands
- tools/hires.py: shared 2.4 um access and transform helpers.
- tools/hires_step_count.py: profile extraction.
- tools/hires_step_eval.py: feats / calib / evaluate.
- data/scratch/hires_step/:
  - prof_z3000/, prof_z4500/ (+ stream_log.json);
  - vfeats_*.pkl;
  - calib_z4500.json (+ _models.pkl);
  - eval_z3000.json / .txt;
  - gm_matched_z3000.json;
  - coverage.json;
  - figs/ (spot_check.png, look1.png, under_z4500*.png);
  - scripts: spot_check.py, look_steps.py, proto_shear.py, calib_xbetween.py, gm_matched.py, coverage.py.
```
uv run python tools/hires_step_count.py profile --band z4500 --workers 6     # ~14 min, ~9 GB streamed
uv run python tools/hires_step_count.py profile --band z3000 --workers 6     # ~13 min
uv run python tools/hires_step_eval.py feats --band z4500 && uv run python tools/hires_step_eval.py feats --band z3000
uv run --with scikit-learn python tools/hires_step_eval.py calib       # z 4500-6000 labels only
uv run --with scikit-learn python tools/hires_step_eval.py evaluate    # band 1 scoring
uv run --with scikit-learn python data/scratch/hires_step/calib_xbetween.py
uv run python data/scratch/hires_step/gm_matched.py
```

## 8. Caveats and what would change the verdict
- The counter is deliberately simple: 1-D profiles with fixed features plus logistic regression. A learned 2.4 um surface
  model, i.e. 2.4 um masks run through the same graph, might do better. That is a GPU project with the fibre-texture problem
  above, and the 2026-10-16 contact check already found native 2.4 um splits sheets into 7-15 um strands.
- Truth is the human meshes. Band 1 was also scored against the CT-supported relabel, which leaves #2167 unfixable (hence
  the "outside #2167" columns). z 4500-6000 has no CT relabel, so calibration used the clean subset.
- Next step instead: re-pick the gm raise threshold on z 4500-6000 against CT-corrected counts (needs a z 4500-6000 CT audit
  or the clean subset), then A/B it in the fitter. Section 4c suggests ~89-90 % precision with |drift| ~0.4-0.6 is reachable
  without new evidence. That is still short of 92 % and zero drift.
