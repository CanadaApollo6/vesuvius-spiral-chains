# 2026-10-18: winding-field model (campaign W2)

## Why
Tracing, winding assignment and mesh correction cost ~25 h per wrap of human time (notes/2026-10-18-sota-pipeline.md).
They all answer one question: which wrap does this piece of sheet belong to? Real labels cannot answer it through
contacts. The sim can, exactly. The 7.91 um ablation showed the sim adds nothing to plain surface segmentation, where
662 real labelled patches already exist (notes/2026-10-17-training.md). So the sim moves to the label real data
cannot have.

## The target: winding coordinate s
- `Scroll.sample()["wind"]`: s = (wrap index k + seam shift) + (r - R_k) / (R_{k+1} - R_k).
- Wrap index is relative to the cell reference, like `instance`. Material radius r, recto face R_k.
- Properties:
  - continuous through sheets, gaps and exact contacts;
  - +1 per wrap;
  - recto faces are its integer level sets;
  - in-sheet floor(s) + 2 == instance (checked: 0 mismatches on all cells).
- `sim/add_winding.py` adds `wind_7p91` to existing cells by rebuilding each scroll from its stored ScrollParams and
  reference. Tables are deterministic in the seed.
- If predicted well, sheets fall out as level sets with identity attached, so surface prediction, tracing and winding
  assignment collapse into one regression.
- A fused contact is a missing unit step in s.
- Globally, the field must gain exactly +1 per turn around the umbilicus. That loop-closure constraint gives a
  label-free consistency signal on real scrolls, usable for QA and for self-training (next step).

## Model (train/winding.py)
- nnU-Net PlainConvUNet (Dataset101 plans), 1 input channel, 1 output channel, 128^3 patches.
- Initialized from the real350 surface model (282/292 tensors), so it starts from real-CT features.
- Loss: Huber regression, invariant to a constant offset and to sign.
  - A 1 mm patch knows neither the absolute wrap number nor where the umbilicus lies.
  - Both are fixed globally afterwards, so no umbilicus input is needed.
- Augmentation: flips, axis permutations, nnU-Net-like intensity augmentation (blur, noise, brightness, contrast, gamma).
- Split: scroll seeds 119-120 held out.
- Speed: ~70 s/epoch on the local RTX 3090.

## Metrics
- **Sim:** wrap error rate = share of in-sheet voxels with |s_hat - s| > 0.5 after the best sign and median shift per
  crop; reported overall and at contacts (gap < 2 um).
- **Real (eval/winding_sites.py):** the jeong200 band pairs of eval/separation.py (same sites).
  - Correct if the step |s_hat(b) - s_hat(a)| between adjacent GT bands is 0.5-1.5.
  - Merge if < 0.5; split if >= 1.5.
  - Compressed (d < 10.5 vox) vs wide (16-24 vox), patch-bootstrap CI.
  - Comparable in spirit to sep_cc of the surface models (real350: 53.0% compressed, 79.0% wide). The bar is to beat
    that at compressed sites.
  - **Flawed (found 2026-10-18, see Results):** the 0.5-1.5 window is gameable. Sites are classed by distance d
    (compressed d is 9-10.2 vox, wide 16-24), so a constant field delta = d/16 with no image input scores 99.9% / 100%.
    Replacement: the PHerc0139 instance set with pairs at 0, 1 and >= 2 wraps, |delta - expected| < 0.25, and
    constant-pitch and per-crop oracle-pitch baselines scored by the same code. A model resolves sheets only if it
    beats the oracle pitch.
- Smoke check (40 training steps, 2 patches): wide sites already have a median step of 1.04.

## Runs
- Pod vss-wind (4x L40S), 2026-10-18, ~42 s/epoch per lane. Logs: data/winding/<tag>/log.jsonl on the pod;
  site CSVs in data/preds/wind_<tag>.csv.
- w1: 60 epochs, init real350. w1s: same from scratch. w1l (150 epochs) and w1p (160^3 patches) still running.

## Results (2026-10-18)
| run | sim val err (all / contact) | jeong200 0.5-1.5 rule (compr / wide) | tight +-0.25 (compr / wide) | median step (compr / wide) |
|---|---|---|---|---|
| w1 (init real350) | 36.6% / 46.0% | 73.9 [70.9, 76.8] / 80.8 | 37.0 / 51.4 | 0.68 / 1.08 |
| w1s (scratch) | 53.2% / 59.4% | 49.8 [45.8, 54.0] / 61.8 | 23.2 / 34.2 | 0.52 / 0.87 |
| constant pitch 16 vox | - | 99.9 / 100.0 | 0.0 / 52.3 | - |
| constant pitch 10 vox | - | 99.9 / 0.0 | 99.9 / 0.0 | - |
| per-patch oracle pitch | - | - | 28.9 / 74.7 | - |

- Real-data init matters a lot (w1 vs w1s, every metric).
- w1 does not yet resolve compressed sheets: the median step at compressed sites is 0.68, not 1. It partly merges.
- Its correct rate is the same whether real350 separated, missed or fused the site (75 / 72 / 75%). It behaves like
  a context/pitch estimator, not a sheet resolver.
- The jeong200 site metric cannot separate those two behaviours (table above). No improvement claim from these runs.
- Sim val error is high (37% of in-sheet voxels off by > 0.5 wrap per crop): an absolute coordinate over a 128^3
  crop drifts. Next formulation to try: predict local quantities (gradient of s, or phase plus step-at-face) and
  integrate.

### Pitch-proof checks (2026-10-18)
- Sim (train/winding.py validate, new): voxel pairs along the true winding direction, 2-30 vox, correct if
  |delta_hat - delta| < 0.25. Baseline = per-crop oracle pitch (resolves nothing).
  - w1: 60.4% (oracle 59.6%); across contacts 46.4% (oracle 46.9%). w1s: 49.1% / 36.0%.
  - So w1 is a pitch estimator on exact sim labels too. Profiles (data/scratch/wind_w1_profiles.png): it tracks open
    regions closely and drifts 0.5-1 wrap through runs of fused contacts.
- Real PHerc0139 9.362 um set (notes/2026-10-18-p0139-testset.md; 87 sites, 24,972 pairs, steps 0-4, distance
  balanced), |delta - expected| < 0.25 pooled:

| method | pooled | step 0 | step 1 | step >= 2 | contact | open |
|---|---|---|---|---|---|---|
| per-crop oracle pitch (the bar) | 52.0 | 84.4 | 40.1 | 31.6 | 36.4 | 54.4 |
| best constant pitch | 48.4 | 86.3 | 34.9 | 23.8 | 34.9 | 50.4 |
| w1 (7.91 sim, init real350) | 38.2 | 50.6 | 35.8 | 28.2 | 28.5 | 39.6 |
| w1s (scratch) | 26.4 | 32.6 | 27.3 | 19.4 | 22.5 | 27.0 |

  - w1 vs oracle: -13.8 pts, CI [-16.2, -11.4]. Not resolving sheets on real data either.
  - Weakest on same-wrap pairs (50.6%): the field is not constant along a sheet. Trained at 7.91, tested at 9.362.
- Gradient-loss variants (w2g grad_w 1, w2g4 grad_w 4) running on the pod; 9 um sim cells (data/sim/v9) generating.

### Gradient loss: no (2026-10-18)
- w2g (grad_w 1) / w2g4 (grad_w 4), 60 epochs: sim pair 55.8% / 56.1% (w1 60.4%, oracle 59.6%); contact 41.9% / 42.5%.

### Are contacts visible in sim CT? (eval/sim_contact_sep.py)
- Local test (25^3 window, components in the window, as eval/separation.py) on the 40 Dataset102 fold-0 validation
  cells. Points on sheet i next to sheet i+1, touching (< 2 um) or apart (>= 16 um).
- simpre (sim-trained surface model): touching pairs separated 46.1% (merged 53.0% given hit); apart 95.6%.
  real350 misses almost all sim sheets (texture domain gap), so it says nothing here.
- Separation does not track interface_dip or contact_loose monotonically (few scrolls, confounded).
- So contacts are partly visible in sim and a local face model uses that. The winding regression does not
  (pair score at contacts = oracle pitch).

### Phase head (runs wp1, wp3)
- Integer level sets of s are recto faces, so frac(s) is absolute; only the sign is ambiguous. wind_loss subtracts
  the mean and throws that anchor away.
- `--phase-w`: 3 outputs (s, cos 2 pi s, sin 2 pi s); phase_loss = wind_loss + w * phase MSE, shared best sign.
- Decode (decode_phase): shift the coarse channel onto the phase anchor (circular mean), then unwrap:
  s = s_c + wrap(phi - s_c). Validation reports the decoded pair score and the coarse one (pair_coarse).
- Runs (pod, 60 epochs, init real350): wp1 (w 1), wp3 (w 3) on 7.91 sim; w9p (w 1) and w9 (plain) on the
  9.362 um sim cells (data/sim/v9: 400 cells, 192^3, seeds 101-120, per-cell stage jitter).

#### Results
Sim validation (pair score; oracle = per-crop pitch baseline on the same pairs):

| run | sim | pair | oracle | contact | oracle | coarse |
|---|---|---|---|---|---|---|
| wp1 (phase w 1) | 7.91 | 73.0 | 59.6 | 56.4 | 46.9 | 56.1 |
| wp3 (phase w 3) | 7.91 | 67.1 | 59.6 | 53.5 | 46.9 | 45.9 |
| w9p (phase w 1) | 9.362 | 77.3 | 58.8 | 59.7 | 34.6 | 57.6 |
| w9 (plain) | 9.362 | 69.4 | 58.8 | 49.7 | 34.6 | - |

So the phase anchor is the first thing that beats pitch on sim, including across contacts (+25 pts for w9p).

Real PHerc0139 9.362 um (pooled, step 0 / 1 / >= 2):

| model | pooled | step 0 | step 1 | step >= 2 |
|---|---|---|---|---|
| w9p | 36.4 | 47.0 | 34.7 | 27.5 |
| w9 | 40.3 | 56.3 | 35.7 | 28.8 |
| wp1 | 40.7 | 51.7 | 40.0 | 30.4 |
| w1 | 38.2 | 50.6 | 35.8 | 28.2 |
| oracle pitch (bar) | 52.0 | 84.4 | 40.1 | 31.6 |

None transfers. data/scratch/wind_real_vs_sim.png: on sim the decoded integer level sets follow the sheets; on real
9 um crops w9p's level sets are fragmented (the phase channel locks onto texture, not faces) and w9's are smooth but
cut across sheets. Real 9 um sheets are fibrous, ragged, irregularly spaced, with wide gaps; the sim renders clean
regular bands. The blocker is the sim->real gap in appearance and geometry, not the loss.

Next: (a) anchor the phase on real surface predictions (real350 / a 9 um surface model) instead of learning it from
sim CT; (b) adapt on real unlabelled 9 um crops; (c) sim realism at 9 um (fibre texture, spacing irregularity, gaps).
PHerc0139 stays held out for all of them.

### Surface-first winding beats pitch on real data (eval/p0139_crossings.py)
- delta = number of predicted surface crossings on the segment A -> B (entries into the mask after an off-run >=
  min_gap; first `skip` vox ignored; segment runs `skip` past B), median over the straight path and 6 two-leg paths
  through jittered midpoints (sigma 1.5 vox). Mask: real350 (nnU-Net, real Dataset059 = Scrolls 1/4/5 at 7.91 um;
  PHerc0139 never seen) run on the 9.362 um crops at spacing 9.362/7.91 (resampled to its training grid).
- Tuning (eval/p0139_xing_sweep.py): sites sorted by stem, even = tune, odd = report. Frozen: skip 3, min_gap 2.5,
  dilate 1, 7 paths sigma 1.5 (now the scorer defaults). Gains plateau beyond (skip 5 / gap 3.5: 68.1).
- Report half (43 sites): 67.9% vs oracle pitch 51.9%, +16.1 pts, CI [11.7, 20.2]. Step 0 / 1 / >=2:
  70.0 / 76.9 / 56.9 (pitch 84.6 / 39.7 / 31.3).
- Untuned defaults (skip 2, gap 1.5, no dilation, 1 path), all 87 sites: 59.3% (+7.3, CI [4.8, 9.5]); at native
  spacing 1.0 (no resampling) 57.5%.
- Estimator ceiling with the labelled surfaces (instancesTr > 0), untuned defaults: 83.4%; contact pairs 56.8%
  (touching bands leave no gap). Unlabelled sheets between labelled ones also cap it.
- data/scratch/xing_real350.png: real350 finds many sheets the labels leave out and sometimes draws two lines per
  sheet (split faces), which is what the min_gap / dilation tuning suppresses.
- Consequence for W2: drop dense winding regression from sim CT; go surface-first. Levers: a surface model trained on
  real 9 um data from other scrolls (PHerc0814 and PHerc0500P2 have 9.362 um volumes and meshes; PHerc0009B, PHerc1447
  at 8.64 um), contact separation (sim contacts as augmentation), then a global winding assignment from surfaces.
- Frozen config, all 87 sites (includes the tune half), pooled [CI]; step 0 / 1 / >=2; contact / open:

| mask | pooled | step 0 | step 1 | step >= 2 | contact | open |
|---|---|---|---|---|---|---|
| labelled surfaces (ceiling) | 83.2 [81.3, 85.3] | 92.9 | 91.8 | 65.0 | 49.5 | 88.3 |
| real350 | 67.6 [64.5, 70.6] | 69.2 | 77.5 | 56.0 | 46.2 | 70.8 |
| real350 + mirror TTA | 67.8 | 72.0 | 77.1 | 54.2 | 45.3 | 71.2 |
| real250 | 66.2 | 69.8 | 76.1 | 52.5 | 43.4 | 69.6 |
| squeeze250 | 66.2 | 67.9 | 76.1 | 54.5 | 44.9 | 69.4 |
| simft | 67.6 | 69.3 | 76.7 | 56.7 | 45.5 | 70.9 |
| 5-mask majority vote | 67.8 | 71.3 | 76.9 | 55.3 | 45.6 | 71.2 |
| oracle pitch (bar) | 52.0 | 84.4 | 40.1 | 31.6 | 36.4 | 54.4 |

  TTA and ensembling add nothing: the models' errors are shared. Next gains need new training data (real 9 um) or a
  contact-aware estimator (the ceiling itself is 49.5% at contacts: touching bands give no off-run).
- Contact-aware count (count_thick: each run counts round(thickness along the normal / T) sheets): no gain (best T 5,
  report half 68.0 vs 67.95; T 3 collapses to 47). At contacts real350 does not draw a double-thick band; the second
  sheet is just missing. Contact separation has to come from the surface model.

### Real 9 um fine-tunes and signed crossings
- Fine-tunes on Dataset103_Real9 (pod vss-real9, 150 epochs, cloud/real9_pod.sh), scored at native spacing with
  the frozen estimator (all 87 sites): r9ft 65.6, r9ftsq (squeeze aug) 66.3, r9sim (sim init) 64.5; real350 67.6.
  Retuned per model on the tune half, r9ft reaches 67.5 on the report half (real350 67.95). No gain from real
  9 um labels so far (mesh-raster labels are loose; 405 crops).
- Error split (real350, frozen): same-wrap pairs over-count 29% open / 47% contact (labelled surfaces 7-8%):
  the path drops out of a fragmented band and re-enters it. Step >= 2 under-counts (36% open, 66% contact).
- Signed count (count_signed): entries signed by sign(u . n), normals oriented onto the crop's dominant sheet normal,
  entries with |u . n| < graze dropped, delta = |sum|. Grazing exit/re-entry is dropped, and out-and-back cancels.
  real350, graze 0.15 (tuned on the even half): report half 72.75% vs pitch 51.9 (+20.9, CI [16.7, 25.1]);
  step 0 / 1 / >=2: 86.3 / 75.8 / 56.2; contact 49.8, open 75.8. Same-wrap pairs now beat pitch (84.6).
- Finer graze grid: real350 signed, graze 0.2: report half 73.3% (+21.5, CI [17.4, 25.5]); step 0 / 1 / >=2:
  91.0 / 72.9 / 56.0; contact 50.2, open 76.4. r9ft signed: 71.6. Current best estimator.
- Ridge count on real350 probabilities (eval/p0139_ridges.py, hysteresis peaks, signed): best 72.8% on the report
  half (step 0 92.6, step 1 67.5, >=2 58.3). Same as the signed mask count: the probabilities carry no extra
  separation. At compressed sites the intermediate sheet is absent from the output (real350 says 1 where the
  labelled surfaces say 2 in 20% of step-2 open pairs; compressed sites 57% correct vs wide 83%).
- Hypothesis: every training target is a thick band (DS059 ~4.5 vox, mesh band ~3 vox), so sheets closer than
  that cannot be separated. Next: thin targets (1-vox mesh raster, gap voxels between touching segments explicit
  background, a 1.5 vox shell ignored), Dataset104_Real9Thin.
- Scorer default is now the signed count (eval/p0139_crossings.py --graze 0.2). All 87 sites: real350 73.1%
  [70.2, 76.0] (vs pitch +21.0, CI [18.2, 23.7]); labelled-surface ceiling 83.2%. r9ft300 (300 epochs on
  Dataset103) 70.5% on the report half: longer real 9 um fine-tuning makes it worse.
- Why the real 9 um fine-tunes cannot help at compressed sites: the Dataset103 dedupe dropped every raster within
  4 vox of a larger segment (meant for duplicate tracings), so the set has no sheets < 4 vox apart (median spacing
  p10 10 vox). New sets (notes/2026-10-18-real9-trainset.md): Dataset104 thin (1-vox raster, 1.5 vox shell
  ignored), Dataset105 band + CT-checked dedupe (sheet-gap-sheet profile keeps a close tracing; gap voxels taught
  as background; 296 crops with pairs < 4 vox), Dataset106 thin + the same.
- Lanes on vss-real9 (real350 init, 150 epochs): r9thin (104), r9thinsq (104, squeeze), r9gap (105), r9thingap
  (106, squeeze).
- Thin / gap fine-tunes (vss-real9, real350 init, 150 epochs), signed count graze 0.2, report half:

| model | labels | report half | step >= 2 | contact | all 87 (frozen) |
|---|---|---|---|---|---|
| real350 (no fine-tune) | DS059 band | 73.3 | 56.0 | 50.2 | 73.1 |
| r9gap | band + CT-checked gaps (105) | 72.3 | 56.7 | 47.7 | 71.9 |
| r9thinsq | thin, squeeze (104) | 68.7 | 48.0 | 46.1 | 68.0 |
| r9thin | thin (104) | 68.4 | 47.2 | 45.8 | 68.0 |
| r9thingap | thin + gaps, squeeze (106) | 68.2 | 45.9 | 45.9 | 67.2 |

  Ridge counting on their probabilities is lower still (66-71). Thin targets hurt: the output fragments and
  intermediate sheets go missing (step >= 2 drops ~9 pts). Teaching gaps neither helps nor hurts (the compressed
  pairs add 0.3% of surface voxels). The thick-target hypothesis is not supported; no real 9 um fine-tune beats
  real350. Remaining misses at compressed sites look like an information limit of local 9 um CT for this model
  class. Next lever: global consistency (winding assignment over a surface graph, where a missed sheet is implied
  by continuity around the turn) rather than better local masks.

### Global winding assignment on a patch graph (eval/p0139_graph.py)
- Per crop: real350 mask (dilated 1) cut into blocks; 26-connected components per block = patches. Edges to the k
  nearest patches within a radius; d_ij = signed crossing count rep_i -> rep_j (count_signed absolute=False, sign
  along the crop-oriented normal; median of 3 paths), weight 1 / (1 + length / 10). Winding per patch by IRLS L1,
  rounded. Pair delta = |w(patch B) - w(patch A)| (nearest patch <= 3 vox, else the direct count).
- First grid (k 30), report half: block 12, radius 40: 77.3% vs direct signed 73.4% vs pitch 51.9;
  step 0 / 1 / >=2: 91.3 / 76.6 / 64.0 (direct 91.4 / 72.7 / 56.0); contact 50.7, open 80.8.
  block 16 r40 77.0, block 24 r40 76.3. Accuracy rises with radius (r16 74.9, r24 75.9, r40 76.0 tune), so a
  missed sheet is recovered from paths around it.
- Extension (block 12): radius and k saturate. k 30: r56 77.29 (vs direct +3.9, CI [2.85, 5.03]); k 60: r40 / r56 /
  r80 tune 75.85 / 75.77 / 75.89, best r80 k60 report 77.38 (vs direct +4.0, CI [2.95, 5.12]); step >= 2 63.7,
  contact 50.7. Frozen: block 12, radius 40, k 30 (cheapest at the plateau).
- Contact pairs stay at ~50 %: at a contact both points snap to the same merged patch, so the graph cannot see the
  sheet between them. Tested next: split patches thick along their normal into two layers (split_thick, thresholds
  4-8 vox), and fall back to the direct count when A and B share a patch (`_sp` tags).
- Both are null (data/preds/graph_split.json). Split thresholds 4 / 5 / 6 / 8 tune 76.19 / 76.11 / 76.18 / 76.03 vs
  75.96 unsplit. Best (t4) report 77.32 vs 77.28; contact 51.3 vs 50.7. The same-patch fallback never changes a
  pair. Kept unsplit.
- **Frozen: block 12, radius 40, k 30, no split. All 87 sites: 76.6 % (site-bootstrap CI [73.3, 79.6]).** Compare
  direct signed 73.15, oracle pitch 52.0. Graph minus direct: +3.46 pp, paired CI [2.69, 4.26]. Step 0 / 1 / >= 2:
  90.9 / 76.8 / 62.1; contact 50.4, open 80.6. The GT-mask ceiling is 83.2.
- Contacts (~50 %) remain the gap and no post-processing of real350 masks has moved them. The sheet between two
  touching papyri is not in the mask, so neither the graph nor the direct count sees it.
- **Contacts are capped by binary masks, not by the network.** The frozen graph on GT surface masks (instances > 0,
  data/preds/graph_gt.json) reaches 84.9 % on all 87 sites, with contact 49.6 % and open 90.2 %. Direct count on GT:
  83.3 %, contact 48.5 %. real350 + graph is already at that cap on contacts (50.4 %). What separates it from GT is
  open pairs: 80.6 vs 90.2, and open pairs are ~88 % of the set. Contacts would need instance-separated surfaces,
  not a binary mask.
- **m7 (the bucket's surface prediction, PHerc0139 volume 20250728140407, L0 th0.2, scripts/p0139_m7.py) + frozen
  graph: 80.5 % on all 87 sites** (direct 77.9; contact 53.8, open 84.5; report half 80.1). real350 gives 76.6.
  Caveat: m7's training data is not documented (metadata: s3://scrollprize-reconstruction/models/m7_nnunet/), and
  it beats the GT masks on contacts, so it may have seen PHerc0139 labels. This is not a held-out result. m7 is a
  public resource, so using it as a tool for reading other scrolls is legitimate.
- m7 variants, report half: dilate 0 gives 80.43 (dilate 1: 80.10). The union with real350 gives 79.04. m7 alone
  stays.
- **m7 distillation (held-out route):** Dataset107_M7Distill (scripts/m7_distill_set.py) has 1713 crops of 192^3
  on m7's chunk grid. Scrolls: PHerc0814 300, 0500P2 213, 1447 300, 0826 300, 0211 300, 0125 300; never PHerc0139.
  Labels are m7 > 0; native resolution, spacing 1. Dataset108_M7Mix is 107 + 103 (mesh labels).
  Pod vss-m7d (4x A6000) runs four lanes from real350: m7d (150 ep), m7d300, m7dsq (squeeze), m7mix (108, 150 ep).
- **Distillation result (held out, PHerc0139 never in training).** Masks at spacing 1.1836 beat native 1.0 on the
  tune half (m7d: graph 80.15 vs 78.40; m7mix: 79.88 vs 78.61), so 1.1836 is chosen. Frozen graph:

| surfaces | report half | all 87 | site CI | vs real350 graph (paired, all 87) | contact | open |
|---|---|---|---|---|---|---|
| real350 | 77.28 | 76.61 | [73.3, 79.6] | - | 50.4 | 80.6 |
| **m7d** (107, 150 ep) | 80.22 | **80.19** | [77.6, 82.7] | **+3.57 [2.13, 5.12]** | 55.3 | 84.0 |
| m7mix (108, 150 ep) | 80.47 | 80.17 | [77.6, 82.6] | +3.56 [2.20, 4.93] | 54.8 | 84.0 |
| m7 itself (maybe not held out) | 80.10 | 80.48 | [77.3, 83.2] | +3.87 [2.67, 5.22] | 53.8 | 84.5 |

  Our own checkpoint, trained only on m7 pseudo-labels of six other scrolls, matches m7 on PHerc0139 (+3.6 pp over
  real350, CI above 0). Contacts rise above the GT-binary cap (55 vs 49.6): m7-style masks separate some touching
  sheets that the dilated GT raster merges. Direct count (no graph): m7d 78.51, m7mix 78.30 (real350 73.15).
  Bookkeeping: the first graph runs wrote graph_<tag>_s1.json/csv for both spacings (Path.with_suffix ate ".1836");
  the reruns are graph_<tag>_s11836.*, and eval_m7d.sh now strips the dot.
- **m7d300 (Dataset107, 300 epochs) is the best of the four on the tune half** (graph 81.08 at spacing 1.1836; m7dsq
  80.61, m7d 80.15, m7mix 79.88). Report half 81.13. **All 87: 81.11 % (CI [78.6, 83.5]), +4.49 pp over real350 +
  graph (paired CI [3.03, 5.97]); contact 55.8, open 84.9.** This is above m7 itself (80.48). The squeeze
  augmentation did not help (80.24 report). Longer training did, so the next round scales epochs and data
  (Dataset109: 19 scrolls). Paired comparisons: eval/p0139_paired.py.
- **Round 2 (Dataset109: 19 scrolls, 4297 crops; pod vss-m7w, 4x RTX4090) plateaus at m7d300.** Frozen graph,
  s1.1836, all 87, paired vs m7d300:
  - w300 (real350 init, 300 ep): 80.76, -0.34 [-0.92, +0.17];
  - w300c (m7d300 init, +300 ep): 81.05, -0.06 [-0.54, +0.39].
  - On hf2um (notes/2026-10-19-hf2um-benchmark.md), compressed separation: m7d300 57.7, w300c 55.9, w300 54.1.
    Official blend: 0.476 / 0.478 / 0.476.
  - More scrolls do not help, nor does continuing past 300 epochs. The student sits at the teacher's level
    (m7 80.5). w600 and w900 are still training; they are expected to be flat.
  - Further gains need a signal beyond m7 pseudo-labels: graph-consistency self-training, or instance-separated
    targets for contacts.
  - A majority vote of m7d300, w300c and w300 masks with the frozen graph gives 81.17, +0.06 [-0.38, +0.48]: null.
    The students' errors are shared.
  - Decision: pull w600 when it finishes, then terminate vss-m7w. w900 (~$14 more) is stopped as an expected null.
  - w600 (real350 init, 600 ep, Dataset109): 81.30 on PHerc0139, +0.20 [-0.39, +0.83] vs m7d300. hf2um compressed
    separation 56.2 (m7d300 57.7), official 0.474. Flat, as expected.
  - m7d300 stays the release candidate: equal within noise, and trained on the smaller, documented set.

### Certifier signal from the graph itself (W5, eval/p0139_certify.py, 2026-10-19)
- `eval/p0139_graph.py` now also writes per-pair c_* features. They use no GT:
  - c_res: max node mean L1 residual;
  - c_frac, c_wdiff_frac: non-integral windings;
  - c_deg: min node degree;
  - c_disagree: |graph - direct count|.
  The frozen m7d300 run reproduces 81.11 exactly (data/preds/graphc_m7d300.csv).
- Single-feature AUCs for a wrong pair delta: residual 0.69, winding-difference non-integrality 0.69,
  graph/direct disagreement 0.67.
- A logistic model fitted on the tune half and scored on the report half reaches AUC 0.76.
  - Flagging the 30 % least confident pairs catches 64 % of the errors; the other 70 % are 90.2 % right
    (all pairs: 81.1).
  - Flagging 50 % catches 80 %; the kept half is 92.5 % right.
- Where graph and direct disagree (17 % of pairs), both are poor (graph 50.7 %, direct 41.4 %). Where they agree
  they are 87.5 % right, so falling back to direct does not help; the signal is for triage.
- Use: a confidence map for the spiral fitter and for human time (Grand Prize, 8 h). Spend human checks on the
  flagged ~30 % and trust the rest. The same signal can filter pseudo-labels for self-training.

## Residual-driven patch split (2026-10-19): null
- Contact pairs are capped by the masks, not the graph: with GT binary masks the frozen graph scores 49.6 % at
  contacts (direct 48.5 %), 39.7 % of them undercounted by one. m7d300 masks: 55.8 %.
- Tried `--rsplit tau` (eval/p0139_graph.py): solve, split every patch whose mean edge residual exceeds tau into two
  layers along its normal, re-solve. m7d300 masks, s1.1836:
  | tau | tune | report | contact (report) |
  |---|---|---|---|
  | 0 (frozen) | 81.08 | 81.13 | 58.18 |
  | 0.15 | 80.89 | 80.61 | 57.90 |
  | 0.25 | 81.25 | 81.03 | 58.32 |
  | 0.4 | 81.06 | 81.22 | 58.73 |
- Tune pick (0.25) reports -0.1 pp. Kept off. (The run's CSVs were lost to an output-name bug, now fixed; numbers
  are from the logs data/scratch/rsplit_m7d300_r*.log.)
- So the contact lever has to come from the model (a contact-aware target), not the graph.
