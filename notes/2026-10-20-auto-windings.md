# 2026-10-20: automatic relative-winding annotations for the spiral fitter (tools/auto_windings.py)

The organisers say human relative-winding annotations
"seem to have a great impact on the spiral fit" and that automating them "will boost scalability". This tool
turns surface masks into `relative_windings.json` with the frozen patch-graph solver (eval/p0139_graph.py, block 12,
r40, k30) plus a step-confidence certifier. Everything ran on local CPU. No GPU, no pods, $0.

## 0. Headline
- **Site validation** (87 PHerc0139 test crops, 2,119 radial rays, 5,092 steps with both ends on a human label):
  - m7 masks, confidence model scored out of fold (2-fold by site half): **98.0 % step precision
    (site-bootstrap CI [96.5, 99.1]) at 58 % coverage** (frozen threshold p_err <= 0.0461). Raw precision with no
    filter is 92.8 %.
  - Outside windings w033-w037 (the band villa #2167 questions): 99.2 % [98.5, 99.8] at 65 % coverage with the same
    threshold, or 98.0 % at 94 % coverage with its own threshold.
  - Held-out masks (m7d300, never trained on PHerc0139) at the frozen threshold: 98.8 % at 49 % coverage, 99.7 %
    outside w033-w037. m7 run at 1.18x (m7x): 98.2 % at 70 %.
- **Band z 3000-4500** (m7 bucket masks, 24 angles x 23 z = 552 rays, 4,929 crops, ~2 h on 11 CPU workers):
  - Frozen threshold: **3,697 chains / 20,770 points** (median 4, max 47 per chain).
  - End-to-end vs chains cut from the human meshes on the same rays: **94.8 % step precision (5,025 matched steps)**;
    96.3 % on clean steps outside w033-w037. **The 98 % site precision does not transfer to the band.**
  - `--conf 0.014`: 1,865 chains / 5,694 points at 97.4 % (1,221 steps). 0.014 was picked on the 30-deg angles
    (98.3 % there); on the 15-deg-offset angles, unseen when it was picked, it gives **96.3 %**.
- **JSON**: parsed by the pinned loader (villa-0fb38c45 `point_collection.load_point_collection`) plus a replica of
  fit_spiral's strip preparation. All points annotated, no chain with decreasing wind_a, max consecutive dtheta
  1e-4 rad.
- **Controls on the same probes:** oracle chains from the human meshes (17,766 points / 1,178 chains), and a corrupt
  copy of them with 10 % of steps flipped (89.6 % step agreement with the oracle, as designed).

## 1. Fitter format (verified in the pinned volume-cartographer/scripts/spiral)
- `{"vc_pointcollections_json_version": "1", "collections": {"<id>": {"name", "points": {"<pid>": {"p": [x, y, z],
  "wind_a": k, "creation_time": 0}}, "metadata": {}, "color": [0, 0, 0]}}}`.
  - Loader: point_collection.py:29.
  - p -> zyx: fit_spiral.py:1219.
  - Mixed annotated/unannotated points are stripped: normalise_pcl_winding_annotations, point_collection.py:544.
  - Dataset discovery: fit_session.py:286 (`relative_windings.json`).
- With input_disable_patches every pcl is an unattached strip (fit_spiral.py:1310-1420; losses.py:1467
  get_unattached_pcl_strip_losses). The fitter subtracts wind_a x dr from the unwrapped shifted radius and pulls the
  result to a constant, so wind_a must increase outward (sample_spiral.py:76). Consecutive points need |dtheta| < pi.
  Our chains are radial, so dtheta = 0 between consecutive points.
- **Override `pcl_unattached_pcl_min_point_spacing` to 0** (default 16 vox, config.py:313). With the default, the
  greedy decimation drops 14 % of our points (5 % at --conf 0.014): sheets are 10-20 vox apart.
- Point ids are globally unique and in chain order. Each chain is one collection with wind_a 0, 1, 2, ... and >= 2
  points.

## 2. Method
- **Probes.** A probe is a horizontal ray from the umbilicus at fixed (z, theta).
  - Crops are 192^3, centred every 96 vox of radius, and aligned in z to the 192-chunk layers of the m7 zarr, so each
    crop reads exactly one chunk layer.
  - Each crop serves three rays, at z = layer centre and +-64, so the z spacing is 64.
  - The radial extent of each ray comes from m7 level 3.
- **Per crop.**
  - Mask dilated by 1, then the frozen graph solve.
  - Ray crossings: runs of the dilated mask along the ray. Runs closer than 2.5 vox merge, as in the frozen signed
    count. Runs truncated at the segment end are dropped.
  - Each crossing takes the solved winding of its patch. The point is placed at the centre of the raw-mask run, so it
    lies on the sheet.
  - Sign: + along the ray. The crop's dominant oriented normal is compared with the ray direction, and the ray is
    skipped in that crop if |cos| < 0.3.
- **Steps.**
  - d = w_next - w_prev.
  - round(d) = 0: the two crossings merge (same sheet).
  - round(d) < 0: the chain breaks.
  - round(d) > 1: never emitted (`--max-step 1`). Graph-claimed missing sheets are right only 62 % of the time
    (118/189). The kept ones were 30 % right.
- **Band assembly.**
  - Crossings are deduplicated across overlapping crops, keeping the copy from the crop whose ray-segment centre is
    nearest.
  - Each step comes from the most central crop that holds both ends.
  - When a second crop also holds both ends and disagrees, the step is dropped (`--require-agree 1`). This is nearly
    inert: only ~20 % of steps have a second crop, and those see the same mask, so it dropped 0 kept steps in the band.
- **Confidence.** Logistic model on:
  - log node residual, log non-integrality of d, log degree, step >= 2, gap, run length, grazing, multi-winding run;
  - ray-segment aggregates: mean dfrac, mean and max residual, share of steps >= 2.
  - The ray-segment aggregates lift report-half AUC from 0.74 to 0.79; gradient boosting was no better.
  - Fitted on all 87 sites and frozen in `tools/auto_windings_conf.json`. The threshold is the largest out-of-fold
    p_err cut with cumulative precision >= 98 %.
  - Chains are split at every step below threshold. `--max-len` caps chain length; split chains share a point.
- **Umbilicus.**
  - `umbilicus_9p362()` and `affine_2p399_to_9p362()` reuse eval/spiral_eval.py (`Umbilicus`, `affine`): each
    2.399 um control point is mapped whole with the published affine (p9_xyz = A p2399_xyz + t) and interpolated
    linearly in z, as the fitter's json_umbilicus_z_to_yx does.
  - A copy in the fitter's json format is written to `<out>/umbilicus_9p362.json`.
- **Truth for validation.**
  - The winding of the nearest human surface voxel (surfacesTr) within 3 vox.
  - Ambiguous if another instance is within +1 vox of that distance.
  - A step is scorable only if both ends are labelled.

## 3. Validation tables (data/scratch/auto_windings/val_*.json, CSVs alongside)
Out-of-fold coverage / precision, m7 masks. Coverage is kept steps over scorable steps.

| kept (of candidates) | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % | 100 % |
|---|---|---|---|---|---|---|---|---|---|---|
| precision | 98.6 | 98.8 | 98.5 | 98.2 | 98.1 | 97.9 | 97.5 | 96.9 | 95.7 | 93.0 |

At the frozen threshold (p_err <= 0.0461):

| masks | coverage | precision (CI) | outside w033-w037 | touching w033-w037 | #2167 sector |
|---|---|---|---|---|---|
| m7 (OOF) | 58.1 % | 98.0 [96.5, 99.1] | 99.2 % at 65 % | 94.4 % at 43 % | 100 % (49/49) |
| m7d300 (held out) | 49.4 % | 98.8 | 99.7 % at 55 % | | |
| m7x (m7 at 1.18x) | 69.6 % | 98.2 | 99.1 % at 74 % | | |

- Steps whose label difference skips a winding absent from the site (likely label errors) barely matter: 98.2 % without
  them.
- By site class at the threshold: compressed 98.0, mixed 97.4, wide 98.4.
- Chains emitted on the sites: 1,472, median 3 points, 96 % fully correct. Chains are short here because a site ray
  is at most 144 vox.
- Raw errors are mostly undercounts (dr 1, truth 2: 270 of 356). The m7 mask misses or merges a sheet in a way the
  graph cannot see. Half of all steps have residual < 0.01 and dfrac < 0.01, and 2.8 % of those are still wrong.
  That sets the ceiling of this certifier.
- **Errors concentrate in w033-w037:** 12.9 % raw error vs 4.1 % elsewhere, almost all undercounts, where the human
  labels have one winding more than the masks show.
  - This is the band villa #2167 questions (notes/2026-10-20-p0139-label-audit.md: GT-mask graph 78.6 % there vs
    90.3 % elsewhere).
  - It is either a label problem or genuine compression; the CT has not been checked.
- **#2167 sector.** Taken from eval/spiral_eval.py (`gap_sectors(human_segments(1700, 8400), Umbilicus())`, reused
  rather than re-derived; the affine and umbilicus helpers are reused the same way). It gives 106 slabs of 50 z
  (z 2800-8100) where w034/w036 exist and w035 does not. For z 3000-4500 the gap starts at 144-152 deg and is
  105-119 deg wide (atan2(y - cy, x - cx), 9.362 frame). Scored with a +-10 deg margin.
  - Only 49 scorable site steps fall inside it: 100 %. Outside it: 97.97 %. The sector is not where the test sites
    lie.

## 4. Band z 3000-4500
Outputs: data/scratch/auto_windings/band_m7_d30/ (frozen threshold), band_m7_d30_c014/ (`--conf 0.014`), and
band_m7/ and band_m7_c014/ (full 15-deg grid). Each has relative_windings.json, chains.json, steps.csv, summary.json and
compare_band_oracle.json.
- **Probes.** z 3000-4500: 23 z levels (64 apart, 8 chunk layers) x 12 angles (30 deg) = 276 rays, 2,482 192^3 crops,
  ~3,986 cached 192^3 mask chunks (0.5 GB packbits). Crops are checkpointed (fsynced) under band_m7/crops/, so the run
  survived two machine reboots. `--reassemble --dtheta 30` re-cuts any subset in 3 s.
- **Runtime.** ~1.45 s per crop wall time with 11 CPU workers (~16 core-s per crop). That is ~1 h for the 30-deg
  subset and ~2 h for the 15-deg grid, CPU only.
- **Counts at the frozen threshold.** 20,452 candidate steps, 8,635 kept (42 %), 551 with no crop holding both ends.
  That gives 1,811 chains / 10,446 points (median 4, max 47). The 0.014 cut gives 1,944 steps, 923 chains and
  2,867 points (median 3).
- **Loader check** (pinned load_point_collection + normalise + replicated strip decimation). Both files parse with all
  points annotated, 0 decreasing chains and dtheta <= 1e-4 rad. Default spacing 16 would drop 14 % / 5 % of points.
- **End-to-end vs the mesh oracle.** Steps whose both points match a mesh crossing within 4 vox, uniquely; 36 % of
  points match.

| cut | steps | all | in #2167 | out #2167 | clean (no w041/42/45/46) | clean, outside w033-w037 |
|---|---|---|---|---|---|---|
| p_err <= 0.0461 (frozen) | 2,584 | 95.3 | 94.5 (721) | 95.6 (1,863) | 96.2 | 96.7 (2,176) |
| p_err <= 0.014 | 650 | 98.3 | 99.1 (109) | 98.2 (541) | 98.9 | 98.8 (561) |

  - Curve at the frozen-threshold chains, by p_err quantile of the matched steps: best 25 % = 98.3, 50 % = 96.5,
    75 % = 95.9, 100 % = 95.3.
  - **119 of 122 errors are undercounts** (auto +1, meshes +2 or more); 3 are auto +1 where the meshes say 0. In 98
    the mesh has a crossing between the two points that the mask merged or missed. In 21 the mesh winding jumps with
    no crossing between (a label gap, e.g. w035 in #2167); excluding those gives 96.1 %.
  - Errors cluster by angle (theta 120 / 150 deg: 81 % / 86 %, inside the #2167 gap start; 240-330 deg: 97-98 %), by
    z (4064-4192: ~90 %) and by winding (w034 75 %, w040 47 %, w041 50 %, w049 55 %, w026 88 %; most others >= 97 %).
  - The band is harder than the 87 sites: longer rays, more compressed sheets at r 600-1300, and more w033-w049
    trouble. The 98 % threshold learned on the sites does not transfer.
- **Controls** (same probes, 15-deg grid). The oracle has 552 rays, 1,178 chains, 17,766 points and 671 chain breaks
  where mesh windings fail to increase. Corrupt-10 % has 1,732 of 16,588 steps flipped and 89.6 % agreement. Both
  parse with the pinned loader.
- **Full 15-deg grid** (band_m7/, all 4,929 crops; solved across three sessions because of two reboots, ~1.45 s per
  crop wall time):

| cut | chains / points | matched steps | all | in #2167 | out #2167 | clean | clean, outside w033-w037 |
|---|---|---|---|---|---|---|---|
| p_err <= 0.0461 (frozen) | 3,697 / 20,770 | 5,025 | 94.8 | 93.4 (1,444) | 95.4 (3,581) | 95.7 | 96.3 (4,294) |
| p_err <= 0.014 | 1,865 / 5,694 | 1,221 | 97.4 | 96.2 (211) | 97.6 (1,010) | 98.0 | 97.8 (1,098) |

  - 0.014 was chosen on the 30-deg angles: 98.3 % (650) there vs **96.3 % (571) on the unseen odd angles**. Use 96-97 %
    as the honest band number at that cut.
  - Matched-step quartiles by p_err (frozen chains): best 25 % = 97.2, 50 % = 96.0, 75 % = 95.6, all = 94.8.
  - Errors: 260, of which 243 are undercounts. 44 are mesh label gaps; precision excluding them is 95.7 %. By angle,
    theta 240-345 deg gives 95-98 % and theta 105-165 deg (around the #2167 gap) gives 69-89 %.
  - Loader check: all points annotated, 0 decreasing chains, max dtheta 2e-4 rad. Spacing 16 would drop 14.1 % /
    4.7 %.
  - Steps touching w041/42/45/46 are 67 % "correct". The mesh oracle itself is wrong in places, so the true precision
    sits somewhat above these numbers, by an unknown margin.

## 5. Commands
```
uv run python tools/auto_windings.py sector
uv run python tools/auto_windings.py validate --fit --out data/scratch/auto_windings/val_m7          # ~5 min, 14 procs
uv run python tools/auto_windings.py validate --seg data/preds/xing_m7d300_s11836_seg --out data/scratch/auto_windings/val_m7d300
uv run python tools/auto_windings.py band --z 3000 4500 --dtheta 15 --out data/scratch/auto_windings/band_m7
uv run python tools/auto_windings.py band ... --reassemble --conf 0.02 --max-len 20   # re-cut without re-solving
uv run python tools/auto_windings.py oracle --z 3000 4500 --dtheta 15 --out data/scratch/auto_windings/band_oracle
uv run python tools/auto_windings.py corrupt --src data/scratch/auto_windings/band_oracle/relative_windings.json --frac 0.1 --out data/scratch/auto_windings/band_corrupt10
uv run python tools/auto_windings.py compare --auto data/scratch/auto_windings/band_m7_d30 --oracle data/scratch/auto_windings/band_oracle
uv run --with tqdm --with einops python tools/auto_windings.py check data/scratch/auto_windings/band_m7_d30/relative_windings.json
uv run python tools/auto_windings.py band --z 3000 4500 --dtheta 30 --reassemble --from-crops data/scratch/auto_windings/band_m7/crops --out data/scratch/auto_windings/band_m7_d30   # add --conf 0.014 for the strict cut
```
- **Runtime note.** THP is "always" on this machine. numpy's madvise(HUGEPAGE) made the workers spend 2x more time in
  the kernel than in user code (compaction under memory pressure). The tool sets NUMPY_MADVISE_HUGEPAGE=0, which
  halves the wall time.

## 6. Caveats and next steps
- **m7 may have seen PHerc0139 labels.** The m7d300 result is the held-out number, and it is as good or better.
- **Truth is the human meshes, flaws included.**
  - Loose meshes: ~2 vox off.
  - Coincident pairs: w041/w042 and w045/w046.
  - The w033-w037 band.
  - Band-level oracle comparisons report a "clean" subset that excludes steps touching the blacklisted pairs and
    w033-w037.
- **Band certifier gap.** Sites give 98 %; the band vs meshes gives 94.8 % (frozen) and 96-97 % (0.014). Next:
  recalibrate p_err on band mesh-oracle steps with leave-z-slab-out folds, add radius/compression features, and
  score against the CT where the meshes are suspect (w034, w040/41, w049).
- **Not yet tested: whether the fitter improves.** Next is the A/B fit on the band, auto vs oracle vs corrupt vs
  none, scored with eval/spiral_eval.py, with pcl_unattached_pcl_min_point_spacing = 0.
- **Levers.**
  - Better masks: m7x/m7d300 give 70 % / 49 % coverage at the same precision, and m7x would need a GPU pass over the
    band.
  - Allowing step 2 where a second crop agrees.
  - Linking chains across rays.

## Why auto chains didn't move the fit (2026-10-20 late; data/scratch/annot_diag/diag.py, lookups in arm0's fit)

| chains | n | points | median length | median winding span | steps arm0 already agrees with | chains fully consistent with arm0 |
|---|---|---|---|---|---|---|
| auto (band_m7) | 3,697 | 20,770 | 4 | 3 | 84.5 % | 48.9 % |
| auto strict (c014) | 1,865 | 5,694 | 2 | 1 | 85.2 % | 70.2 % |
| oracle | 1,178 | 17,766 | 14 | 17 | 71.1 % | 7.4 % |

- Auto chains are short and mostly restate what the fit already does; oracle chains span ~17 windings and nearly all
  contradict arm0 somewhere. The fit result (oracle +14.5 pp, corrupt10 +13.2 pp at ~90 % step precision, auto ~0)
  points at chain length / long-range counting, not step precision.
- Next arms: long auto chains (per-ray, stitched across crops, loose cut), oracle labels on auto's layout, oracle cut
  into length-4 pieces.

## Long-chain arms for the fit A/B (2026-10-20 night)
New in tools/auto_windings.py:
- `band --bridge 1`: steps with no common crop, negative steps and steps above --max-step count +1 instead of cutting.
- `relabel`: oracle windings on another chain layout.
- `split --n 4`: consecutive pieces share their boundary point, so every step is kept.
- `compare` now also reports pair precision by true winding distance, error classes, error clustering, and chain
  length/span.

Chains are already stitched along the whole ray: crossings are deduplicated across the 96-vox crop overlaps, and each
step comes from the most central crop holding both ends. Chains were short only because of the cut rules.

Cut sweep (long_sweep/, 15-deg grid, vs band_oracle):

| cut | bridge | max step | chains | median len / span | step precision | pairs 6-10 / 11-20 exact |
|---|---|---|---|---|---|---|
| 0.0461 | 0 | 1 | 3,697 | 4 / 3 | 94.8 | 65 / 37 |
| 0.2 | 0 | 2 | 3,831 | 7 / 6 | 90.3 | 49 / 21 |
| 0.35 | 0 | 2 | 2,725 | 10 / 9 | 89.0 | 45 / 18 |
| **0.35** | **1** | **2** | **1,895** | **14 / 13** | **88.9** | **45 / 19** -> band_m7_long |
| 1.0 | 0 | 2 | 1,634 | 15 / 15 | 88.2 | 44 / 18 |
| 1.0 | 1 | 2 | 664 | 65 / 68 | 88.1 | 45 / 19 -> band_m7_ray |

- **Errors in band_m7_long** (1,058 of 9,535 matched steps):
  - 717 undercounts at contacts (the missed mesh sheet is <= 8 vox from a neighbour);
  - 121 spaced undercounts;
  - 122 label gaps;
  - 98 overcounts.
  - Errors are not clustered much: 25 % sit on the worst 10 % of rays.
- **The errors are biased, not random.** Over true distances of 10-20 windings, the mean signed error is -1.48
  windings for band_m7_long and -1.55 for band_m7_ray, with |err| <= 1 for 59 % / 56 % of pairs. corrupt10 gives
  +0.05 with 86 % of pairs within 1. Auto chains compress the count; corrupt10's errors average out.
- **Run length can't flag merged sheets.** 6.9 % of mask runs cover two mesh sheets, and their run length mostly
  looks single. Flagging runs of >= 10 vox catches 13 % of merged runs at 43 % precision.
- **Long chains need `pcl_unattached_pcl_min_point_spacing = 0`**: the default 16 drops 22 % of their points.
- **diag.py** (arm0 agreement with the chain steps / chains fully consistent with arm0):
  - band_m7_long: 78.1 % / 11.9 %;
  - band_m7_ray: 77.0 % / 0.2 %;
  - band_oracle_on_auto: 79.9 % / 50.5 %;
  - band_oracle_len4: 71.1 % / 47.3 %;
  - for reference, band_oracle: 71.1 % / 7.4 % and corrupt10: 65.1 % / 6.1 %.

## New band masks, same rays and cut (2026-10-21; band_m7x_long, band_m7sep03x_long)
- **Setup.** `band --mask-url <local zarr> --cache <own chunk cache>`. Driver: data/scratch/auto_windings/run_newmasks.sh
  (resumable).
  - Masks: data/scratch/p0139_spiral/masks/{m7x,m7sep03x}.
  - Checked: z chunks 14-24 cover the band, values are {0, 255}, and the level-3 radial extent is reused from the m7
    cache, so all 4,929 crops are identical.
  - Run time: ~1.6 h per mask on 11 CPU workers.
- **Results** at cut 0.35 / bridge 1 / max step 2, vs band_oracle:

| arm | chains | median len / span | step precision | under at contacts | spaced under / label gap / over | drift over 10-20 windings |
|---|---|---|---|---|---|---|
| band_m7_long | 1,895 | 14 / 13 | 88.9 | 7.5 % | 1.3 / 1.3 / 1.0 % | -1.48, 59 % within +-1 |
| band_m7x_long | 1,403 | 23 / 23 | 90.0 | 6.6 % | 0.9 / 1.2 / 1.3 % | -1.20, 67 % |
| band_m7sep03x_long | 1,797 | 15 / 15 | 89.3 | 7.1 % | 1.1 / 1.3 / 1.2 % | -1.26, 66 % |

  - Sweep tables: long_sweep_m7x/ and long_sweep_m7sep03x/.
  - The frozen cut (0.0461) gives 95.0 / 95.1 % step precision.
- **Site check at the frozen threshold:** m7sep03 (xing_m7sep03_s11836_seg) gives 97.6 % at 59.3 % coverage, and
  99.3 % at 65 % outside w033-w037. m7 gives 98.1 % at 58 %; m7x gives 98.2 % at 70 %.
- **diag.py:**
  - m7x_long: arm0 agrees 79.3 %, chains fully consistent with arm0 6.4 %.
  - m7sep03x_long: 78.6 % / 10.5 %.

## Fixing the undercount drift with lasagna grad_mag (2026-10-21; band_m7x_ray, band_m7x_{ray,long}_gm[20])
- **band_m7x_ray** (cut 1.0 / bridge 1 / max step 2, re-assembled from the m7x crop solves): 645 chains / 42,194
  points, median span 71, step precision 89.3 %, drift over 10-20 windings -1.23 with 66 % within +-1.
- **Field.** data/p0139/inputs/PHerc0139_grad_mag.ome.zarr: 4-vox grid in the 9.362 frame, density = u8 / 1000
  windings per voxel. For each chain step, ∫ density dl is taken between the two points (nearest sample, 1-vox steps;
  band cached as gm_band_z747_1128.npy).
  - Raw ∫ is ~1.2x high: median 1.24 on true single steps, 1.64 on true double steps (wide overlap).
  - Per step it is weak. Among chain steps of 1, AUC for "truth is 2" is 0.69 for ∫ alone, 0.67 for ∫ relative to
    neighbours, and 0.57 for gap relative to neighbours.
  - Counting with grad_mag alone over 10-20 windings is useless: sd 3.0 vs 1.28 for the chain, and smoothed hybrids
    stay at sd ~2.7.
  - Mask run thickness adds nothing. The CT profile was skipped because the CT isn't local.
- **Rule.**
  - A logistic P(true = 2 | chain = 1) on: ∫, ∫ relative to neighbours, gap relative to neighbours, log gap, local ∫
    level, log p_err, and bridged.
  - Fitted on even angle indices only, with the threshold set for zero drift on that half (0.14). AUC on odd angles
    is 0.73.
  - `gmfit` writes tools/auto_windings_gm.json; `gmfix --src <chains> [--thr]` applies it.
- **Held-out (odd angles), vs band_oracle:**

| arm | step precision | under (at contacts) | over | drift over 10-20 windings | within +-1 | sd |
|---|---|---|---|---|---|---|
| m7x_ray | 88.8 | 9.3 (7.1) | 1.9 | -1.30 | 63 % | 1.28 |
| m7x_ray_gm (0.14) | 84.0 | 6.6 (5.0) | 9.4 | +0.15 | 74 % | 1.67 |
| m7x_ray_gm20 (0.2) | 87.0 | 7.5 (5.7) | 5.5 | -0.53 | 77 % | 1.35 |
| m7x_long | 89.4 | 9.2 (7.0) | 1.4 | -1.29 | 64 % | 1.26 |
| m7x_long_gm | 84.3 | 6.6 (4.9) | 9.1 | -0.06 | 78 % | 1.50 |
| m7x_long_gm20 | 87.5 | 7.5 (5.6) | 5.0 | -0.64 | 78 % | 1.25 |
| corrupt10 (all rays) | 89.6 | | | +0.05 | 86 % | 1.07 |

- The fix trades a biased error for a roughly zero-mean one. About 30 % of contact undercounts are caught, at the
  cost of ~4 overcounts per caught undercount. The spread stays wider than corrupt10's.
- Across all rays, the rule raises 17 % of single steps (6,843 of 39,483 on ray_gm). Some of these lie inside the
  radius where the human meshes start (r < ~580), so they can't be checked.
- **Leakage.** 9 coefficients and the threshold were fitted on even-angle human labels in the same band. The odd-angle
  rows are the honest numbers.
- All gm arms load with the pinned loader: all points annotated, 0 decreasing chains. Spacing 16 would drop 24-25 %
  of points.
- **diag.py** (arm0 agrees / chains fully consistent):
  - m7x_ray: 78.8 % / 0.6 %;
  - ray_gm: 74.0 % / 0.2 %;
  - long_gm: 74.5 % / 2.8 %;
  - ray_gm20: 77.1 % / 0.5 %;
  - long_gm20: 77.7 % / 4.9 %.

## Leakage-free grad_mag rule: fitted on z 4500-6000, applied to z 3000-4500 (2026-10-21; band_m7_ray_gmx)
- **Fitting band.**
  - band_m7_z4500 (bucket m7 masks, same ray settings): 4,811 crops / 576 rays, ~1.7 h. The ray cut has 657 chains /
    43,404 points at 88.7 % step precision against band_oracle_z4500 (human meshes, 1,174 chains / 19,160 points).
  - The local grad_mag store only reached z ~4800. I ran `spiral/prep_inputs.py all --z0 4500 --z1 6000`
    read-only into data/scratch/auto_windings/lasagna_z4500. Where the stores overlap they are identical.
    `gm_volume` fills rows absent from data/p0139/inputs from that store.
- **Model.** `gmfit --half -1` on that band only gives tools/auto_windings_gm_z4500.json (11,656 single steps, 1,033
  truly double), zero-drift threshold 0.19. `gmfix` on band_m7_ray (z 3000-4500, unchanged) writes band_m7_ray_gmx:
  664 chains / 41,322 points, 7,270 of 38,562 single steps raised. It loads with the pinned loader.
- **z 3000-4500 vs band_oracle:**

| arm | model fitted on | step precision (even / odd angles) | contact under | over | drift over 10-20 windings | within +-1 |
|---|---|---|---|---|---|---|
| band_m7_ray | - | 88.8 / 87.4 | 7.4 / 8.1 % | 1.4 / 1.8 % | -1.55 | 56 % |
| band_m7_ray_gm_inband | z3000 even angles, thr 0.16 | 84.5 / 82.9 | 5.1 / 5.5 % | 8.8 / 9.8 % | -0.03 / +0.01 | 74 / 71 % |
| band_m7_ray_gm_inband_all | z3000 all rays, thr 0.16 | 84.6 / 83.1 | 5.1 / 5.5 % | 8.6 / 9.5 % | -0.03 | 72 % |
| **band_m7_ray_gmx** | **z4500-6000, thr 0.19** | **85.1 / 84.0** | **5.2 / 5.6 %** | **8.1 / 8.6 %** | **-0.09** | **73 %** |

- **Transfer.**
  - With the z4500 model, the threshold that would zero the drift on z 3000-4500 is ~0.18; the transferred 0.19 leaves
    -0.09. Sensitivity is ~0.15 windings per 0.01 of threshold (0.16: +0.40; 0.21: -0.30).
  - Coefficients (intercept, I, Ir, Lr, logL, Iloc, logperr, bridged):
    - z4500: -2.62, -0.28, 0.97, -0.39, 0.05, 0.22, 0.62, -0.31;
    - z3000 all rays: -2.74, 0.32, 0.43, -0.07, -0.33, 0.08, 0.61, -0.29.
  - I and Ir are collinear, so the two fits split the weight differently, but the predictions agree.
- **Per-slab zero-drift threshold (500 z):** 0.16 / 0.17 / 0.15 in z 3000-4500 (in-band model) and 0.18 / 0.19 / 0.19
  in z 4500-6000. Stable to +-0.02 within a band.

## Step confidence for gmx chains: precision vs drift (2026-10-21; band_m7_ray_gmx_{p93,dsafe})
Goal: raise gmx step precision (84.6 %) to >= 92 % keeping >= 60 % of points and |drift 10-20| <= 0.2, with nothing
fitted or thresholded on z 3000-4500 labels. CPU only, ~1 min per stage.
- **Code.** tools/auto_windings_stepconf.py. `feats --band z3000|z4500` re-assembles the ray chains from the crop
  checkpoints, checked identical to band_m7_ray / band_m7_z4500_ray, and to band_m7_ray_gmx after the gm rule. It
  keeps every graph feature per step. Driver scripts and outputs are in data/scratch/auto_windings/stepconf/:
  sel_z4500.py, sweep2_z4500.py, mask_labelfree.py, apply_z3000.py, write_sets.py, table_z3000.json.
- **Training data: z 4500-6000 only.** band_m7_z4500_ray + band_oracle_z4500.
  - The gm rule is applied out of fold there: it is refitted with each 500-z slab left out, threshold 0.19. Its
    precision is then 86.6 % (z 3000-4500: 84.6 %), and gm_p behaves as it does on an unseen band.
- **Features** (per step):
  - d, raised, bridged, graph-claimed 2;
  - gm: gm_p, ∫, ∫ relative to neighbours, gap, local ∫ level, coverage, ∫/level - d;
  - graph: residuals (step, ends, ray), dfrac, degree, run lengths, grazing, multi, second-crop agreement and d
    difference, crops holding the step, mask runs merged into the step, offset from the crop centre;
  - neighbour rays (z +-64, theta +-15): this step's d minus the neighbour's interpolated winding change over the same
    radii, and the same over a +-48 vox window;
  - chain context: raised steps within +-3, gm_p of adjacent steps, mean residual.
  - The site certifier's logperr is left out of the model (`NO_SITE`): with it the AUC is identical.
- **Model.** Logistic P(step correct), 3-fold by z slab on z 4500-6000.
  - OOF AUC: 0.774 (gradient boosting 0.773; gm-only features 0.776). It is almost all d and gm_p. Within final d = 1
    steps the AUC is 0.60.
  - Refitted on all of z 4500-6000 and frozen: AUC 0.832 on z 3000-4500.
- **Policies.** (a) drop steps below the threshold, splitting the chain; pieces of >= 2 points are kept. (b) the same,
  then drop pieces of < 5 points. Class thresholds use t1 for d = 1 steps and t2 for d >= 2 steps. Mask consensus is a
  parameter-free rule fixed before scoring: drop a step if the m7x and/or m7sep03x crossings strictly inside
  (r_a + 3, r_b - 3) on the same ray do not number d - 1 (`any` = either mask disagrees, `both` = both disagree).
- **(c) Per-point weights: not supported** by the pinned fitter (villa-0fb38c45).
  - point_collection.py:61-69 reads only p / wind_a / creation_time.
  - The strip loss weights are global: loss_weight_unattached_pcl_radius 2.0 and _dt 4.0 (config.py:394-395).
  - pcl_sampling_weights is per source-file stem (losses.py:22-40). member_weights is the internal component size
    (fit_spiral.py:995).
  - Weighting points or chains needs a code change.

z 3000-4500 vs band_oracle. The model and thresholds come from z 4500-6000; the labels here are used only to score.

| policy | kept steps | kept points | chains | median len (p90) | step precision | drift 10-20 | within +-1 | pairs 10-20 |
|---|---|---|---|---|---|---|---|---|
| gmx, no filter | 100 | 100 | 664 | 65 (91) | 84.6 | -0.09 | 73 % | 47,564 |
| (a) top 90 % of steps | 90.0 | 96.3 | 3,205 | 6 (33) | 87.6 | -0.51 | 79 % | 31,731 |
| (b) top 90 % | 84.3 | 87.4 | 1,816 | 13 (45) | 87.8 | -0.51 | 79 % | 31,731 |
| (a) top 80 % | 80.0 | 90.5 | 4,878 | 5 (17) | 91.9 | -0.87 | 82 % | 13,757 |
| (b) top 80 % | 70.4 | 75.6 | 2,616 | 9 (22) | 92.4 | -0.87 | 82 % | 13,757 |
| (a) top 70 % | 70.0 | 81.4 | 5,188 | 4 (14) | 93.0 | -0.79 | 85 % | 8,861 |
| (b) top 70 % | 58.8 | 63.9 | 2,482 | 9 (19) | 93.7 | -0.79 | 85 % | 8,861 |
| (a) top 60 % | 60.0 | 71.7 | 5,246 | 4 (12) | 93.8 | -0.76 | 87 % | 5,663 |
| (a) top 50 % | 50.0 | 62.1 | 5,351 | 3 (10) | 94.6 | -0.62 | 92 % | 2,493 |
| (b) top 50 % | 36.4 | 40.6 | 1,961 | 7 (14) | 95.3 | -0.62 | 92 % | 2,493 |
| (b) class thr, P-first at z4500 thresholds (t1 0.886, drop d >= 2) | 50.5 | 55.2 | 2,272 | 8 (17) | 94.2 | -0.76 | 87 % | 6,333 |
| **(b) p93**: P-first, t1 0.857 reset label-free to 65 % of points | 59.9 | 65.0 | 2,519 | 9 (19) | **93.6** | **-0.79** | 85 % | 8,998 |
| (a) D-safe (t1 0.894, t2 0.235) | 74.2 | 86.3 | 5,464 | 3 (15) | 85.9 | -0.21 | 82 % | 14,042 |
| **(b) dsafe**: D-safe, z4500 thresholds | 61.5 | 65.7 | 2,170 | 9.5 (25) | **87.4** | **-0.21** | 82 % | 14,042 |
| (a) mask consensus, any | 79.5 | 91.1 | 5,342 | 5 (15) | 92.4 | -0.80 | 84 % | 11,782 |
| (b) mask consensus, any | 68.4 | 73.9 | 2,738 | 9 (20) | 93.1 | -0.80 | 84 % | 11,782 |
| (a) mask consensus, both | 85.7 | 95.5 | 4,642 | 5 (19) | 91.1 | -0.85 | 80 % | 16,659 |
| (b) mask consensus, both | 77.2 | 82.4 | 2,684 | 10 (24) | 91.7 | -0.85 | 80 % | 16,659 |

- **No set meets all three targets, and none of these step filters can.**
  - On z 4500-6000 (OOF, 396 combinations of t1, t2 and min piece length), every set with >= 92 % precision and
    |drift| <= 0.2 keeps <= 22 % of points at min piece length 5. With min piece length 2 it keeps <= 43 % (logistic)
    or <= 54 % (boosting), and median pieces are 2 points long, so no linkage is left.
  - With >= 60 % of points and |drift| <= 0.2, the best precision is 88.4 % at min piece length >= 5. Boosting reaches
    91.4 % only with median pieces of 3 points, where 502 of 64,628 pairs 10-20 survive.
  - Reason (z 4500-6000 composition):
    - Final d = 1 steps are 92.5 % correct. Their errors are undercounts (6.8 % under, 0.8 % over), and the best
      10 % by score still undercount 2.4 %.
    - Raised steps are 34 % correct; the best decile is only 64 %.
  - So >= 92 % means keeping mostly d = 1 steps, and that leaves the one-sided undercount: about -0.05 x 14 windings
    ≈ -0.6 to -0.8. Cancelling it needs about 11 % of kept steps to be raised ones, which caps precision at ~88-89 %.
  - The other masks do not see the missing sheet either. Between the ends of a d = 1 step, m7x / m7sep03x show an
    extra crossing in only 1.7 % / 1.3 % of steps. So mask consensus mostly drops raised steps (69 % have no extra
    crossing) and ends up in the same place as the confidence filter.
- **Chosen sets** (format identical to band_m7_ray_gmx; pinned loader: all points annotated, 0 decreasing chains;
  spacing 16 would drop 21 % / 20 %). They are the two ends of the trade-off, for a fitter A/B:
  - **band_m7_ray_gmx_p93**:
    - z 3000-4500: 2,519 chains / 26,862 points (65.0 %), median 9 / span 8. 93.6 % step precision (clean 94.7 %,
      clean outside w033-w037 95.6 %); pairs 11-20 exact 33.6 % vs 28.2 % for gmx. **Drift -0.79**, back towards
      band_m7_ray's -1.55.
    - z 4500-6000 OOF: 93.0 % / 74.5 % / -0.67.
    - Rule: drop every d >= 2 step and every d = 1 step with P < t1, keeping pieces of >= 5 points. t1 was reset on
      the unlabelled z 3000-4500 scores to keep 65 % of points, because the z4500 t1 kept only 55 % there.
  - **band_m7_ray_gmx_dsafe**:
    - z 3000-4500: 2,170 chains / 27,160 points (65.7 %), median 9.5 / span 10. 87.4 % (+2.9 pp), **drift -0.21**.
    - z 4500-6000 OOF: 88.2 % / 74.4 % / -0.19.
    - All thresholds come from z 4500-6000.
- **Leakage audit.**
  - No z 3000-4500 label was used to fit the model, the thresholds or the min piece length.
  - Soft spots:
    1. The site certifier p_err (tools/auto_windings_conf.json) was fitted on the 87 PHerc0139 test sites, ~35 of
       them at z 2900-4600. It is out of the new model, but it still reaches it through gm_p (logperr is a gmx
       feature, pre-existing). Its weight is small: the AUC is unchanged without it.
    2. p93's t1 was reset after I had seen z 3000-4500 scores for the z4500 thresholds. The reset uses only a
       label-free point count, but the precision of the un-reset rule had already been seen.
    3. The m7 / m7x / m7sep03x masks may have seen PHerc0139 labels in training (as before).
    4. The mask-consensus rules have no fitted parameters, but they could only be scored on z 3000-4500 (m7x and
       m7sep03x are not on disk for z 4500-6000). Both variants are reported, not picked.
- **Next.**
  - A/B p93 vs dsafe at weight 10 / spacing 0 on one L40S each (~$4.5 per arm), to learn whether switches (label
    errors) or wrong area (drift) responds.
  - Beating the frontier needs new evidence of contact undercounts. Candidates: cross-ray sheet linking (z +-64,
    where contacts often separate), the CT profile, or a mask that separates contacts. Per-chain weights would need a
    fitter patch.
