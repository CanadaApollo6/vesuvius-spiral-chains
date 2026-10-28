# 2026-10-21: spiral-fit annotation round 2 (PHerc0139 z 3000-4500, vs arm0; eval/spiral_eval.py)

All at unattached radius weight 10, min point spacing 0. B - arm0 [95 % CI], * = CI excludes 0.

| arm | chains | pair acc (matched) | coverage | acc_all | contact | wrong area (seam) | switches/1000 |
|---|---|---|---|---|---|---|---|
| 0 baseline | none | 78.0 | 62.9 | 49.1 | 57.7 | 48.8 | 48.5 |
| 2 oracle | human, len 14 | +14.1* | +2.9 | +11.6* | +9.9* | -13.5* | +8.2* |
| 6 oracle 10 % flipped | | +13.2* | +3.2 | +11.2* | +7.6 | -11.6* | +15.2* |
| 4 auto short | m7, len 4 | -1.2 | -5.4* | -4.9 | +1.1 | -2.4* | +4.7* |
| 7 auto long | m7, len 14, cut 0.35 | +10.0* | -8.6* | -1.2 | +5.7 | +1.8 | +12.0* |
| 8 auto ray | m7, len 65 | **+12.1*** | +2.5 | **+9.9*** | +3.0 | **+14.8*** (worse) | +6.9* |
| 9 oracle on auto layout | human labels, auto points | +10.5* | -1.1 | +5.6* | +10.2* | -6.3* | +2.5* |
| 10 oracle in 4-pt pieces | | +8.0* | -2.9 | +2.5 | +8.4* | -5.1* | +10.5* |

Reading:
- Long auto chains fix local counting: ray chains get 84 % of the oracle's matched-pair gain and most of its
  acc_all gain.
- But they wreck the global count: wrong-winding area +14.8 pp. Their undercount drift (-1.5 windings per 10-20,
  mostly touching sheets fused in the mask) compresses the fit radially. Random errors (arm 6) don't do this.
- Auto's layout is fine (arm 9 keeps 70 % of the pair gain with true labels), so the labels' bias is the lever.
- Shared-point 4-point pieces still link (arm 10 keeps a good part of the gain), so chain "length" is really
  linkage.
- Next: remove the undercount bias with a mask-independent count (lasagna grad_mag integral along each step),
  ray chains from m7x masks (drift -1.20), lower chain weight so tracks hold the global count.

## Round 2 run log (runner): arms 7-12, full table, masks, cost

Full paired table, arms 0-12 (`python3 spiral/arm_table.py data/scratch/spiral_eval/arms/arm0_vs_arm{1..12}.json`;
eval/spiral_eval.py, z 3000-4500, tol 4, 4-vox exports, 33 sites):

| arm | pair acc (matched) | coverage | acc_all | contact acc | wrong area % (seam) | switches/1000 (seam) | unmatched % |
|---|---|---|---|---|---|---|---|
| arm0 (A) | 78.0 | 62.9 | 49.1 | 57.7 | 48.8 | 48.5 | 21.5 |
| arm1 | 92.5<br>+14.5 [+9.3, +20.8]* | 61.1<br>-1.8 [-7.2, +3.3] | 56.5<br>+7.4 [+3.5, +11.5]* | 70.2<br>+12.5 [+3.9, +20.0]* | 38.3<br>-10.5 [-12.8, -8.1]* | 57.0<br>+8.4 [+5.8, +11.8]* | 21.8<br>+0.3 [-0.2, +0.7] |
| arm2 | 92.1<br>+14.1 [+8.0, +21.2]* | 65.8<br>+2.9 [-1.5, +7.5] | 60.6<br>+11.6 [+5.8, +16.8]* | 67.6<br>+9.9 [+2.0, +17.1]* | 35.3<br>-13.5 [-16.0, -10.8]* | 56.7<br>+8.2 [+5.5, +11.4]* | 20.6<br>-0.9 [-1.3, -0.5]* |
| arm3 | 77.5<br>-0.5 [-2.9, +2.0] | 60.9<br>-2.0 [-5.2, +1.1] | 47.2<br>-1.8 [-4.8, +1.2] | 58.9<br>+1.2 [-4.9, +7.2] | 48.4<br>-0.5 [-0.7, -0.2]* | 52.1<br>+3.6 [+2.6, +4.5]* | 22.1<br>+0.6 [+0.2, +0.9]* |
| arm4 | 76.9<br>-1.2 [-7.2, +5.1] | 57.5<br>-5.4 [-10.7, -0.8]* | 44.2<br>-4.9 [-10.0, +0.6] | 58.9<br>+1.1 [-8.9, +11.5] | 46.4<br>-2.4 [-3.3, -1.5]* | 53.2<br>+4.7 [+3.4, +6.1]* | 21.4<br>-0.1 [-0.5, +0.2] |
| arm5 | 78.5<br>+0.5 [-6.7, +7.6] | 49.4<br>-13.5 [-19.2, -8.6]* | 38.8<br>-10.3 [-16.5, -4.0]* | 59.1<br>+1.4 [-9.5, +9.9] | 54.6<br>+5.8 [+4.4, +7.2]* | 53.0<br>+4.5 [+1.8, +7.2]* | 24.0<br>+2.4 [+1.6, +3.2]* |
| arm6 | 91.3<br>+13.2 [+7.5, +19.8]* | 66.1<br>+3.2 [-1.2, +7.3] | 60.3<br>+11.2 [+5.6, +16.6]* | 65.3<br>+7.6 [-0.6, +15.0] | 37.2<br>-11.6 [-14.1, -9.0]* | 63.7<br>+15.2 [+12.3, +18.6]* | 21.1<br>-0.4 [-0.9, -0.1]* |
| arm7 | 88.0<br>+10.0 [+5.6, +14.4]* | 54.3<br>-8.6 [-14.0, -3.6]* | 47.8<br>-1.2 [-4.6, +1.8] | 63.5<br>+5.7 [-1.9, +12.6] | 50.6<br>+1.8 [-0.2, +3.7] | 60.6<br>+12.0 [+8.0, +16.5]* | 23.4<br>+1.9 [+1.4, +2.3]* |
| arm8 | 90.1<br>+12.1 [+5.3, +19.2]* | 65.4<br>+2.5 [-2.3, +7.0] | 58.9<br>+9.9 [+3.0, +16.5]* | 60.7<br>+3.0 [-7.8, +15.0] | 63.6<br>+14.8 [+11.8, +17.6]* | 55.4<br>+6.9 [+2.5, +11.4]* | 24.2<br>+2.7 [+2.2, +3.2]* |
| arm9 | 88.5<br>+10.5 [+6.0, +15.3]* | 61.8<br>-1.1 [-6.6, +4.0] | 54.7<br>+5.6 [+0.6, +10.3]* | 67.9<br>+10.2 [+1.2, +20.4]* | 42.5<br>-6.3 [-8.3, -4.3]* | 51.0<br>+2.5 [+0.7, +4.6]* | 20.8<br>-0.7 [-1.1, -0.3]* |
| arm10 | 86.0<br>+8.0 [+4.4, +12.3]* | 60.0<br>-2.9 [-7.5, +1.9] | 51.6<br>+2.5 [-0.9, +6.0] | 66.1<br>+8.4 [+2.1, +13.7]* | 43.8<br>-5.1 [-7.1, -3.1]* | 59.0<br>+10.5 [+7.5, +14.0]* | 20.9<br>-0.6 [-1.0, -0.3]* |
| arm11 | 86.1<br>+8.0 [+3.3, +12.4]* | 70.5<br>+7.6 [+1.7, +13.4]* | 60.6<br>+11.6 [+6.1, +16.9]* | 62.8<br>+5.1 [-9.3, +15.7] | 60.4<br>+11.6 [+10.0, +13.2]* | 60.4<br>+11.9 [+8.5, +15.5]* | 17.5<br>-4.0 [-4.7, -3.3]* |
| arm12 | 87.0<br>+8.9 [+1.2, +17.6]* | 58.9<br>-4.0 [-10.6, +2.9] | 51.3<br>+2.2 [-6.4, +11.1] | 64.1<br>+6.4 [-3.8, +15.3] | 47.5<br>-1.4 [-3.2, +0.3] | 47.4<br>-1.1 [-4.1, +2.3] | 25.4<br>+3.9 [+3.1, +4.6]* |

Cells: B value, then B - arm0 [95 % CI]; * = CI excludes 0.

What each new arm is:

| arm | input change vs arm0 | strips (chains / points / median) | weight |
|---|---|---|---|
| arm7 | + band_m7_long | 1,895 / 40,758 / 14 (step precision 88.9 % vs oracle) | 10 |
| arm8 | + band_m7_ray | 664 / 41,322 / 65 (88.1 %) | 10 |
| arm9 | + band_oracle_on_auto | 1,463 / 7,151 / 4 (100 %) | 10 |
| arm10 | + band_oracle_len4 | 5,974 / 22,562 / 4 (100 %) | 10 |
| arm11 | our m7 masks at 1.18x (tracks + SDT), no strips | - | - |
| arm12 | our m7sep03 masks at 1.18x (tracks + SDT), no strips | - | - |

Strips were active in arms 7-10.
- The fit logs show `pcls: 0 cross-patch, N unattached` with N = 1895 / 664 / 1463 / 5974 (every chain).
- `unattached_pcl_radius` at step 29,800: 9.2 / 3.0 / 0.2 / 14.4.
- Satisfied strip points: 83.2 / 94.1 / 97.3 / 91.3 %.
- Tracks dropped by strip exclusion: 3.6k / 3.8k / 0.6k / 1.3k of 814,845.
- Final total loss: 259 / 241 / 220 / 251 (arm0 215).
- Fit times: 12,642 / 12,768 / 12,623 / 12,842 s. arm11: 12,923 s; arm12: 12,321 s.

Masks for arms 11-12 (`spiral/predict_band.py`, `mask_pod.sh`, 8xA40 pod):
- Settings: nnU-Net fold 0, spacing = plans spacing x 9.362/7.91 (1.18x), argmax surface class 1 -> uint8 {0,255}.
  Blocks are 384^3 with a 32-vox margin. Blocks whose CT level 2 is all zero are skipped (608 in-scroll blocks).
- Coverage: z 2688-4800, full XY. That spans the L0 chunk rows 14-24 used for track extraction (z 2744-4756) and
  the L1 rows used for the SDT.
- Layout matches the bucket mask exactly: zarr v2, levels 0 and 1, chunks 192^3, blosc zstd 1 shuffle,
  dimension_separator "/", L1 = L0[::2,::2,::2].
- m7 used checkpoint_best (the only one). m7sep03 used **checkpoint_final**: eval/infer_nnunet.py load_predictor
  prefers final, so that was the only checkpoint uploaded.
- Throughput: 24 workers (3 per GPU) gave ~0.2 blocks/s. nnU-Net's CPU resampling is the bottleneck. m7 ran
  03:55-04:45; m7sep03 (16 workers, started 6 min later and run concurrently) finished at 05:05.
- Local copies, 1.3 GB each:
  - `data/scratch/p0139_spiral/masks/m7x/m7x.zarr`;
  - `data/scratch/p0139_spiral/masks/m7sep03x/m7sep03x.zarr`;
  - `.blocks/` holds per-block timing and surface fraction.

Alignment checks:
- Before the run, 3 local blocks of m7 1.18x vs the bucket mask: dice 0.759 / 0.774 at shift (0,0,0); all
  flips and swaps were far worse.
- The third block scored 0.302 because of a solid filled-chunk artifact outside the scroll in the bucket mask.
- The pod output for block 3456_3072_1920 was identical to the local test (dice 0.9999).
- m7sep03 vs bucket on two blocks: dice 0.834 / 0.820 (m7x vs bucket: 0.774 / 0.759).
- Surface fraction over the band: m7x 9.5 %, m7sep03x 13.4 %. On the two check blocks the bucket (th 0.2) had
  18.4 / 14.8 %, so our argmax m7 sheets are thinner.

Tracks from the new masks (ROI = tracks fully inside z 3000-4500):

| mask | tracks extracted | in ROI | directed crossing records | final loss | satisfied track points |
|---|---|---|---|---|---|
| bucket m7 (arm0) | 1,159,745 | 814,807 | 12,978,382 | 215 | 74.7 % |
| m7x (arm11) | 1,109,504 | 810,868 | 5,250,278 | 103 | 81.1 % |
| m7sep03x (arm12) | 1,082,298 | 758,007 | 5,681,112 | 204 | 67.8 % |

Reading for arms 11-12:
- **Our m7 masks alone (arm11) beat the bucket masks on counting but not on global count.**
  - acc_all +11.6, coverage +7.6, matched +8.0, unmatched -4.0. That matches arm2 (oracle strips) on acc_all.
  - Wrong area is +11.6 pp worse and switches +11.9, the same signature as arm8 (ray chains).
  - Thinner sheets give fewer than half the crossing records and a much lower final loss (103 vs 215), so the tracks
    constrain less. The fit is locally consistent but drifts in the global winding count.
- **m7sep03 masks (arm12)**: matched +8.9, but the CI is wide (+1.2 to +17.6). Area -1.4 and switches -1.1 are not
  significant, coverage is -4.0 and unmatched +3.9. It is the only arm so far that does not raise switches.

Oddities:
- **Bootstrap without windings.** On round 2's first launch the arms 7-10 pods had no windings.
  - The sets were not passed to bootstrap_pod.sh. I uploaded them by hand at 04:05, before run_arm.sh reached the
    dataset step (tracks/SDT finished at 04:15). The fits started with all strips; no rerun was needed.
- **Slow local scoring.** It took 1,000-1,370 s per arm instead of ~500 s.
  - An unrelated `tools/auto_windings.py band ... --mask-url .../m7sep03x.zarr` job (11 workers) was running on
    this machine at the same time and pushed it into 20 GB of swap.
- **arm0's loss.** The final total loss of arm0 is 215 in fit.log; the earlier note rounded it to ~205.

Cost, round 2 (all pods terminated; `prime pods list` shows no vss-p0139 pods):

| item | time | cost |
|---|---|---|
| 4 x L40S, arms 7-10 | 03:41-07:59, 17.20 GPU-h x $1.1068 | $19.04 |
| 8xA40 mask pod | 03:52-05:09, 1.29 h x $4.7368 | $6.11 |
| 2 x L40S, arms 11-12 | 04:05-08:51, 9.54 GPU-h x $1.1068 | $10.56 |
| **total** | | **~$35.7** (budget $90) |

## Round 3 (2026-10-21): drift-corrected ray chains, arms 13-19

All on bucket m7 tracks/SDT unless noted, min point spacing 0. Chains: `data/scratch/auto_windings/<set>`.
Full table: `uv run python spiral/arm_table.py data/scratch/spiral_eval/arms/arm0_vs_arm{1..19}.json`.

| arm | chains (weight) | pair acc (matched) | coverage | acc_all | contact | wrong area (seam) | switches/1000 |
|---|---|---|---|---|---|---|---|
| 0 baseline | none | 78.0 | 62.9 | 49.1 | 57.7 | 48.8 | 48.5 |
| 2 oracle | human (10) | +14.1* | +2.9 | +11.6* | +9.9* | -13.5* | +8.2* |
| 8 | m7 ray (10) | +12.1* | +2.5 | +9.9* | +3.0 | +14.8* | +6.9* |
| 13 | m7x ray (10) | +14.6* | +2.5 | +11.5* | +9.6 | +5.4* | +7.5* |
| 14 | m7 ray (2) | +12.4* | -6.7* | +1.8 | +4.8 | +14.4* | +4.9* |
| 15 | m7x ray (2) | +13.1* | -0.7 | +7.6* | +4.6 | +4.7* | +8.3* |
| 16 | m7x ray (10), m7x tracks/SDT | +13.9* | -0.2 | +8.5* | +10.5* | +5.1* | +5.7* |
| 17 | m7x ray + gm (fit on band, even angles) (10) | +12.4* | -4.3 | +3.9 | +10.4* | -4.0* | +21.7* |
| 18 | m7x ray + gm, threshold 0.20 (10) | +4.1* | -6.8* | -3.1 | +4.6 | -5.3* | +13.8* |
| **19** | **m7 ray + gmx (fit on z 4500-6000 only) (10)** | **+12.3* [7.9, 17.2]** | -3.1 | **+4.9*** | **+12.0*** | **-4.7* [-7.2, -2.1]** | +17.6* |

Reading:
- **arm19 is the claim arm and it holds.** The gm model saw no labels from the scored band. It keeps 87 % of the
  oracle's matched-pair gain and 85 % of its contact gain, and turns wrong-winding area from +14.8 (arm8, same
  chains uncorrected) to -4.7 (35 % of the oracle's -13.5). First fully automatic arm better than arm0 on both
  local and global winding.
- Cost: sheet switches +17.6 per 1000 (arm0 48.5). The gm-corrected arms switch most; label errors drive it
  (arm6, 10 % random flips: +15.2; arm9, true labels on auto layout: +2.5). Step precision is 84.6 % for gmx vs
  100 % oracle, so the next lever is precision (drop or down-weight uncertain steps), not more chains.
- m7x chains (13, 15, 16) cut the drift damage by two thirds vs m7 chains (+5 vs +15 area) but still compress.
- Weight 2 vs 10 changes little (14 vs 8, 15 vs 13); tracks do not hold the global count on their own.
- gm threshold 0.20 (arm18) over-corrects: area improves slightly more but pair accuracy collapses to +4.1.

Cost, round 3 (all pods terminated 15:34Z; `prime pods list` empty):

| item | time | cost |
|---|---|---|
| 3 x 2xL40S, arms 13-18 | 09:41-13:59, 25.8 GPU-h x $1.1068 | ~$28.6 |
| 1 x L40S, arm19 | 11:37-15:34, 3.95 h x $1.1068 | ~$4.4 |
| **total** | | **~$33** (budget $55) |
