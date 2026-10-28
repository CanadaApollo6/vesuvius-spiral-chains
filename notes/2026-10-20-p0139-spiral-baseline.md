# 2026-10-20: PHerc0139 baseline spiral fit (arm 0), z 3000-4500 on 20250728140407 (9.362 um)

This is arm 0 of the A/B: the villa spiral fit using only published inputs. Later arms swap in
our masks (tracks + SDT) and our automatic winding annotations (`relative_windings.json`), and keep everything else
the same.

Code is in `spiral/`:
- `prep_inputs.py`: umbilicus and lasagna moved from the 2.399 frame to the 9.362 frame. Runs locally.
- `sanity_checks.py`: the checks below.
- `extract_tracks.py`: tracks for the band.
- `fit_ckpt.py`: the pinned fit CLI plus autosave and resume.
- `run_arm.sh`: one arm on the pod, end to end and resumable.
- `arms/arm0.env` and `arms/TEMPLATE.env`.
- `export_fine.sh`: re-exports an arm's windings on a finer grid (default 4 vox) from its `checkpoint_fitted.ckpt`.
- `pull_arm.sh`: copies an arm's outputs back, including fine-grid exports.
- `env/pod_setup.sh` gained `FIT_ONLY=1`, which skips VC3D and the ink checkpoints.

Pins: villa `0fb38c45` (fit, SDT, extractor, packer) with the pherc343 shell-optional patch. Recipe overrides come
from `config/recipe.toml` (no patches, phase spacing, 30k steps). Sense is ACW, from the catalog (z top-to-bottom
True, left-handed False).

## Blocker found and how it was handled: no published tracks in the band
The published tracks (`dl.ash2txt.org/.../PHerc0139/20250728140407/tracks/*.extract.json`) have
`effective_z_bounds = [4500, 17500]`. The band z 3000-4500 has **no published tracks**.

Arm 0 therefore extracts tracks for the band from the bucket m7 mask, with the pinned villa extractor
(`extract_surface_tracks.py`, its own slab functions unchanged) and the published parameters:
- downsample 2;
- ribbons 2 deep every 8 slices;
- vertical slabs 4 thick every 16;
- dust 400, max area 640000;
- interjoint paths of >= 10 vertices.

Details:
- Extraction range: z 2744-4756, i.e. the band +- 256. The fitter keeps only tracks entirely inside [3000, 4500), as
  it does with the full published DBM.
- The wrapper reads the mask once in chunk-aligned blocks and farms slabs out to 12 processes, with Kimimaro
  `parallel=1` per process.
- After extraction: the pinned packed store, then the pinned `build_track_crossings.py` over the whole DBM.
- Result: 1,159,740 tracks in 388 s. Of these, 814,807 tracks and 61.4 M points fall in the ROI, with 6.49 M
  crossings.
- This is also what an "our masks" arm needs, so arms 0 and 1+ share one extractor. Published tracks for z >= 4500
  were never used.

## Inputs moved from 2.399 to 9.362
The affine is `data/eval/surface_labels_2026_07_sample/transform_0139_2p399_to_9p362.json`, with
p9 = A p2 + t (xyz). Derived from the matrix:
- det > 0, isotropic scale 0.25708 (singular values 0.25722 / 0.25709 / 0.25693);
- a proper rotation of 177.44 deg about an axis 0.35 deg off z.
- The normal (covector) map is L^-T ~ (1/s) [[-0.999, -0.038, 0.011], [0.038, -1.000, -0.006], [0.011, -0.005,
  1.000]]. So nx and ny change sign.

What was moved:
- **Umbilicus:** all 391 points, transformed. z runs 1440-20435 in the 9.362 frame, monotonic, with 14 points in the
  band. Written to `data/p0139/inputs/umbilicus.json`.
- **Lasagna nx/ny/grad_mag:** group 4 (16x of 2.399) resampled to a 9.362 store at lasagna_scale 4 (group "2", shape
  5244x1656x1656, 32^3 blosc-lz4, the same layout as PHerc0125). Only target rows 22-36 (z 2816-4736) are written.
  - Nearest neighbour; target voxel i sits at 4i, the fitter's convention.
  - Normals: n9 = normalize(L^-T n2), then negated where n9z < 0, because the stores keep only nx and ny and nz is
    taken as +sqrt.
  - Source rows were mirrored to `data/dl_p0139/lasagna_2p399/` (1.0 GB). The output is 1.05 GB. 6 min on CPU.
- **grad_mag units.** lasagna writes u8 = 1000 x windings per model-input voxel. `<run>.lasagna.json`
  `grad_mag_factor` converts that to per-L0-voxel: 0.25 for the PHerc0139 store (the model ran on L2), 1.0 for
  PHerc0125.
  - The fitter decodes u8 x 0.25 / 1000 x lasagna_scale (4), i.e. u8 / 1000 per working voxel, which is the
    PHerc0125 convention. A check on PHerc0125: median u8 111 gives a 9 vox pitch, which matches its ~10 vox gaps.
  - So u8_9 = u8_2 x 0.25 x |L^-T n| (~0.97 x). The median goes from 55 to 53, and 0 % saturates.
  - **The recipe's phase mode never reads grad_mag**, so this matters only for a grad_mag-mode arm.

## Sanity checks (before any fit), `data/scratch/p0139_spiral/`

### Umbilicus
`umbilicus_overlay.png`: CT L3 overview plus a 512^2 L0 inset at z 3000, 3500, 3750, 4000 and 4496. In every inset the
transformed umbilicus sits inside the hollow innermost turn.

Free radius = distance to the first m7 sheet, measured over 72 rays on m7 L1:

| z | umbilicus (median / min) | mirrored point (missing flip), median | 200 random in-scroll points (p50 / p99) |
|---|---|---|---|
| 3000 | 86 / 43 | 69 | 11.5 / 74.5 |
| 3500 | 75 / 30 | 14.5 | 11.2 / 52.7 |
| 3750 | 43 / 10 | 26.5 | 11.5 / 87.0 |
| 4000 | 43.5 / 30 | 12.0 | 10.5 / 54.5 |
| 4496 | 81.5 / 58 | 37.0 | 11.2 / 73.1 |

(vox)

Opposite-ray crossing asymmetry is 0.08-0.18 at the umbilicus and 0.32-0.46 at the mirrored point. This measure's own
argmin lands 345-403 vox away, in -y. The scroll is crushed asymmetrically, so equal layer counts on opposite rays
do not mark the core. I do not use that argmin; the insets settle the question. Offset from the CT-mask centroid is
0.10-0.18 R.

### Normals
44,713 m7 sheet voxels with structure-tensor coherence > 0.2, in 21 crops of 96^3 at z 3150 / 3750 / 4350 across 8
angular sectors. Comparison with the resampled lasagna normal:

| variant | median abs cos | frac > 0.9 | median on tilted sheets (abs(nz) > 0.25, n = 19,777) |
|---|---|---|---|
| **as built** | **0.9973** | **0.970** | **0.9977** |
| source vector, not rotated | 0.9085 | 0.521 | 0.6389 |
| nx <-> ny swapped | 0.7759 | 0.322 | 0.7933 |
| x-only sign flip | 0.6777 | 0.166 | 0.7140 |
| positions scaled, not rotated | 0.7423 | 0.243 | 0.7485 |

Because the stores are sign-free (nz = +sqrt), skipping the xy flip changes only the sign of nz. That shows up on
tilted sheets (0.64), and there are many in this band. Per-crop medians are 0.978-0.999.

### grad_mag
- Pitch along the structure-tensor normal on m7 (rising edges over +-40 vox): median 16.0 vox, n = 206.
- Pitch from the resampled grad_mag (1000 / u8): 11.9 vox. The ratio is 1.31.
- m7 undercounts sheets that touch, so its pitch is biased high. The units are consistent; a wrong frame would be off
  by ~3.9x.

## Pod run
Pod `<pod-id>` (vss-p0139-arm0):
- runpod L40S 48 GB, $1.1068/h, 32 vCPU (cgroup quota 27.2), driver 595.91 (CUDA 13.2), so the pinned cu130 torch
  2.13 was used.
- Root disk 120 GB; we used < 10 GB. `--disk-size 250` went to the network volume.
- Created 18:18:59 UTC. `pod_setup.sh` took ~1 min (the image had caches). `torch.cuda.is_available()` was True at
  setup.

| step | wall |
|---|---|
| pack lasagna normals/grad_mag (verify 500: all match) | 18 s |
| CT L4 (201 MB), m7 L0 rows 14-25 (1.2 GB), m7 L1 rows 7-12 (245 MB) | 14 s |
| tracks (extract, pack, crossings) | 6 min 31 s |
| surf-SDT (make_surf_sdt, level 1, CT L4 occupancy) | 66 s |
| fit, 30k steps, ~3.4 it/s, OMP 4 | 2 h 48 min 24 s (10,104 s; 18:31:57-21:20:21 UTC) |

Fit inputs as loaded: 814,807 tracks in the ROI; normals pool 0.78 GiB; SDT pool 2.9 GiB (the fitter built it
itself); 0 patches; no point collections.

Resume:
- `fit_ckpt.py` writes `checkpoint_autosave.ckpt` (686 MB) every 2500 steps.
- `run_arm.sh` resumes from the newest one.
- Tested on the pod: a second process loaded the step-2500 autosave and continued at 2500/30000 (shown as x/27,500).
  It was then killed by PID.

Two local reboots (~19:40 and ~20:03 UTC) killed the local session and watchers. The detached pod job was not
affected and was not restarted.

## Results (arm 0)
- **Fit:** 30,000 steps in 2 h 48 min. No errors and no restart.
- **Windings:** w010-w129, 120 windings (`save_mesh fitted: winding range [10, 130)`). Each has a plain and a
  `_spliced` tifxyz. With 0 patches, `_spliced` is byte-identical to the plain one (checked at step 4).
- **Satisfaction** (`run/satisfied_fitted.json`): 310,748 / 814,807 tracks satisfied (38.1 %); 45.8 M / 61.4 M track
  points (74.7 %). There are no patches to score.
- **Public-grid meshes:** `data/scratch/p0139_spiral/arm0/meshes/` at output_step_size 20 (scale 0.05; e.g. w060 is
  91 x 277), 83 MB.
- **Fine-grid meshes:** `data/scratch/p0139_spiral/arm0/meshes_step4/`, made by `export_fine.sh` at
  output_step_size 4 (scale 0.25; w060 is 455 x 1383), 2.0 GB, 39 s on the pod.
  - Method: resume from `checkpoint_fitted.ckpt` with only `output_step_size` overridden. The loop runs 0 steps;
    output_step_size is not a model-shaping key.
  - Same fitted model and same winding numbers as the 20-vox export.
  - Check: step-20 vertices lie on the step-4 surface, with nearest-neighbour distance p50 0.02-0.11 vox, p99
    0.16-0.42, max 2.7 (w020/060/100/125).
  - Valid fraction ~0.82 per grid. The invalid (-1) rows are the z-margin rows outside [3000, 4500).
- **Also in `arm0/`:**
  - `run/checkpoint_fitted.ckpt` (654 MB; re-render or resume);
  - `fit.log`, `overrides.json`, `arm_done.json`, `ds/` (spiral-scroll.json, band.json, umbilicus.json);
  - `logs/timings.txt`;
  - `export_step4/`;
  - the tracks `extract.json`.

## Cost
Pod `<pod-id>`, created 18:18:59 and terminated 21:23 UTC: 3.07 h x $1.1068/h = **~$3.40**. No other pods.

## Rerun an arm with other inputs
On the pod, after `FIT_ONLY=1 bash env/pod_setup.sh` and uploading `data/p0139/inputs/` to
`~/data/PHerc0139/inputs/`:

    cp spiral/arms/TEMPLATE.env spiral/arms/arm1.env   # set ARM, MASK_TAG and MASK_S3/MASK_LOCAL,
                                                                     # or TRACKS_DBM/SURF_SDT; REL_WINDINGS; FIT_OVERRIDES
    source ~/spiral_env.sh
    setsid nohup bash ~/vss/spiral/run_arm.sh ~/vss/spiral/arms/arm1.env > ~/work/p0139/arm1.out 2>&1 &
    # locally afterwards:
    # fine-grid (4 vox) re-export once arm_done.json exists, then pull:
    bash ~/vss/spiral/export_fine.sh ~/vss/spiral/arms/arm1.env 4
    bash spiral/pull_arm.sh arm1 root@<ip> <port>

How each arm variable maps:
- `MASK_*`: tracks and SDT are rebuilt from that mask, cached per `MASK_TAG`. The mask needs levels 0 and 1.
- `TRACKS_DBM` / `SURF_SDT`: become `paths.tracks_dbm` / `paths.surf_sdt` directly.
- `REL_WINDINGS`: copied to `<ds>/relative_windings.json`.
- `FIT_OVERRIDES`: JSON merged over the recipe overrides into `FIT_SPIRAL_CONFIG_OVERRIDES`.

## Arm 0 scored (eval/spiral_eval.py, z 3000-4500, tol 4, 4-vox export; data/scratch/spiral_eval/arms/arm0.*)
- Primary pairs (7,471 in band, 33 sites, none inside the #2167 gap): matched accuracy 78.0 [71.3, 84.0], coverage
  62.9 [57.1, 67.6], all-pairs accuracy 49.1; contact 57.7 [50.3, 65.3], open 80.2.
- Vertices (3.63M, stride 2): seam-aware wrong-winding area 48.8 % [46.4, 51.6], switches 48.5 / 1000 [39.7, 58.0],
  unmatched 21.5 %; contact 65.9 % wrong, 83 switches / 1000.
- Excluding the #2167 gap changes vertex numbers little (wrong 47.5 %, switches 43.3).
- So the baseline fit gets about half the human surface area onto the wrong winding in this band: large headroom
  for the annotation and mask arms. Runtime 681 s, 10.9 GB peak RSS for the 4-vox fit.

## Annotation arms 1-6 (2026-10-20/21)
Same band, bucket-m7 tracks and SDT, and recipe as arm 0. The only difference is `relative_windings.json` in the
dataset root (from `data/scratch/auto_windings/`). Overrides: `pcl_unattached_pcl_min_point_spacing` = 0.0 and
`loss_weight_unattached_pcl_radius` = 2.0 or 10.0. Both key names were checked in the pinned config.py (lines 313
and 394).

| arm | annotations | points | radius weight |
|---|---|---|---|
| arm1 | band_oracle (chains cut from the human meshes) | 1,178 chains / 17,766 points | 2.0 |
| arm2 | band_oracle | 1,178 / 17,766 | 10.0 |
| arm3 | band_m7 (auto, frozen cut, ~95 % step precision) | 3,697 / 20,770 | 2.0 |
| arm4 | band_m7 | 3,697 / 20,770 | 10.0 |
| arm5 | band_m7_c014 (auto strict, ~97 %) | 1,865 / 5,694 | 10.0 |
| arm6 | band_corrupt10 (oracle with 10 % of steps flipped) | 1,178 / 17,766 | 10.0 |

The weights are as specified. All six ran in parallel, so there was no early w2-vs-w10 read to adapt from. Weight 10
did not destabilise any fit: final total losses were 218-251, against arm 0's ~205.

**The strip loss was active in every arm.**
- The fit log shows `pcls: 0 cross-patch, N unattached` with N = 1178 / 1178 / 3697 / 3697 / 1865 / 1178, i.e.
  every chain. No chains were dropped for the z-roi, and spacing 0 means no decimation.
- `unattached_pcl_radius` is nonzero from step 0. At step 0 it is 74.5 (arm1), 135.6 (arm5) and 354.2 (arm6,
  corrupt). At step 29,800 it is 1.7 / 2.4 / 7.6 / 3.8 / 0.1 / 5.0.
- Final satisfied strip points: 93.6 / 96.3 / 81.7 / 91.4 / 95.8 / 95.1 %.
- With strips present, the fitter drops tracks within 16 vox of strip points: 400-1,700 of 814,845 tracks
  (`track radius loss: ... survive exclusion`).
- Re-extraction on the new pods gave 1,159,745 tracks, against 1,159,740 on the arm 0 pod. 814,845 of them are in
  the ROI (814,807 in arm 0). The extractor is slightly nondeterministic across pods; the difference is negligible.

**Scores** (eval/spiral_eval.py, z 3000-4500, tol 4, 4-vox exports, paired against arm 0, 33 sites;
`data/scratch/spiral_eval/arms/arm0_vs_armN.json`, table by `spiral/arm_table.py`):

| arm | pair acc (matched) | coverage | acc_all | contact acc | wrong area % (seam) | switches/1000 (seam) | unmatched % |
|---|---|---|---|---|---|---|---|
| arm0 (A) | 78.0 | 62.9 | 49.1 | 57.7 | 48.8 | 48.5 | 21.5 |
| arm1 | 92.5<br>+14.5 [+9.3, +20.8]* | 61.1<br>-1.8 [-7.2, +3.3] | 56.5<br>+7.4 [+3.5, +11.5]* | 70.2<br>+12.5 [+3.9, +20.0]* | 38.3<br>-10.5 [-12.8, -8.1]* | 57.0<br>+8.4 [+5.8, +11.8]* | 21.8<br>+0.3 [-0.2, +0.7] |
| arm2 | 92.1<br>+14.1 [+8.0, +21.2]* | 65.8<br>+2.9 [-1.5, +7.5] | 60.6<br>+11.6 [+5.8, +16.8]* | 67.6<br>+9.9 [+2.0, +17.1]* | 35.3<br>-13.5 [-16.0, -10.8]* | 56.7<br>+8.2 [+5.5, +11.4]* | 20.6<br>-0.9 [-1.3, -0.5]* |
| arm3 | 77.5<br>-0.5 [-2.9, +2.0] | 60.9<br>-2.0 [-5.2, +1.1] | 47.2<br>-1.8 [-4.8, +1.2] | 58.9<br>+1.2 [-4.9, +7.2] | 48.4<br>-0.5 [-0.7, -0.2]* | 52.1<br>+3.6 [+2.6, +4.5]* | 22.1<br>+0.6 [+0.2, +0.9]* |
| arm4 | 76.9<br>-1.2 [-7.2, +5.1] | 57.5<br>-5.4 [-10.7, -0.8]* | 44.2<br>-4.9 [-10.0, +0.6] | 58.9<br>+1.1 [-8.9, +11.5] | 46.4<br>-2.4 [-3.3, -1.5]* | 53.2<br>+4.7 [+3.4, +6.1]* | 21.4<br>-0.1 [-0.5, +0.2] |
| arm5 | 78.5<br>+0.5 [-6.7, +7.6] | 49.4<br>-13.5 [-19.2, -8.6]* | 38.8<br>-10.3 [-16.5, -4.0]* | 59.1<br>+1.4 [-9.5, +9.9] | 54.6<br>+5.8 [+4.4, +7.2]* | 53.0<br>+4.5 [+1.8, +7.2]* | 24.0<br>+2.4 [+1.6, +3.2]* |
| arm6 | 91.3<br>+13.2 [+7.5, +19.8]* | 66.1<br>+3.2 [-1.2, +7.3] | 60.3<br>+11.2 [+5.6, +16.6]* | 65.3<br>+7.6 [-0.6, +15.0] | 37.2<br>-11.6 [-14.1, -9.0]* | 63.7<br>+15.2 [+12.3, +18.6]* | 21.1<br>-0.4 [-0.9, -0.1]* |

Cells: B value, then B - arm0 [95 % CI]; * = CI excludes 0.

Reading:
- **Oracle chains help a lot.**
  - Matched pair accuracy rises from 78 to 92 (+14.5 at weight 2, +14.1 at weight 10).
  - Wrong-winding area falls from 48.8 % to 35-38 % (-10.5 / -13.5).
  - Weight 10 beats weight 2 on area (-13.5 vs -10.5), acc_all (+11.6 vs +7.4) and coverage. Its matched accuracy is
    the same.
  - **Caveat: these chains are cut from the same human meshes that the eval scores against.** This is an
    upper bound for perfect annotations at these probe positions, not a fair arm.
- **Robust to random errors.** With 10 % of steps flipped (arm6), most of the gain survives: +13.2 accuracy and -11.6
  area. Switches rise to +15.2 / 1000.
- **Auto chains give almost nothing.**
  - arm3/arm4: wrong area -0.5 / -2.4 (significant but small). Accuracy is flat; coverage is -2 / -5.4.
  - Switches are +3.6 / +4.7.
  - The strict cut (arm5) is worse than the baseline: coverage -13.5, area +5.8.
- **The gap is not explained by step precision.** corrupt10 (~90 % steps right) beats auto (~95 %) by a wide margin.
  Candidate causes, none tested yet:
  - auto chains are short (median 4 points vs ~15 for oracle chains), so each ties few windings together;
  - auto errors are systematic, the same m7 confusions the tracks already contain, while the corrupt10 flips are
    independent;
  - auto coverage is uneven over the windings that the human meshes cover.
- **Every arm raises seam-aware switches** (+3.6 to +15.2 / 1000). The strips pull locally, and the surface changes
  winding around them.

Wall time and cost:
- Fit times: 14,102 / 12,996 / 12,941 / 12,740 / 12,496 / 12,919 s (3.5-3.9 h; arm 0 took 10,104 s). Mean speed
  was ~2.3 it/s against ~3 it/s for arm 0. The extra cost comes from the strip loss plus sharing one physical runpod
  host.
- Step-4 exports: about 40 s each.
- Pods: 4 x L40S (arms 1-4) and 1 x 2xL40S (arms 5+6, one GPU each), on runpod NL.
  - A sixth single L40S was created and terminated unused after 2 min, once the 2x offer was taken.
  - Created 21:39-21:41 UTC; terminated 02:17 (arm1 pod) and 02:42 (the rest).
- Cost: 19.76 single-GPU h x $1.1068 + 5.02 h x $2.1968 = **~$32.9**.
  - About $3 of that is idle time: the first collector's `ssh` inside `while read` consumed the pod list, so pulls
    were serialised behind scoring. Fixed with `ssh -n` / `< /dev/null`.
- Scoring: local CPU, 495-529 s and up to 15.9 GB RSS per arm.

New code (spiral/):
- `arms/arm1..6.env`;
- `arms/pods_annot.tsv` (arm -> pod);
- `bootstrap_pod.sh`: copies spiral/, inputs and windings, and starts pod_setup;
- `pod_arms.sh`: several arms per pod, one GPU each; shared tracks/SDT are built once;
- `collect_and_score.sh`: pull, verify, terminate, then score one arm at a time; resumable via markers;
- `arm_table.py`.

Outputs: `data/scratch/p0139_spiral/arm1..6/`. Each holds `meshes/` (20 vox), `meshes_step4/`,
`run/checkpoint_fitted.ckpt`, `fit.log`, `ds/` (including relative_windings.json) and `logs/`.
