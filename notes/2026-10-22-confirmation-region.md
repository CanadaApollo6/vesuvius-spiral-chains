# 2026-10-22: clean confirmation region for the spiral-fit annotation method (prep only; no pods, $0 spent)

**Decision: PHerc1667, volume 20251217075048 (2.399 um) at level 2 (9.596 um), z 6500-8000 (L2) over the full
slice.** That is 2.399 z 26000-32000. PHerc0139 z 6000-8250 is the prepared backup (section 7).

Why PHerc1667:
- It is the only other scroll with dense public human winding labels. The merged_v4 mesh is the full read, and 16
  single-wrap segments give an independent check.
- The chain models (conf/gm) were fitted only on PHerc0139, so on the method side a cross-scroll test is clean.
- No band of PHerc1667 has been looked at with any fit.

PHerc0172 and the other candidates have no winding-labelled meshes at about 8-9.4 um (bucket survey in
data/scratch/confirm_region/bucket_survey.json).

## 1. Region and inputs

| item | PHerc1667 status | how we got it |
|---|---|---|
| human meshes with winding labels | merged_v4 (2.399 tifxyz, 2061 x 30097 grid) | Unwrapped to k 11..41 by `eval/p1667_labels.py`. Labels are k = 39 - floor((Phi - 144)/360), with sign, seam and K0 fitted to agree with the single-wrap segments (81 % vertex agreement). 0 conflicting links; one component holds 99.999 % of vertices. |
| bucket m7 | **none** for this scroll | Made locally: `spiral/predict_band.py --level 2 --um 1 --ref-um 1 --threshold 0.2`, the bucket "-L2-th0.2" recipe from its metadata.json (m7_nnunet on CT level 2, native spacing, no TTA, P > 0.2). Output: `data/p1667/masks/m7_L2_th0.2.zarr`, levels 0-3 (levels 0/1 by predict_band; 2-3 nearest-downsampled), z 6144-8256. It covers 261 blocks (13.7 Gvox) and took 38 min of compute on the local 3090. Mean surface fraction is 7.6 %. Recipe check on PHerc0332 (same scanner setup, 2.399 µm): dice against bucket m7-L2 is 0.815 on an in-scroll block, the best shift is (0,0,0), and surface fraction is 19.6 % vs 20.2 %. The second block (dice 0.63 inside the CT) sits at the scroll edge, where the bucket is 52 % filled outside the CT (an artifact). |
| lasagna nx/ny/grad_mag | public, 20251217075048-lasagna-20260724 group 4 | `prep_inputs.py all --scroll PHerc1667 --z0 6500 --z1 8000 --margin 256`. Group 4 maps 1:1 onto target group 2 (L = 0.25 I). Verified identical: nx/ny 99.98 % exact (max diff 1), grad_mag 100 %. Rows 48..65, 454 MB fetched. |
| umbilicus | **none published** | Derived from the human mesh: a Kasa circle fit to the innermost winding per 64 z (`data/eval/p1667/umbilicus_p1667_L2.json`, 128 points). Must be disclosed. |
| CT for the SDT occupancy (recipe ct_mask_level 4) | 2.399 has no level 6 | `prep_inputs.py ct`: 2.399 level 5 rows of the band ±512, max-pooled 2x, giving `data/p1667/inputs/ct.zarr/4` with scale 16. Exact for make_surf_sdt's CT ≥ 1 test. Checked with villa's `_load_ct_occupancy`: scale (16,16,16), band occupancy 0.30. |
| scan geometry | L2 shape 9269 x 3808 x 3808 (zyx), 128³ chunks, uncompressed | spiral_outward_sense ACW, the same as PHerc0139 (one-turn radius check, sense_check.py) |

The band:
- 30 windings (k 11..41) cover the full 360°.
- 2,283,248 reference vertices at stride 2.
- 72 bootstrap blocks (30° x 250 z).
- 136,691 ray pairs.
- The core is hollow (r < ~450 vox).
- Sheet pitch is 26-28 vox for k ≤ 27 and 15-23 vox for k 28-41, against 10-16 in PHerc0139.

Why this band: it is the tallest 1,500-z band with every winding labelled over the full 360° on the merged mesh. Below
z ~6250 the outer k thin out (coverage table in data/scratch/confirm_region/p1667_coverage.json).

Local data:
- downloads in data/dl_p1667/ (1.3 GB: merged mesh, lasagna rows, CT L5 rows);
- derived inputs in data/p1667/ (inputs/, masks/).

## 2. Code changes (backward compatible)
- `eval/spiral_eval.py`:
  - `--scroll {p0139,p1667}` on main, build-oracle and corrupt.
  - p1667 loads the merged mesh labels as one surface per k, reads the p1667 umbilicus, and scores ray pairs
    (`--ray-dz 25 --ray-dtheta 3`) instead of frozen sites.
  - The reference cache is `ref_p1667_z...`.
  - **Check:** the PHerc0139 z 3000-4500 oracle rerun matches the stored `oracle.json` in all 1,362 values. Only
    peak RSS and the new `scroll` key differ.
- `release/fit_vertex_dump.py`: added `--scroll` and `--out-dir`.
- `eval/subset_wrong_area.py`: added `--scroll p1667 --z0 --z1`.
- `eval/label_subsets.py`: added `--scroll p1667`.
- `eval/ct_gap_audit.py`: added `--scroll p1667`, which uses CT L2, our m7 and PHerc1667 grad_mag, and avoids the 144° label seam.
- `spiral/prep_inputs.py`:
  - `--scroll PHerc1667` (identity affine, human-mesh umbilicus);
  - new `ct` subcommand.
- `spiral/predict_band.py`: added `--level` and `--threshold`. The defaults are unchanged.
- `spiral/run_arm.sh`:
  - SCROLL/VOL/UM/SENSE can be overridden from the arm file or env;
  - input names are keyed by $SCROLL;
  - `$DATA/inputs/ct.zarr` is used when present;
  - spiral-scroll.json and band.json carry the scroll.
- `bootstrap_pod.sh`: SCROLL, LOCAL_INP, MASK_DIR and WIND_ROOT.
- `collect_and_score.sh`: reads the scroll from band.json and passes `--scroll`. spiral_eval now runs under `flock /tmp/vss_eval.lock`.
- New `eval/p1667_labels.py`, which builds the labels and the umbilicus.
- Scratch scripts are in data/scratch/confirm_region/.

## 3. Scorer validation
Scores at z 6500-8000, tol 4: `score_oracle_vs_segs_z6500.json`, 80 s, 2.3 GB peak.

| "fit" scored | ray-pair acc (matched) | coverage | wrong area (seam) | switches /1000 | unmatched |
|---|---|---|---|---|---|
| merged-mesh oracle (31 surfaces) | 99.999 [99.997, 100] | 100 % | 0.00 [0, 0.001] | 0.007 | 0.00 % |
| 16 independent single-wrap segments | 88.7 [85.8, 91.4] | 33.4 % | 8.01 [6.7, 9.3] | 45.5 | 41.4 % |

- The oracle ceiling is 100 %, so the scorer machinery is sound on PHerc1667.
- The independent segments show the size of the human-label noise floor.
  - 58 % of their wrong vertices are overlap duplicates: two segments within 4 vox claim the same sheet. Without them
    the wrong area is 3.5 %.
  - The rest are mostly ±1. The 29.8k matched vertices on k 13-27, which no segment covers, are wrong by
    construction.
  - Own-offset subsets work. With a split at k 34: inner 11.2 %, outer 5.3 %
    (`p1667/subset_oracle_vs_segs.json`).
- **Pair sites: rely on vertices plus ray pairs; we did not build frozen sites.** Every k is labelled over the full
  360°, so ray pairs give 136k pairs over all 72 blocks with no hand-picked sites, and adjacent contacts are flagged by
  Δr < 4.
  - Building frozen sites by hand adds a selection step and gives about 100 times fewer pairs.
  - Pairs stay secondary because the labels come from one merged read; vertices are primary.

## 4. Label audit (before any fit)

### 4a. Spacing consistency (`label_subsets.py --scroll p1667` -> data/scratch/label_audit/subsets_bandp1667.json, sha 37a7f179…)
- Consistent: k 11-16, 18-31, 34.
- Inconsistent: 17, 32, 33, 35-41.
- No coincident partners. Every k has 174-180 tested angle bins.
- The k ≥ 35 failures are not random. Radial order r(k+1) > r(k) is violated only for pairs with k ≥ 34
  (`p1667_order_check.py`):
  - pair 34: 4 %, 35: 12 %, 36: 14 %, 37: 8 %, 38: 10 %, **39: 40 %**, 40: 11 %;
  - the violations sit in sectors 260-360° and 0-40°.
- The independent segments agree with the merged labels on k 34-41 as well as on k 28-33 (3.5-7 % wrong both).
  The outer labels are therefore at least corroborated. The order violations look like outer sheets that are not
  star-shaped about the umbilicus, or genuine local errors. The CT audit (4b) decides.

### 4b. CT audit (`ct_gap_audit.py --scroll p1667`)
Setup:
- 6 z levels (6592-7842) x 24 angles, avoiding the 144° seam; 145 ray segments.
- Inputs: CT L2, our m7L2 mask, PHerc1667 grad_mag.
- Outputs: `ct_counts_bandp1667.json` (sha 236bbe01…) and `ct_profiles_bandp1667.npz`.

Every label comes from one merged read, so dk = 1 pairs calibrate grad_mag (0.816 per labelled turn). Spans test
drift.

Expected m7 runs strictly between two labels = dk - 1.

| span | labels apart | m7 runs between (expected) | gm turns (expected dk) | reading |
|---|---|---|---|---|
| dk 1, k 11-31 (each) | 1 | 0.01-0.20 (0) | 0.9-1.27 | clean; w17-w18 and w20-w21 (gaps 32-38 vox) have 0.20 / 0.08 extra runs, gm 1.23 / 1.27 |
| dk 2, k 11-29 | 2 | 1.02-1.27 (1) | 1.97-2.46 | ok |
| dk 3, k 11-29 | 3 | 2.06-2.33 (2) | 3.0-3.6 | ok |
| w11-w21, w21-w31 | 10 | 9.61, 9.13 (9) | 11.8, 12.1 | ok: no drift over 20 turns, at most +0.6 over 10 |
| w29-w32, w32-w35 | 3 | 1.93, 1.68 (2) | 3.3, 3.0 | ok / soft |
| w31-w36 | 5 | 3.39 (4) | 4.75 | about 0.6 turn short |
| dk 1, k 37-41 | 1 | 0.01-0.06 | **0.63, 0.55, 0.33, 0.28** | gaps 9-15 vox; gm sees a fraction of a turn per label |
| w37-w39, w39-w41 | 2 | **0.37, 0.14** (1) | 1.12, 0.52 | **over-labelled** |
| w38-w41 | 3 | **0.35** (2) | 0.77 | **over-labelled** |
| w36-w41 | 5 | **1.22** (4) | 2.48 | CT sees about 2-2.5 turns where the labels claim 5 |
| w31-w41 | 10 | **5.28** (9) | 7.86 | 3-4 turns short, all of it outside k 36 |

**Verdict.**
- The labels of k 11-33 are CT-supported, with no cumulative drift over 20 turns. There is at most one partial extra
  sheet (w17-w18 and w20-w21 on about 10-20 % of rays).
- k 34-36 are soft.
- **k ≥ 37 are not CT-supported.** Both m7 and the independent lasagna density see about half the labelled turns.
  - The reading is outer layers traced with more labels than turns: duplicated or fused sheets. In p0139 at the same
    10-16 vox pitch, m7 resolved the sheets (0.93 turns per label).
  - This agrees with 4a: the order violations start at pair 34 and the spacing failures at 35.
  - The single-wrap segments share the same reading, so they cannot arbitrate here.
- No relabel map is proposed. The outer labels are excluded from the claim instead (section 8).

## 5. Arm env templates (spiral/arms/)
All arms share these lines:

```
SCROLL=PHerc1667
VOL=20251217075048
UM=9.596
SENSE=ACW
Z0=6500
Z1=8000
MASK_TAG=m7L2
MASK_LOCAL=$HOME/data/PHerc1667/masks/m7_L2_th0.2.zarr
TRACKS_DBM=
SURF_SDT=
STEPS=
```

Each arm then sets:
- **c1667_base.env** (baseline = arm0/arm20): `ARM=c1667_base`, `REL_WINDINGS=`, `FIT_OVERRIDES=`.
- **c1667_oracle.env** (ceiling):
  - `ARM=c1667_oracle`;
  - `REL_WINDINGS=$HOME/data/PHerc1667/windings/band_oracle_p1667/relative_windings.json`;
  - `FIT_OVERRIDES='{"pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}'`.
  - The chains come from `data/scratch/confirm_region/p1667_oracle_chains.py`: auto_windings oracle with the same
    probes, dθ 15. That gives 805 chains, 15,950 points, median length 27, and 96 % of steps are +1. sha 37c1157e….
- **c1667_method.env** (claim, placeholder):
  - `ARM=c1667_method`;
  - `REL_WINDINGS=$HOME/data/PHerc1667/windings/<SET>/relative_windings.json`;
  - the same FIT_OVERRIDES as the oracle.
  - `<SET>` comes from the frozen auto_windings commands of the claimed method, run with these p1667 hooks:
    - **m7 source:** `M7Chunks(url=data/p1667/masks/m7_L2_th0.2.zarr, cache=<own dir>)`. `radial_extent` reads
      `M7_URL` level "3" directly. Levels 2-3 (nearest, L1[::2] and L1[::4]) have been added to our mask over the band
      by `data/scratch/confirm_region/mask_add_levels.py`, so only the URL needs the hook.
    - **umbilicus:** call `se.set_scroll("p1667")` first; `umbilicus_9p362()` then follows it. Replace
      `write_umbilicus_9p362` with a copy of the p1667 json.
    - **grad_mag:** `GM_ZARR = data/p1667/inputs/PHerc1667_grad_mag.ome.zarr`.
    - **caches:** `CACHE` must be a new directory. The gm_band_z*.npy and m7 chunk caches are keyed by z only and
      would mix scrolls.
    - **models:** the clean PHerc0139 conf/gm jsons, unchanged.
    - **geometry:** crop, ray and probe parameters stay frozen in voxels; 9.596 vs 9.362 µm needs no rescale. The pitch
      is 1.5-2.5x larger, which is a scale-transfer risk the test is meant to expose. Do not retune.

Pod commands, as on band 2:
- local: `SCROLL=PHerc1667 LOCAL_INP=data/p1667/inputs MASK_DIR=data/p1667/masks WIND_ROOT=data/scratch/confirm_region/p1667/windings bash spiral/bootstrap_pod.sh <host> <port> band_oracle_p1667`
- on the pod: `pod_arms.sh arms/c1667_*.env`
- collection: `collect_and_score.sh` with a pods tsv whose base column is `c1667_base`.

## 6. Cost and time per arm
- Fits run a fixed 30k steps. On PHerc0139 that took 2.7-3.7 h of pod time per arm, setup to termination.
- Our estimate for PHerc1667 is about the same per arm:
  - 30 windings instead of 33-37;
  - a smaller slice (3808² vs 6621²);
  - track extraction over a similar mask volume.
  - Unmeasured: L2 mask L0 is 9.6 µm like the 9.36 µm PHerc0139 L0, but the sheets are thicker in voxels.

| arm | pod | est. hours | est. cost |
|---|---|---|---|
| c1667_base | 1x L40S ($0.82-1.79/h) | ~3 | $2.5-5.5 |
| c1667_method | 1x L40S | ~3.5-4 | $3-7 |
| c1667_oracle | 1x L40S | ~3.5 | $3-6.5 |
| **total** | 3 pods in parallel | ~4 h wall | **~$9-19** (cap proposal $60) |

- Local prep is done; the mask took about 45 min on the 3090, at $0.
- Scoring is about 1.5 min and 2.3 GB per comparison.
- No pod is needed for prep.

## 7. Backup: PHerc0139 z 6000-8250 (fork report; files in data/scratch/confirm_region/p0139_z6000/ and data/scratch/label_audit/*band3*)
**Inputs**
- Lasagna rows 45-66 are resampled; the umbilicus is unchanged.
- Bucket m7 L0 rows 29-44 and L1 rows 15-21 are present, and CT L4 is local.
- 33 windings per 250 z (w050 is missing at 8000-8250).
- The #2167 gap flags 27.3 % of vertices.

**Sites:** 10 sites, 2,287 mix pairs. They fall in two clusters, z 6108-6621 and z 8022-8248. c043 is 287 of 288
flagged.

**Oracle:** pair acc 97.86 [95.0, 100], wrong area 0.00. oracle9 is 96.36, wrong area 2.44.

**Label audit:**
- subsets: 21 of 37 consistent;
- CT: w034=w035 on about half the rays, w041=w042 are one sheet, and w047-w052 has **one label too many**;
- the band-1 relabel map is rejected here (`relabel_band3.json`: supported / as_labelled / outer_minus2).

**Leak-free refits:**
- conf model refitted on the 77 sites outside [6000, 8250);
- gm model refitted on chains from crops kept below z 6000.

**Weaknesses relative to PHerc1667:**
- few clustered sites;
- a large flagged sector;
- a contested relabel (w047-w052 sets most pair outcomes);
- same scroll as development (the solver hyperparameters were chosen on all 87 sites).

## 8. Pre-registration: analysis plan FROZEN 2026-10-21T23:48:55Z (UTC), before any PHerc1667 fit

Frozen now: the analysis plan, the label split and the files below. The method arm stays a placeholder until it is
frozen (about Oct 25-26). Its chains, conf/gm model shas and auto_windings commands will be appended then as an
addendum, before its pod is created. The baseline and oracle arms launch after this freeze. Their results must not
change anything in this section.

**Frozen files (sha256)**

| file | sha256 |
|---|---|
| data/eval/p1667/labels_merged_v4.npz (labels) | 8ee42ee187fc3fdbba4c5c1093c8b1c756b861f7cdb886c89c20060d741e46fa |
| data/eval/p1667/umbilicus_p1667_L2.json | e6a862b8bb20bece4f3fdcf9ae46f391359682d93c522b4c1b877a3e0dd3d5ac |
| data/scratch/spiral_eval/cache/ref_p1667_z6500-8000_s2_m0_v2.npz (reference sample) | 74082855ca9f70eb54af28875751955cd888c43ea2c1af0a7e7dedc5f8dd211f |
| data/scratch/label_audit/subsets_bandp1667.json | 37a7f17999dac036ebb3d770e7f08a7307a4cd76794fbca86955dbc9b3bc1cda |
| data/scratch/label_audit/ct_counts_bandp1667.json | 236bbe01b6acb07b6b5a4e7c3fbea165dd3a7f4014cbeb312f2bcf74d0e7de0c |
| data/scratch/confirm_region/p1667/windings/band_oracle_p1667/relative_windings.json (oracle chains) | 37c1157ef438da802f2d38df12980f5a6c7fe6392b2b54e6d5bc278d1d827e9a |
| eval/spiral_eval.py (scorer) | 7922129529d58d1d1f06509681cc169b0b7d533498354dcabfc77449310e9ca9 |
| eval/subset_wrong_area.py (primary endpoint) | 45f3eb8302f27c31262521e6840eff9449cc09c3710107b89519990dd6a8a29f |
| eval/p1667_labels.py | f90e8b9a8a8dd56812a65785aea73f79ac0cbaa3e0b839084422ffcc715960bf |
| config/recipe.toml (fitter recipe) | 9ba8494ef25ba7fd6d60f492f51827015130f8ae55fd7d0a9d63285d762db1f1 |
| spiral/arms/c1667_base.env | e75b9dd1553210f437c951019ed3d6bf0641e568d626136dd1121ee02fcf44c6 |
| spiral/arms/c1667_oracle.env | 77c2825151de9dcf7cb83c99586b8bb93a57a71969a39c985b094fc96b235285 |

**Plan**
1. **Arms:** c1667_base, c1667_oracle and c1667_method. Each is fit once (pinned villa 0fb38c45 fit_spiral, 30k
   steps, recipe above) and exported at step 4.
   - There are no reruns, except that a failed pod is resumed from its checkpoint with the same env.
   - All arms use the same local inputs: m7L2 mask, lasagna, CT L4, umbilicus.
2. **Scoring:**
   - `eval/spiral_eval.py --scroll p1667 --fit <base> --fit-b <arm> --z0 6500 --z1 8000 --tol 4`, stride 2, run
     under `flock /tmp/vss_eval.lock`;
   - then release/fit_vertex_dump.py (`--scroll p1667`) for each arm;
   - then `eval/subset_wrong_area.py --scroll p1667 --z0 6500 --z1 8000 --subsets subsets_bandp1667.json --split 34
     --own-offsets --dump-a <base> --dump-b <arm>`.
3. **Label split:** inner = k < 34 (k 11-33, 1,553,468 vertices); outer = k ≥ 34. This is fixed from 4a and 4b.
4. **Primary endpoint:** method − base, seam-aware wrong-winding area on the inner windings, with offsets refitted on
   the inner vertices alone (`metrics_own_offsets.inner.wrong_area_seam_diff`). The paired block bootstrap is
   spiral_eval.BootV (30° x 250 z, 1000 resamples, seed 1).
   - **Success requires the upper end of the 95 % CI to be below 0.**
5. **Switch margin (also required for the claim):** method − base vertex switches per 1000 must be at most +5
   (point estimate). The measure is spiral_eval `vertices.metrics.all.all.switch_per1000_seam_diff`.
6. **Uninformative rule (checked first):** compute the primary endpoint with oracle in place of method. If the upper
   CI of oracle − base is not below 0, the region is uninformative for this method class. We then report it so, with
   no claim either way, whatever the method arm shows.
7. **Secondary read-outs (reported, not part of the claim):**
   - own-offset wrong area for all vertices, outer, and consistent/inconsistent subsets (subsets file above);
   - ray-pair accuracy (matched and all) and coverage, all pairs and pairs with both labels k < 34. The k < 34 filter
     is `eval/pairs_ksplit.py` on `--dump-pairs`, a post-processing script, so the scorer sha above stays unchanged;
   - global-offset wrong area;
   - unmatched fraction;
   - method − oracle.
8. **Disclosures:**
   - the umbilicus is derived from the human mesh;
   - the mask is our reproduction of the bucket m7-L2-th0.2 recipe (dice 0.815 against bucket m7 on a PHerc0332
     block);
   - m7's training data may include PHerc1667;
   - the labels are one merged human read, unwrapped by eval/p1667_labels.py;
   - outer labels (k ≥ 37) are not CT-supported.

## 9. Method-arm tooling and frozen-tool dry run (diagnostics only; NOT the method freeze; no fit)
Written 2026-10-22T01:05Z, after the section-8 freeze. These numbers use the k < 34 oracle labels of the confirmation
band. They are reported as asked. Choosing the method variant from them would tune on the confirmation labels; any
such choice must be disclosed in the method addendum.

**Hooks (backward compatible), tools/auto_windings.py sha 8f097312…ddafa44edb:**
- Global `--scroll p1667` (given before the subcommand) calls `set_scroll`. It switches:
  - M7_URL to the local data/p1667/masks/m7_L2_th0.2.zarr. Level 0 is the L2 frame; `radial_extent` reads its level 3
    (8x) without anonymous-S3 options;
  - CACHE to data/scratch/auto_windings/p1667/ (m7_L0 chunks, l3_*, gm_band_*, sector);
  - M7_SHAPE to 9269x3808x3808;
  - GM_ZARR to data/p1667/inputs/PHerc1667_grad_mag.ome.zarr (level 2 = 4 vox; rows cover z 6144-8288), with no
    extras;
  - the umbilicus to spiral_eval `--scroll p1667`; `write_umbilicus_9p362` copies umbilicus_p1667_L2.json;
  - `load_meshes9` to the merged human labels (`se.p1667_segments`).
- `_band_crop` workers re-apply the scroll if they were spawned. M7Chunks defaults are read at call time.
- `compare --kmax K` adds `inner_k_lt_K` steps, pair precision by true winding distance, and drift (3-6, 10-20)
  restricted to oracle windings < K. On p1667 the p0139 blacklist {41, 42, 45, 46} is empty.
- New eval/pairs_ksplit.py (sha a721e1df…): the k-split ray-pair read-out from `--dump-pairs` CSVs, using spiral_eval's
  own Boot and summ.
- **Checks:**
  - p0139 reassembly of band_m7_z1750 crops reproduces band2_m7_ray_c (04bcb7e2…), and gmfix reproduces
    band2_m7_ray_gmxc (33968f63…);
  - p0139 `compare` reproduces the 79.96 % in notes/2026-10-21-second-band-plan.md;
  - `band --limit 24` on p0139 gives 24 of 24 crop pickles identical to band_m7_z1750/crops;
  - `--scroll p1667 oracle` reproduces the frozen oracle sha 37c1157e… (the old monkeypatch script is no longer needed);
  - pairs_ksplit "all" on arm0_pairs.csv matches arm0.json pairs_mix all/all exactly, CIs included.

**Dry-run chains (frozen recipe, clean models: conf da3515a8…, gm 11e1166f…), z 6500-8000, dtheta 15:**
- Commands:
  - `--scroll p1667 band --workers 6 -> p1667/band_m7_p1667` (2,275 crops, 61 min locally);
  - `band --reassemble --conf 1.0 --bridge 1 --max-step 2 --conf-json <clean> -> band_m7_p1667_ray_c`;
  - `gmfix --model <gm clean> (thr 0.18) -> band_m7_p1667_ray_gmxc`.
- Both sets: 552 rays, 577 chains, 18,670 points, median 34 points per chain.
- Pinned loader: all annotated, 0 decreasing chains.

| chains (data/scratch/auto_windings/p1667/) | relative_windings.json sha256 |
|---|---|
| band_m7_p1667_ray_c (no gm) | 85dcbfb46a66b192927cfef1adac91a080b9b1171a7690f34c6493129e0804b5 |
| band_m7_p1667_ray_gmxc (frozen recipe = arm21 pipeline) | 74b4edacda757ae30c7ff368cdc500995929da364bd74d3a63abfc14af91bc45 |

Diagnostics against band_oracle_p1667, k < 34. Sources: `compare --kmax 34` and
data/scratch/confirm_region/p1667_dryrun_diag.json.

| | step precision k<34 (steps) | dr=2 steps precise | drift 10-20 windings (within +-1) | drift 3-6 | raised 1->2 |
|---|---|---|---|---|---|
| ray_c (no gm) | 95.82 % (8,594) | 26 % of 392 | +0.95 (74.7 %) | +0.31 | - |
| ray_gmxc (frozen) | 86.86 % (8,594) | 14 % of 1,428 | +2.74 (30.6 %) | +0.86 | 2,385 / 17,291 (13.8 %) |
| band 2 p0139 gmxc, for reference | 79.96 % (all) | | -0.18 (60.2 %) | | 25 % |

Mean chain winding increment over labelled spans of n windings, compared with the CT audit's own rays (k < 34):

| span | labels | ray_c | ray_gmxc | m7 runs (n_m7+1) | CT peaks (n_ct+1) | grad_mag turns (median) |
|---|---|---|---|---|---|---|
| c1 | 1 | 1.07 | 1.17 | 1.08 | 1.73 | 1.05 |
| c3 | 3 | 3.20 | 3.56 | 3.12 | 4.62 | 3.40 |
| c10 | 10 | 10.72 | 11.99 | 10.34 | 15.09 | 11.97 |

**Scale transfer:**
- True sheet spacing on the oracle rays is median 26.4 vox (p10-p90 13-54) on PHerc1667, against 16.2 (8-32) on PHerc0139
  band 2. That is 1.6x wider, at almost the same voxel size.
- The crop and graph geometry carries over: 192³ crops, radius 40, 2.5-vox gap merge. The no-gm chains are 95.8 %
  step-precise and track the m7 run count (c10 10.7 against m7 10.3, labels 10).
- The grad_mag raise rule does not transfer.
  - It was fitted at the zero-drift point on PHerc0139, where tight packing makes m7 merge contact sheets, so true
    2-steps are common there.
  - On PHerc1667, m7 rarely merges sheets (m7 runs per labelled gap: 1.08). The rule still raises 13.8 % of single
    steps, 86 % of them wrongly. This turns +0.95 into +2.74 windings of over-count per 10-20 windings.
  - grad_mag density itself counts about 12 turns per 10 labelled windings here (gm_scale_per_turn 0.816).
- The CT peak count over-counts most (1.7 per gap) because of noise peaks at prominence 0.15. It is not a usable
  reference for drift; labels and m7 agree within 3 % on k < 34.
- Fitter note: 13.7 % of points would be dropped at the default min spacing of 16. The method arm keeps
  `pcl_unattached_pcl_min_point_spacing 0` (as the oracle env).

## 10. Baseline and oracle arms (run 2026-10-21T23:52Z - 2026-10-22T04:05Z)
- Pods vss-x-base (<pod-id>) and vss-x-oracle (<pod-id>), 1x L40S runpod at $1.11/h. Both terminated after pull
  (02:36Z, 03:40Z). Cost about $3.0 + $4.2 = **$7.2** (cap $25).
- Base: 30k steps in 2.7 h. Oracle: 3.8 h.
- Scored with `flock`: data/scratch/spiral_eval/arms/c1667_base_vs_c1667_oracle.json, vertex dumps in
  data/scratch/confirm_region/p1667/vertex/c1667_{base,oracle}.npz, and the own-offset read-out
  data/scratch/confirm_region/p1667/subset_base_vs_oracle.json.
- **Uninformative rule (section 8.6): the oracle PASSES** (oracle − base, inner own-offset seam wrong area, CI upper
  < 0). The region is informative. Nothing in section 8 was changed after seeing these numbers.
- Secondary pair read-outs need a `--dump-pairs` rerun of the same spiral_eval command (scorer sha unchanged). They are
  not run yet.

## 11. Method addendum: FROZEN 2026-10-22T13:26:00Z (UTC), before any method-arm fit
Section 8 (analysis plan) is unchanged. This addendum names the method arms, seeds and decision rule, as decided by the
coordinator on PHerc0139 evidence only (notes/2026-10-22-fit-seed-variance.md, -hitl-budget.md sections 4-8,
-context-raise-rule.md, -robust-fitter.md). There is no new method candidate, and the human-calibration session is
not part of the method.

**(a) Disclosure.**
- The coordinator saw the step-level dry run of section 9 before this freeze: step precision and drift on the k < 34
  confirmation labels, for the gmxc chains vs the no-gm chains.
- The baseline and oracle arms of section 10 had also been scored, and only the informativeness pass was reported.
- No fit-level outcome of any method arm has been seen. None has been run.

**(b) Arms.** All arms use the same setup as c1667_base / c1667_oracle (section 5). Chain arms use weight 10 and
spacing 0, as the oracle. Seeds are set with `optimizer_random_seed` in FIT_OVERRIDES (villa default 1).

| arm | role | windings set (sha256 of relative_windings.json) | seeds (env files in spiral/arms/) |
|---|---|---|---|
| M1 | **primary**: the pre-registered frozen recipe | band_m7_p1667_ray_gmxc -> p1667_m7_ray_gmxc (74b4edac…91bc45) | 1 c1667_m1s1, 2 c1667_m1s2 |
| M2 | secondary; **informed by the dry run** | band_m7_p1667_ray_c -> p1667_m7_ray_c (85dcbfb4…0804b5) | 1 c1667_m2s1, 2 c1667_m2s2 |
| base | baseline (no windings) | - | 1 c1667_base (run, section 10), 2 c1667_bases2, 3 c1667_bases3 |
| oracle | informativeness only (single) | band_oracle_p1667 (37c1157e…) | 1 c1667_oracle (run) |

- The placeholder c1667_method.env is superseded by the four M env files and was removed.

**(c) Decision rule** (eval/method_table.py):
- For every method-seed x baseline-seed pair (2 x 3 = 6 per arm), compute the primary endpoint of section 8.4: method
  − base, seam wrong area on k < 34 with own offsets, using paired BootV.
  - A pair passes if its 95 % CI upper is below 0 AND its switch difference is at most +5 per 1000 (point estimate;
    section 8.5).
- **An arm passes only if all 6 of its pairs pass AND Holm rejects for it across M1 and M2 at alpha 0.05.**
  - The Holm p per arm is the maximum over its 6 pairs of the two-sided bootstrap p, 2 min(P(d ≥ 0), P(d ≤ 0))
    (intersection-union).
  - Holm thresholds are 0.025 for the smaller p, then 0.05.
- The section-8.6 informativeness rule was checked first; it passed (section 10).
- Every combination is reported, whatever the outcome.
- **Computation from per-vertex dumps** (meshes are deleted after dumping, to save disk):
  - the primary endpoint is subset_wrong_area's own-offset code;
  - the switch difference is spiral_eval's `vertices.metrics.all.all.switch_per1000_seam_diff` computed from the
    dumps;
  - checked on base vs oracle, the dump-based switch difference equals the paired spiral_eval json exactly (estimate
    and CI).

**(d) Secondary read-outs** (not part of the claim):
- each arm gets a single-fit `spiral_eval --dump-pairs` run, with the same command as section 8.2 and no `--fit-b`;
- `eval/pairs_ksplit.py --pairs <base>_pairs.csv --pairs-b <arm>_pairs.csv --split 34` per pair (all / inner /
  outer ray-pair accuracy and coverage, B − A);
- the oracle's pair read-out vs c1667_base;
- subset_wrong_area metrics for all subsets (own and shared offsets);
- method − oracle.

**Inputs (sha256):**

| file | sha256 |
|---|---|
| data/scratch/confirm_region/p1667/windings/p1667_m7_ray_gmxc/relative_windings.json | 74b4edacda757ae30c7ff368cdc500995929da364bd74d3a63abfc14af91bc45 |
| data/scratch/confirm_region/p1667/windings/p1667_m7_ray_c/relative_windings.json | 85dcbfb46a66b192927cfef1adac91a080b9b1171a7690f34c6493129e0804b5 |
| spiral/arms/c1667_m1s1.env | 868eb252a05b7ca7df45532b700ec3e2d7abb5a48c5e698406fe876215ddc139 |
| spiral/arms/c1667_m1s2.env | a30b9edb0b5c939b7dfe4dc79b9241ac8038d5a708caab3a8faa77707d5d12f3 |
| spiral/arms/c1667_m2s1.env | 6750641eace0071e2b267f3771d89defaa24af4236739d7d9d28f1d0ce5fd140 |
| spiral/arms/c1667_m2s2.env | ff7af38da5dfc256aeaf77cc4f2fa964a8ab5fb72640123aca94848f1e1b1b4b |
| spiral/arms/c1667_bases2.env | 952f1d44040d3775bc8e8d6766ce00df8bf757b18c12169618ec0fb8dd97fdb0 |
| spiral/arms/c1667_bases3.env | 283cc8dcf21a7f059177a95351a5ea7c2c3e955ec28b605e3bae326602f16057 |
| spiral/arms/c1667_base.env (unchanged since section 8) | e75b9dd1553210f437c951019ed3d6bf0641e568d626136dd1121ee02fcf44c6 |
| spiral/arms/c1667_oracle.env (unchanged since section 8) | 77c2825151de9dcf7cb83c99586b8bb93a57a71969a39c985b094fc96b235285 |
| tools/auto_windings.py (chain tool, p1667 hooks) | 8f0973120146e10f30811f28dab6e01c921be96f39273724869b01ddafa44edb |
| tools/auto_windings_conf_clean.json | da3515a85b9bb31efb3cb677b7930fbd97e8369cff6ad8571a8e0ffe8ddf3d2d |
| tools/auto_windings_gm_z4500_clean.json | 11e1166f6b5cfdc580621b0198bd069196292c8f9fa5c2b9bf0c1e5633f5428c |
| eval/method_table.py | 6cfc15dfaf2c4a140b50a99238259180bcc40b1fed3000d5300606b17ad7609e |
| eval/pairs_ksplit.py | a721e1dfe620b48b629a9400a51a0112fb9e12610d5abd389b77730d9d9969f7 |
| eval/subset_wrong_area.py (unchanged since section 8) | 45f3eb8302f27c31262521e6840eff9449cc09c3710107b89519990dd6a8a29f |
| eval/spiral_eval.py (unchanged since section 8) | 7922129529d58d1d1f06509681cc169b0b7d533498354dcabfc77449310e9ca9 |
| release/fit_vertex_dump.py | cecb55139bc0e8be7b598aad12c6d2ad1aaf264402e1c47840e67a8498e24136 |
| spiral/run_arm.sh | 0310894f63fd5a2c04f08de79af7d8df664d51a8dc27a322fe690f2bda4366dc |
| spiral/pod_arms.sh | 81c5117b1908911644eb4c67bc0bad1c9288ad0de86c7b9c611f15ab617a4249 |
| spiral/bootstrap_pod.sh | 1d29ec4e1193be04dcfde4005384e34f6fd41950967867308d79ac46caa4cb87 |
| data/scratch/label_audit/subsets_bandp1667.json | 37a7f17999dac036ebb3d770e7f08a7307a4cd76794fbca86955dbc9b3bc1cda |
| data/eval/p1667/labels_merged_v4.npz | 8ee42ee187fc3fdbba4c5c1093c8b1c756b861f7cdb886c89c20060d741e46fa |
- Check added 2026-10-22T13:40Z: single-fit `--dump-pairs` CSVs of c1667_base and c1667_oracle, put through
  pairs_ksplit, reproduce the paired spiral_eval pairs_ray all/all differences exactly (acc_matched, coverage, acc_all;
  estimates and CIs). After that, the base and oracle meshes were deleted; their dumps and CSVs are kept.

## 12. Method-arm results (fits 2026-10-22T13:36Z - 22:05Z; table built 22:15Z; method_table.json sha256 9586d3a3…7ae5f6; script sha 6cfc15df… unchanged from §11)

Output: data/scratch/confirm_region/p1667/method_table.json (log: method_table.log next to it), from
`eval/method_table.py` as frozen in section 11. Primary endpoint = inner (k<34) own-offset seam wrong-area diff
(method - base, %, 30°x250z block bootstrap, 1000 resamples). Switch = switch_per1000_seam_diff.

| combo (base vs method) | diff est | 95% CI | p | switch | pass |
|---|---|---|---|---|---|
| base s1 vs M1 s1 | -0.39 | [-0.83, +0.11] | 0.120 | -20.0 | no |
| base s2 vs M1 s1 | +0.17 | [-0.40, +0.75] | 0.596 | -19.9 | no |
| base s3 vs M1 s1 | -1.55 | [-2.11, -0.98] | <0.001 | -22.5 | yes |
| base s1 vs M1 s2 | -0.46 | [-0.93, +0.04] | 0.078 | -20.0 | no |
| base s2 vs M1 s2 | +0.10 | [-0.48, +0.68] | 0.778 | -19.9 | no |
| base s3 vs M1 s2 | -1.63 | [-2.18, -1.04] | <0.001 | -22.5 | yes |
| base s1 vs M2 s1 | -28.16 | [-29.37, -26.90] | <0.001 | -63.3 | yes |
| base s2 vs M2 s1 | -27.60 | [-28.92, -26.30] | <0.001 | -63.2 | yes |
| base s3 vs M2 s1 | -29.32 | [-30.50, -28.12] | <0.001 | -65.9 | yes |
| base s1 vs M2 s2 | -27.99 | [-29.25, -26.69] | <0.001 | -62.9 | yes |
| base s2 vs M2 s2 | -27.43 | [-28.77, -26.08] | <0.001 | -62.8 | yes |
| base s3 vs M2 s2 | -29.15 | [-30.36, -27.91] | <0.001 | -65.4 | yes |

- **M1 (gmxc, primary): FAIL.** 2 of 6 combos pass; p_arm 0.778; Holm threshold 0.05; not rejected.
- **M2 (ray_c, secondary, informed by the dry run): PASS.** 6 of 6 combos pass; p_arm < 0.001 (0 of 1000); Holm threshold 0.025; rejected.
- Levels, inner own-offset wrong area (%):
  - base s1/s2/s3: 78.9 / 78.3 / 80.1;
  - M1: 78.5 / 78.4;
  - M2: 50.7 / 50.9;
  - oracle: 46.3.
  - M2 recovers about 86 % of the base-to-oracle gap, and M2 - oracle is +4.5 to +4.6 (CI about [4.0, 5.1]).
- Secondary pair read-outs (pairs_ksplit, inner, acc_matched_seam diff vs base):
  - M1: +4.6 to +7.6, with coverage -3 to -6;
  - M2: +24.9 to +27.8, with coverage -3 to 0;
  - oracle vs base s1: +23.0;
  - M2 - oracle: +2.3 (CI about [1.1, 3.9]).
  - Files: data/scratch/spiral_eval/arms/*_pairs_k34.json.
- GPU caveat: the seed-1 base and the oracle ran on runpod L40S.
  - M1 s1/s2, M2 s1 and base s3 ran on massedcompute L40S.
  - M2 s2 ran on a datacrunch A6000 48GB; base s2 ran on a massedcompute A6000 48GB (6 vCPU).
  - The GPU type is confounded with seed for those two arms. The A6000 arms sit inside the L40S seed spread (M2 s2 vs s1: 50.9 vs 50.7; base s2 78.3 vs s1/s3 78.9/80.1), so this has no visible effect on the verdict.
- Cost of the 6 arms (pod create to terminate x listed price): about $18.3 (m1s1 2.86, m1s2 2.41, m2s1 2.43, m2s2 4.67, bases2 3.75, bases3 2.16); cap $30.
  - Base/oracle (section 10): $7.2.
  - All vss-x pods are terminated (checked 22:12Z).
  - Pulled meshes were deleted after scoring; vertex dumps are in data/scratch/confirm_region/p1667/vertex/, and single-fit scores and pair CSVs are in data/scratch/spiral_eval/arms/.
