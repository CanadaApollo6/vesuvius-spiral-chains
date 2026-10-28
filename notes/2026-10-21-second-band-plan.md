# 2026-10-21: confirmation band for the arm19 claim (plan; nothing launched, $0 spent)

**Goal.** Repeat arm0 vs arm19 on a second PHerc0139 region with the pipeline frozen exactly as in round 3. The region
must not have been used to fit or tune anything. z 3000-4500 is the first test band. z 4500-6000 trained the gm
correction (`tools/auto_windings_gm_z4500.json`).

**Decision: z 1750-3000** on 20250728140407 (9.362 um). Fallback: z 7000-8500. No other scroll qualifies
(section 1c).

## 1. Candidate regions

### 1a. Human labels by z (local checks; scripts in data/scratch/second_band/)
`coverage.py` reads the 37 public 2.399 um meshes, mapped with the published affine, the same way spiral_eval does.
`pairs_per_band.py` counts the frozen `scoring_pairs.csv` (kind mix, skip pairs dropped).

| z | windings with mesh | median angle covered per winding | vertices (stride 2) |
|---|---|---|---|
| 1250-1500 | 3 | 42 deg | ~1k |
| 1500-1750 | 14 | 211 deg | 69k |
| 1750-2000 | 22 | 304 deg | 245k |
| 2000-2500 | 22 | 360 deg | 763k |
| 2500-3000 | 27-32 | 360 deg | 904k |
| 3000-8000 | 33 | 360 deg | ~600k per 250 z |
| 8000-8250 | 32 | 359 deg | 386k |
| 8250-8500 | 18 | 241 deg | 77k |

- Below z ~2600, w035-w038 and w040-w050 have no meshes. They start at z 2525-2998; w039 runs from z 1794.
  - So z 1750-2600 is labelled only on w023-w034, w039 and w051-w059. There is a ~16-winding unlabelled gap in the
    middle.
  - The vertex metrics still work: the per-bin seam offset is a mode over all labelled vertices. In effect they test
    whether the fit counts across the unlabelled gap correctly.
- Bucket m7 mask L0 (S3 listing): rows 6-7 (z 1152-1536) hold only 5-12 MB, row 8 (z 1536-1728) 55 MB, and rows 12 and
  up ~115 MB. So the scroll's bottom cap is below z ~1750. The umbilicus starts at z 1440.

| band | mix pairs | sites | contact | windings | pairs touching w033-37 | site classes (pairs) |
|---|---|---|---|---|---|---|
| 3000-4500 (band 1) | 7,471 | 33 | 9.8 % | 24-58 | 30.8 % | 3,127 compressed / 2,099 mixed / 2,245 wide |
| **1750-3000** | **5,872** | **21** | 16.1 % | 25-56 | 51.8 % | 1,152 / 2,986 / 1,734 |
| 6000-7500 | 1,360 | 5 | 7.8 % | 49-57 | 0 % | 558 / 0 / 802 |
| 6750-8250 | 927 | 5 | 36.8 % | 28-58 | 40.3 % | 862 / 0 / 65 |
| 7000-8500 | 1,995 | 7 | 26.0 % | 26-58 | 21.3 % | 1,710 / 0 / 285 |

- Every site below z 3000 has its origin at z >= 1760. Bands 1500-3000, 1600-3000 and 1750-3000 therefore get the
  same 5,872 pairs. Starting at 1750 drops the cap and the partly labelled 1500-1750 slab, and loses no pairs.
- **#2167.** The gap sector exists only where w034 and w036 both have meshes and w035 does not. That starts at
  z ~2750 (w036 starts at 2761, w035 at 2826).
  - In z 1750-2750 there is no gap at all.
  - In 2750-3000, spiral_eval flags a pseudo-gap ~300 deg wide, because w035 only partly exists there. The
    `excl_flag*` variants carry it: 279 of 5,872 pairs and 16.5 % of vertices are flagged.
  - Above z 3000 the real gap is 117 deg wide at z 3000, narrowing to ~65 deg at z 7000-7750.

### 1b. Scorer check on z 1750-3000 (done, local, 47 s)
Ran `spiral_eval.py build-oracle` on z 1750-3000, then scored that oracle: the human 2.4 um meshes as the fit
(`data/scratch/spiral_eval/oracle_z1750.json`).
- mix pairs: acc_matched 97.82 [95.6, 99.3], contact 86.6, open 99.98, coverage 100.
- legacy pairs: 98.69. Vertices: switches 0.11 / 1000, wrong area 0.01 %, unmatched 0.01 %.
- 1.91M vertices, 21 sites. spiral_eval runs unchanged on this band.
- The pair ceiling is 97.8, against 99.5 in band 1. The loss is all on contact pairs, which carry per-site
  registration error (as in band 1), and there are more contact pairs here (16 % vs 10 %).

### 1c. Other scrolls: none usable without a new scorer front end
- **PHerc1667 and PHerc0172.** Both have human meshes, at 7.91 um.
  - PHerc1667 has 20 segments. real350 trained on it (Scroll 4).
  - PHerc0172 has 44 windings. It is in our Dataset111 and in real350 (Scroll 5).
  - Neither has a bucket m7 mask at the scan level we use, published tracks or a 9.362 frame.
- **PHerc0814, 0500P2, 0009B and 1447.** These have human meshes but are in our Real9 / m7-distill training sets.
  The arm19 pipeline uses none of our trained models, so this matters less than it sounds. Still, each would need
  umbilicus, lasagna, sites and pairs, and spiral_eval is hard-wired to PHerc0139 (its sites.json, affine, umbilicus
  and mesh directory).
- **PHerc0343P and MANBp.** No public meshes.
- A cross-scroll test would be the strongest confirmation, but it is 1-2 days of engineering. It is not this run.

### 1d. Leakage status of z 1750-3000 (what the frozen pipeline has seen)

| component | fitted / tuned on | sees z 1750-3000? |
|---|---|---|
| gm model + threshold 0.19 (`auto_windings_gm_z4500.json`) | z 4500-6000 rays vs meshes | no |
| ray cut (conf 1.0, bridge 1, max-step 2), weight 10, spacing 0 | z 3000-4500 | no |
| step confidence model (`auto_windings_conf.json`, its logperr is a gm feature) | all 87 sites, 21 of them in this band | **yes, site level** |
| graph solver (block 12, r40, k30) | frozen on the 87 sites | same as above |
| bucket m7 mask | organisers (may include PHerc0139 Kaggle labels) | both arms equally |

- The conf model's status is the same as in band 1 (it saw 33 sites there). It must be disclosed.
- Optional post-hoc check, run after the claim numbers are frozen:
  - refit the conf model without the 21 band sites (a temporary copy of the JSON, never committed into the pipeline);
  - re-run gmfix;
  - report how many gm raises flip.
  - If < 2 %, the leak is immaterial. This check never replaces the pre-registered arm.

## 2. What the run needs (z 1750-3000)

| input | status | action |
|---|---|---|
| bucket m7 L0/L1 | present (L0 rows 7-16, L1 rows 4-7) | run_arm.sh fetches rows from Z0/Z1 itself |
| CT L4 (SDT occupancy) | whole level fetched already | none |
| umbilicus | z 1440-20435, ~25 control points in band | same `data/p0139/inputs/umbilicus.json` |
| lasagna nx/ny/grad_mag (9.362 store) | local store covers target rows 22-36 (z 2816-4736) only; the source (2.399 group 4) has data from z9 ~1280 | **run prep_inputs.py additively for rows 12-21** (z 1536-2816). Existing rows 22+ have `.done` markers and are skipped. |
| grad_mag for gmfix | `gm_volume` reads `data/p0139/inputs` + `lasagna_z4500` | covered by the same additive resample; no code change. The cache file is keyed by z (a new `gm_band_z<a>_<b>.npy`), so nothing stale is reused. |
| published tracks | none below z 4500 | extracted from the mask as in arm0 |
| sites / pairs | frozen `scoring_pairs.csv` (21 sites in band) | none; spiral_eval filters by `--z0/--z1` |
| auto chains | none | `tools/auto_windings.py band` on the new band (unchanged code, defaults, same chunk cache) |
| oracle chains (ceiling arm only) | none | `tools/auto_windings.py oracle --z 1750 3000` (reads the meshes directly) |

**Script issues found (no code changed):**
- **run_arm.sh**
  - It sources the arm file before `Z0=${Z0:-3000}`, so `Z0=1750` / `Z1=3000` can go in the env file. OK.
  - The SDT cache path `$DATA/sdt_$MASK_TAG/PHerc0139_surf_sdt.ome.zarr` is not keyed by z. Use a new
    `MASK_TAG=m7pub_z1750`, so no pod can reuse a band-1 SDT.
- **collect_and_score.sh** hard-codes `arm0` and `--z0 3000 --z1 4500`. Either add `BASE`/`Z0`/`Z1` env defaults (3
  lines), or pull and score by hand (commands below).
- **sanity_checks.py** defaults `--out data/scratch/p0139_spiral`, which would overwrite band-1 PNGs. Pass a new `--out`.
- **prep_inputs.py** rewrites `data/p0139/inputs/PHerc0139.lasagna.json`; only its `p0139.rows` stats block changes. Back
  it up first.
- **arm_table.py** labels the base row "arm0 (A)". This is cosmetic.

**Arm files** (copy the pattern of arm0.env / arm19.env; new names, so pull dirs don't collide):
```
# arms/arm20.env: band 2 baseline (= arm0 on z 1750-3000)
ARM=arm20
Z0=1750
Z1=3000
MASK_TAG=m7pub_z1750
MASK_S3=s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr
TRACKS_DBM=
SURF_SDT=
REL_WINDINGS=
FIT_OVERRIDES=
STEPS=

# arms/arm21.env: band 2 claim arm (= arm19): same as arm20 plus
ARM=arm21
REL_WINDINGS=$HOME/data/PHerc0139/windings/band_m7_z1750_ray_gmx/relative_windings.json
FIT_OVERRIDES='{"pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}'

# arms/arm22.env (optional ceiling, = arm2): same as arm21 with
ARM=arm22
REL_WINDINGS=$HOME/data/PHerc0139/windings/band_oracle_z1750/relative_windings.json
```

## 3. Pre-registration (freeze before any pod starts)
- **Primary comparison:** arm21 - arm20, `eval/spiral_eval.py --z0 1750 --z1 3000 --tol 4`, 4-vox exports, variant
  "all".
- **Confirmation** = both of these:
  - (a) matched pair accuracy (mix): the 95 % site-bootstrap CI lies above 0;
  - (b) seam-aware wrong-winding area: the CI lies below 0.
- **Reported, not gating:** acc_all, coverage, contact accuracy, switches/1000 (expected to rise, as in band 1:
  +17.6), and the `excl_flag` variants.
- **No re-tuning after any look at band-2 results.** That covers threshold, weight, cut and gm model. The chain
  file's sha256 goes into this note before bootstrap.
- **Power.** 21 sites against 33 widens the pair CI by about sqrt(33/21) = 1.25x. Band 1 gave +12.3 [7.9, 17.2]. A
  similar effect would give roughly +12 +- 6, which still clears 0. The wrong-area CI uses 60 vertex blocks and should
  stay tight.

## 4. Ordered commands

Local (CPU, $0, ~2.5 h wall, mostly step 3):
```
# 1. lasagna rows for the band (additive; ~0.8 GB of lasagna chunks streamed, ~10 min)
cp data/p0139/inputs/PHerc0139.lasagna.json data/p0139/inputs/PHerc0139.lasagna.band1.json.bak
uv run python spiral/prep_inputs.py all --z0 1750 --z1 3000 --dl data/dl_p0139 --out data/p0139/inputs
#    check: the log shows rows 12..21 resampled, valid_frac > 0, grad median ~50; rows 22-24 skipped

# 2. sanity (umbilicus inside the innermost turn; normals vs m7, as in band 1; streams small CT/mask crops)
uv run python spiral/sanity_checks.py --umb-z 1760 2000 2500 2990 --normal-z 1900 2400 2900 \
    --out data/scratch/p0139_spiral_z1750

# 3. automatic chains, frozen pipeline (resumable crop checkpoints; ~1.5-2 h on 11 workers)
uv run python tools/auto_windings.py band --z 1750 3000 --dtheta 15 --workers 11 --out data/scratch/auto_windings/band_m7_z1750
uv run python tools/auto_windings.py band --z 1750 3000 --dtheta 15 --conf 1.0 --bridge 1 --max-step 2 --reassemble \
    --from-crops data/scratch/auto_windings/band_m7_z1750/crops --out data/scratch/auto_windings/band_m7_z1750_ray
uv run python tools/auto_windings.py gmfix --src data/scratch/auto_windings/band_m7_z1750_ray \
    --model tools/auto_windings_gm_z4500.json --out data/scratch/auto_windings/band_m7_z1750_ray_gmx   # summary must say thr 0.19
uv run --with tqdm --with einops python tools/auto_windings.py check data/scratch/auto_windings/band_m7_z1750_ray_gmx/relative_windings.json
sha256sum data/scratch/auto_windings/band_m7_z1750_ray_gmx/relative_windings.json   # paste into section 3

# 4. oracle chains (ceiling arm, and a diagnostic compare that may not feed back into anything)
uv run python tools/auto_windings.py oracle --z 1750 3000 --dtheta 15 --out data/scratch/auto_windings/band_oracle_z1750
uv run python tools/auto_windings.py compare --auto data/scratch/auto_windings/band_m7_z1750_ray_gmx \
    --oracle data/scratch/auto_windings/band_oracle_z1750

# 5. write spiral/arms/arm20.env, arm21.env (arm22.env) as in section 2
```

Pods (needs go-ahead; follow prime-pod-gotchas: re-list pods after any failed create, check
`torch.cuda.is_available()` right away):
```
# 6. one 2xL40S (arm20 + arm21 share tracks/SDT via pod_arms.sh); optionally one 1xL40S for arm22
bash spiral/bootstrap_pod.sh root@<ip> <port> band_m7_z1750_ray_gmx band_oracle_z1750
# on the pod, once ~/pod_setup.log ends with the GPU name:
source ~/spiral_env.sh; setsid nohup bash ~/vss/spiral/pod_arms.sh ~/vss/spiral/arms/arm20.env \
    ~/vss/spiral/arms/arm21.env > ~/work/p0139/pod_arms.out 2>&1 < /dev/null &
# (arm22 pod: same, pod_arms.sh with arm22.env only)
# record spiral/arms/pods_z1750.tsv: arm host port pod_id
```

Collect and score (local):
```
# 7. after ~/work/p0139/<arm>/export_step4.json exists: pull, verify the winding count, then terminate the pod
bash spiral/pull_arm.sh arm20 root@<ip> <port>; bash spiral/pull_arm.sh arm21 root@<ip> <port>
uvx --from prime prime --plain pods terminate <pod_id> --yes
# 8. score (~6-10 min each; band-2 vertex set is 1.9M vs 3.6M)
uv run python eval/spiral_eval.py --fit data/scratch/p0139_spiral/arm20/meshes_step4 \
    --fit-b data/scratch/p0139_spiral/arm21/meshes_step4 --z0 1750 --z1 3000 --tol 4 \
    --out data/scratch/spiral_eval/arms/arm20_vs_arm21.json
uv run python eval/spiral_eval.py --fit data/scratch/p0139_spiral/arm20/meshes_step4 \
    --fit-b data/scratch/p0139_spiral/arm22/meshes_step4 --z0 1750 --z1 3000 --tol 4 \
    --out data/scratch/spiral_eval/arms/arm20_vs_arm22.json
uv run python spiral/arm_table.py data/scratch/spiral_eval/arms/arm20_vs_arm2{1,2}.json
```
(Or add `BASE=${BASE:-arm0}; Z0=${Z0:-3000}; Z1=${Z1:-4500}` to collect_and_score.sh and run it on pods_z1750.tsv.)

## 5. Time and cost
- **Per-arm fit time.** Band 1 took 10,104 s for arm0 (no strips) and 12,500-14,100 s for strip arms; arm19's pod
  ran 3.95 h in total.
  - The step count is fixed (30k), so per-step cost, not band height, dominates.
  - Band 2 is 1,250 z and has a smaller scroll cross-section at the bottom, so a fit should take at most as long
    as in band 1.
  - Assume ~3.1 h of pod time for the baseline and ~4.0 h for a strip arm, both including setup, tracks, SDT, export
    and pull.

| plan | pods | pod-h | cost |
|---|---|---|---|
| baseline + claim (minimum) | 1 x 2xL40S ($2.20/h), ~4.2 h | 8.4 GPU-h | **~$9** |
| same on 2 single L40S ($1.11/h) | 3.1 h + 4.0 h | 7.1 | ~$8 |
| + oracle ceiling arm22 | + 1 x L40S, ~4.0 h | +4.0 | **+$4.5 (total ~$13-14)** |

- **Wall time.** Local prep is ~2.5 h (step 3 dominates; the baseline pod could start after step 1 if we want to save
  ~2 h). Pods take ~4.2 h in parallel, and scoring ~0.5 h. **About 7 h end to end, or ~5 h if the baseline starts
  early.**
- This is far below the ~$1k check-in level. Pods still need the user's go-ahead.

## 6. Fallback band z 7000-8500 (if z 1750-3000 fails a sanity check)
- 7 sites, 1,995 pairs, mostly compressed. Too few sites for the pair CI to be strong, so lead with the vertex
  metrics there.
- Inside the #2167 z range (gap ~65 deg at ~183 deg). Mesh coverage drops above z 8250.
- Same recipe:
  - `prep_inputs.py all --z0 7000 --z1 8500` (rows 54-67, additive);
  - `band --z 7000 8500`;
  - `Z0=7000 Z1=8500 MASK_TAG=m7pub_z7000`;
  - arms 23 and 24.
- Mask rows 35-45 are present.
- Same cost (~$9 minimum).

## 7. Pre-registration, leakage-free version (frozen 2026-10-21T17:30Z, before any pod)
This replaces section 3 where they differ. One change from the plan above: the conf-model leak (section 1d) is
removed instead of disclosed.

**Leakage-free models (new files; the old ones and the old chain dirs are untouched).**
- `tools/auto_windings_conf_clean.json`: the step-confidence model, identical recipe (same 12 features, same logistic
  fit, threshold = largest 2-fold out-of-fold p_err cut with cumulative precision >= 98 %), fitted only on the 33
  sites whose 192^3 crop lies entirely outside z 1750-4500 (all have origin z >= 4500; none lies below 1750). The 54
  dropped sites include the 9 that straddle z 4500.
  - Command: `tools/auto_windings.py validate --fit --exclude-z 1750 4500 --rows-csv data/scratch/auto_windings/val_m7.csv
    --conf-json tools/auto_windings_conf_clean.json --out data/scratch/auto_windings/val_m7_clean`. The rows are the cached
    per-step features of val_m7 (same m7_seg masks, same rays).
  - Recipe check: the same command on all 87 sites reproduces `auto_windings_conf.json` exactly (thr 0.0461,
    max |coef difference| 0).
  - Result: threshold 0.0023 (98.0 % at 8 % coverage on 33 sites). The ray pipeline runs at `--conf 1.0`, so this
    threshold is unused. Only the model's p_err enters the gm rule, as the logperr feature.
- `tools/auto_windings_gm_z4500_clean.json`: the gm rule, identical recipe (`gmfit --half -1` on z 4500-6000 rays vs
  band_oracle_z4500, zero-drift threshold on z 4500-6000 only).
  - Its input chains `band_m7_z4500_ray_c` were re-assembled from the same z4500 crops with the clean conf model. The
    points and windings are byte-identical to band_m7_z4500_ray; only per-step perr differs.
  - Recipe check: the same gmfit on the frozen-conf chains reproduces `auto_windings_gm_z4500.json` exactly (thr 0.19,
    n 11,656).
  - Result: n 11,656 (1,033 truly double), **zero-drift threshold 0.18** (the old model: 0.19). Per-slab
    thresholds are 0.17 / 0.18 / 0.18.
- Code change: `tools/auto_windings.py` gains `--conf-json` (band, validate), `--exclude-z` and `--rows-csv`
  (validate). The defaults are unchanged and reproduce every old output.

**Clean chains.** Same ray reassembly as band_m7_ray (`--conf 1.0 --bridge 1 --max-step 2`, require-agree 1, dtheta
15), then `gmfix` with the clean gm model at its own threshold of 0.18:
- band 1: `band_m7/crops -> band_m7_ray_c -> band_m7_ray_gmxc`. Points are identical to band_m7_ray_gmx;
  1,745 of 40,658 steps differ (963 raises dropped, 782 added); 7,089 raises in all (gmx: 7,270).
- band 2: `band_m7_z1750/crops (new, 3,781 crops, 480 rays, frozen code) -> band2_m7_ray_c -> band2_m7_ray_gmxc`: 597 chains / 31,223 points, median 59 points, 7,232 of 28,711 single steps raised (25 %; band 1: 18 %). Loads with the pinned loader (all annotated, 0 decreasing chains). Oracle band_oracle_z1750: 756 chains / 9,274 points (the unlabelled w035-w050 gap makes it sparse).
- Diagnostics only. Nothing below was or will be changed because of these numbers.

| chains | step precision vs oracle (all) | drift over 10-20 windings | within +-1 | pairs 10-20 |
|---|---|---|---|---|
| band_m7_ray_gmx (old, band 1) | 84.55 | -0.093 | 73.1 % | 47,564 |
| band_m7_ray_gmxc (clean, band 1) | 84.23 | -0.083 | 72.3 % | 47,564 |
| band2_m7_ray_gmxc (clean, band 2) | 79.96 (no gm: 88.22) | -0.178 (no gm: -2.049) | 60.2 % | 6,249 |

**Arms** (env files in spiral/arms/; all bucket m7 tracks/SDT, recipe unchanged; strip arms at weight 10, min
point spacing 0; 30k steps; scored on 4-vox exports, meshes_step4):

| arm | band | chains | role |
|---|---|---|---|
| arm20 | z 1750-3000 | none (= arm0 setup) | band-2 baseline |
| arm21 | z 1750-3000 | band2_m7_ray_gmxc | **pre-registered claim** |
| arm22 | z 1750-3000 | band_oracle_z1750 (human meshes) | ceiling (reported) |
| arm23 | z 3000-4500 | band_m7_ray_gmxc | secondary check vs arm0 (= arm19 with the clean chains) |
| arm24 | z 3000-4500 | band_m7_ray_gmx_p93 (other agent, step filter) | **dev**, not pre-registered |
| arm25 | z 3000-4500 | band_m7_ray_gmx_dsafe (other agent, step filter) | **dev**, not pre-registered |

**Primary test (the only gating one).** arm21 vs arm20, `eval/spiral_eval.py --fit arm20/meshes_step4 --fit-b
arm21/meshes_step4 --z0 1750 --z1 3000 --tol 4`, variant "all", 95 % site/block bootstrap CIs as printed by
spiral_eval. **Confirmed if and only if both hold:**
- (a) matched pair accuracy (pairs_mix, all, acc_matched_diff): CI lower bound > 0;
- (b) seam-aware wrong-winding area (vertices, all, wrong_area_seam_diff): CI upper bound < 0.

Anything else is a fail. That includes one criterion passing alone, and a run that cannot be scored.
- **Reported, not gating:** acc_all, coverage, contact accuracy, switches/1000, unmatched, the excl_flag variants,
  and arm22 vs arm20.
- **Secondary (not gating):** arm23 vs arm0 at z 3000-4500, same two criteria. arm23 is arm19 with the leak removed.
- **Dev (not gating, not pre-registered):** arm24 and arm25 vs arm0 at z 3000-4500. Question: do switches come from
  label errors (p93 should cut them), or does drift dominate wrong area (dsafe)?
- **No retuning after looking.** No threshold, weight, cut, model, chain file, arm setting or metric changes after
  any band-2 or arm result is seen. A failed or crashed arm may be rerun only with byte-identical inputs.

**sha256 (frozen inputs).**

| file | sha256 |
|---|---|
| tools/auto_windings_conf_clean.json | da3515a85b9bb31efb3cb677b7930fbd97e8369cff6ad8571a8e0ffe8ddf3d2d |
| tools/auto_windings_gm_z4500_clean.json | 11e1166f6b5cfdc580621b0198bd069196292c8f9fa5c2b9bf0c1e5633f5428c |
| tools/auto_windings_conf.json (old, unchanged) | b1dc7bda628a78c2961e7707991b9b650aa23c4bb2cd63361b69068b7941c233 |
| tools/auto_windings_gm_z4500.json (old, unchanged) | 18d5ce24c4df61b3dcd7f818d342e1889c621f7ad575b518d2bacc388f31df7f |
| band2_m7_ray_gmxc/relative_windings.json (arm21) | 33968f6326040add69e2103db1646cc916975eacb7b44eb633119cb629870732 |
| band_oracle_z1750/relative_windings.json (arm22) | 2ca25e02d61c66f699d114fd65185d92ef0bec17c42b475f18a665cf0395bc76 |
| band_m7_ray_gmxc/relative_windings.json (arm23) | b012c8839efe397c848c4e60392f16ff4d2fd6cd82b630c62ad29baaa49593ed |
| band_m7_ray_gmx_p93/relative_windings.json (arm24, dev) | 6cc2ec1352074bb7b6d247209cf958ce5a30ab957d610e8f4fc28ba60b9edde4 |
| band_m7_ray_gmx_dsafe/relative_windings.json (arm25, dev) | 2c4356b8327a5f505768bed4d05e02ce690d7062f253a8bedc06acc4f00bc27f |
| band_m7_ray_gmx/relative_windings.json (arm19, reference) | 232e07f3bb0678c1fc60dabe21a04af5a52734517d1b3a2c7771380f8a4ed4a0 |
| tools/auto_windings.py | e817dd425eac74dc5ecfc0802a911090f08fc497a2b6e3c3cdb96432a82849b1 |

**What remains leaky (disclosed).**
- The graph solver's three hyperparameters (block 12, r40, k30) were chosen on all 87 sites.
- The bucket m7 mask may have been trained on PHerc0139 labels; this affects both arms equally.
- The ray cut and weight 10 / spacing 0 were chosen on band 1. That matters for band 1 (arm23-25), not for band 2.

**Inputs.**
- The lasagna rows for z 1536-2816 were resampled additively; rows 22+ were skipped and the umbilicus is
  byte-identical. The old stats json is at `data/p0139/inputs/PHerc0139.lasagna.band1.json.bak`.
- Sanity (`data/scratch/p0139_spiral_z1750/`):
  - Umbilicus free radius is 66-92 vox at z 1760-2990, against a p50 of ~10 for random points inside the scroll.
  - Normals vs m7: median |cos| 0.997; controls 0.63-0.88.
  - grad_mag pitch ratio 1.51 (band 1: 1.31). The recipe's "phase" spacing mode does not read grad_mag.

**Script fixes (backward compatible).**
- run_arm.sh: the SDT cache is keyed by band when Z0/Z1 are not 3000/4500.
- collect_and_score.sh: optional 5th tsv column = base arm, band read from the arm's ds/band.json, output
  `<base>_vs_<arm>.json`; terminations are appended to pods_arms.txt.
- bootstrap_pod.sh: verifies the sha256 of every windings file on the pod.
- arm_table.py: the base row is named from the file.
- sanity_checks.py: run with a new --out.

### 7a. Amendment (2026-10-21T17:50Z; arm20-22 pods started 17:32-17:37Z, no fit output exists or has been seen)
Reason: the band-1 figure work (notes/2026-10-21-fit-figures.md) found that the seam-aware wrong-area metric is
sensitive to winding *numbering* errors in the human meshes. On band 1, w041/w042 and w045/w046 are coincident pairs
and w040->w043, w046->w047 are two turns apart; a naive relabel (k >= 43 -> k-1) flips arm19 - arm0 wrong area from
-4.7 to +13.5 pp. Band 2 has an unlabelled w035-w050 gap, so its vertex metric may be equally numbering-sensitive.

Changes: **none to the primary test, arms, inputs or the gating criteria (a) and (b).** Added, reported and not gating:
- Wrong area per winding subset: windings whose human-mesh labels are geometrically consistent (no coincident
  partner, spacing to both labelled neighbours within 0.6-1.6x the local median) vs the rest. The subsets are fixed
  from the human meshes alone before any band-2 fit output is read (file + sha256 to be recorded in 7b).
- Numbering sensitivity: the (b) statistic recomputed under each label correction the CT audit (7b) supports. If the
  audit is inconclusive, report the naive relabels as bounds.
- Interpretation rule, fixed now: if (a) passes and (b) passes only under the original numbering but not under a
  CT-supported correction, the write-up claims local winding (pairs) only and reports the global-count result as
  label-dependent.

### 7b. Label-consistency subsets (frozen 2026-10-21T17:48:20Z, from the human meshes only; no arm20-22 output, fit.log or score had been opened or listed)
- Script: `uv run python eval/label_subsets.py --band 1|2`. It reads only the human meshes (sites.json segments,
  2.399 tifxyz -> published affine, blacklisted included), the umbilicus and nothing else. Rules are in its docstring.
  - Rays = 50 z x 2 deg bins around the umbilicus. A mesh that crosses a ray twice (its own >360 deg overlap) is not used
    in that bin.
  - Coincident partner: |dr| < 0.25 x local median on >= 20 % of shared bins.
  - Per 2 deg angle bin: z-median per-turn spacing to the nearest scorer-used label on each side (blacklisted and
    partners skipped), compared with the local median (label-consecutive pairs, |dk| <= 3, labels within k +- 6,
    +-2 angle bins).
  - Consistent = no partner, >= 20 angle bins, and >= 80 % of angle bins with every neighbour in 0.6-1.6x.
  - One change before freezing, made on human data only: per (z, angle) bins were tried first. In this scroll,
    adjacent-sheet spacing at a single ray varies 10-30 vox, so all 37 windings failed. The z-median per angle bin
    is used instead.

| file | sha256 |
|---|---|
| data/scratch/label_audit/subsets_band1.json | 20bbd3c70e1a1552e27239b48e85b1a5b8274ce257b58d65101e16d75f97a720 |
| data/scratch/label_audit/subsets_band2.json | c6dd5789b89ca4ca4643250317dc34827487f8aef18158c90c5b3e22fb2565f0 |
| eval/label_subsets.py | ef89fdef696515f553ac5d338a85bb0c288b3e72902d035a2747ac6e0b3613a1 |

| band | consistent | inconsistent |
|---|---|---|
| 1 (z 3000-4500) | 16: w023, w029-w033, w037, w044, w047, w048, w052-w054, w057-w059 | 21: w024-w028, w034-w036, w038-w043, w045, w046, w049-w051, w055, w056 |
| 2 (z 1750-3000) | 18: w023, w026, w028-w033, w036, w051-w059 | 19: w024, w025, w027, w034, w035, w037-w050 |

#### 7b, band-2 CT audit (2026-10-21T18:10Z; human meshes, CT, m7 and grad_mag only; still blind to arm20-22)
- Method: the same as the band-1 audit in notes/2026-10-21-fit-figures.md ("Label audit, band 1").
  - eval/ct_gap_audit.py --band 2: 24 angles x 6 z (1856-2898), seams avoided.
  - Counted by m7 runs and by grad_mag turns, calibrated on dk=1 controls.
  - "Local" estimates are calibrated per ray on w051-w059. That span itself gives gm 7.94 for labels 8, so the
    controls behave.
  - Output: data/scratch/label_audit/ct_counts_band2.json.

| span | labels | CT turns: local gm / m7 [95 % ray bootstrap] | verdict |
|---|---|---|---|
| w023-w028 (w024, w025 and w027 are spacing-inconsistent) | 5 | 5.3-5.6 | labels right; spacing irregular |
| w041 vs w042 (z >= 2689, where both exist) | 0 | coincident, as in band 1 | same sheet |
| w040-w043 (z >= 2689) | 3 | 2 turns on 59 % of rays | 2 turns, as in band 1 |
| w034-w036 | 2 | 1 turn (6 rays only) | w036 directly outside w034 (thin evidence) |
| **w034-w051, all rays** | 17 | **14.60 [14.04, 15.23] / 15.43 [14.82, 15.88]** | **original numbering rejected** (both CIs exclude 17) |
| w034-w051, outside 145-260 deg | 17 | 15.5 / 15.4 [CIs 14.7-16.6] | between -1 and -2 |
| w034-w051, #2167 sector | 17 | gm 12.5 | about -4.5 (sector; use excl_flag) |
| w033-w051 | 18 | 15.7-16.7 | consistent with the above |

- **Bounds**, in data/scratch/label_audit/relabel_band2.json
  (sha256 f87f5fb886db0fd7659325c1a28598c543c5a0e8da9a92eb97e205e6a5c800d0):
  - `minus1` is the band-1 analogue: w041, w042 -> 41, and k >= 43 -> k - 1. This puts w051 16 turns outside w034.
  - `minus2`: w036-w040 -1, w041, w042 -> 40, and w043+ -2. This puts w051 15 turns outside w034.
  - CT cannot separate -1 from -2: outside the sector the shift is about -1.5 +- 0.5. Both maps are bounds, not a single
    correction.
- Band-1 counterpart: data/scratch/label_audit/relabel_band1.json
  (sha256 c5507cbfbfd0bf616d85d88d72d27f5271553f5113053350ffa3392abad12f66). `supported` is the CT-majority map;
  `outer_minus2` is the other bound.
- **Instruction for the band-2 (b) read-out.** This adds to section 7 and 7a; it does not replace them.
  1. Make the per-vertex dumps:
     `release/fit_vertex_dump.py --z0 1750 --z1 3000` for arm0 and for the band-2 arms.
  2. Report (b) three times: under the original labels, under `minus1` and under `minus2`. Use
     `uv run python eval/subset_wrong_area.py --band 2 --dump-a <arm0> --dump-b <armX> --subsets data/scratch/label_audit/subsets_band2.json --split 41 [--relabel data/scratch/label_audit/relabel_band2.json --relabel-key minus1|minus2] [--excl-flag] [--own-offsets]`.
  3. Apply 7a's interpretation rule:
     - The pooled sign counts as label-robust only if it agrees under both bounds.
     - Otherwise the label-order-independent numbers are primary: inner (k < 41) and outer (k >= 43) with
       `--own-offsets`, on the consistent subset.
  - Band 1 shows why this is needed. The pooled arm19 - arm0 is -4.69 pp under the original labels, +13.51 under
    `supported` and +0.55 under `outer_minus2`. The own-offset inner (-19.3) and outer (+9.5) are unchanged by relabeling.
- Scripts (sha256):
  - eval/subset_wrong_area.py: 10186ccacb385a5fda6a041909168df7c7c8d7c4e22ee0280abaf02c01d20e0e
  - eval/ct_gap_audit.py: 8a444125248900f9919f9a4e1f397a2d0a53c8b6d21dc54664917a80855cf2e3
  - eval/label_audit_figs.py: b2bd7b28e32371c8f84947e6fe415af422e1417cdd5088bac7ffde637f37d96e

### 7c. Amendment 2 (written before any arm20-25 score exists; time in the line below)
Reason: the CT audit (7b) shows the human numbering is wrong in both bands (band 1: w041==w042, w045==w046,
w040->w043 is 2 turns; outer windings 1-2 too high; band 2: w034->w051 is ~14.6-15.4 turns vs 17 labelled). Pair
truth deltas come from the same numbering, so pairs spanning those gaps carry wrong truth.
Changes: none to the gating test. Added, reported and not gating: (a) matched pair accuracy recomputed (i) excluding
pairs whose ends straddle a CT-flagged gap or duplicate, and (ii) under relabel_band2.json minus1 and minus2. The same
interpretation rule as 7a applies to pairs: a claim survives only if it holds on the excluded-pairs set.
- 7c frozen at 2026-10-21T18:13:07Z

## Confirmation band (results; written 2026-10-21T22:05Z, after 7b's sha256 lines existed)

All six arms ran to step 4, 240 windings each. Each arm was scored with `spiral_eval --tol 4`, one arm at a time.
Outputs are in data/scratch/spiral_eval/arms/<base>_vs_<arm>.json. The tables are `spiral/arm_table.py` output.

### Band 2 (z 1750-3000), base arm20

| arm | pair acc (matched) | coverage | acc_all | contact acc | wrong area % (seam) | switches/1000 (seam) | unmatched % |
|---|---|---|---|---|---|---|---|
| arm20 (A) | 73.3 | 48.8 | 35.8 | 59.5 | 55.9 | 61.1 | 30.0 |
| arm21 | 68.9<br>-4.4 [-9.5, +1.9] | 59.6<br>+10.8 [+6.0, +16.3]* | 41.1<br>+5.3 [+1.1, +10.2]* | 56.1<br>-3.4 [-8.4, +3.4] | 47.0<br>-9.0 [-13.2, -5.1]* | 91.9<br>+30.8 [+26.2, +35.6]* | 28.3<br>-1.6 [-2.5, -0.8]* |
| arm22 | 71.7<br>-1.6 [-5.9, +3.8] | 59.3<br>+10.5 [+5.3, +16.6]* | 42.5<br>+6.7 [+2.3, +12.1]* | 57.4<br>-2.1 [-7.4, +4.9] | 36.7<br>-19.3 [-24.1, -14.9]* | 69.0<br>+7.9 [+3.1, +12.6]* | 27.3<br>-2.7 [-3.4, -2.0]* |

Cells: B value, then B - arm20 [95 % CI]; * = CI excludes 0.

**Pre-registered test (section 7): FAIL.**
- (a) arm21 - arm20 matched pair accuracy is -4.4 [-9.5, +1.9]. The CI includes 0 and the estimate is negative.
- (b) wrong area (seam) is -9.0 [-13.2, -5.1]. This passes as pre-registered.
- The claim needs both, so it fails.
- The oracle ceiling arm22 does not improve matched accuracy on this band either: -1.6 [-5.9, +3.8]. So (a) has little
  or no headroom here. That is an observation, not a re-reading of the test.

7b/7c read-outs (added, not gating). Tools:
- `eval/subset_wrong_area.py --band 2 --split 41 --own-offsets` on vertex dumps from `release/fit_vertex_dump.py`
  (data/scratch/fit_figures/vertex/arm2{0,1,2}.npz). The arm20 dump reproduces the scorer: 55.94 / 29.96.
- `eval/pairs_exclude.py spiral` with flagged_pairs_spiral_band2.csv: 545 CT-flagged pairs excluded; minus2 changes the
  truth of 57 pairs and minus1 changes none.
- Outputs: data/scratch/label_audit/swa_band2_arm20_arm2{1,2}_*.json and spiral_ct_fix_band2_arm2{1,2}.json.
- Runner and table scripts: data/scratch/second_band/labelfix/.

Wrong area (seam) B - arm20, pp:

| arm | labels | all | consistent | inner own-off (cons) | outer own-off (cons) |
|---|---|---|---|---|---|
| arm21 | original | -9.0 [-13.2, -5.1]* | -9.0 [-13.4, -5.0]* | +4.2 [+2.1, +6.4]* | +3.3 [+0.7, +5.8]* |
| arm21 | minus1 | +13.1 [+11.1, +15.2]* | +14.0 [+11.6, +16.2]* | (same) | (same) |
| arm21 | minus2 | +14.6 [+11.4, +18.6]* | +13.8 [+10.4, +18.2]* | +4.3 | +3.3 |
| arm22 | original | -19.3 [-24.1, -14.9]* | -15.4 [-20.3, -10.9]* | -4.5 [-7.3, -1.8]* | -1.7 [-4.1, +0.7] |
| arm22 | minus1 | +7.3 [+5.3, +9.4]* | +12.8 [+10.3, +15.3]* | (same) | (same) |
| arm22 | minus2 | +8.6 [+5.4, +12.4]* | +12.3 [+8.7, +17.0]* | -4.4 | -1.7 |

`--excl-flag` (#2167 sector dropped) changes none of these signs.
- Under 7a's rule, the pooled (b) improvement is **not label-robust**: it flips sign under both bounds.
- The label-order-independent own-offset numbers on the consistent subset are therefore primary. They say arm21 is
  **worse** than arm20: inner +4.2*, outer +3.3*.
- The oracle arm22 is better inside (inner -4.5*) and flat outside.

Pairs (7c), matched accuracy B - arm20:

| arm | all pairs | CT-flagged pairs excluded | minus2 truth, excl |
|---|---|---|---|
| arm21 | -4.4 [-9.5, +1.9] | -5.0 [-8.9, -1.0]* | -4.9 [-8.6, -1.0]* |
| arm22 | -1.6 [-5.9, +3.8] | -2.3 [-5.7, +1.2] | -1.9 [-5.3, +1.5] |

acc_all stays positive throughout: arm21 +5.4*, arm22 +6.7*.

Net on band 2: the claim fails as pre-registered, and with the label corrections it is worse:
- matched accuracy is significantly negative on the excluded-pairs set;
- the only robust wrong-area numbers point against arm21.

### Band 1 (z 3000-4500), base arm0
- arm23 = clean gmxc chains: pre-registered secondary check.
- arm24 = p93 and arm25 = dsafe: DEV arms, not pre-registered.
- arm19 is the earlier gmx result, shown for reference.

| arm | pair acc (matched) | coverage | acc_all | contact acc | wrong area % (seam) | switches/1000 (seam) | unmatched % |
|---|---|---|---|---|---|---|---|
| arm0 (A) | 78.0 | 62.9 | 49.1 | 57.7 | 48.8 | 48.5 | 21.5 |
| arm19 | 90.3<br>+12.3 [+7.9, +17.2]* | 59.8<br>-3.1 [-9.3, +2.5] | 54.0<br>+4.9 [+0.5, +9.3]* | 69.8<br>+12.0 [+2.1, +20.9]* | 44.1<br>-4.7 [-7.2, -2.1]* | 66.1<br>+17.6 [+13.6, +21.9]* | 21.2<br>-0.3 [-0.8, +0.2] |
| arm23 | 93.0<br>+15.0 [+9.5, +21.3]* | 61.8<br>-1.0 [-5.9, +4.0] | 57.5<br>+8.4 [+3.5, +12.8]* | 72.2<br>+14.5 [+5.6, +23.3]* | 36.9<br>-11.9 [-14.7, -8.9]* | 63.7<br>+15.2 [+11.5, +19.5]* | 21.7<br>+0.2 [-0.4, +0.8] |
| arm24 (dev) | 75.7<br>-2.3 [-8.8, +4.4] | 58.7<br>-4.2 [-9.5, +0.6] | 44.5<br>-4.6 [-10.6, +1.0] | 59.6<br>+1.9 [-8.6, +14.3] | 46.4<br>-2.5 [-3.4, -1.5]* | 56.6<br>+8.1 [+6.5, +9.9]* | 21.2<br>-0.4 [-0.7, -0.0]* |
| arm25 (dev) | 85.1<br>+7.0 [+3.7, +11.0]* | 61.4<br>-1.5 [-6.4, +3.1] | 52.2<br>+3.1 [-1.0, +7.0] | 66.1<br>+8.3 [+2.5, +13.9]* | 44.7<br>-4.1 [-6.5, -1.5]* | 56.8<br>+8.3 [+5.0, +12.2]* | 21.2<br>-0.3 [-0.7, +0.2] |

Cells: B value, then B - arm0 [95 % CI]; * = CI excludes 0.

- **Secondary check (arm23 vs arm19): passes on both criteria and is stronger than arm19.**
  - Only 1,745 of 40,658 steps (4.3 %) differ from gmx, yet wrong area is -11.9 vs -4.7.
  - So fit-to-fit variance on these metrics is large compared with the chain change.
- Band-1 7b read-out (relabel_band1 `supported` / `outer_minus2`), wrong area B - arm0:

  | arm | original | supported | outer_minus2 | own-off inner / outer (cons) |
  |---|---|---|---|---|
  | arm19 | -4.7* | +13.5* | +0.5 | -15.6* / +9.8* |
  | arm23 | -11.9* | +8.7* | -4.0* | -19.1* / +9.5* |
  | arm24 | -2.5* | +4.4* | +1.0* | +2.1 / +0.3* |
  | arm25 | -4.1* | +12.9* | +1.0 | -12.8* / +5.3* |

  - The pooled signs are not label-robust for any arm.
  - The robust pattern is the one 7b found for arm19: big gains inside k < 41 and losses outside.
- Band-1 pairs: the CT-flagged exclusion removes no band-1 spiral pair, the same as for arm19. Only
  supported_plus_2167 changes the truth: arm23 +10.8*, arm25 +3.0, arm24 -2.4.
- Dev-arm reading:
  - p93 (arm24) removes most of the gain.
  - dsafe (arm25) keeps about half of it and has about half the switches of arm19/arm23.

### Cost (Prime, start to termination; 6 x 1xL40S)

| arm | provider | price | hours | cost |
|---|---|---|---|---|
| arm20 | massedcompute | $0.82/h | 2.73 | $2.23 |
| arm21 | datacrunch | $1.79/h | 3.66 | $6.55 |
| arm22 | runpod | $1.11/h | 2.88 | $3.20 |
| arm23 | massedcompute | $0.82/h | 2.92 | $2.40 |
| arm24 (dev) | massedcompute | $0.82/h | 2.98 | $2.44 |
| arm25 (dev) | massedcompute | $0.82/h | 3.42 | $2.81 |
| total | | | | **~$19.6** (cap $60) |

All six pods were terminated by collect_and_score.sh, and the terminations are logged in
data/scratch/p0139_spiral/pods_arms.txt. A `prime pods list` at 21:56Z shows no vss-* pod (only flint-tb2, not ours).

### Deviations from section 7
- Pods:
  - 6 single-L40S pods from 3 providers instead of 2xL40S pods, because no 2xL40S offer existed. Prices are mixed.
  - Datacrunch and massedcompute images ship python 3.10, so bootstrap installs a uv python 3.12 `python3` shim.
- arm22's first launch (17:53:03Z) raced the input copy. It was killed after 9 s, before any fit output, and relaunched
  at 17:53:45Z.
- Band-2 arms use MASK_TAG m7pub with a band-keyed SDT cache (run_arm.sh) instead of a separate m7pub_z1750 tag.
- The clean conf threshold (0.0023) is unused at `--conf 1.0`; only p_err enters, through gm. The clean gm zero-drift
  threshold is 0.18 (old model 0.19).
- 7b/7c read-outs for arm23-25 were added as well (not gating).
