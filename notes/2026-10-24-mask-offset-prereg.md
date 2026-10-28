# 2026-10-24: m7 mask->CT offset: cause, label-free correction, and a dev gate on the chain-switch cost (fixed before any fit)

**Gate frozen: see addendum G0 (UTC timestamp and sha256 of sections 0-5).** Sections 0-5 are written after the
label-free offset analysis (section 1) and the label-free construction of the corrected chain sets, and before any
fit with corrected chains and before any label-based score of them. Nothing in sections 0-5 changes after G0; the log
and results below are append-only and timestamped. Run directory `data/scratch/mask_offset/`; pods `vss-off-*`.

## 0. Question and hypothesis
Test 7 (notes/2026-10-24-f7-switch-aware-cut.md, L3 post-hoc) found that the cut recipe's extra seam switches (+6-7 per
1000 pairs at z 7000) do not depend on which chain pieces are kept: within 16 vox of kept chain points the local switch
rate is 177-238 per 1000 against 63-85 for base fits under every variant. So the cost looks like the weight-10 chain
radius term itself. The fiber-contacts work (notes/2026-10-24-fiber-contacts-gate.md, L4) found that the m7 mask sits
about 1.6-1.9 vox inward of the CT sheets. **Hypothesis H:** the chain points are m7 run centres, so if the fitter's
other evidence sits on the CT sheet, the radius term pulls the surface about 2 vox off that evidence near every anchor
and causes local switches; moving the chain points onto the CT sheet removes that cost.

## 1. Findings before the gate (label-free; scripts in `data/scratch/mask_offset/`)

### 1a. The offset is real and not a frame artefact
- **Grids are identical.** The PHercParis4 m7 store (`...-surface-m7-L2-th0.2.zarr`, metadata.json: inference on
  `resolution "2"` of `2.4um_PHerc-Paris4_masked.zarr`, output OME levels by "nearest") has level-0 shape
  (18946, 8174, 8174), equal to CT level 2 of 20260411134726. Both OME `.zattrs` carry scale-only transforms (no
  translation). PHerc0139 m7-L0 (`resolution "0"` of 20250728140407) has shape (20974, 6621, 6621), equal to CT level 0.
  m7 and CT are read from the same voxel grid in both regions, so no level-alignment, half-voxel or mean-pool origin
  term can enter between them. (Fiber/native conversions do not apply here.)
- **The shift is radial, not constant, not scale** (`frame_check.py` on the 768 raw 128^3 block shifts per region of
  `reg_<region>.json`; fit d_yx = c + a u_rad, bootstrap over zones):

  | region | radial mean | a (radial coef) [95 %] | constant c_y, c_x | radial mean by 90-deg sector | by radius tercile |
  |---|---|---|---|---|---|
  | PHerc0139 b1 | −1.91 (87 % inward) | −2.01 [−2.31, −1.72] | +0.07, +0.26 | −1.58 / −2.73 / −1.65 / −2.12 | −1.75 / −2.07 / −1.91 |
  | Paris4 z9000 | −1.64 (78 % inward) | −1.67 [−1.98, −1.32] | +0.23, +0.48 | −0.62 / −2.06 / −2.21 / −1.72 | −1.83 / −2.01 / −1.08 |

  The dy/dx means change sign with the quadrant while the radial component stays negative on every side, the constant
  term is <= 0.5 vox, and a scale term is not consistent (−0.43 / +0.43 per 1000 vox). A frame artefact would be a
  constant xyz shift (sector means of alternating radial sign) or grow with radius. Neither is seen.
- **Directly at the chain points, along the ray** (`mean_profile.py`, CT averaged over 9 parallel lines, aligned on
  each chain point, t > 0 outward): the m7 run is centred on the chain point (m7 centroid 0.00, half-max at about
  ±2 vox), but the CT is not.
  - PHerc0139 b1 (z 3360/3424, 3,607 points): mean CT 74 at t = −4, rises to its peak 106.3 at **t = +2.0**, back to 102
    at +4. The m7 band [−2, +2] sits on the inner flank of the sheet.
  - Paris4 z9000 (z 9504/9568, 4,345 points): steep inner flank (84 at −4 → 98 at 0), peak 99.7 at **t = +1.0**, slow
    outer decay (96 at +4, 89 at +8). The bright band that matches the m7 width best is centred at **+1.5 to +2.0**.
  - Per-point CT peak in ±4 vox (noisy single profiles): median +0.75 (PHerc0139) / +0.45 (Paris4), p10-p90 about
    −2.7 … +3.3. Neighbourhood-averaged (the `field` estimator below): median **+2.30 / +2.06**, 94 % / 87 % outward.

### 1b. Cause: m7 marks one face of the sheet (its training target), and that face points inward here
- m7 is the nnU-Net component of the 1st-place Kaggle "Surface Detection" solution (`scrollprize/surface_m7_nnunet`,
  Dataset100_VesuviusSurface, labels background / surface / ignore). The organisers' sibling predictions are explicitly
  named `surface-recto-*`.
- Our measurement of those training-type labels (notes/2026-10-18-w1-data-recon.md, `label_stats.json`): on the Kaggle
  9 um patches the surface label is a 1-3 vox band whose ridge lies 1-2 vox **off** the CT intensity peak, on the side
  facing air, 1.8-2.7 vox inside the CT half-max edge ("on the bright recto skin"). The 2.4 um human meshes do the
  same (label at t = 0, CT peak at +5 to +8 vox of 2.4 um).
- So m7 learned a face band, not a sheet centre. In these rolls the labelled face (recto) faces the umbilicus, so the
  m7 band sits on the inner flank of each sheet and the CT sheet centre is 1.5-2 vox outward. The extra local ±2-4 vox
  variation of the block shifts is partly estimator noise (sd 2.2-2.4 around the radial fit) and partly real
  (oblique sheets, merged contacts, places where the band picks the other face).

### 1c. What the fitter actually uses (villa 0fb38c45, our recipe; `spiral/run_arm.sh`, recipe.toml)
- **Tracks** (`tracks_dbm`): kimimaro skeletons of the **same m7 mask** (level 0 of `m7_L2_th0.2.zarr`, downsample 2,
  `spiral/extract_tracks.py`). Loss weights track_radius 50, track_dt 10 (from step 10,000).
- **Surf-SDT** (`surf_sdt`): signed distance of the **same m7 mask** at level 1, threshold 128 (`make_surf_sdt.py`). In
  `dense_spacing_mode = "phase"` it drives the whole phase bundle (band registration, crossing count, min spacing,
  dense_spacing_density weight 12).
- **Lasagna normals** nx / ny (CT-derived, level 2): orientation only (dense_normals 100); they carry no sheet position
  at the 2-vox scale. grad_mag is not loaded in phase mode (`grad_mag_spacing_enabled` needs mode `grad_mag`).
- Patches off (`input_disable_patches`), shell off. Chains: unattached strips, radius weight 10, spacing 0.
- **Consequence for H: the offset is shared.** Every positional input of the fitter (tracks, SDT bands, chain points)
  is in the m7 frame, on the same inner face. Chain points do not sit 2 vox off the fitter's other evidence; they sit
  on it. Correcting the chains alone moves them about 2 vox *off* the tracks and SDT bands, so H as stated is weakened a
  priori, and the corrected arm may add switches instead of removing them. **Prediction recorded now: no switch
  improvement (switch Δ stays near +6.5); a small increase is possible.** The test is still run (one fit) because it is
  cheap and decides the question directly, and because the near-anchor switch cost could still depend on where inside
  the m7 band the anchor sits.

## 2. Correction (label-free; new module, default off)
`tools/autowind/offset.py` (`python -m autowind.offset`; sha in G0). Modes `off` (default: byte-for-byte copy),
`ctpeak` (per-point CT peak, noisy, not used), **`field`** (used), `radial` (constant). `field`, per chain point:
the raw CT profiles along the ray (9 lines, 0.25 vox, trilinear) of all chain points within 64 vox in z (adjacent ray
levels), 15 deg (adjacent rays) and 48 vox in radius are averaged, each aligned on its own point; the shift is the
centre of the window of width = the point's m7 run length (rl, clipped to [2, 8]) with the highest mean CT, over
|shift| <= 4 vox, clipped to the midpoints to the chain's neighbouring points minus 0.5 vox (order kept; never onto a
neighbour's sheet). Only point positions change: wind, steps and pieces are unchanged.
- Reproduction (checked before G0): `--mode off` and `--mode field --window 0` reproduce the test-6 cut sets byte for
  byte: z 7000 `p4z7_f6` 03c014bdf5e28d0e / chains b5d20c1da2056e76; z 9000 `p4z9000_cut263` 0954c72a8ed7df88 /
  chains bdbc7bce34cc7c5c. tools/auto_windings.py, eval/p0139_graph.py and tools/autowind/postproc.py are unchanged.
- Corrected sets (built before G0, label-free; shas and shift statistics in G0):
  `data/scratch/mask_offset/windings/p4z7_f6_off4` (z 7000) and `.../p4z9000_cut263_off4` (z 9000), command
  `python -m autowind.offset --chains <test-6 cut set> --mode field --ct s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr --ct-level 2 --out <dir>`.

## 3. GATE (fixed now)
Scorer `data/scratch/mask_offset/dev_table_off.py` (= `data/scratch/f7_switch/dev_table_f7.py` unchanged, sha
cc7786b7…, with this gate and vertex dir; sha in G0): the stored dumps of the band's 3 base seeds (z 7000: f6z_base /
f6z_bases2 / f6z_bases3), own-offset seam wrong area on PRIMARY_K `13,16-32,34-44,47-61,63-108`, paired BootV, switches
per 1000 adjacent pairs.

Arms (z 7000-8500, burned dev band; weight 10, spacing 0, seed 1, villa 0fb38c45):
- **as-is:** the test-6 cut chains `p4z7_f6` = the stored fit **f6z_ms1** (same chains and settings; not re-fitted).
  Reference with the same code: −7.85 / −7.93 / −7.95 primary, switch +6.51 / +7.10 / +6.72, nonprimary +5.47 / +3.77 /
  +3.79; local switch rate 237.5 /1000 at < 16 vox and 109.8 at 16-32 vox of kept chain points (base 85.2 / 74.1).
- **corrected:** `p4z7_f6_off4`, arm **off_f6z7** (`spiral/arms/off_f6z7.env` = f6z_ms1.env with ARM and
  REL_WINDINGS changed). One fit.

**PASS iff, against EACH of the 3 base seeds:** (1) primary own-offset upper 95 % CI < 0; (2) primary estimate <= −5.0
pp; (3) switch Δ <= +2.0 per 1000 (point estimate).

Reported whatever the outcome: corrected − as-is (f6z_ms1) primary and switch deltas, nonprimary own-offset Δ, global
seam Δ, and the switch decomposition (`switch_diag_off.py`, = f7 `switch_diag.py` unchanged): local switch rate per
1000 matched pairs at < 16 and 16-32 vox (3-D) from the arm's own kept chain points, against base and against f6z_ms1
(test 7's near-chain diagnostic).

Reading of the result, fixed now:
- PASS -> H is supported on this band despite the shared frame; stage B below, then the confirmation in section 4.
- FAIL with near-chain local rate (< 16 vox) within ±30 /1000 of f6z_ms1 -> the m7 offset does not explain the
  near-chain switch cost (consistent with section 1c).
- FAIL with a clearly higher near-chain rate (> f6z_ms1 + 30) -> the correction conflicts with the fitter's m7-frame
  evidence (tracks / SDT), as predicted in 1c; a test would need the fitter inputs moved too (a different test).

**Cap and order (at most 3 dev fits in this test; expected 1-2):**
- Stage A: off_f6z7 only.
- Stage B (only if stage A passes): **off_f6z9** on z 9000-10500 (`p4z9000_cut263_off4`, env off_f6z9.env = f6_cut.env
  with ARM and REL_WINDINGS changed) against p4_base / p4_bases2 / p4_bases3 with the same three items (PRIMARY_K
  `21-32,34-60,62-87,90-101,106-108`); f6_cut is the as-is reference. If stage A passes but stage B fails, the result is
  "z 7000 only" and no confirmation follows from this note.
- No re-tuning of the correction (window, neighbourhood, mode) after any fit is scored. A third fit is used only to
  repeat a failed / crashed pod run of the same arm.

**Secondary (no pods; label use limited to the two dev regions' existing pilot tables):** the contact-detector CT pilot
(notes/2026-10-24-contact-detector-gate.md L3; `ctfeats.py` features, `ct_pilot_auc.py`, `ct_pilot_fit.py`) is re-run
with the CT step profiles sampled between **corrected** chain points (`field` shifts of the no-raise chains
`band_m7_ray_c` / `paris4_m7_ray_c`, computed with the neighbour levels z 3296-3488 / 9440-9632), on the same pilot
steps (PHerc0139 b1 z 3360/3424, Paris4 z9000 z 9504/9568). Reported: per-feature AUC vs true diff == 2, and the
cross-pilot (train on one region, score on the other = LORO over the two CT regions) precision at 2/4/6/9/12 % raised,
for the gm7 and "all" feature sets, before vs after. Not gated; it says whether better-placed CT samples carry more
contact signal.

## 4. If it passes: confirmation (not run under this note)
The fresh region PHercParis4 z 10800-11800 (notes/2026-10-24-f7-confirm-prereg.md, stage 0 frozen 18:32:29Z, S_mask
19.0) is still unused. A separate two-stage pre-registration would freeze the method (test-6 cut + `autowind.offset
--mode field`) and the 6-combo pass rule (every combo upper CI < 0 and switch Δ <= +5, oracle informative) before the
chains exist. **Nothing label-based is run on z 10800-11800 under this note.**

## 5. Budget and rules
- Spend cap $30 (pods `vss-off-*`, REST via `data/scratch/mask_offset/pods/prime_rest.py`, which refuses to delete any
  pod not named `vss-off-*`). Terminate right after the pull; log in data/scratch/p0139_spiral/pods_arms.txt. Never touch
  vss-ink2-*, vss-hitl-*, sm86-speed, kv4-*, flint-* or any pod not created here.
- Frozen files untouched: tools/auto_windings.py, eval/p0139_graph.py, tools/autowind/postproc.py, tools/check_viewer.py,
  tools/check_viewer_ui.html and the files frozen in notes/2026-10-24-hitl-session-prereg.md.
- Vertex dumps under `flock /tmp/vss_eval.lock` unless MemAvailable > 20 GB (logged in
  `data/scratch/mask_offset/eval_lock_exceptions.log`).
- No commits, pushes or uploads. z 10800-11800 untouched.

---

## Log (append-only)
### G0 (2026-10-24T23:29:34Z): gate frozen
sha256 of this file up to and including the "## Log (append-only)" line:
3ce100c2ef2daa9f65755e853f5fab06996fcf5752451ec3c99435ea361e5e67 (snapshot `data/scratch/mask_offset/freeze/gate_note_snapshot.md`;
all shas in `data/scratch/mask_offset/freeze/gate_freeze.sha256`).
- tools/autowind/offset.py 28ea92dc…; dev_table_off.py 2511d904…; switch_diag_off.py a8fae598…; dev_table_f7.py cc7786b7…
  (unchanged); switch_diag.py 0acace32… (unchanged); off_f6z7.env 67b32d28…; off_f6z9.env fd3b7327….
- Corrected sets (`--mode field`, W 4, neighbourhood 64 vox / 15 deg / 48 vox):
  - z 7000 `p4z7_f6_off4`: relative_windings.json **4039ee7d63e3ddd8…**, chains.json 395b2be5…. 5,768 chains, 53,749
    points (same chains, winds and names as p4z7_f6); shift median +2.12 vox (p10 +0.58, p90 +3.85), 93.6 % outward,
    3.4 % inward; 7,822 points (14.6 %) at the +4 cap; 6,225 CT chunks.
  - z 9000 `p4z9000_cut263_off4`: relative_windings.json **a48e7198a8aa6435…**, chains.json e5b6bcf6…. 6,209 chains;
    shift median +2.12.
  - Checks: radii stay strictly increasing in every chain; z, wind, collection names and wind_a identical to the input.
- auto_windings.py 8f097312…, p0139_graph.py fb7553b1…, postproc.py af573101… unchanged.
- No fit with corrected chains exists; no vss-off pod has been created; no label-based number of the corrected sets has
  been computed.
### L1 (2026-10-24T23:31Z): stage A pod created
`off_f6z7` (chains p4z7_f6_off4, rw sha 4039ee7d…): pod <pod-id> vss-off-off_f6z7, 1xA6000 48 GB massedcompute,
$0.54/h, created 23:30:16Z by `data/scratch/mask_offset/pods/run_arm.sh` (REST; boot and launch follow in the same
script). Collector `pods/collect.sh` (N_ARMS=1: pull, terminate, vertex dump) and a spend watchdog (deletes the pod at
2026-10-25T10:30Z if it still exists) run locally. Only one vss-off pod exists.
### L2 (2026-10-24T23:33Z): secondary result, contact-detector CT pilot at corrected positions (no pods)
`data/scratch/mask_offset/ct_pilot_off.py` (ctfeats.chain_feats unchanged, between field-corrected no-raise chain
points; same pilot steps: PHerc0139 b1 719 labelled single steps / 53 true-2, Paris4 z9000 1,817 / 290). Output
`ct_pilot_off.json` / `.log`.

| | PHerc0139 b1 orig -> corrected | Paris4 z9000 orig -> corrected |
|---|---|---|
| best CT feature AUC (c_midmean) | 0.667 -> 0.670 | 0.592 -> 0.606 |
| c_midmax AUC | 0.595 -> 0.629 | 0.545 -> 0.568 |
| c_dblA / c_dblB (double-sheet at a crossing) | 0.563 / 0.586 -> 0.548 / 0.543 | 0.574 / 0.581 -> 0.574 / 0.569 |

Cross-pilot (train on one region, score the other):
| model | b1 -> Paris4: AUC; precision at 2/4/6/9/12 % raised | Paris4 -> b1 |
|---|---|---|
| CT only (8), orig | 0.612; 8 / 11 / 19 / 21 / 21 % | 0.624; 21 / 11 / 12 / 14 / 12 % |
| CT only (8), corrected | 0.627; 28 / 25 / 24 / 21 / 23 % | 0.668; 21 / 18 / 16 / 16 / 14 % |
| all 21 (gm7 + CT + nb + mask), orig | 0.755; 8 / 25 / 34 / 42 / 40 % | 0.762; 43 / 29 / 33 / 28 / 24 % |
| all 21, corrected | 0.771; 8 / 26 / 34 / 41 / 39 % | 0.788; 43 / 36 / 30 / 28 / 26 % |
| gm7 alone (no CT; reference) | 0.765; 28 / 38 / 41 / 42 / 44 % | 0.735; 57 / 32 / 28 / 23 / 20 % |

- Sampling CT between corrected points lifts the CT features slightly (AUC +0.003 to +0.044, mostly the mid-gap
  brightness features, because the step's ends now sit on the sheet centres and the gap profile is cleaner). The
  combined model's held-out precision moves by −3 … +7 pp at fixed raised fractions; the precision changes are within
  the noise of these small pilots (2 % of 719 steps is 14 steps).
- The contact detector's ceiling stays where it was (held-out precision <= ~45 % at useful raise volumes against the
  70 % gate bar). The offset was a real but minor contributor to the weak CT pilot (AUC <= 0.67), not the reason for it.
### L3 (2026-10-25T00:20Z): off_f6z7 fit started
The boot verified the windings sha on the pod (4039ee7d63e3ddd8…) and launched at 23:35:36Z. Tracks (m7, z 6744-8756)
and surf-SDT were built on the pod. The fit started at 00:19:43Z with the f6z_ms1 overrides
(`pcl_unattached_pcl_min_point_spacing` 0.0, `loss_weight_unattached_pcl_radius` 10.0, 30k steps, z 7000-8500).
Mechanics check of the scorer (before any off_f6z7 number existed): dev_table_off.py reproduces f6z_ms1 against the 3
bases exactly (−7.85 / −7.93 / −7.95, switch +6.51 / +7.10 / +6.72), `dev_table_check.json`.

### L4 (2026-10-25T05:35Z): stage A result, off_f6z7 scored
Fit finished at about 05:15Z, exported, pulled (240 windings), and the pod was terminated at 05:20:57Z (REST status
TERMINATED; `prime_rest.py list` shows no vss-off pods). Vertex dump at 05:31Z, run without flock (MemAvailable 52 GB,
logged). Frozen shas re-checked against `freeze/gate_freeze.sha256` after scoring: all OK. Outputs:
`data/scratch/mask_offset/dev_table_off_z7.json` and `.log`, `switch_diag_z7000_off.json` and `.log`.

Against each base seed (own-offset seam wrong area on PRIMARY_K, paired BootV 95 % CI; switches per 1000 adjacent pairs):

| base | arm | primary Δ [95 % CI] | switch Δ | nonprimary Δ | global Δ [95 % CI] | gate items (CI / est / switch) |
|---|---|---|---|---|---|---|
| f6z_base   | off_f6z7 (corrected) | −7.07 [−8.16, −5.89] | +6.19 | +9.60 | −2.73 [−3.90, −1.32] | yes / yes / **no** |
| f6z_bases2 | off_f6z7 (corrected) | −7.15 [−8.20, −5.98] | +6.79 | +7.90 | −3.23 [−4.26, −1.95] | yes / yes / **no** |
| f6z_bases3 | off_f6z7 (corrected) | −7.16 [−8.19, −6.01] | +6.41 | +7.92 | −3.24 [−4.29, −1.97] | yes / yes / **no** |
| f6z_base   | f6z_ms1 (as-is)      | −7.85 [−9.01, −6.68] | +6.51 | +5.47 | −6.06 [−7.76, −4.42] | yes / yes / no |
| f6z_bases2 | f6z_ms1 (as-is)      | −7.93 [−9.07, −6.78] | +7.10 | +3.77 | −6.56 [−8.12, −4.95] | yes / yes / no |
| f6z_bases3 | f6z_ms1 (as-is)      | −7.95 [−9.08, −6.80] | +6.72 | +3.79 | −6.57 [−8.13, −4.97] | yes / yes / no |

Corrected − as-is (same for all 3 bases to ±0.01): primary **+0.78 pp** (worse; arm wrong area 65.4 vs 64.6 %), switch
**−0.31 /1000**, nonprimary **+4.13 pp** (worse), global seam **+3.33 pp** (worse).

Switch decomposition (`switch_diag_off.py --band z7000_off`; local rate per 1000 matched pairs by 3-D distance to the
arm's own kept chain points; mean of the 3 bases as reference):

| dchain | base | f6z_ms1 (as-is) | off_f6z7 (corrected) | corrected − as-is |
|---|---|---|---|---|
| < 16 vox  | 85.2 | 237.5 | **232.6** | −4.9 |
| 16-32 vox | 74.2 | 109.8 | **106.7** | −3.1 |
| total switch Δ vs mean(base) | — | +6.78 | +6.46 | −0.32 |

Elsewhere the decomposition matches f6z_ms1 bin by bin (dcut, dtheta_ray, dz_ray, sector, k within about ±1 /1000).

## VERDICT (stage A): FAIL
- Items (1) and (2) pass against all three bases. Item (3) fails against all three: switch Δ is +6.19 / +6.79 / +6.41,
  where the gate needs ≤ +2.0. The switch cost of the as-is chains (+6.5 … +7.1) is essentially unchanged.
- Under the frozen reading rule: FAIL, with the near-chain (< 16 vox) rate 232.6 vs 237.5, inside ±30 of f6z_ms1.
  **The m7 mask→CT offset does not explain the near-chain switch cost**, as section 1c predicted from the shared frame
  (the fitter's tracks and surf-SDT come from the same m7 mask, so chains and fitter evidence were already in one frame).
- The correction also costs: nonprimary +4.1 pp and global seam +3.3 pp worse than as-is, primary +0.8 pp worse. Moving
  the chains about 2 vox outward, onto the CT sheet centre, takes them out of the m7 frame that the tracks and SDT
  still use. That conflict shows up as wrong area, not as extra switches.
- Stage B (off_f6z9) is **not run** (it needed stage A to pass). Dev fits used: 1 of 3. No confirmation follows from this
  note. PHercParis4 z 10800-11800 stays untouched. `autowind.offset` stays default off (`--mode off` reproduces the
  chain sets byte for byte).
- What remains open (not tested here): moving the fitter's own m7-derived inputs (tracks, surf-SDT) together with the
  chains, which would be a different test with a new pre-registration. The offset is still a real, documented frame
  fact (−1.7 … −2.0 vox radial, recto-face target) for anyone who samples CT at m7 positions.

Spend: one A6000 pod, 23:30:05Z to 05:20:55Z = 5.85 h × $0.54 = **$3.16** (cap $30). Watchdog stopped; collector exited.
