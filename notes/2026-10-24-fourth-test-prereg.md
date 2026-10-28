# 2026-10-24: fourth pre-registered test, spacing-aware count rule (`autowind --raise auto`) on PHercParis4 z 7000-8500 (L2)

Status: **STOPPED AT THE DEV STAGE: NO-GO (2026-10-23T22:10Z).** Stage 0 (region choice) was frozen; stage 1 was never
frozen, and no confirmation pod was launched. Sections 1-9 are the stage-1 draft as it stood at the go/no-go and are
kept for the record. The outcome is in Results. Template: notes/2026-10-23-third-region-prereg.md
(Paris4 z 9000-10500). Run directory: `data/scratch/fourth_test/`.

## 0. Stage 0: region choice (frozen before any chain is built on it)
**Region: PHercParis4 (Scroll 1), volume 20260411134726, L2 (9.6 um), z 7000-8500, full slice.**

Fixed at the timestamp in addendum S0, before any chain, crop, fit or label-based metric on this band exists. Choice by
the brief's criteria:
- **Fresh.** The only computations ever made on this band are label-free S_mask (Paris4 draft §1.4, 2026-10-23) and
  label coverage (`data/eval/paris4/cover_z7000-8500.json`). No chain, crop, fit, step metric or fit metric.
- **Tight.** S_mask 17.5 [17.0, 18.0], slabs 17.0-18.0, sectors 15.0-21.5. The tightest Paris4 band scouted, below the
  dev regions (PHerc0139 19.0, Paris4 z 9000 19.0) and far from PHerc1667 (25.0).
- **Labels.** The organiser 20260701 winding-group meshes cover all 120 windings k 10-129 with >= 95 % angle cover
  (`cover_z7000-8500.json`), the same label source as Paris4 z 9000.
- **Scorer hooks.** `eval/spiral_eval.py --scroll paris4` takes the band as `--z0/--z1`. `label_subsets.py` and
  `ct_gap_audit.py` get a backward-compatible `--scroll paris4z7` choice (same labels, band 7000-8500, own m7/grad_mag
  rows).
- **Separation from dev data.** 500 z from the burned Paris4 band (z 9000-10500). The autowind crops of z 7000-8500
  reach at most z ~8600, those of z 9000-10500 start at z ~8900, so no crop overlaps.
- **Not chosen:** Paris4 z 12500-14000 and 15500-17000 (S_mask 18.0) lie inside the 2023 GP label range (L2
  11832-17625), where m7's training data may include Scroll 1. PHerc0139 z 6000-8500 is sparsely labelled. Paris4
  z 4500-6000 is not tight (S_mask 22.0).
- **Disclosed:** the confirmation band is on the same scroll (and the same lasagna run and m7 run) as a dev region.
  It tests transfer to a fresh, tighter band, not to a new scroll.

## 1. Method: `autowind chains --raise auto`, one spacing-aware count rule for any scroll
**Rule (label-free; computed from the mask, then grad_mag only if the rule raises):**
1. Solve the autowind crops of the band (frozen recipe, `tools/auto_windings.py` untouched).
2. **S_pack** = median radial gap between consecutive deduplicated mask crossings on the probe rays of the solved crops
   (`tools/autowind/packing.py`, a port of `data/scratch/third_region/packing_stat.py` sha 594863ce…; 30° × 250 z block
   bootstrap, seed 1, reported only).
3. **If S_pack < 21.0 vox: raise with the frozen grad_mag logistic at threshold 0.263. Otherwise: no raise.**
   - 21.0 = log-midpoint of PHerc0139 (18.5) and PHerc1667 (24.0), fixed in the 2026-10-23 Paris4 draft
     (`data/scratch/paris4_run/prereg_draft_snapshot.md`, sha 208dce44…) before any Paris4 chain existed.
   - 0.263 = the PHerc0139 zero-drift threshold against CT-supported counts (notes/2026-10-22-hitl-budget.md §3), fixed
     2026-10-22 before any Paris4 data existed.
   - Neither number was tuned on any label in this test. The dev evidence below decided only between "use this rule",
     "use it with 0.35" and "stop".

**Dev evidence (labels on development regions only: PHerc0139 bands 1-2, PHerc1667, PHercParis4 z 9000-10500).**

(a) Step diagnostics (`data/scratch/fourth_test/dev_steps.py`, `sweep_raw.py` -> `sweep_raw.json`): step precision,
drift over 10-20 true windings (d10; chain minus label), raised fraction of single steps.

| region | S_pack [95 %] | rule picks | no raise: prec / d10 / within ±1 | gm 0.263: prec / d10 / within ±1 / raised | gm 0.35: prec / d10 |
|---|---|---|---|---|---|
| PHerc0139 b1 (CT `supported`) | 18.5 [18.0, 19.0] | gm 0.263 | 88.8 / −0.95 / 70.2 | 87.7 / −0.16 / 77.4 / 10.4 % | 88.6 / −0.48 |
| PHerc0139 b2 (`minus1`) | 18.5 | gm 0.263 | 88.6 / −1.54 / 58.1 | 86.2 / −0.51 / 67.3 / 14.9 % | 87.8 / −0.88 |
| PHerc1667 (k < 34) | 24.0 [23.5, 25.0] | **off** | 95.8 / +0.95 / 74.7 | 91.0 / +2.02 / 45.1 / 8.3 % | 93.1 / +1.68 |
| PHercParis4 z 9000 (k < 109) | 18.5 [18.0, 19.0] | gm 0.263 | 78.4 / −1.87 / 45.5 | 76.2 / −0.12 / 53.8 / 24.6 % | 78.2 / −0.92 |

- PHerc1667 over-counts at every threshold, so "off" is right there, and the rule leaves it at no raise (the
  `--raise auto` set is byte-identical to the confirmed M2 set).
- At the three tight regions no raise under-counts (d10 −0.95 to −1.87); gm 0.263 brings d10 to −0.12 … −0.51 at a cost
  of 1-2 pp step precision. 0.35 sits halfway.
- Paris4's grad_mag scale is ~1.44× PHerc0139's (median single-step integral 1.89 vs 1.31), yet its raw zero-drift
  threshold (≈ 0.255) matches 0.263, so no normalisation was adopted.
- S_pack separates the regions: 18.5 at all three tight regions vs 24.0 at PHerc1667.

(b) Existing fit evidence (own-offset wrong area, from earlier notes):
- PHerc1667: no raise (M2) PASSED its pre-registered test (−27 to −29 pp); the gm raise (M1) failed.
- PHerc0139 band 2: gm 0.263 (arm26) inner consistent −5.3 [−7.7, −3.0] vs base; no raise ties it (−0.3 [−4.3, +3.3]).
- PHerc0139 band 1: gm 0.263 (arm29) k 23-40 −17.2 [−20.0, −14.3] vs base, the same as no raise (−16.9); gm 0.18
  (arm23) was better inner (−32.5) and worse at k 43-50 (+21.6).
- PHercParis4 z 9000: no raise FAILED (+10.4 … +10.6 vs base, switches +16 … +18), with under-counting chains.

(c) Development fits on the burned Paris4 z 9000-10500 band (`data/scratch/fourth_test/dev_table.py`, the Paris4
endpoint vs the three Paris4 base fits; $ in section 8):

d4_r263 = `--raise auto` on Paris4 z 9000 (S_pack 18.5 -> gm 0.263; windings sha 77949c5f…), seed 1, L40S, scored
against the stored Paris4 dumps (`data/scratch/fourth_test/dev_table.json`):

| d4_r263 − X | primary own-offset wrong area (Paris4 PRIMARY_K) | p | switches / 1000 | global seam |
|---|---|---|---|---|
| p4_base | −0.32 [−2.31, +1.48] (69.2 vs 69.5 %) | 0.67 | **+40.8** [34.2, 46.9] | +1.2 [−0.3, +2.5] |
| p4_bases2 | −0.45 [−2.45, +1.36] | 0.59 | **+39.0** | +1.1 |
| p4_bases3 | −0.38 [−2.38, +1.39] | 0.61 | **+40.1** | +1.1 |
| p4_ms1 (no raise) | **−10.87 [−12.10, −9.66]** | <0.001 | +23.4 | −1.2 [−2.3, +0.0] |
| p4_ms2 (no raise) | −10.96 [−12.20, −9.67] | <0.001 | +22.4 | −1.7 |
| p4_oracle | +24.35 [+22.55, +26.16] | <0.001 | +58.7 | +26.9 |

The raise removes all of no raise's +10.5 pp loss, but the result only **ties** the no-chain base, and the fit has
about 40 more seam switches per 1000 pairs than base. Against the pre-registered PASS criterion (upper CI < 0, switch
≤ +5) it fails all 3 combos, on both conditions.

**Go / no-go (fixed before the dev fits were scored):** go with gm 0.263 if d4_r263 beats all three Paris4 bases on
the primary endpoint (upper CI < 0, switch ≤ +5); else go with 0.35 if d4_r35 does; else stop and report that no rule
was found, without a confirmation run.

## 2. Region and inputs
- **Volume:** PHercParis4 (Scroll 1), volume 20260411134726 (2.400 um). Working frame L2 = 9.6 um, p_L2 = p_2400 / 4.
- **Band:** z 7000-8500 (L2), full slice (stage 0, section 0).
- **m7 mask:** the public bucket `PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr`;
  only the chunk rows covering the band ± 256 were byte-copied (L0 rows 34-46, L1 18-22, L2 9-12, L3 4-6; 2.6 GB) into
  `data/paris4_z7000/masks/m7_L2_th0.2.zarr` (`data/scratch/fourth_test/prep/fetch_z7000.sh`).
- **Lasagna nx / ny / grad_mag, CT occupancy:** the same public lasagna run as Paris4 z 9000, built by
  `spiral/prep_inputs.py all --scroll PHercParis4 --z0 7000 --z1 8500 --margin 256` into
  `data/paris4_z7000/inputs/`. grad_mag is a fitter input for every arm and, if the rule raises, a chain input.
- **Umbilicus:** `data/eval/paris4/umbilicus_paris4_L2.json` (public, / 4), unchanged. **Spiral sense CW**, as Paris4.

## 3. Reference and scorer
- **Labels:** the same organiser set 20260701 (`data/eval/paris4/labels_w0701.npz`) as Paris4 z 9000; same provenance
  caveat (an organiser spiral-fit output, not raw human traces).
- **Scorer:** `eval/spiral_eval.py --scroll paris4 --z0 7000 --z1 8500` (sha 813860d7…, unchanged), 4x upsampled
  reference at stride 2: 12,403,350 vertices, 24,682,157 adjacent pairs, 5.8 % contact. Cache `ref_paris4_z7000-8500_s2_m0_v2.npz`.
- **Oracle ceiling** (merged-mesh oracle, `--min-edge 8`): ray pairs acc_matched 99.96 [99.94, 99.98] %, coverage 99.88 %; vertex wrong area (seam) 0.08 [0.05, 0.12] %
  (contacts 1.40 %, decimation); switches 1.16 / 1000; unmatched 0.05 %; 862 s, peak 6.8 GB
  (`data/scratch/fourth_test/score_oracle_z7000.json`). The scorer is sound on this band.
- tol 4, seam-aware residuals, BootV (30° × 250 z, 1000 resamples, seed 1), `release/fit_vertex_dump.py --scroll paris4`
  (sha 80e8b55c…) for the primary input, single-fit `--min-edge 8` runs for ray pairs: all as Paris4 z 9000.

## 4. Label audit and the primary winding set (from audits only; no fit and no chain metric exists)
### 4.1 Label consistency
`eval/label_subsets.py --scroll paris4z7` -> `subsets_bandparis4z7.json` (sha 864b3984…): **105 of 120
windings consistent**; inconsistent 10, 14, 15, 33, 45, 46, 62, 122-129.
### 4.2 m7 / grad_mag audit
`eval/ct_gap_audit.py extract/count --scroll paris4z7` -> `ct_counts_bandparis4z7.json`; as Paris4 (n_ct = 0;
m7 + grad_mag), gm calibrated to 0.726 per turn on the dk = 1 median.

| span | rays | m7 runs between (expected) | gm turns (expected) |
|---|---|---|---|
| all dk = 1 | 12,377 | 0.10 (0); P(0) 0.91 | 1.00 |
| all dk = 2 | 6,156 | 0.91 (1) | 2.09 |
| all dk = 3 | 3,968 | 1.82 (2) | 3.16 |
| all dk = 5 | 2,320 | 3.66 (4) | 5.35 |
| all dk = 10 | 1,244 | 8.06 (9) | 10.76 |

- m7 undercounts uniformly again (−0.09 / −0.18 / −0.34 / −0.94), the contact-merge signature seen at z 9000.
- grad_mag per-label scale rises outward (dk = 10 spans: 9.6-10.2 for k 30-60, 11.2-12.8 for k 60-110, 16.7 at
  w110-w120).
### 4.3 Primary winding set
The Paris4 rule, unchanged (`data/scratch/fourth_test/primary_kset.py` = `data/scratch/paris4_run/primary_kset.py` with
the z7000 file names): CT-reliable (every dk = 1 pair: mean extra m7 runs ≤ 0.3, median gm turns in [0.6, 1.5]; no
dk = 10 span where m7 and gm both drift > 1 in the same direction) ∩ label-consistent.
- Failing dk = 1 pairs: gm 10-12, 109-112, 114-129 (m7 also 116-124). Drifting span: w110-w120 (m7 +1.34, gm +6.73).
- CT-reliable: 13-108 (96 windings).
- **PRIMARY_K = 13, 16-32, 34-44, 47-61, 63-108: 90 windings** (2,665,374 gen-19 vertices; hard-coded in
  `eval/method_table_p4z7.py` and `controls/common.py`). Output: `data/scratch/fourth_test/primary_kset.out`.
- The singleton 13 is kept because the rule outputs the union of ranges without a minimum length, as at z 9000.

### 4.4 Controls (built from labels / traces only, before this freeze)
As Paris4 addendum A2, band z 7000-8500 (`data/scratch/fourth_test/controls/`):
- **2023 traces:** 20231005123336 is outside the band (z ≥ 10932); the other three start at z ≈ 7350. 185,894 trace
  pairs from 118,538 vertices, 60 blocks, n = 1-5. The reference meets them only at windings 30-43 (inner).
  Floor (labels_w0701): acc_all 63.7 [61.4, 66.5] %, acc_matched 95.4 %, coverage 66.8 %.
- **Generation-19:** 4,419,975 vertices with ID 10-135 in the band (2,665,374 with ID in PRIMARY_K). Floor: own-offset
  primary wrong area 63.1 [61.5, 64.6] %; global agreement 26.8 % (different winding count, as at z 9000; reported
  as is, no claim).

## 5. Arms (env files `spiral/arms/f4_*.env`; shas in section 10)
Shared lines are those of the p4_* arms with `Z0=7000`, `Z1=8500`; the pod gets the z 7000 mask rows and inputs
(`boot_and_launch.sh` takes MASK_DIR next to INP_DIR).

| arm | chains | FIT_OVERRIDES | seed |
|---|---|---|---|
| f4_base, f4_bases2, f4_bases3 | none | none | 1 / 2 / 3 |
| f4_oracle | `band_oracle_p4z7`: auto_windings oracle probes from the labels (`paris4_oracle_chains.py --z 7000 8500`) | chain overrides | 1 |
| f4_ms1, f4_ms2 (method) | `p4z7_auto` (the frozen command below) | chain overrides | 1 / 2 |

- Chain overrides `{"pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}`.
- **Method chain command (frozen).** The crop solve ran label-free before this freeze
  (`autowind chains --raise off … --out data/scratch/fourth_test/cropsolve_p4z7`; its chains were never inspected). The
  method set is made from those crops, read-only:
  ```
  uv run python -m autowind chains --mask data/paris4_z7000/masks/m7_L2_th0.2.zarr \
    --umbilicus data/eval/paris4/umbilicus_paris4_L2.json --z0 7000 --z1 8500 --voxel-um 9.6 \
    --raise auto --grad-mag data/paris4_z7000/inputs/PHercParis4_grad_mag.ome.zarr \
    --crops-ro data/scratch/fourth_test/cropsolve_p4z7/crops --no-solve --work data/scratch/fourth_test/cropsolve_p4z7/work \
    --out data/scratch/fourth_test/windings/p4z7_auto
  ```
  Defaults `--spack-thr 21.0 --auto-gm-thr 0.263`. Whatever S_pack it reports, its decision is the method; there is
  no override.
- **Fit:** villa 0fb38c45 fit_spiral, config/recipe.toml, run_arm.sh, export_fine.sh step 4. No reruns except a
  checkpoint resume with the same env.

## 6. Analysis plan (`eval/method_table_p4z7.py`; all six fits scored before any label-based chain diagnostic)
As Paris4 z 9000 (notes/2026-10-23-third-region-prereg.md §6), with this band, these arms and this PRIMARY_K:
1. **Oracle informativeness first:** oracle − base on the primary endpoint for each base seed; any upper CI ≥ 0 ->
   uninformative, no claim, everything still reported.
2. **Primary endpoint**, 2 method seeds × 3 base seeds = 6 combos: method − base own-offset seam wrong area on
   PRIMARY_K (`subset_wrong_area.sub_scores`, offsets refitted on those vertices), paired BootV 95 % CI, two-sided
   bootstrap p.
3. **Switch margin:** method − base seam switches per 1000 adjacent pairs (all vertices), point estimate ≤ +5.
4. **PASS iff all 6 combos have upper CI < 0 and switch ≤ +5, p_arm = max p < 0.05, and the region is informative.**
   One method arm: no Holm step.
5. **Secondaries** (reported whatever the outcome): global (shared-offset) seam wrong area; ray pairs (acc_matched,
   coverage; all / primary); subsets (consistent / inconsistent / primary / nonprimary, own and shared offsets);
   method − oracle; the generation-19 control (own-offset wrong area on gen-19 IDs in PRIMARY_K) and the 2023-segment
   trace-pair control, built as Paris4 addendum A2 (`data/scratch/fourth_test/controls/`, band z 7000-8500); reference
   floors reported. If a control cannot be built it is reported "not done".
6. **Blinding:** no label-based chain or step metric on this band before all six fits are scored (the oracle chains are
   label-derived by construction). The step diagnostics of section 1 are run on this band only after the method table.

## 7. What was computed on z 7000-8500 before the freeze (disclosed)
- S_mask and label coverage (stage 0); label subsets, m7 / gm audit, primary set (section 4).
- Reference build and oracle ceiling; oracle chains (label-derived).
- The label-free crop solve (`--raise off`) for the method chains; its chain outputs were not opened and no metric was
  computed on them. The `--raise auto` command (section 5) runs after this freeze; its output sha is appended in stage 2.
- Control inputs (gen-19 vertices, 2023 trace pairs): labels / traces only.

## 8. Pods and cost
- **Order:** f4_base (smoke), f4_oracle, f4_bases2, f4_bases3 after stage 1; f4_ms1, f4_ms2 only after the stage-2
  addendum. 1x L40S preferred, A6000 / RTX6000Ada / A100 fallback; every pod logged in
  `data/scratch/p0139_spiral/pods_arms.txt` (names `vss-f4-<arm>`).
- **Cap: $80 total for this test, dev fits included.** Running spend at the stop: **$2.40** (d4_r263 only; section Results).
- Pod rules as Paris4 (never touch others' pods; re-list after failed creates; terminate after pull; score under the
  lock; delete meshes after the vertex, control and pair dumps, keeping f4_base and f4_ms1).

## 9. Disclosures
- **Same scroll as a dev region.** The band is 500 z from the burned Paris4 z 9000 band, same m7 and lasagna runs, same
  label source. This tests transfer to a fresh, tighter band of a scroll whose z 9000 band shaped the go/no-go; it is
  not a new-scroll test.
- **The rule was motivated by Paris4 z 9000's failure** (no raise under-counted there). Its two numbers predate Paris4,
  but the decision to register the rule (rather than drop it again) was made after that failure and after the dev fits.
- **Code changes since the Paris4 freeze**, all outside the frozen files: `tools/autowind/{packing.py (new), core.py,
  cli.py}` (`--raise auto`, `--spack-thr`, `--auto-gm-thr`), byte-identical reproduction of every published set
  re-checked (release/AUTOWIND.md); `eval/label_subsets.py` and `eval/ct_gap_audit.py` gain a backward-compatible
  `--scroll paris4z7` choice (band 7000-8500; m7/gm paths of the z 7000 rows). `tools/auto_windings.py` (8f097312…),
  `eval/p0139_graph.py` (fb7553b1…), spiral_eval, subset_wrong_area, pairs_ksplit and fit_vertex_dump are unchanged.
- The reference is an organiser spiral-fit; m7's training data is undocumented; the band is below the GP label z range.
- The villa shell_outer_winding_idx / gap-expander defaults (130) are unchanged, as Paris4.


## Addenda
### S0 (2026-10-23T18:26:00Z; stage 0 frozen)
Region choice above frozen at 2026-10-23T18:26:00Z. Nothing computed on z 7000-8500 so far except S_mask and label coverage; the m7
rows and lasagna inputs are being fetched (label-free). Next: label audits and oracle (label-derived, as Paris4), then
the label-free crop solve for the chains, all before the stage-1 freeze of the rule.

### D1 (2026-10-23T21:41Z; dev stage, operational)
- Two machine shutdowns (an unplanned restart about 19:30Z, then an orderly shutdown 19:39:50Z-20:33Z) killed the
  local collector and the z 7000 crop solve. Both were restarted. The crop solve resumes from atomic checkpoints, so
  its output is unaffected. The /tmp backups of the four edited files were lost; they were rebuilt byte-exact from the
  edit history into `data/scratch/fourth_test/orig/` (shas a42a9c2e, 46331cf6, 5a63f5a3, baeb21b3 match).
- **Deviation from the lock rule (approved by the coordinator):** the d4_r263 vertex dump (one dev-stage job) ran
  outside `flock /tmp/vss_eval.lock`. After the reboot the machine reported 62 GB RAM with about 40 GB available, and
  the crop solve held the lock until about 23:30Z. A watchdog would have requeued the dump under the lock if
  MemAvailable fell below 8 GB.
- The d4_r35 grabber was stopped at 18:52Z because there were no L40S / A6000 / RTX6000Ada offers. d4_r35 now runs
  only if d4_r263 fails the go criterion.

## Results (2026-10-23T22:10Z): NO-GO at the dev stage. No confirmation was run.
**Decision.** The go criterion (section 1, fixed before d4_r263 was scored) failed. d4_r263 ties all three Paris4 base
fits on the primary endpoint (−0.3 to −0.5 pp, every CI spans 0, p ≥ 0.59), and its switches are +39 to +41 per 1000
against a +5 margin.

**d4_r35 was not run (deviation from the plan's wording, disclosed).** The plan said to try gm 0.35 next. Both
pre-registered conditions rule it out on data already in hand:
- **Primary.** It moves monotonically with raise strength on Paris4 z 9000: no raise +10.5, gm 0.263 −0.4. The step
  diagnostics put 0.35 between the two (d10 −0.92, halfway between −1.87 and −0.12), so its primary estimate is
  expected between 0 and +10.
- **Switches.** Both ends already fail the margin: no raise +16 to +18, gm 0.263 +39 to +41.
- **Same classifier.** At 0.35 the raise is the same weak per-step classifier: raise precision 37.8 % at 0.35 vs
  35.6 % at 0.263 (table below).

It would have cost about $3-9 for a predictable fail. The grabber had also found no L40S / A6000 / RTX6000Ada offer in
30 minutes.

**Post-hoc diagnosis** (dev labels only, `dev_steps.py`). Of the chain steps the raise turns from 1 into 2, the share
that is truly a 2-winding step:

| region | gm 0.263: raised steps (labelled) | raise precision | merged-step recall | gm 0.35: precision / recall |
|---|---|---|---|---|
| Paris4 z 9000 | 3,510 | 35.6 % | 34.9 % | 37.8 % / 19.0 % |
| PHerc0139 b1 | 557 | 34.3 % | 23.9 % | 38.2 % / 15.0 % |
| PHerc0139 b2 | 353 | 33.1 % | 34.5 % | 38.1 % / 22.4 % |
| PHerc1667 | 485 | 7.0 % | 42.5 % | 9.2 % / 33.8 % |

- **The gm raise fixes the mean, not the steps.** At every threshold, about two of every three raises are wrong. So
  the mean drift goes to zero (d10 −1.87 -> −0.12), but whole-chain consistency does not improve: within ±1 over 10-20
  windings only rises from 45 % to 54 %, and step precision falls 78.4 -> 76.2 %.
- **The fitter's switches.** It reads the misplaced +1s as local winding jumps, so switches rise (+23 vs no raise).
  Fixing the mean removes no raise's systematic loss, and that is all; the tie with base follows.
- **The rule's branch logic held.** S_pack 18.5 vs 24.0 separates the tight regions from PHerc1667, and the rule
  leaves PHerc1667 at no raise. The binding limit is the raise itself: no threshold of the frozen logistic gives local
  counts good enough to beat the no-chain base at Paris4 packing.
- **What would be needed.** A better contact detector than the integrated grad_mag logistic: higher raise precision,
  not a different threshold. That is outside this test.

**State of the confirmation band (PHercParis4 z 7000-8500).** It stays fresh for chains, fits and chain metrics:
- No chain set exists. The label-free crop solve was stopped at 3,413 / 6,139 crops (checkpoints only, no
  `relative_windings.json` written), and no fit was run.
- Label-side assets exist: stage 0, subsets, m7 / gm audit, primary set 13,16-32,34-44,47-61,63-108, oracle ceiling,
  oracle chains, controls and floors. They can be reused by a future pre-registration, which must disclose this note.

**Spend and pods.** One pod: vss-f4-d4_r263 (L40S massedcompute, $0.82/h), created 18:23:58Z, terminated 21:19:26Z
right after the pull: 2.92 h, **$2.40 of the $80 cap**. Pods list empty at 22:10Z.

**Deviations.**
- (1) d4_r35 skipped (above).
- (2) d4_r263 vertex dump and dev_table ran outside the eval lock (D1; approved by the coordinator; min MemAvailable
  25 GB).
- (3) Two machine shutdowns. Restarts as in D1; no result affected.
- (4) The crop solve on z 7000 was stopped at the NO-GO.
