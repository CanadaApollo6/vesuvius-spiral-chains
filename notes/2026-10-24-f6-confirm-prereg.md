# 2026-10-24: f6 confirmation, chain cut at tight packing (A4) on PHercParis4 z 7000-8500 (L2): pre-registration

Status: **STAGE 1 (frozen at the timestamp in addendum S1).** Stage 2 (selected recipe, method chain sha) is appended
before any method pod is created. Dev evidence and gates: notes/2026-10-24-f6-bias-screen.md (section 3, R1-R3).
Template and stage 0: notes/2026-10-24-fourth-test-prereg.md (stage 0 frozen 2026-10-23T18:26:00Z, sha in
data/scratch/fourth_test/freeze_stage0.txt). Run directory: `data/scratch/f6_screen/confirm/`.

## 0. Why this test, and the recipe-selection clause
- f6_cut went on the Paris4 z 9000 dev region (primary Δ vs the 3 base seeds −5.41 / −5.54 / −5.47, worst upper CI
  −3.94; switches +3.8 / +2.0 / +3.1). f6_w1 did not (R3). f6_w03 (weight 0.3) is still running at this freeze.
- **The method is fixed by the screen note's frozen selection rule, applied to dev data only:** if f6_w03 does not go,
  the method is **A4 cut** (sections 1, 5). If f6_w03 goes AND has a lower worst-case upper CI than f6_cut (−3.94), or
  ties on it with fewer worst-case switches, the method is **A1 w = 0.3** (section 1). Stage 2 states which branch the rule
  picked, with the f6_w03 numbers. No confirmation number exists when the branch is picked (base and oracle fits are
  not scored before all six fits are in).
- Base and oracle arms do not depend on the branch, so they start after this freeze, as in the fourth test.

## 1. Method (S_pack-branched; label-free)
Step 1 (both branches; frozen command, read-only on the label-free crops resumed after the f6_cut GO):
```
uv run python -m autowind chains --mask data/paris4_z7000/masks/m7_L2_th0.2.zarr \
  --umbilicus data/eval/paris4/umbilicus_paris4_L2.json --z0 7000 --z1 8500 --voxel-um 9.6 \
  --raise auto --grad-mag data/paris4_z7000/inputs/PHercParis4_grad_mag.ome.zarr \
  --crops-ro data/scratch/fourth_test/cropsolve_p4z7/crops --no-solve --work data/scratch/fourth_test/cropsolve_p4z7/work \
  --out data/scratch/f6_screen/confirm/windings/p4z7_auto
```
Defaults `--spack-thr 21.0 --auto-gm-thr 0.263`. Whatever S_pack it reports, its decision is used; no override.

Step 2:
- **A4 cut** (default branch): `cd tools && ../.venv/bin/python -m autowind.postproc --run
  ../data/scratch/f6_screen/confirm/windings/p4z7_auto --mode cut --thr 0.263 --out
  ../data/scratch/f6_screen/confirm/windings/p4z7_f6` (tools/autowind/postproc.py sha bd072174…). S_pack < 21: every
  no-raise single step with gm_p ≥ 0.263 is cut (no shared point; pieces < 2 points dropped; nothing raised).
  S_pack ≥ 21: the run's no-raise set, byte for byte. Fit weight 10, spacing 0.
- **A1 w = 0.3** (only if the clause in section 0 picks it): p4z7_f6 = the run's no-raise set (noraise/ if S_pack < 21,
  else the run's set); fit weight 0.3 iff S_pack < 21, else 10; spacing 0.

## 2-4. Region, reference, primary set, controls
As the fourth test, sections 2-4, unchanged: PHercParis4 vol 20260411134726, L2 9.6 um, z 7000-8500, full slice;
labels_w0701; scorer `eval/spiral_eval.py` sha 813860d7…, `release/fit_vertex_dump.py` sha 80e8b55c…,
`eval/subset_wrong_area.py` sha 8890d16d…; oracle ceiling 0.08 % wrong area (sound); **PRIMARY_K = 13, 16-32, 34-44,
47-61, 63-108** (90 windings); gen-19 and 2023-trace controls in data/scratch/fourth_test/controls/.

## 5. Arms (env files `spiral/arms/f6z_*.env` = the f4_*.env lines with ARM renamed; method arms also with the
f6 windings set)

| arm | chains | FIT_OVERRIDES | seed | env sha |
|---|---|---|---|---|
| f6z_base | none | none | 1 | d010f0b0… |
| f6z_bases2 | none | seed 2 | 2 | d8781fe3… |
| f6z_bases3 | none | seed 3 | 3 | dc2622dc… |
| f6z_oracle | band_oracle_p4z7 (sha fc8a00a5…, copied from data/scratch/fourth_test/windings) | spacing 0, weight 10 | 1 | 96017e35… |
| f6z_ms1 | p4z7_f6 (stage 2 sha) | spacing 0, weight 10 (A4) | 1 | a98ac1e3… |
| f6z_ms2 | p4z7_f6 | spacing 0, weight 10, seed 2 (A4) | 2 | c2e63224… |

If A1 is picked, f6z_ms1/2 change only in the weight (0.3 iff S_pack < 21); their new shas go in stage 2.
Fit: villa 0fb38c45 fit_spiral, config/recipe.toml, run_arm.sh, export_fine.sh step 4. No reruns except a
checkpoint resume with the same env, or a relaunch of a pod that died before fitting.

## 6. Analysis (`data/scratch/f6_screen/confirm/method_table_f6z7.py`, sha 65624a72…: eval/method_table_p4z7.py with
only the arm names, dump dir and pairs json suffix changed)
1. **Oracle informativeness first:** oracle − base, primary endpoint, each base seed; any upper CI ≥ 0 -> uninformative,
   no claim, everything still reported.
2. **Primary:** method − base own-offset seam wrong area on PRIMARY_K, paired BootV (30° × 250 z, 1000 resamples,
   seed 1), 2 method seeds × 3 base seeds = 6 combos; two-sided bootstrap p.
3. **Switch margin:** method − base seam switches per 1000 adjacent pairs (all vertices), point estimate ≤ +5.
4. **PASS iff all 6 combos have upper CI < 0 and switch ≤ +5, p_arm = max p < 0.05, and the region is informative.**
   One method arm, no Holm step.
5. **Secondaries** (reported whatever the outcome): as the fourth test §6.5 (global seam wrong area, ray pairs,
   subsets, method − oracle, gen-19 and 2023-trace controls).
6. **Blinding:** no label-based chain or step metric on z 7000-8500 until all six fits are scored. Label-free counts of
   the method set (S_pack, steps cut, pieces) are recorded in stage 2.

## 7. What exists on z 7000-8500 before this freeze (disclosed)
- Everything listed in the fourth test §7 (stage 0 metrics, label audits, primary set, reference, oracle chains,
  controls), and the first 3,413 crops of the label-free crop solve.
- After the f6_cut GO (2026-10-24T06:03Z): the remaining 2,726 crops (data/scratch/f6_screen/prep/q5_resume2.sh, exit 0
  at 07:19:35Z; 6,139 crops). The solve's own `--raise off` chain file was written and not opened.
- No fit on this band exists. No f6 chain set on this band exists yet.

## 8. Pods and cost
- **Order:** f6z_base, f6z_oracle, f6z_bases2, f6z_bases3 after this freeze; f6z_ms1, f6z_ms2 only after stage 2.
  Names `vss-f6-f6z_*`; L40S / A6000 / RTX6000Ada / A100, ≤ $1.85/h; every pod logged in
  data/scratch/p0139_spiral/pods_arms.txt and terminated right after its pull.
- **Budget:** the f6 cap $70, dev fits included (dev spend at this freeze ≈ $17 projected, f6_w03 included).
  Check in before exceeding it.
- Vertex, control and pair dumps under the eval lock (or > 20 GB available, logged); meshes deleted after the dumps
  except f6z_base and f6z_ms1.

## 9. Disclosures
- **Same scroll, and the same band as the stopped fourth test.** Its stage 0, primary set and controls are reused; the
  fourth test's dev stage stopped there (no fit on this band). The recipe was chosen on Paris4 z 9000 (500 z away,
  same m7 and lasagna runs, same label source). Transfer to a fresh band of the same scroll, not a new-scroll test.
- **Branch.** The cut acts only when S_pack < 21; at wide packing the method is the confirmed M2 recipe, unchanged.
- **Cut sampling.** The fitter samples 13 strips per step whatever their length, so short pieces are visited less
  often per point; recorded at dev (screen note L2), not changed.
- The cut threshold reads gm_p as stored in chains.json (4 decimals); at dev this cut one more step than the raise rule
  would have raised (10,904 vs 10,903; screen note L1).
- **Code since the fourth test:** only `tools/autowind/postproc.py` (new module). tools/auto_windings.py (8f097312…),
  eval/p0139_graph.py (fb7553b1…), packing.py (3ea55390…), spiral_eval, subset_wrong_area and fit_vertex_dump are
  unchanged; every published chain set still reproduces (postproc is a separate entry point).

## Addenda (append-only)
### S1 (2026-10-24T07:32:59Z): stage 1 frozen
sha256 of this file up to and including the "## Addenda (append-only)" line: 6a4442419ea7d3460eaa8f17fa6a8a51e3c9b60db22244bc171a496127763521
(data/scratch/f6_screen/confirm/freeze_stage1.txt).
### S2 (2026-10-24T08:40:52Z): stage 2 frozen: method A4 cut; method chain sha
- **Branch picked by the section 0 clause: A4 cut.** f6_w03 did not go (primary −2.24 / −2.37 / −2.30, upper CI
  −0.20 / −0.33 / −0.34, but switches +17.7 / +15.9 / +17.0 > +5; screen note R4). f6_w1 did not go (R3). Only f6_cut
  went. f6z_ms1/2 envs are unchanged from stage 1 (a98ac1e3…, c2e63224…).
- **Step 1** (section 1 command, run 07:34:31-07:34:45Z under the lock; log data/scratch/f6_screen/confirm/build_step1.log):
  `data/scratch/f6_screen/confirm/windings/p4z7_auto`: **S_pack 17.25** (boot95 17-18; 576 rays, n 56,243; slab
  medians 17.0-17.5) -> decision **gm** (tight). packing.json fe1fa1af…, noraise/chains.json 44b843a2…, chains.json
  89702dea…, relative_windings.json 8371da0b….
- **Step 2** (postproc cut, thr 0.263): `data/scratch/f6_screen/confirm/windings/p4z7_f6`:
  **relative_windings.json sha256 03c014bdf5e28d0e383122f10d98e173c35d46f5795cf6beac14455618319370**, chains.json
  b5d20c1da2056e76b974126fc85f0e83cb3ca9289a07feef1dc2c5f8760380ba. Re-run to a second dir: byte-identical.
  Label-free counts: 52,605 single steps, 8,329 cut (15.8 %), 5,768 pieces, 53,749 points, 3,378 singleton points
  dropped, median piece 6 points. (Paris4 z 9000 dev: 24.6 % of single steps cut, 6,209 pieces.)
- No label-based metric on these chains has been computed. No f6z fit has been scored.
- Pods so far: f6z_base (L40S datacrunch $1.81/h, created 07:51:18Z), f6z_oracle (L40S massedcompute $0.82/h,
  07:57:47Z), f6z_bases2 (A6000 datacrunch $0.79/h, 07:58:26Z); f6z_bases3 waiting for an offer. Method pods
  (f6z_ms1, f6z_ms2) are created only after this addendum.
- Correction to S2 (written 2026-10-24T08:41:10Z): f6z_bases3 was already created at 08:29:55Z (A6000 massedcompute $0.54/h) and launched at 08:36:57Z; S2's "waiting for an offer" was out of date.

## Results (2026-10-24T17:20Z): FAIL on the pre-registered criterion (switch margin). The primary endpoint improved strongly.
All six fits were pulled, dumped and scored before any label-based chain metric on this band was computed.
Table: `data/scratch/f6_screen/confirm/method_table.json` (method_table_f6z7.py, run under the eval lock);
controls: `data/scratch/f6_screen/confirm/control_table.json` (control_table_f6z.py = the fourth test's table with the
f6z arm names).

**1. Oracle informativeness: informative.** Oracle − base, primary: −31.34 [−33.85, −28.75], −31.42 [−33.96, −28.85],
−31.44 [−33.98, −28.87] (base seeds 1/2/3).

**2-4. Combos** (own-offset seam wrong area on PRIMARY_K, %; switch Δ per 1000 seam pairs):

| combo | base | method | primary Δ [95 % BootV] | p | switch Δ [CI] | pass |
|---|---|---|---|---|---|---|
| f6z_ms1 − f6z_base | 72.42 | 64.57 | −7.85 [−9.01, −6.68] | < 0.001 | +6.51 [1.93, 11.39] | no |
| f6z_ms1 − f6z_bases2 | 72.50 | 64.57 | −7.93 [−9.07, −6.78] | < 0.001 | +7.10 [2.95, 11.68] | no |
| f6z_ms1 − f6z_bases3 | 72.52 | 64.57 | −7.95 [−9.08, −6.80] | < 0.001 | +6.72 [2.71, 11.20] | no |
| f6z_ms2 − f6z_base | 72.42 | 64.04 | −8.39 [−9.58, −7.15] | < 0.001 | +7.63 [3.08, 12.14] | no |
| f6z_ms2 − f6z_bases2 | 72.50 | 64.04 | −8.47 [−9.61, −7.26] | < 0.001 | +8.23 [4.03, 12.54] | no |
| f6z_ms2 − f6z_bases3 | 72.52 | 64.04 | −8.48 [−9.62, −7.28] | < 0.001 | +7.85 [3.87, 12.06] | no |

- **Verdict: FAIL.** Every primary CI is far below 0 (p_arm < 0.001), but every switch Δ (+6.5 … +8.2) exceeds the
  pre-registered +5 margin, so no combo passes. No claim of a confirmed improvement is made.
- Effect size vs dev: primary −7.9 … −8.5 here vs −5.4 … −5.5 at Paris4 z 9000; switches +6.5 … +8.2 here vs
  +2.0 … +3.8 at dev. The oracle's own switch Δ is +1.8 … +2.4 (CI spans 0).

**5. Secondaries** (reported whatever the outcome; vs base seed 1 unless stated):
- Global (shared-offset) seam wrong area, all vertices: −6.06 [−7.76, −4.42] (ms1), −6.01 [−7.71, −4.39] (ms2);
  −6.0 … −6.6 over all 6 combos.
- Subsets, ms1 − base: own offsets: consistent −6.74, inconsistent −6.17, primary −7.85, **nonprimary +5.47
  [1.71, 8.71]**. Shared offsets: primary −11.45, inconsistent −1.65 [−4.28, 1.17], **nonprimary +8.66 [3.94, 13.25]**.
  So the cut fits are worse on the non-primary windings (k 10-12, 14, 15, 33, 45, 46, 62 and 109-129: label-inconsistent or
  CT-unreliable).
- Ray pairs, primary: acc_matched 71.3 -> 83.5 (ms1) / 83.4 (ms2); coverage 72.6 -> 71.8 / 70.9. All rays:
  acc_matched 69.0 -> 79.5 / 79.4; coverage 64.1 -> 65.5 / 64.9. Oracle: acc_matched 88.0, coverage 77.2.
- Method − oracle: global +21.73 [19.03, 24.24] (ms1), +21.79 (ms2); own primary +23.48 / +22.95. The cut set closes
  about a quarter of the base-to-oracle gap (7.9-8.5 of 31.4 points).
- 2023 trace pairs (floor 63.7 % acc_all): base 29.6-29.9 %, ms1 48.7 %, ms2 49.5 %, oracle 61.9 %;
  Δ acc_all +18.8 … +19.9 (p < 0.001, all 6 combos).
- Generation-19 own-offset primary wrong area (floor 63.1 %): base 76.4-76.8, ms1 75.2, ms2 74.8, oracle 65.2;
  Δ −1.3 … −2.0 (p ≤ 0.004, all 6 combos).
- Step diagnostics on this band (allowed now): not run.

**Pods and cost.** All six were terminated right after their pulls (pods_arms.txt; the pod list shows 0 pods at
17:05Z). f6z_base $6.96, f6z_oracle $2.49, f6z_bases2 $4.80, f6z_bases3 $3.84, f6z_ms1 $4.89, f6z_ms2 $3.96 =
**$26.94** (hours × listed rate). With the dev fits ($15.19): **$42.13 of the $70 cap.**
