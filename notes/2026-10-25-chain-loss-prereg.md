# 2026-10-25: chain-loss tolerance (test 12): keep the count fix, lose the anchor switches. Dev gate fixed before any fit

## 0. Question and hypotheses
Relative-winding chains fix the winding count (test 6 confirm f6z_ms1/2: primary −7.85…−8.48 pp on Paris4 z 7000), but
they add seam switches: +6.5…+8.2 per 1000, mostly within 32 vox of kept chain points (local rate 237 vs 85 per 1000
at < 16 vox). In villa 0fb38c45 (`spiral/run_arm.sh`, recipe.toml, our weight 10 / spacing 0) a chain enters
the fit through two terms:
- **Strip radius term:** for each sampled row (32 points of one chain), `relu(|normalised radius − row mean| −
  0.025 dr)`, an L1 hinge with weight 10, on from step 0. Normalised radius = unwrapped spiral radius − wind_a·dr.
- **Strip DT term:** each point is pulled to the snapped sheet round(row median) + wind_a·dr in scroll space. Euclidean
  distance, margin 0.025 dr, power 3 within the row and 0.5 across rows, weight 4. It is on only for `iteration >
  loss_start_patch_dt` = 25000 of 30000 steps.

The tests:
- **H1, dead zone:** a 0.025 dr margin pins each point's exact radius (0.4 vox at dr ≈ 16 vox). A wide margin of
  0.3 dr keeps the count, since a point one turn off is still outside the zone, but stops pinning positions that
  the sheet evidence disputes.
- **H2, no late snapping:** the DT term snaps chain points onto the sheet that round(median) picks, after step 25000.
  Late snapping at a decaying learning rate may leave local folds (switches).
- **H3, anneal:** the count is set early. If the chain weight drops late, the regularisers and sheet evidence can
  heal the small corrected islands while the global numbering persists.

## 1. Offline diagnostics ($0; dev band Paris4 z 7000-8500, burned; scripts in `data/scratch/chain_loss/`)
### 1a. Before or after step 25000?
- **No intermediate state was ever pulled.** No run has autosave checkpoints or meshes before the end, and
  satisfaction is logged only at the end. So "does the switch excess exist at step 25000" cannot be answered offline.
  It becomes a pre-registered secondary of the H2 arm (§3d): the 25000 autosave is the exact pre-DT state, because
  DT weight affects nothing before step 25000.
- **Fit logs:** mean loss-term values over steps 23-25k → 25.2-27.2k → 28-30k (track_radius, the main geometry
  term):

  | fit | before DT | just after DT | late |
  |---|---|---|---|
  | f6z_base seeds (no chains) | ~159-163 | flat | flat |
  | f6z_ms1 | 172.0 | 182.4 | 177.1 (unattached DT 16.7 → 12.0) |
  | f6z_oracle | 173.0 | 177.9 | 174.5 |
  | f7_c10z7 | 159.1 | 166.2 | 163.0 |
  | c1667_m2s1 | 71.8 | 76.2 | 73.4 |

  So DT onset visibly fights the sheet evidence wherever chains disagree with it. At z 9000, f6_w1 and f6_w03 (radius
  weight 1 or 0.3, DT still 4) had late DT losses of 183 and 285 and switch Δ of +10…+12 and +16…+18. That fits "late
  snapping of poorly satisfied chains makes switches".
- **Counter-evidence:** f7_c10z7 conflicts less (its pre-DT track term equals base) yet has the same +6.4 switch Δ.
  The timing evidence is therefore mixed.

### 1b. Residuals of chain points under the stored fits
`chain_resid.py` maps every chain point through the fitted transform; `e` is in units of the fit's own dr.

| fit / chain set | points | satisfied | median abs(e_mean) | > 0.3 dr | > 0.5 dr |
|---|---|---|---|---|---|
| f6z_ms1 / p4z7_f6 | 53,749 | 44,337 (fit log 44,362) | 0.034 | 4.4 % | 2.3 % |
| f6z_ms2 / p4z7_f6 | 53,749 | 43,781 | 0.035 | 4.4 % | 2.2 % |
| f7_c10z7 / own | 22,141 | 21,133 | 0.018 | 0.3 % | 0.1 % |
| f6z_oracle / oracle | 68,325 | 66,645 | 0.036 | 0.8 % | 0.2 % |
| f6z_base / p4z7_f6 (no chains) | 53,749 | 18,002 | 0.70 | 74 % | 60 % |

- The chain fits obey their chains almost exactly.
- The chain-free base disagrees with the chains on the count for 55 % of the points. Its in-turn position
  disagreement f_base = abs(e − round(e)) has median 0.18 dr and p90 0.43.

### 1c. Residuals near switches (`resid_vs_switch.py`)
Each matched adjacent reference pair within 16 vox of a chain point is assigned to its nearest point. The excess is
method − mean of the 3 base seeds, per 1000 pairs of the whole band.

- **f6z_ms1:** near excess +4.77 of about +6.8 in total; local rate 237 vs 85.
- **The excess sits on obeyed points.** By r_fit (= abs(e_mean)):
  - < 0.025 dr: +1.53 (local 204)
  - 0.025-0.1 dr: +2.03 (local 234)
  - 0.1-0.2 dr: +0.71 (local 329)
  - ≥ 0.2 dr: +0.5 (local 338-389)

  Poorly obeyed points switch more per pair, but they are few.
- **Position disagreement with the chain-free geometry matters only mildly.** By f_base: local rate 204 at < 0.05 dr
  vs 270 at 0.35-0.5 dr. A rough bound: only about 20 % of the near excess scales with f_base, so H1 alone predicts
  ≲ 1 /1000 gain.
- **Count-changing points carry most of it.** Points where the chain asks for another count than base (c_base):
  other_count +3.57 (local 298 vs 90) vs same_count +1.20 (164 vs 80).
- **f6z_oracle shows the same pattern:** local 168 vs 81, 80 % count-changing points.
- **f7_c10z7:** near excess +1.43, local 177 vs 63.

### 1d. Interpretation, choice and order of arms (fixed now)
- Most near-chain switches are boundaries of small count-corrected islands: rays where the chain moved the count and
  the surrounding sheet did not follow. They are not mainly position-pinning errors.
- No loss change can drop the count corrections themselves without losing the gain. What can help is letting the
  surroundings follow, or not sharpening the boundaries late:
  - **H2:** removes the late snap of island points to their own sheet.
  - **H3:** keeps the early count, then relaxes so that regularisers and track/lasagna evidence can smooth island
    edges.
  - **H1:** removes the residual pinning. The weakest evidence (≲ 1 /1000).
- **Arms** (3, all on the same chains, weight, spacing and seed as f6z_ms1, `VILLA_PATCH=chain_loss_tol.patch`):
  1. **cl_dt0z7, H2:** `loss_weight_unattached_pcl_dt` 0. Nothing else changes, and steps 0-25000 run the same
     configuration as f6z_ms1 (only GPU nondeterminism differs).
  2. **cl_tolz7, H1 + H2:** DT 0, `unattached_pcl_radius_loss_margin` 0.3, `unattached_pcl_radius_center` "median"
     (detached). Reasons:
     - 0.3 rather than 0.35: a point free inside the zone stays at least 0.2 dr from the ±0.5 dr count boundary
       (0.15 dr at 0.35), and inside the fitter's 0.45 dr satisfaction tolerance. Sheet evidence therefore cannot
       drift it into the wrong count before the hinge resists.
     - Median: in a 1-D toy (`tests/toy_count.py`), a 0.3 dead zone around the mean stalls count-fixing when 40-50 % of
       a row is a turn off (0.50-0.59 of runs end correct). A non-detached median stalls the same way, because the
       selected element takes the whole centre gradient. The detached median gives 1.00 at every fraction.
     - The hinge is kept, so the constant L1 force outside the zone still fixes counts. A periodic (count-blind) loss
       was rejected because it drops the count information that makes chains worth having.
  3. **cl_annz7, H3 + H2:** DT 0, chain weight ×1 until step 15000, linear to ×0.1 at 22500, ×0.1 to the end
     (`unattached_pcl_weight_final_scale` 0.1, decay 15000-22500). At step 15000 the learning rate is about 0.55 of its
     start value, so the count has had half the run to settle. The 7500-step ramp and 7500 steps at ×0.1 give the
     regularisers time to act. f6_w1 (weight 1 from step 0) had unfixed counts and late DT; here DT is off and the
     early weight is the full 10.
- H2 is in every arm. The data does not separate H2's late snapping from the main-phase radius term, and DT-on
  versions of H1/H3 would spend the 3-arm budget on a confound. Arm 1 measures H2 alone; arms 2 and 3 measure H1 and
  H3 on top of it.
- **Prediction (recorded honestly):**
  - cl_dt0z7: switch Δ falls by 0-2 /1000 (from +6.5…+7.1), primary about unchanged.
  - cl_tolz7: at most a further 1 /1000.
  - cl_annz7: the most uncertain, between −4 and +2 /1000 in switches, with a risk of losing part of the count gain.
  - Most likely outcome: no arm passes the switch item (≤ +2), because the islands are a count effect.

## 2. Patch (`spiral/patches/chain_loss_tol.patch`, sha256 c0657b76d594276e6644684def45b1bc36d688f0574275efe2b59a2d9c6c31bb)
- **Form:** an extended copy of the existing stack. It is robust_unattached.patch (049ebefe…), which is itself
  bitwise identical to villa at defaults (test 9), plus the new options. Applied alone by run_arm.sh (one
  VILLA_PATCH).
- **Increment:** `data/scratch/chain_loss/patch_incremental.diff`. New config keys, all off by default:
  - `unattached_pcl_radius_loss_margin` (null = patch_radius_loss_margin)
  - `unattached_pcl_radius_center` (mean | median; median = detached)
  - `unattached_pcl_weight_final_scale` (1.0)
  - `unattached_pcl_weight_decay_start` / `unattached_pcl_weight_decay_end` (0 / 0); the schedule code runs only if
    final_scale != 1.
- **Equivalence proof** (`tests/run_equiv.sh` → `tests/equiv_report.json`): on CPU, `tests/loss_equiv_cl.py` runs villa's
  `get_unattached_pcl_strip_losses` on 60 synthetic ray chains (dr 16, noise 0.08 dr, wrong steps, single points a turn
  off), 4 sampled steps, DT off and on. It compares float.hex of the radius and DT losses and of the gradients of dr,
  a scalar and a 3×3 transform parameter, plus a sha of every point's gradient.
  - Bitwise identical to unpatched villa: the robust tree; the cl tree with no keys; all new keys at their defaults
    given explicitly; margin 0.025; schedule final_scale 1.0.
  - Different, as intended: margin 0.3 with mean or with median. Points with gradient go from 232-247 at default to
    81-149 (mean) and 11-77 (median).
  - Schedule values at steps 0 / 14999 / 15000 / 18750 / 22499 / 22500 / 29999: 1 / 1 / 1 / 0.55 / 0.1001 / 0.1 / 0.1.
  - Robust tests (loss_equiv weights1/weights, cut_test) are identical on the cl and robust trees.
  - method_test.py differs between runs on the same tree (its toy fit is not deterministic), so it is a smoke test,
    not part of the proof.
- **H2 needs no code:** with DT weight 0, villa skips the DT target cache and multiplies the DT term by 0.

## 3. GATE (fixed now)
### 3a. Arms and scorer
- Band Paris4 z 7000-8500 (burned dev band), chains `p4z7_f6` (relative_windings.json 03c014bd…, the same file as in
  f6z_ms1's dataset).
- Arm files `spiral/arms/cl_dt0z7.env`, `cl_tolz7.env`, `cl_annz7.env` (= f6z_ms1.env with ARM, FIT_OVERRIDES
  and VILLA_PATCH changed).
- Scorer `data/scratch/chain_loss/dev_table_cl.py`: `data/scratch/f7_switch/dev_table_f7.py` unchanged (cc7786b7…),
  with this gate and vertex dir.
  - Endpoints: own-offset seam wrong area on PRIMARY_K `13,16-32,34-44,47-61,63-108`, paired BootV, switches per 1000
    adjacent pairs, nonprimary own-offset wrong area.
  - Comparison: against each of the 3 stored base seeds f6z_base / f6z_bases2 / f6z_bases3.
- References (same code): f6z_ms1 −7.85 / −7.93 / −7.95 primary, switch +6.51 / +7.10 / +6.72, nonprimary +5.47 /
  +3.77 / +3.79. Near-chain local rate 237.5 (< 16 vox) and 109.8 (16-32) vs base 85.2 / 74.1 (reproduced by
  `switch_diag_cl.py` on 2026-10-25).

### 3b. Dev gate (per arm)
**PASS iff, against EACH of the 3 base seeds:**
1. primary upper 95 % CI < 0;
2. primary estimate ≤ −5.0 pp;
3. switch Δ ≤ +2.0 per 1000 (point estimate);
4. nonprimary own-offset Δ ≤ +3.0.

- **If several pass:** choose the lowest worst-case (max over seeds) switch Δ; on a tie within 0.5, the better
  worst-case primary.
- **Always reported:** primary, switch and nonprimary per seed; arm − f6z_ms1 deltas; near-chain local switch rate at
  < 16 and 16-32 vox from kept chain points (`switch_diag_cl.py` = test 7's `switch_diag.py` unchanged, 0acace32…),
  vs base and vs f6z_ms1.
- **Reading, fixed now:**
  - An arm whose near-chain rate (< 16 vox) is within ±30 of f6z_ms1's 237 did not change the anchor mechanism,
    whatever its band totals.
  - A drop below 180 with the primary kept means the mechanism is loss-driven, and the remaining gap to the gate is
    what a loss change cannot reach.

### 3c. PHerc1667 non-regression (only for the selected dev winner)
- **Arm:** M2 chains `p1667_m7_ray_c` (relative_windings.json 85dcbfb4…, the same file as c1667_m2s1's dataset) on
  PHerc1667 z 6500-8000, with the winner's overrides.
- **Env:** `spiral/arms/cl_{dt0,tol,ann}_1667.env`, written now = c1667_m2s1.env with ARM, FIT_OVERRIDES and
  VILLA_PATCH changed. Only the winner's runs.
- **Scorer:** `data/scratch/chain_loss/nonreg_table_cl.py`, which reuses eval/method_table.py's endpoint code
  unchanged: inner k < 34 own-offset wrong area; switches per 1000.
  - Comparison: against the stored dumps c1667_base / bases2 / bases3 and M2 s1 / s2.
  - Smoke test with M2 s1 as the "new" arm reproduces the stored numbers: −28.16 / −27.60 / −29.32 and switch −63.3 /
    −63.2 / −65.9.
- **PASS iff, against EACH base seed b:**
  1. upper CI < 0;
  2. primary estimate ≤ 0.8 × mean(M2 s1, M2 s2 vs b), i.e. it keeps ≥ 80 % of M2's gain;
  3. switch Δ ≤ mean(M2 s1, M2 s2 vs b) + 2.0.

### 3d. Secondary: timing (no extra fit)
- **Method:** on the cl_dt0z7 pod, `spiral/snap_export.sh` keeps the step-25000 autosave. After the arm
  finishes, it exports that state on the 4-vox grid (optimizer_num_training_steps 25000, so zero steps are run) as
  **cl_dt0k25z7**.
- Up to step 25000 that fit runs the f6z_ms1 configuration, so the snapshot is "f6z_ms1 just before DT" (another GPU
  realisation of it). It is scored by the same scorer, but it is not an arm and not counted as a fit.
- **Reading, fixed now:**
  - Snapshot switch Δ ≥ +4.6 against every base seed (≥ 70 % of f6z_ms1's) → the excess forms before DT, in the
    main phase.
  - ≤ +2.0 against every seed → it forms after step 25000.
  - Otherwise: both phases.
  - cl_dt0z7 final − snapshot = what steps 25000-30000 do without DT.

### 3e. Cap, order, no re-tuning
- **At most 4 scored fits:** the 3 dev arms, run in parallel, plus 1 PHerc1667 non-regression.
- A pod that fails before the fit writes meshes may be relaunched once (the spend counts).
- No setting is changed after any fit is scored: no other margin, schedule or weight. If no arm passes, the test
  ends at dev.

## 4. If dev and non-regression pass: confirmation (separate two-stage pre-registration, not run under this note)
- **Region:** fresh Paris4 z 10800-11800. Prepared under notes/2026-10-24-f7-confirm-prereg.md stage 0 (frozen
  18:32:29Z): m7 rows, lasagna inputs, oracle chains band_oracle_p4z11, crop solve cropsolve_p4z11, f7z_{base,bases2,
  bases3,oracle}.env. No fit and no label metric has been computed there.
- **Stage 1** freezes inputs, scorer and primary set; **stage 2** freezes the blind-built method chain set (test-6 cut
  recipe, f6_cut) by sha before any fit.
- **Arms:** base ×3, oracle, method ×2 seeds with the winning setting.
- **Pass:** every one of the 6 method×base combos has upper CI < 0 and switch Δ ≤ +5.
- **Primary set = CT-supported windings,** by two label-free-of-fits tests applied together:
  1. **Local CT turn counts** (data/scratch/f7_switch/primary_kset.py on ct_counts_bandparis4z11.json, built in f7
     stage 0). Winding k is CT-reliable iff every dk = 1 pair touching it has m7 extra runs ≤ 0.3 and gm turns in
     [0.6, 1.5], with no drifting dk = 10 span. It must also be label-consistent. Result: 21-47, 49-60, 63-99,
     111-115, 119-122.
  2. **The audit's outer-winding finding** (notes/2026-10-24-paris4-label-audit.md). Rays crossing the labels show
     k > 110 unreliable in every audited band: 43.5 % of decided cells flagged at z 7000, 37.6 % at z 3000-7000,
     33.5 % at z 11848-18000. Sheets are skipped at k 111-124 and labels sit in air at k 124-129. Local turn counts
     miss this, because a skipped sheet keeps dk = 1 pairs looking normal. The cut is k ≤ 110. The audit did not
     touch z 10752-11848; the cut is taken from the bands on both sides and is the same in every band. Inside
     k ≤ 110 the flagged share stays ≤ 11 % (k 80-99 7 %, k 100-109 11 % in z 11848-18000), and k 100-110 already
     fail test 1.
  - **Planned primary set:** 21-47, 49-60, 63-99 (75 windings). Stage 1 freezes it, with the script and the shas of
    its inputs.
- **No label-based chain metric** on z 10800-11800 until all 6 fits are scored.

## 5. Budget and rules
- **Spend cap $70 for the whole test,** pods `vss-cl-*`. Expected: dev 3 × ~$3.5 (A6000 $0.54/h, ~6.5 h including
  setup), non-regression ~$3.5, confirmation 6 × ~$3.5; at most $1.0/h offers.
- **Pod handling:**
  - REST only, via `data/scratch/chain_loss/pods/prime_rest.py`. Its delete refuses any pod not named `vss-cl-*`.
  - Re-list after any failed or timed-out create.
  - One `watchdog.sh` per pod with a deadline (launch + 9 h).
  - `collect.sh` terminates each pod right after its pull. Log: data/scratch/p0139_spiral/pods_arms.txt.
  - Never touch sm86-speed, kv4-*, flint-*, or any pod not created here.
- **Frozen files untouched:** tools/auto_windings.py, eval/p0139_graph.py, tools/autowind/postproc.py,
  tools/check_viewer*, and every file whose sha is frozen in a prereg note. The villa checkout is not modified
  (patches only).
- **Memory:** vertex dumps run under `flock /tmp/vss_eval.lock`. Runs without it only with MemAvailable > 20 GB,
  logged in `data/scratch/chain_loss/eval_lock_exceptions.log`.
- No commits, pushes or uploads.

---

## Log (append-only)
### G0 (2026-10-25T16:44:45Z): gate frozen
- sha256 of this file up to and including the "## Log (append-only)" line:
  d1ef2418232a9bc6738ab23603c149074c176895cc9b4463511498988768f579. Snapshot:
  `data/scratch/chain_loss/freeze/gate_note_snapshot.md`. All shas: `data/scratch/chain_loss/freeze/gate_freeze.sha256`.
- **Patch and envs:** chain_loss_tol.patch c0657b76…; cl_dt0z7.env 61fdaaa5…; cl_tolz7.env c0767c06…; cl_annz7.env
  44c94d3f…; cl_dt0_1667.env 76058051…; cl_tol_1667.env 9ee3b8c6…; cl_ann_1667.env 2ed69d09…; snap_export.sh
  fd650c96….
- **Scorers:** dev_table_cl.py ffded9b7…, switch_diag_cl.py 675afa22…, nonreg_table_cl.py 9e4f958a…. Unchanged:
  dev_table_f7.py cc7786b7…, switch_diag.py 0acace32…, method_table.py 6cfc15df….
- **Chains:** p4z7_f6 03c014bd…; p1667_m7_ray_c 85dcbfb4….
- **Frozen tools unchanged:** auto_windings.py 8f097312…, p0139_graph.py fb7553b1…, postproc.py af573101….
- **State at freeze:** no vss-cl pod exists, and no fit with any test-12 setting exists.
### L1 (2026-10-25T16:47Z): dev pods created
- cl_dt0z7 d31fc64e…, cl_tolz7 eed68d10…, cl_annz7 60faf2ad…: all vss-cl-*, 1×A6000 48 GB massedcompute at $0.54/h,
  created 16:45:22Z, 16:46:04Z and 16:46:48Z by `data/scratch/chain_loss/pods/run_arm.sh`.
- Watchdogs delete each pod at 2026-10-26T01:50Z if collect.sh has not.
- collect.sh runs with N_ARMS=3 and SNAP_ARMS=cl_dt0z7.
- 18:12Z: watchdog deadlines moved to 2026-10-26T05:30Z (fit ETAs 6.5-8 h at 18:11Z).
### L2 (2026-10-25T17:46-17:56Z): fits started
- Patch applied on every pod: `applied villa patch chain_loss_tol.patch (c0657b76d594276e)`.
- overrides.json on each pod matches its env (DT 0; margin 0.3 with median; schedule 0.1 over 15000-22500).
- The step-25000 autosave of cl_dt0z7 was kept at 22:31:36Z (completed_iterations 25000). Its export resumed "at
  iteration 25000" and ran zero optimisation steps.

### L3 (2026-10-25T23:40Z - 2026-10-26T00:37Z): pulled, pods terminated, dumped
- Fit times: dt0 20,819 s, tol 20,938 s, ann 23,798 s; 120 spliced windings each.
- Terminated by collect.sh: d31fc64e… and eed68d10… at 23:40:30Z / 23:40:32Z, 60faf2ad… at 00:37:15Z. The watchdogs
  then exited and no vss-cl pod remains.
- **Spend:** 6.92 + 6.91 + 7.84 = 21.67 pod-h at $0.54/h = **$11.70**.
- Final chain-point satisfaction (fit logs; f6z_ms1 44,362 = 82.5 %):
  - dt0: 36,461 / 53,749 (67.8 %)
  - tol: 35,243 (65.6 %)
  - ann: 34,795 (64.7 %)

  Late DT is what puts the last 15 % of chain points on their sheet.
- Vertex dumps ran under flock in collect.sh. Scoring ran with the eval lock held by the dump, logged in
  `eval_lock_exceptions.log` (MemAvailable ≥ 50 GB). The final table ran under flock.

### L4 (2026-10-26T01:0xZ): dev table (`dev_table_cl_z7.json` / `.log`) and switch decomposition (`switch_diag_z7000_cl.json`)
Δ against f6z_base / f6z_bases2 / f6z_bases3; primary = own-offset seam wrong area on PRIMARY_K (pp); switches per
1000 adjacent pairs; near-chain = local switch rate per 1000 matched pairs at < 16 / 16-32 vox from kept chain points
(base 85.2 / 74.1).

| arm | primary Δ [95 % CI] | switch Δ | nonprimary Δ | near-chain < 16 / 16-32 | gate |
|---|---|---|---|---|---|
| f6z_ms1 (ref) | −7.85 [−9.01, −6.68] / −7.93 / −7.95 | +6.51 / +7.10 / +6.72 | +5.47 / +3.77 / +3.79 | 237.5 / 109.8 | fail |
| **cl_dt0z7** (H2) | −7.89 [−9.07, −6.71] / −7.97 / −7.99 | +4.85 / +5.45 / +5.07 | +5.38 / +3.68 / +3.70 | 223.9 / 103.5 | **FAIL** (switch, nonprimary) |
| **cl_tolz7** (H1+H2) | −9.84 [−11.81, −8.01] / −9.92 / −9.94 | +7.21 / +7.81 / +7.43 | +8.25 / +6.55 / +6.57 | 218.4 / 105.0 | **FAIL** (switch, nonprimary) |
| **cl_annz7** (H3+H2) | −7.85 [−9.02, −6.66] / −7.93 / −7.95 | +2.23 / +2.82 / +2.44 | +5.28 / +3.58 / +3.60 | 186.7 / 93.3 | **FAIL** (switch by 0.23-0.82, nonprimary) |
| cl_dt0k25z7 (step-25000 snapshot) | −7.91 / −7.99 / −8.01 | +4.13 / +4.72 / +4.34 | +5.42 / +3.72 / +3.74 | 224.2 / 104.0 | (not an arm) |
| f6z_oracle (ref) | −31.34 / −31.42 / −31.44 | +1.83 / +2.43 / +2.05 | −2.59 / −4.29 / −4.27 | | |

Arm − f6z_ms1:
- dt0: primary −0.04, switches −1.65.
- tol: primary −1.99, switches +0.70, nonprimary +2.8.
- ann: primary 0.00, switches −4.28 (−63 % of ms1's excess), nonprimary −0.19.

Pre-registered readings:
- **Near-chain rate (§3b):**
  - dt0 (224) and tol (218) are within ±30 of 237, so the anchor mechanism did not change.
  - ann (187) is a 51-point drop. That is outside ±30, but above the 180 mark for "loss-driven"; between the
    pre-registered bands.
  - Its switch reduction is spread over all distances: < 16 vox excess 4.69 → 2.98, 16-32 vox 3.78 → 2.02.
- **Timing (§3d):** the snapshot's switch Δ is +4.13 / +4.72 / +4.34. Not all are ≥ +4.6 and none is ≤ +2.0, so the
  reading is **both phases**.
  - About two thirds of f6z_ms1's excess (+6.5…+7.1) already exists at step 25000 (main phase, radius term only, DT
    off), with the full primary gain.
  - Steps 25000-30000 without DT add +0.7 (dt0 final − snapshot).
  - With DT they add about +2.4 (f6z_ms1 − snapshot; the pre-25000 states are separate GPU realisations of the same
    configuration).
  - The snapshot's near-chain rate (224) equals dt0's final, so the islands near anchors form before DT. DT sharpens
    them further (237).

## VERDICT (dev): FAIL, no arm passes. Test 12 ends at dev
- **Per §3e:** no non-regression fit and no confirmation on z 10800-11800. Nothing touched that region. 3 of the 4
  allowed fits were used; spend $11.70 of $70.
- **Primary is not the problem:** every arm keeps the count gain (upper CI < 0, estimate ≤ −5 against all seeds).
- **Two items fail:**
  1. Switches. Only cl_annz7 gets close, at +2.2…+2.8 against a limit of +2.0.
  2. Nonprimary, for every arm including f6z_ms1. The label audit attributes most of that penalty to label errors
     (about 0 on trusted vertices), so it is mostly outside a loss change's reach.
- **What was learned:**
  - **H2, late DT snapping,** accounts for about 1.6-2.4 /1000 of the switch excess. Removing DT alone keeps the
    primary.
  - **H1, dead zone 0.3 dr with median centre,** is harmful. It strengthens the count fix (primary −9.9), with more
    switches and a worse nonprimary. A free in-turn position lets points drift with the sheet evidence, and the
    count fix then shows up as more and bigger islands. This matches §1c: position pinning was not the cause.
  - **H3, anneal,** is the only lever that works on the islands: −63 % of the switch excess at an identical primary.
    The diagnosis holds: the count is set in the first half, and relaxing the chain weight late lets the surroundings
    heal the island boundaries.
- **Not a pass and not re-tuned under this note:** a stronger or earlier anneal is the obvious next test (for example
  final scale 0.03, or decay 12500-20000). It needs its own pre-registration on this burned band, because §3e forbids
  re-tuning here.

### Pointer (2026-10-26T01:18Z)
- The follow-up anneal test (test 13) is pre-registered separately in notes/2026-10-26-anneal-prereg.md (G0
  2026-10-26T01:17:12Z). Nothing above this line was changed.
