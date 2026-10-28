# 2026-10-26: annealed chain weight (test 13): push the test-12 anneal further. Dev gate fixed before any fit

Fixed before any test-13 fit. Results and deviations go below the "Log" line, timestamped. Test 12
(notes/2026-10-25-chain-loss-prereg.md) stays frozen; its log only gets a pointer to this note.

## 0. Question
Test 12's cl_annz7 (chain weight ×1 until step 15000, linear to ×0.1 at 22500, DT 0) kept the count fix (primary
−7.85 / −7.93 / −7.95, the same as f6z_ms1). It cut switches to +2.23 / +2.82 / +2.44 per 1000 (f6z_ms1 +6.51 / +7.10 /
+6.72; cl_dt0z7, the same run without the anneal, +4.85 / +5.45 / +5.07). That removed 63 % of the switch excess left
by DT 0, but it missed gate item 3 (≤ +2.0) against every seed.

The question is whether an anneal that is lower, earlier, or both still keeps the count and gets switches to ≤ +2.0
against every base seed.

## 1. What test 12 showed, and the choice of arms
- **Timing.** The step-25000 snapshot of cl_dt0z7 already had switches of +4.13 / +4.72 / +4.34, about 2/3 of
  f6z_ms1's excess. Most switches therefore form while the chain weight is at full strength, before 25000. An anneal
  that starts before the bulk of them form should do more than one that only ends lower.
- **Final level.** Under cl_annz7, the window-averaged raw chain hinge (weighted loss / schedule / 10) relaxed while
  the weight fell and kept rising at ×0.1: 0.885 (12.5-15k), 0.59 (17.5-20k), then 0.75, 0.88, 0.95, 0.99 by
  27.5-30k. The primary was kept. At ×0.1 the chains are therefore close to unenforced late in the fit, yet the
  count holds. The remaining ×0.1 pull is ~1/3 of the full-weight residual force, so going to ×0.03 should cut what
  is left of that pull by ~70 %. Its effect on switches should be smaller than the effect of timing.
- **Count risk.** A low weight from step 0 loses the count (test 6 on z 9000: f6_w1 primary +1.5, f6_w03 −2.3). The
  count therefore needs early full weight, but how long that weight is needed is not known. No snapshot before
  25000 exists. In cl_dt0z7 the hinge was still 0.95 at 12.5-15k, so the count may not be fully settled by 12500.
  An earlier start is the riskier change. The learning rate at 12500 is still 0.605× its initial value, against 0.548×
  at 15000.
- **Arms (2; DT 0, chains p4z7_f6, all other settings = cl_annz7):**
  - **an_f03z7:** ×1 until 15000, linear to **×0.03** at 22500. This is cl_annz7's timing with a lower final level, so
    an_f03z7 − cl_annz7 isolates the final scale. It is the count-safe arm.
  - **an_e03z7:** ×1 until **12500**, linear to **×0.03** at **20000**. This is earlier and lower, so an_e03z7 −
    an_f03z7 isolates the timing. It is the arm that is likelier to reach the switch target, at some risk to the count.
  - Not chosen: a start at 10000 or earlier, because of the count risk above and the lack of any data before 12500.
    Also not chosen: a final scale of 0, because the count may need a residual hold, and an arm that is 0 after
    22500 cannot tell those apart.
- **Predictions (honest, written before any fit; per seed f6z_base / f6z_bases2 / f6z_bases3):**
  - **an_f03z7:** primary about −7.8 (−7.5 to −8.1) per seed. Switches about +1.9 / +2.5 / +2.1 (each ±0.4).
    S1 nonprimary about −2.2 (passes item 4). Most likely outcome: it fails item 3 on f6z_bases2. P(pass) ≈ 10 %.
  - **an_e03z7:** primary about −7.5 (−6.0 to −8.0; a partial count loss would show here first). Switches about
    +1.6 / +2.2 / +1.8 (each ±0.5). S1 nonprimary about −2. P(pass) ≈ 25 %.
  - P(at least one arm passes) ≈ 30 %.
  - **The binding item is switches against f6z_bases2.** The oracle chains themselves give +2.43 there, so a pass
    needs fewer switches than the oracle-chain fit. The prediction is that the dev gate most likely fails again, but
    by less than 0.5.
  - **Near-chain local rate (< 16 vox):** cl_annz7 was 186.7 (f6z_ms1 237.5, base 85.2). Prediction: an_f03z7 about
    175, an_e03z7 about 160.
  - **PHerc1667 non-regression, if reached:** the M2 chains there give much larger gains (about −28 pp), so the
    80 % rule allows a loss of about 5.6 pp. Prediction: an_f03 keeps ≥ 90 % of the gain. an_e03 carries more risk
    (P(fail) ≈ 25 %) because PHerc1667's count may settle later.

## 2. Code and files (no new patch)
- **Patch:** `spiral/patches/chain_loss_tol.patch` (test 12, sha256
  c0657b76d594276e6644684def45b1bc36d688f0574275efe2b59a2d9c6c31bb), unchanged.
  - Options used: `unattached_pcl_weight_final_scale`, `unattached_pcl_weight_decay_start`, `_end`.
  - Schedule: 1 before the start, linear to the final scale at the end, the final scale after it.
  - Bitwise identical to villa 0fb38c45 at defaults.
- **Envs:** `spiral/arms/an_f03z7.env`, `an_e03z7.env` (= cl_annz7.env with ARM and the schedule in
  FIT_OVERRIDES changed). `an_f03_1667.env` and `an_e03_1667.env` (= cl_ann_1667.env, likewise).
- **Pod tooling:** `data/scratch/anneal/pods/`, a copy of `data/scratch/chain_loss/pods/` with names vss-an-*.
  `prime_rest.py delete` refuses any other name. Vertex dumps go to `data/scratch/anneal/vertex/`.

<!-- GATE-BEGIN -->
## 3. GATE (fixed now)
### 3a. Arms and scorer
- **Band and chains:** Paris4 z 7000-8500 (burned dev band), chains `p4z7_f6`, the same as cl_annz7 and f6z_ms1.
- **Scorer:** `data/scratch/anneal/dev_table_an.py`.
  - **Primary and switches on the ORIGINAL labels** (gate items 1-3):
    - code: `data/scratch/f7_switch/dev_table_f7.py`, unchanged (cc7786b7…);
    - primary: own-offset seam wrong area on PRIMARY_K `13,16-32,34-44,47-61,63-108`, paired BootV;
    - switches: per 1000 adjacent pairs, all vertices;
    - comparison: against each of the 3 stored base seeds, f6z_base / f6z_bases2 / f6z_bases3.
  - **Gate item 4 on the label audit's S1 trusted vertices:**
    - code: `eval/paris4_label_audit.py` `cmd_rescore` + `vertex_trust`, unchanged (76081274…), through a runtime
      `TESTS["t13"]` entry (band z7000, z 7000-8500, same PRIMARY_K, same base dumps);
    - nonprimary = ¬PRIMARY_K ∩ trusted, own offsets refitted on that mask;
    - audit `data/scratch/label_audit/paris4/z7000/audit.json`.
  - **Original-label nonprimary:** always reported next to item 4; not gated.
- **Corrections files frozen by sha** (G0 lists the values):
  - z7000/audit.json (used by item 4);
  - z7000/label_corrections.json and .csv;
  - data/scratch/label_audit/paris4_label_corrections.json (190a061b…).
- **Smoke test (2026-10-26, before any fit)** reproduces both sources:
  - **f6z_ms1:** S1 nonprimary −1.69 / −1.80 / −1.85 (= the t6c rescore). Original labels: primary −7.85 / −7.93 /
    −7.95, switch +6.51 / +7.10 / +6.72, nonprimary +5.47 / +3.77 / +3.79 (= test 12).
  - **cl_annz7:** S1 nonprimary −2.17 / −2.28 / −2.33 (original +5.28 / +3.58 / +3.60).
  - **f6z_oracle:** S1 nonprimary −7.40 / −7.51 / −7.56, switch +1.83 / +2.43 / +2.05.

### 3b. Dev gate (per arm an_f03z7, an_e03z7)
**PASS iff, against EACH of the 3 base seeds:**
1. primary upper 95 % CI < 0 (original labels);
2. primary estimate ≤ −5.0 pp (original labels);
3. switch Δ ≤ +2.0 per 1000 (point estimate, original labels);
4. nonprimary own-offset Δ ≤ +3.0 on S1 trusted vertices.

- **cl_annz7 is scored under the new item 4 as evidence only.** It cannot pass retroactively. Its smoke result fails
  item 3 on all 3 seeds (+2.23 / +2.82 / +2.44) and passes item 4.
- **If both arms pass:** pick the lower worst-case (max over seeds) switch Δ. On a tie within 0.3, pick an_f03z7, the
  smaller change.
- **Always reported per seed:**
  - primary;
  - switch;
  - nonprimary on S1 and on the original labels;
  - S1 primary and S1 switch (secondary);
  - arm − cl_annz7 deltas;
  - near-chain local switch rate at < 16 and 16-32 vox from kept chain points (`data/scratch/anneal/switch_diag_an.py`
    = test 7's `switch_diag.py` unchanged, 0acace32…).

### 3c. PHerc1667 non-regression (only for the selected dev winner)
- **Arm:** M2 chains `p1667_m7_ray_c` on PHerc1667 z 6500-8000 with the winner's schedule
  (`an_f03_1667.env` or `an_e03_1667.env`).
- **Scorer:** test 12's frozen `data/scratch/chain_loss/nonreg_table_cl.py` (9e4f958a…), unchanged, run with
  `--vd data/scratch/anneal/vertex`.
  - Code: eval/method_table.py endpoints; inner k < 34 own-offset wrong area; switches per 1000.
  - Comparison: against c1667_base / bases2 / bases3 and M2 s1 / s2.
- **Env pattern:** as in test 12: SCROLL=PHerc1667, MASK_DIR=data/p1667/masks,
  WR=data/scratch/confirm_region/p1667/windings, inputs data/p1667/inputs.
- **PASS iff, against EACH base seed b:**
  1. upper CI < 0;
  2. primary estimate ≤ 0.8 × mean(M2 s1, M2 s2 vs b), i.e. it keeps ≥ 80 % of M2's gain;
  3. switch Δ ≤ mean(M2 s1, M2 s2 vs b) + 2.0.

### 3d. Confirmation (only if dev and non-regression pass): fresh Paris4 z 10800-11800, two stages
- **Prepared inputs** (f7 stage 0, notes/2026-10-24-f7-confirm-prereg.md): m7 rows, lasagna inputs, oracle chains
  band_oracle_p4z11, f7z_{base,bases2,bases3,oracle}.env. No fit and no label metric has yet been computed in this band.
- **Arms (6 fits):** base ×3 (seeds 1-3, f7z_base / bases2 / bases3), oracle, and the method with the winning
  schedule × 2 seeds. The method chains are built blind with the test-6 cut recipe (f6_cut, as p4z7_f6).
- **Stage 1** (before any method chain exists in that band; frozen in a stage-1 log entry here with shas). It
  freezes the inputs, the scorer and the primary set.
  1. **The audit method, applied to that band as a label-only check.**
     - A new wrapper, `data/scratch/anneal/conf_audit.py`, imports `eval/winding_audit.py` unchanged (51211362…). It
       registers a runtime preset `PHercParis4_conf` = a copy of the `PHercParis4` preset with `forbidden_z` cleared.
     - The guard is lifted for this label-only audit only, on the coordinator's instruction. The frozen file is not
       touched.
     - It then runs `extract --z0 10800 --z1 11800`, `analyse --anchor 30 --outward-ccw` and `report --anchor 30`
       into `data/scratch/label_audit/paris4/conf_z10800/`, with the same steps and settings as band_pipeline.sh.
     - The audit reads only the organiser labels (labels_w0701.npz), the CT, the m7 mask and grad-mag. It uses no
       fit and no chains.
  2. **Primary set = CT-supported windings by the audit method:** vertices whose label k is in the planned k set,
     **and** that are S1-trusted under `vertex_trust`.
     - Planned k set: 21-47, 49-60, 63-99, from test 12 §4: local CT turn counts via primary_kset.py (a2e0d3fd…), cut
       at k ≤ 110 by the audit's outer-winding finding.
     - `vertex_trust` = every label pair touching the vertex's k in its (500-z slab, 30° sector) cell is combined-ok
       with conf ≥ 0.8. It is applied to that band's own audit.json.
     - This is the same rule as dev item 4 (PRIMARY_K ∩ trusted).
     - Stage 1 freezes the band's audit.json sha, the trusted-vertex mask (npz, sha), the kept share, and the
       confirmation scorer (a `TESTS` entry for z 10800-11800 run through `cmd_rescore`, in a new file).
  3. **No label-based chain metric** in that band until all 6 fits are scored. Stage 1 reports only the audit's own
     label-vs-CT summary and the trusted share.
- **Stage 2:** builds the method chain set blind and freezes it by sha before any fit. Then the 6 fits run.
- **Confirmation PASS iff every one of the 6 method×base combos has:**
  - primary upper 95 % CI < 0, on the stage-1 primary set (planned k ∩ S1 trusted);
  - switch Δ ≤ +5 per 1000 (original labels, all vertices, as in the test-3/6 confirmations).
  - Also reported (not gated): primary on original labels within the planned k set, S1 switches and S1 nonprimary,
    and the oracle check.

### 3e. Cap, order, no re-tuning
- **At most 2 dev fits** (both run in parallel), **1 non-regression fit**, and the 6-fit confirmation set.
- A pod that fails before writing meshes may be relaunched once (the spend counts).
- **No re-tuning:** no setting is changed after any fit is scored. If no arm passes, test 13 ends at dev.
<!-- GATE-END -->

## 4. Budget and rules
- **Spend cap: $60 for the whole test.**
  - Expected: dev 2 × ~$4 (A6000 $0.54/h, about 7-8 pod-h including setup), non-regression ~$4, confirmation
    6 × ~$4. Worst case ≈ $36.
  - Offers above $1.0/h are not used.
- **Pods:** `vss-an-*`, created and deleted via REST through `data/scratch/anneal/pods/prime_rest.py` only.
  - Re-list after any failed or timed-out create.
  - Each pod gets one `watchdog.sh` with a deadline of launch + 11 h.
  - `collect.sh` terminates each pod right after its pull. Log: data/scratch/p0139_spiral/pods_arms.txt.
  - Never touch sm86-speed, kv4-*, flint-*, nb-gen3, or any other pod not created here.
  - Never overwrite a running script on a pod.
- **Frozen files stay untouched:** tools/auto_windings.py, eval/p0139_graph.py, tools/autowind/postproc.py,
  tools/check_viewer*, and every file whose sha is frozen in a prereg note (including eval/winding_audit.py and
  eval/paris4_label_audit.py). The villa checkout is not modified (patches only).
- **Memory:**
  - Vertex dumps and scorers run under `flock /tmp/vss_eval.lock`.
  - Runs without the lock are allowed only with MemAvailable > 20 GB, logged in `data/scratch/anneal/mem_log.txt`.
  - No local GPU is used.
- No commits, pushes, posts or uploads. Derived labels stay local.

---

## Log (append-only)
### G0 (2026-10-26T01:17:12Z): gate frozen
- **sha256 of this file** up to and including the "## Log (append-only)" line:
  ed7ca284ac560a92fe863dc05465f8d7fdfe82b0d66b0096c1830f215cc2e2fb.
  - Gate section (GATE-BEGIN..GATE-END): 3a99858bfdc3f53279f70cb1fb9ba9afddf55eb9c1743ef4c999e0b65ff69512.
  - Snapshots and the full sha list: `data/scratch/anneal/freeze/` (gate_note_snapshot.md, gate_section_snapshot.md,
    gate_freeze.sha256).
- **Patch and envs:** chain_loss_tol.patch c0657b76…; an_f03z7.env 6405e13d…; an_e03z7.env 8caf5bdf…;
  an_f03_1667.env 5aa7d4d2…; an_e03_1667.env 946a800e…; template cl_annz7.env 44c94d3f….
- **Scorers:** dev_table_an.py f2c785a6…, switch_diag_an.py ff3568ba…, nonreg_table_cl.py 9e4f958a… (frozen in
  test 12, used with `--vd`).
  - Unchanged: dev_table_f7.py cc7786b7…, switch_diag.py 0acace32…, method_table.py 6cfc15df…,
    paris4_label_audit.py 76081274…, winding_audit.py 51211362…, spiral_eval.py 813860d7…,
    subset_wrong_area.py 8890d16d….
- **Corrections / audit files:** z7000/audit.json 5195362d…; z7000/label_corrections.json ff8221b5…; .csv 8d020bbc…;
  paris4_label_corrections.json 190a061b….
- **Confirmation primary inputs:** primary_kset.py a2e0d3fd…; ct_counts_bandparis4z11.json e4c965f5…;
  subsets_bandparis4z11.json d5ef8cef….
- **Chains:** p4z7_f6 03c014bd…; p1667_m7_ray_c 85dcbfb4….
- **Frozen tools unchanged:** auto_windings.py 8f097312…, p0139_graph.py fb7553b1…, postproc.py af573101….
- **State at freeze:** no vss-an pod exists (the only pod listed is nb-gen3, which is not ours), and no fit with
  either test-13 schedule exists.

### L1 (2026-10-26T01:17-01:26Z): dev pods created and launched
- vss-an-an_f03z7 <pod-id> and vss-an-an_e03z7 <pod-id>, both 1×A6000 massedcompute at $0.54/h. Created at
  01:17:46Z / 01:17:51Z; pod chains launched at 01:23:21Z / 01:25:30Z.
- Watchdogs: deadline 2026-10-26T12:20:00Z. `collect.sh` running with N_ARMS=2.

### L2 (2026-10-26T02:12:40Z / 02:21:38Z): fits started
- **an_f03z7:** fit started 02:12:40Z.
- **an_e03z7:** fit started 02:21:38Z.
- Both pods applied chain_loss_tol.patch (c0657b76…). The pod-side overrides show DT 0 and the frozen schedules
  (0.03 at 15000-22500 and 0.03 at 12500-20000).

### L3 (2026-10-26T07:19Z / 08:30Z): pulled, pods terminated, dumped
- **an_f03z7:** pulled 07:19:04Z (240 windings), pod <pod-id> terminated at once, dump done by 07:34Z.
- **an_e03z7:** pulled 08:30:15Z (240 windings), pod <pod-id> terminated, dump done 08:59Z.
- Fit wall time: about 5.0 h (f03) and 5.9 h (e03).

### L4 (2026-10-26T09:2xZ): dev table (`data/scratch/anneal/dev_table_an_z7.json` / `.log`; `_orig` and `_audit` parts) and switch decomposition (`switch_diag_z7000_an.json` / `.log`)
Per seed f6z_base / f6z_bases2 / f6z_bases3.

**an_f03z7: GATE PASS**
- Primary: −6.94 [−8.07, −5.76] / −7.02 / −7.04.
- Switch: +0.88 / +1.48 / +1.10.
- Nonprimary S1: +1.04 / +0.93 / +0.88. Original labels: +9.34 / +7.64 / +7.66.

**an_e03z7: GATE PASS**
- Primary: −7.79 [−8.97, −6.61] / −7.87 / −7.88.
- Switch: −0.07 / +0.53 / +0.14.
- Nonprimary S1: −2.34 / −2.46 / −2.50. Original labels: +4.99 / +3.29 / +3.31.

**cl_annz7 (evidence only, new item 4):**
- Switch: +2.23 / +2.82 / +2.44, which fails item 3.
- Nonprimary S1: −2.17 / −2.28 / −2.33 (passes item 4). Original labels: +5.28 / +3.58 / +3.60.
- It would not pass.

**Near-chain local switch rate (< 16 / 16-32 vox from a kept chain point):**

| arm | < 16 | 16-32 |
|---|---|---|
| an_f03z7 | 153.5 | 85.2 |
| an_e03z7 | 151.9 | 85.7 |
| cl_annz7 | 186.7 | 93.3 |
| f6z_ms1 | 237.5 | 109.8 |
| base | 85.2 | 74.1 |

- Band totals relative to the mean of the bases: +1.15 (f03), +0.20 (e03), +2.50 (cl_ann), +6.78 (ms1).

**Selection (§3b):** both pass. The lower worst-case switch Δ wins: an_e03z7 at +0.53, against +1.48 for an_f03z7.
an_e03z7 goes to the PHerc1667 non-regression.

**Against the predictions:**
- Switches came out much lower than predicted (predicted e03 +1.6 / +2.2 / +1.8, f03 +1.9 / +2.5 / +2.1).
- e03 kept the count fully. The count risk was on the other arm: f03 lost about 0.9 pp of primary.
- f03's original-label nonprimary rose to +9.3 / +7.6 / +7.7 (cl_annz7 +5.3). On S1-trusted vertices it is +1.0.
  This is a large label-dependent difference, and it is reported here. Under the old item 4 (original labels,
  ≤ +3.0), f03 would fail on every seed, and e03 would fail narrowly on bases2 and bases3 (+3.29 / +3.31) and on base
  (+4.99).
- The pass therefore depends on the item-4 change that the coordinator ordered, which was frozen in G0 before any fit.

### L5 (2026-10-26T09:33Z - 13:58Z): PHerc1667 non-regression pod for an_e03_1667
- No A6000, L40S or RTX6000Ada offer at ≤ $1.0/h existed from 09:33Z to 13:57Z. The only offers were an A10 at $1.29
  and an A100-40GB at $1.99, so no pod was created in that window.
- The grab was restarted once at 13:34Z, after its 240-min window ended.
- vss-an-an_e03_1667 <pod-id> (1×A6000 massedcompute at $0.54/h) was created 13:58:04Z. Watchdog deadline
  2026-10-27T00:58Z; `collect.sh` running with N_ARMS=3.
- an_e03_1667 fit started 2026-10-26T14:09:25Z (chain_loss_tol.patch applied; pod overrides show DT 0 and 0.03 at
  12500-20000).

### L6 (2026-10-26T19:3xZ): PHerc1667 non-regression: PASS (`data/scratch/anneal/nonreg_an_e03.json` / `.log`)
- an_e03_1667 pulled 19:34:35Z (192 windings). Pod <pod-id> was terminated at once.
- Scored with the frozen nonreg_table_cl.py using `--vd data/scratch/anneal/vertex`. Per base c1667_base / bases2 /
  bases3:

| measure | c1667_base | bases2 | bases3 |
|---|---|---|---|
| primary | −28.47 [−29.72, −27.26] | −27.92 [−29.24, −26.58] | −29.64 [−30.82, −28.47] |
| primary threshold | −22.46 | −22.01 | −23.39 |
| M2 mean primary | −28.07 | −27.52 | −29.24 |
| switch Δ | −69.53 | −69.48 | −72.09 |
| switch threshold | −61.09 | −61.04 | −63.65 |
| M2 mean switch | −63.09 | −63.04 | −65.65 |

- All items pass. an_e03 keeps about 101 % of M2's gain, with about 6.4 fewer switches per 1000 than M2.
- Next: confirmation stage 1, §3d.

### S1 (2026-10-26T19:51:01Z): confirmation stage 1 frozen (before any method chain exists in z 10800-11800)
- **Label-only audit of the fresh band**
  - Code: `data/scratch/anneal/conf_audit.py` (11e90582…) wraps eval/winding_audit.py, unchanged (51211362…).
    PHercParis4 `forbidden_z` is cleared in memory only. The preset is edited, not copied under a new name, because
    the audit code keys the Paris4 handling on the preset name. The frozen file is untouched.
  - Commands: extract `--scroll PHercParis4 --z0 10800 --z1 11800` (21 levels 10824-11784 × 72 angles, 1512 rays;
    CT streamed); analyse `--anchor 30 --outward-ccw`; report `--anchor 30`. All exit 0.
  - Output: `data/scratch/label_audit/paris4/conf_z10800/`; audit.json cd86a5b4…, rays.npz 5872669a….
  - Inputs: labels, CT, m7, grad_mag. No fit, no chains.
- **Label-vs-CT summary** (report.log):
  - 65.3 % of ray-pair weight confident. Of the confident weight: ok 98.5 %, duplicate 0.3 %, untraced 1.2 %.
  - Validation: 225 label-clean controls, held-out m7 mean 1.137 (P(1) 0.782), gm median 1.009, false flags 10.7 %.
  - Split half: 206 / 207 ok reproduced.
- **Primary set:** planned k 21-47, 49-60, 63-99 ∩ S1-trusted vertices. Computed by `data/scratch/anneal/conf_trust.py`
  (b2235f16…) → `freeze/conf_trust_z10800.npz` (2b2e91e5…) and `.json` (46e3b283…).
  - 8,201,759 reference vertices. 4,513,492 have k in the planned set; 2,734,660 of those are trusted, so 60.6 % of
    the primary k vertices are kept (dev z7000: 66.1 %).
  - Trusted share: 40.4 % of all vertices, 40.2 % of adjacent pairs.
  - Within the set, k 84-99 have trusted shares of 0.14-0.38. They stay in by the frozen rule; this is reported only.
- **Scorer:** `data/scratch/anneal/conf_table_an.py` (0027a8d3…). It runs cmd_rescore (unchanged) with
  `TESTS["t13c"]` (band conf_z10800).
  - Bases: f7z_base / bases2 / bases3. Oracle: f7z_oracle. Method: an_e03_c11s1 / an_e03_c11s2.
  - PASS iff all 6 combos have S1 primary upper CI < 0 and original-label switch Δ ≤ +5.
- **Envs:**
  - f7z_base 6478ca16…, f7z_bases2 0c8860af…, f7z_bases3 5c072012…, f7z_oracle 045402ed… (f7 stage 0; the oracle
    chains band_oracle_p4z11 are ba26d8a4…).
  - Method: an_e03_c11s1.env 082d6af0… and an_e03_c11s2.env c913bdad…. Each is f7z_base.env with ARM, the chain set
    p4z11_f6 (built in stage 2), the an_e03z7 overrides (seed 2 for s2) and VILLA_PATCH=chain_loss_tol.patch.
- Full sha list: `data/scratch/anneal/freeze/stage1_freeze.sha256`.
- No label-based chain or fit metric has been computed on this band. No method chain set exists: no p4z11_auto or
  p4z11_f6 anywhere under data/.

### S2 (2026-10-26T19:53Z): confirmation stage 2 frozen: method chain sha (built blind, before any fit)
- **Recipe check:** the current tools reproduce the test-6 cut set byte for byte. postproc.py (af573101…) cut on
  p4z7_auto gives relative_windings 03c014bd… (= p4z7_f6), in `data/scratch/anneal/repro/p4z7_f6_now`.
- **Build script:** `data/scratch/anneal/build_p4z11.sh`. It runs the f6 confirmation step 1 and step 2 commands
  with the band (10800-11800), the inputs (data/paris4_z10800) and the label-free crop solve
  (data/scratch/f7_switch/cropsolve_p4z11) swapped in. Run 19:51:51-19:52:01Z, both steps exit 0.
- **Step 1** (`windings/p4z11_auto`): S_pack 18.0 vox (95 % [17.5, 19.0]) < 21, so the decision is gm (tight).
  relative_windings 40bc6a23…, noraise/chains.json 29af621a…, chains.json 8b1c1be0…, packing.json 249aaa14….
- **Step 2** (cut, thr 0.263; `data/scratch/anneal/windings/p4z11_f6`):
  - **relative_windings.json sha256 c3dd4a305b4158fbadef22cc40018372756a9cf08a784579437f1f98ce9780f0**; chains.json
    e5d0eb92….
  - A re-run to a second dir is byte-identical.
  - Label-free counts: 30,569 single steps, 6,243 cut (20.4 %), 3,791 pieces, 30,327 points, 2,906 singleton points
    dropped, median piece 5 points.
- No label-based metric on these chains, and no fit, exists on this band.
- **Next:** the 6 fits: f7z_base, f7z_bases2, f7z_bases3, f7z_oracle, an_e03_c11s1, an_e03_c11s2.

### L7 (2026-10-26T19:53Z - ): confirmation pods waiting for offers
- Six launchers (`data/scratch/anneal/pods/launch_conf.sh`) have been polling since 19:53Z. Each runs up to 4 grab
  rounds of 240 min, and starts a watchdog (deadline creation + 11 h) as soon as its pod exists.
- No GPU offer at ≤ $1.0/h has appeared since then. Up to 23:5xZ the only GPU offered was an H200 at $5.40/h.
- `collect.sh` is running with N_ARMS=9. No vss-an pod exists.

### L8 (2026-10-27T02:1xZ): interim state, reported to the coordinator. The confirmation runs unattended
- Still no GPU offer at ≤ $1.0/h after 6 h of polling. At 02:00Z the only GPU offered was an H200 at $5.40/h.
- **Still running in the background:**
  - The 6 `launch_conf.sh` launchers (grab rounds until about 2026-10-27T11:53Z).
  - Watchdogs, one per pod, started automatically at creation + 11 h.
  - `collect.sh` (N_ARMS=9), which pulls each pod, terminates it and dumps it.
  - `data/scratch/anneal/auto_conf_score.sh` (48bd4b54…), which runs the frozen conf_table_an.py once all 6 dumps
    exist and writes `data/scratch/anneal/conf_table_an.json` / `.log`.
- **Spend so far: $10.17** (18.84 pod-h × $0.54). The 6 confirmation fits are expected to add about $24.

### D1 (2026-10-27T01:59:45Z): hardware-only deviation for the confirmation (coordinator decision), logged before any confirmation pod exists
- **Reason:** no A6000, L40S or RTX6000Ada offer at ≤ $1.0/h has appeared for 6 h.
- **New rule:** the 6 confirmation pods may use any single GPU that meets all of these:
  - ≥ 48 GB VRAM;
  - x86_64 host with NVIDIA driver ≥ 580;
  - not spot;
  - ≤ $2.50/h;
  - ≥ 6 vCPU and ≥ 40 GB RAM.
- **Excluded:** GH200/GB200 (aarch64), provider lambdalabs (old drivers), and H200 at $5.40/h (the user declined
  premium pricing; the $2.50/h cap excludes it anyway). This is the same rule the whole-scroll run uses.
- **Host check:** after boot, each host is checked for x86_64 and driver ≥ 580. A failing pod is deleted, and its
  (provider, gpuType) pair is added to data/scratch/anneal/pods/bad_offers.txt, seeded from the whole-scroll list.
- **Budget, still $60 in total:** a pod is created only if
  spend so far ($10.17) + Σ(price × 8 h) over the confirmation pods already created + this price × 8 h
  + (pods still to create after this one) × $0.54 × 8 h ≤ $58, keeping $2 margin.
  With A6000-class pods that is about $43 in total.
- **Unchanged:** recipe, envs, chains (p4z11_f6 c3dd4a30…), scorer (conf_table_an.py 0027a8d3…), primary set and
  pass rule. The dev and non-regression fits all ran on A6000.
- **Reporting:** each pod's GPU type is recorded. Any method × base combo whose two fits ran on different GPU types is
  flagged as mixed-GPU.
- **Process change:** the old launchers (launch_conf.sh / run_arm.sh / grab_rest.py, one set per arm) were stopped by
  PID. The pods were re-listed afterwards: no vss-an pod exists.
  - The replacement is one process: `data/scratch/anneal/pods/grab_conf.py`. It creates the pods, then for each pod
    runs `boot_conf.sh` (host check, then the unchanged boot_and_launch.sh) and starts a watchdog at creation + 11 h.
  - The whole-scroll run (vss-full-*) is told to yield offers until all 6 confirmation pods exist.

### L9 (2026-10-27T02:02:19Z): D1 launchers started
- `grab_conf.py` (f16b4f5d…) and `boot_conf.sh` (c11585de…) are running. PID file: `data/scratch/anneal/pods/grab_conf.pid`; log: `grab_conf.log`. Created pods are listed in `conf_pods.tsv`, with GPU type and price.
- `collect.sh` and `auto_conf_score.sh` are both still alive (checked via their PID files). No pods exist.
- At start, the only GPU offer on Prime was H200 at $5.40/h, which is excluded.

### D2 (2026-10-27T03:37:13Z): all 6 confirmation fits on the local RTX 3090 (coordinator decision; hardware and scheduling only), logged before any confirmation fit starts
- **Reason:** Prime has had no offer that qualifies under D1 since 2026-10-26 16:45Z. The only GPU offer has been H200 at $5.40/h, which is excluded.
- **Probe (02:57-03:33Z; not a confirmation fit, and no label metric was computed):** f7z_base on the same local stack with probe-only overrides: STEPS=3500 and loss_start_track_dt 1500 (frozen value 10000), so that the DT phase came early.
  - Peak VRAM: fit process 11,200 MiB, flat through the DT phase and the step-2500 checkpoint. The step-4 export peaked at 11,482 MiB. The whole GPU peaked at 13,984 MiB, including about 1.9 GB of desktop apps.
  - Speed: 3.53 steps/s before DT and 2.90 steps/s during DT, so a full fit takes about 2.7 h.
- **Hardware:** every one of the 6 fits runs on the same GPU (local RTX 3090, driver 615.78.08), so no mixed-GPU comparison is possible.
  - The fits run one at a time, in the order f7z_base, an_e03_c11s1, f7z_bases2, an_e03_c11s2, f7z_bases3, f7z_oracle.
  - Each fit runs under GPU lease ~/.gpu-leases/vss-an-conf.json (max_mem_gb 21).
  - nvidia-smi is logged once per second. A fit is stopped if its process reaches 19,000 MiB.
- **Environment, identical to the pods:** data/scratch/anneal/local_probe/setup_local.sh is the FIT_ONLY steps of env/pod_setup.sh, without apt.
  - villa 0fb38c45 plus the pherc343 fit_spiral_shell_optional.patch; Python 3.14; torch cu130 through villa's lasagna bootstrap; the same spiral dependencies; vcnative built with env/build_vc_native.sh.
  - Two villa trees mirror the pods, where each pod carries a single arm:
    - ext/villa stays without chain_loss_tol.patch, for f7z_base, bases2, bases3 and oracle;
    - ext/villa_cl gets the patch (c0657b76…) applied by run_arm.sh through the env's VILLA_PATCH, for an_e03_c11s1 and s2.
- **Recipe unchanged, no probe overrides:** each fit runs the frozen env file itself (spiral/arms/<arm>.env), checked against the frozen sha before it starts. It goes through the unchanged spiral/run_arm.sh (0310894f…) and export_fine.sh 4 (779d504b…), with HOME set to data/scratch/anneal/local_probe/home so that the env's $HOME/data paths resolve locally.
  - After each fit, its overrides.json is checked against recipe.toml plus the env's FIT_OVERRIDES. In particular, optimizer_num_training_steps must be 30000 and loss_start_track_dt must be absent (default 10000).
  - The inputs are local copies of data/paris4_z10800/{inputs,masks}, the same files the pods would receive.
  - The track and SDT caches for the z 10800 band, built by the probe with the same code and inputs, are shared by all 6 arms, as arms sharing a pod would share them.
  - The chain sets are p4z11_f6 (c3dd4a30…) and band_oracle_p4z11 (ba26d8a4…).
- **Outputs, in the pod format:** a local copy of spiral/pull_arm.sh's file selection writes each arm's outputs to data/scratch/p0139_spiral/<arm>/ and creates pulled.ok. Next comes the same release/fit_vertex_dump.py command collect.sh runs (under flock /tmp/vss_eval.lock), writing to data/scratch/anneal/vertex/<arm>.npz. Then auto_conf_score.sh runs the frozen conf_table_an.py (0027a8d3…) once all 6 dumps exist.
- **Unchanged:** scorer, primary set, pass rule, the $60 cap (local compute adds $0), and no label metric on this band before all 6 are scored.

### L10 (2026-10-27T03:37-03:39Z): D2 started
- **03:37Z:** grab_conf.py (PID 733780) and collect.sh (PID 1566774) were stopped by PID. They had no work left, since no pods would be created.
  - The pods were re-listed: no vss-an pod exists, and none was ever created under D1. Spend stays at $10.17.
- **03:37:29Z:** created data/scratch/anneal/pods/CONF_DONE. Before that I checked the name: only the whole-scroll grabber (grab_coord3.py) reads it, with the meaning "test 13 no longer needs pod offers". The whole-scroll run can now resume creating its own pods.
- auto_conf_score.sh is still waiting for the 6 dumps.
- **03:38:57Z:** data/scratch/anneal/local_probe/local_conf.sh started. Its PID is in conf/local_conf.pid and its log is conf/local_conf.log.
  - It first checked the frozen code and chain shas, and the f7z_base env sha.
  - Fit 1 of 6 is f7z_base, running from the band's cached tracks and SDT.
  - Each arm's nvidia-smi log is written to conf/smi_<arm>.csv.

### L11 (2026-10-27T06:37Z): fit 1/6 f7z_base done
- **Fit:** 30000 steps in 10228 s on the local RTX 3090. Run exit 0 at 06:29:53Z. Peak process VRAM 13,026 MiB, below the 19,000 MiB guard.
- **Checks passed:** overrides (recipe plus band only), pull (240 step-4 windings) and dump (vertex/f7z_base.npz, 06:37:09Z).
- **Arm 2:** an_e03_c11s1 started at 06:37:09Z on ext/villa_cl. chain_loss_tol.patch c0657b76… was applied by run_arm.sh, and 3791 chains were loaded. ext/villa was rechecked and is still clean.
- **Disclosure:** while checking that the dump had finished, I printed the last line of vertex/f7z_base.log.
  - That line is release/fit_vertex_dump.py's standard summary. It includes wrong_area_seam 75.7 and wrong_area_global 83.7 for the base fit; collect.sh writes the same line on the pod path.
  - It is a base-arm figure, not a method-chain metric, and it feeds no decision.
  - From now on I check dumps only by file existence and size, without reading the logs, until conf_table_an.py has scored all 6.

### L12 (2026-10-27T09:34Z): fit 2/6 an_e03_c11s1 done
- **Fit:** run exit 0 at 09:27:25Z. Peak process VRAM 12,808 MiB.
- **Checks passed:** overrides (recipe, band and the env's FIT_OVERRIDES), pull (240 windings) and dump (vertex/an_e03_c11s1.npz, 09:34:27Z). Only file existence was checked.
- **Arm 3:** f7z_bases2 started at 09:34:27Z on ext/villa, which is still clean of chain_loss_tol.patch.

### C1 (2026-10-27T12:03:53Z): correction to the tree checks in D2, L11 and L12, plus a by-hand verification
- **The bug, found by the whole-scroll agent:** local_conf.sh checks that ext/villa is clean with `git -C $V apply --reverse --check spiral/patches/chain_loss_tol.patch`. Under `git -C` that relative path does not resolve, and git exits 128 with "can't open patch". So the check could never detect the patch.
  - My by-hand checks recorded in L11 and L12 ("ext/villa still clean") had the same relative-path problem, so they verified nothing either. Those two statements were unverified when I wrote them.
  - The running script is not changed. From now on each tree is checked by hand with absolute paths.
- **Verified with absolute paths at this time** (while f7z_bases2, fit 3/6, was running on ext/villa):
  - **ext/villa:** HEAD 0fb38c45. chain_loss_tol.patch c0657b76… is not applied: its reverse check fails and its forward check applies cleanly. The pherc343 shell-optional patch is applied, and `git diff` contains only that change (fit_spiral.py, the same 2 changed lines as the patch).
  - **ext/villa_cl:** HEAD 0fb38c45 with shell-optional plus chain_loss_tol.patch applied (its reverse check succeeds). config.py, fit_spiral.py, losses.py and point_collection.py are modified.
- **Fits already done, all on the correct tree:**
  - Only one log in any arm's timings.txt records applying chain_loss_tol.patch: an_e03_c11s1's, at 06:37:09Z. run_arm.sh applies the patch only to $VILLA, which local_conf.sh sets to ext/villa_cl for env files with VILLA_PATCH.
  - ext/villa still shows the patch absent, so f7z_base (fit 1) and f7z_bases2 (fit 3, running) used the unpatched tree. an_e03_c11s1 (fit 2) used the patched ext/villa_cl.
- **Remaining fits:** before each one starts (an_e03_c11s2 on ext/villa_cl; f7z_bases3 and f7z_oracle on ext/villa), I repeat this absolute-path check during the preceding arm's dump window and log it here.

### L13 (2026-10-27T12:22Z): fit 3/6 f7z_bases2 done; tree check before fit 4
- **Fit:** f7z_bases2 exited 0 at 12:22:29Z. Peak process VRAM 12,906 MiB.
- **Checks passed:** overrides (recipe, band and seed 2) and pull (240 windings). The dump was running.
- **Tree check at 12:22:42Z** (absolute paths, during the dump, before an_e03_c11s2 started):
  - ext/villa: 0fb38c45, chain_loss_tol absent, 1 modified file (shell-optional).
  - ext/villa_cl: 0fb38c45, chain_loss_tol APPLIED, 4 modified files.
- an_e03_c11s2 runs next, on ext/villa_cl.

### L14 (2026-10-27T15:20Z): fit 4/6 an_e03_c11s2 done; tree check before fit 5
- **Fit:** an_e03_c11s2 ran on ext/villa_cl. run_arm.sh logged "villa patch chain_loss_tol.patch already applied (c0657b76…)". Exit 0 at 15:20:03Z. Peak process VRAM 12,202 MiB.
- **Checks passed:** overrides (recipe, band, the env's FIT_OVERRIDES and seed 2) and pull (240 windings). The dump was running.
- **Tree check at 15:20:09Z** (absolute paths, before f7z_bases3 started):
  - ext/villa: 0fb38c45, chain_loss_tol absent, 1 modified file.
  - ext/villa_cl: chain_loss_tol APPLIED, 4 modified files.
- f7z_bases3 runs next, on ext/villa.

### D3 (2026-10-27T16:49Z): paused for a machine update (requested by the user)
- **Why:** the user needed to update and reboot the machine.
- **Stop sequence:** the pod grabber grab_coord4.py was stopped at 16:42Z; Prime listed 0 pods. The test-13 agent was stopped.
  - After f7z_bases3 wrote its step-15,000 autosave (checkpoint_autosave.ckpt, 12:48 local / 16:48Z, 497,752,203 bytes), I sent SIGTERM to local_conf.sh, auto_conf_score.sh, local_ws.sh, watchdog2.sh and collect.sh. local_conf.sh's trap ended the fit's process group and logged "stopped by signal" at 16:49:08Z. No lease files remain.
- **State:** dumps exist for 4 of 6 arms. f7z_bases3 is at about step 15,081 of 30,000 and will resume from its step-15,000 autosave. f7z_oracle has not started.
- **Effect on the result:** the resumed fit may not match an uninterrupted run bit for bit, because the RNG and data-loader state may not be restored exactly. This is the same resume path run_arm.sh uses on pods. The recipe, seed and scoring are unchanged. No label metric was viewed.
- **Restart:** run local_conf.sh (resumable: arms with dumps are skipped) and auto_conf_score.sh. Then run whole-scroll's local_ws.sh and, optionally, grab_coord4.py, watchdog2.sh and collect.sh.
- **Restart (2026-10-27T17:33:44Z, after the reboot):**
  - The GPU driver is unchanged (615.78.08); torch 2.13.0+cu130 sees CUDA. The sha prefixes of grab_coord4.py and local_ws.sh match the frozen ones.
  - local_conf.sh skipped the 4 dumped arms. f7z_bases3 resumed from checkpoint_autosave.ckpt with 15,000 steps remaining. The overrides are unchanged (seed 3).
  - auto_conf_score.sh, local_ws.sh, collect.sh, watchdog2.sh and grab_coord4.py were relaunched.
  - **Tree check at 17:36Z** (absolute paths, during f7z_bases3):
    - ext/villa: 0fb38c45, chain_loss_tol absent, 1 file changed (shell-optional).
    - ext/villa_cl: 0fb38c45, chain_loss_tol APPLIED, 4 files changed.

### L15 (2026-10-27T19:25Z): fit 5/6 f7z_bases3 done (resumed after D3); tree check at fit 6 start
- **Fit:** f7z_bases3 exited 0 at 19:18:14Z. Peak process VRAM 12,448 MiB.
- **Checks passed:** overrides (seed 3) and pull (240 windings). Dumped at 19:25:40Z.
- **Fit 6:** f7z_oracle started at 19:25:40Z on ext/villa.
- **Tree check at 19:26:31Z** (absolute paths): ext/villa is 0fb38c45, chain_loss_tol absent, 1 file changed (shell-optional). The poll caught fit 6 about 50 s after it started. Nothing writes to ext/villa between fits, and it was also clean at 17:36Z.

### D4 (2026-10-27T23:33Z): second interruption (session restart), relaunched as systemd user units
- **What happened:** at 21:56:55Z the Claude session's systemd scope (t3code-claude-…scope) was stopped. Every process launched from it went down with it: setsid does not leave the cgroup. local_conf.sh logged "stopped by signal". There was no reboot.
- **Effect:** f7z_oracle (fit 6) lost about 5 min of work. It resumes from its step-22,500 autosave, written at 21:51:53Z, with 7,500 steps remaining. Recipe, seed and scoring are unchanged. No label metric was viewed.
- **Relaunch at 23:33:03Z** as independent systemd user units, so a session restart can't kill them again: vss-an-conf, vss-an-score, vss-ws-local, vss-ws-collect, vss-ws-watchdog and vss-ws-grab.
- **Tree check at 23:34Z:** ext/villa: 0fb38c45, chain_loss_tol absent, 1 file changed, 1 insertion(+), 1 deletion(-).

### L16 (2026-10-28T01:04Z): confirmation scored. PASS
- **Inputs:** f7z_oracle exited 0, was pulled (240 windings) and was dumped at 00:58:23Z. auto_conf_score.sh ran conf_table_an.py, which exited 0 at 01:04:34Z. Outputs: data/scratch/anneal/conf_table_an.json and conf_table_an.log.
- **Frozen criterion:** all 6 method × base pairs have S1 primary upper CI < 0 and switch Δ ≤ +5 → **PASS**.
- **Effect:**
  - S1 primary: −2.02 to −3.90 pp. The worst upper CI is −0.32 (c11s1 vs f7z_base). Switches range from −3.03 to +3.83.
  - Original-label primary: −1.74 to −3.35.
  - This is about a third of the dev effect (−7.8). The oracle shows −34 to −37 pp of headroom, so the method captures roughly 6–11% of it on this fresh region.
- **Reported, not gating (honest caveats):**
  - Nonprimary on S1-trusted labels: +0.27 to +2.20. Two seed-1 CIs only just exclude 0 (lower bounds +0.07 and +0.01).
  - Nonprimary on original labels: −3.3 to −4.0, an improvement. That is the opposite sign pattern to dev, where original labels got worse and S1 got better.
  - On the S2 audited subset (29% of primary vertices): primary is −2.1 to −4.5, but nonprimary is +3.2 to +4.9 (CIs exclude 0) and switches are +4.2 to +6.5. Net: the nonprimary effect depends on the labels and is mildly negative on the best-audited labels.
- **Process:** interruptions D3 (user reboot) and D4 (session-scope kill) both resumed from autosaves. Tree checks are logged in C1, L13–L15, D3 and D4. No label metric was viewed before scoring.
