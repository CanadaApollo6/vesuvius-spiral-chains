# 2026-10-24: f7, switch-aware cut recipe: switch diagnosis, variants and the dev gate (fixed before any dev fit)

**Gates frozen: see addendum G0 (UTC timestamp and sha of sections 0-5).** Written after the switch diagnosis (section 1;
labels on the two burned Paris4 bands only) and before any f7 dev fit. Sections 0-5 do not change after G0; the log and
results below them are append-only and timestamped. Run directory `data/scratch/f7_switch/`; pods `vss-f7-*`.
Confirmation region: notes/2026-10-24-f7-confirm-prereg.md (stage 0 frozen 2026-10-24T18:32:29Z: PHercParis4
z 10800-11800, S_mask 19.0).

## 0. Starting point
Test 6 (`cut`: no raise, hard cuts at gm_p ≥ 0.263 when S_pack < 21) failed its confirmation on the switch margin:
primary −7.85 … −8.48 (all CIs < 0) but switches +6.5 … +8.2 per 1000 (> +5). Its dev band (Paris4 z 9000) showed only
+2.0 … +3.8, and the cut fits were worse on non-primary windings (+5.5 own / +8.7 shared at z 7000).

## 1. Switch diagnosis (dev labels only; both Paris4 bands are burned and fully scored)
Scripts: `switch_diag.py` (pair-level additive decomposition of the scorer's switch rate: method − mean of the 3 base
seeds, per 1000 matched pairs, sums to the scorer's point estimate), `piece_diag.py` (excess near each kind of chain
piece), `rules_screen.py` (step/piece label screen of cut rules). Outputs `switch_diag_z{9000,7000}.json`,
`piece_diag_z{9000,7000}.json`, `rules_screen.json`.

| where the cut fit's extra switches are (excess vs mean base, /1000) | z 9000 (f6_cut) | z 7000 (f6z_ms1 / ms2) | z 7000 oracle |
|---|---|---|---|
| total | +2.97 | +6.78 / +7.90 | +2.10 |
| within 32 vox (3-D) of a kept chain point (14 % of pairs) | **+7.46** (local 260 vs 128 /1000 at < 16 vox) | **+8.47 / +8.88** (238 vs 85) | +3.76 |
| farther than 32 vox | −4.50 | −1.70 / −0.98 | −1.65 |
| within 2.5° of a chain ray angle | +9.86 | +9.10 / +9.81 | +6.93 |
| same seam-offset bin / across a bin edge | +10.44 / **−7.47** | +8.00 / −1.22 | +11.31 / −9.21 |
| within 32 vox of a cut step | +3.48 | +3.30 | +2.21 |
| near 2-4-point pieces / ≥ 5-point pieces | +1.80 / +4.88 | +1.07 / +7.42 | |
| near pieces with ≥ 1 label-wrong step / label-clean pieces (labelled subset) | **+4.36 / +0.10** | **+5.21 / +0.97** | +1.75 / +0.05 |
| primary / non-primary windings | +1.42 / +1.54 | +3.63 / +3.15 | −1.10 / +3.21 |
| 30° sectors | no seam-angle concentration | flat (−1.5 … +2.6 each) | |

Findings:
- **The extra switches sit at the chains, around pieces that contain a wrong step.** Near label-clean pieces the cut
  fit has no excess at z 9000 (+0.1) and little at z 7000 (+1.0). Near pieces with a wrong step the local switch rate
  roughly doubles. The fit obeys a wrong piece locally and disagrees with it between rays, so a small wrong island forms
  around the ray (switches bordering patches of 1-100 vertices: +5.5 / +7.1).
- **Not the cut points as such, not short pieces, not seam angles.** Excess near cut steps is just the part of the
  near-chain excess that lies there; 2-4-point pieces carry less than their share; no angular sector stands out.
  Non-primary windings get worse in proportion to their share of pairs, not specifically.
- **Why dev under-predicted the confirmation by ~4.** The z 9000 base fits carry many switches on the seam-offset
  bin grid (local rate 509 vs 327 /1000 at z 7000). The cut fit removed 7.5 of them, which hid a near-chain excess (+10.4
  same-bin) of the same size as at z 7000 (+8.0). On a band whose base fits are cleaner, the near-chain cost shows in
  full. **So the binding band for switches is z 7000; z 9000 understates switch cost.**
- **Even perfect chains cost switches at z 7000.** The oracle fit is +1.8 … +2.4 over base there (near-chain +3.8,
  far −1.7). A switch gate of ≤ +1 on z 7000 asks for fewer chain-induced switches than label-perfect chains at full
  density give. That is possible only by placing fewer constraints, and mostly clean ones. **This is recorded as a
  real risk that no variant can pass.**
- **Lever (step screen, `rules_screen.json`).** The share of kept chain points that lie in a piece with a wrong step
  (the switch-load proxy) by cut threshold, z 9000 / z 7000:

  | rule | kept step precision | points in wrong pieces | within-piece d10 | kept points |
  |---|---|---|---|---|
  | no raise (no cut) | 78.4 / 83.9 | 97 / 97 % | −1.87 / −1.24 | 48,938 / 57,127 |
  | cut 0.35 (fewer cuts) | 81.9 / 86.4 | 58.5 / 60.0 % | −1.14 / −0.68 | 46,192 / 55,602 |
  | **cut 0.263 (test 6)** | 83.7 / 88.0 | 45.0 / 47.4 % | −0.69 / −0.35 | 43,484 / 53,749 |
  | cut 0.20 | 85.5 / 89.6 | 34.3 / 35.1 % | −0.32 / −0.10 | 39,284 / 50,515 |
  | cut 0.15 | 86.8 / 90.8 | 25.6 / 25.5 % | −0.12 / +0.04 | 33,456 / 45,155 |
  | cut 0.10 | 86.8 / 91.5 | 18.3 / 16.7 % | +0.05 / +0.05 | 24,316 / 34,920 |
  | cut 0.15 + drop pieces < 4 pts | 89.0 / 92.2 | 32.3 / 29.5 % | −0.12 / +0.04 | 22,536 / 35,273 |
  | cut 0.10 + drop pieces < 4 pts | 91.4 / 94.5 | 24.5 / 19.4 % | +0.06 / +0.05 | 12,083 / 22,141 |
  | cut 0.263 + drop pieces < 4 pts | 84.6 / 88.5 | 50.0 / 50.3 % | −0.68 / −0.35 | 37,720 / 49,662 |
  | cut 0.263 only if both sides keep ≥ 4 pts | 81.4 / 86.4 | 50.6 / 51.7 % | −0.84 / −0.43 | 48,938 / 57,127 |

  **A higher threshold (fewer cuts) goes the wrong way:** pieces get longer and keep more merged-contact steps. More
  cuts remove wrong steps and the residual bias (d10 → 0), at the cost of linkage and points. Dropping 2-3-point pieces
  removes low-precision constraints (they sit in contact-dense stretches: 64-74 % step precision).
- A crude load model (near-chain excess ∝ points in wrong pieces × 3.8 + points in clean pieces, calibrated on the test-6
  cut and checked against the oracle's +2.1 at z 7000) predicts z 7000 switch Δ ≈ +0.6 for cut 0.10 + min 4 and ≈ +2.7
  for cut 0.15 + min 4. It is a guess, not a gate.
- **Not chosen: lower chain weight.** It loosens every constraint, clean or wrong. f6 showed far-field switches rising
  as the weight drops (z 9000 far-from-chain excess −11.4 at weight 10, −8.1 at 1, +2.5 at 0.3, no-raise chains), so it
  trades switches, it does not remove them.

## 2. Variants (label-free; S_pack-branched; nothing fitted on the confirmation band)
Both are `autowind.postproc --mode cut` with a lower threshold on the frozen grad_mag logistic and a minimum piece
length (new option `--min-pts`, default 2 = the f6 recipe byte for byte). As before: S_pack < 21 -> cut every no-raise
single step with gm_p ≥ thr (gm_p as stored, 4 decimals), no shared point, re-zero wind per piece, then drop pieces with
< 4 points; S_pack ≥ 21 -> the run's no-raise set unchanged. Fit weight 10, spacing 0 (as test 6).

| variant | command (from `tools/`) | z 7000 set (label-free counts) | z 9000 set |
|---|---|---|---|
| **V1 `c10m4`** | `python -m autowind.postproc --run <run> --mode cut --thr 0.10 --min-pts 4 --out <dir>` | `p4z7_c10m4` sha 6903f93d…: 57.1 % of single steps cut, 3,068 pieces, 22,141 points, median 6 | `p4z9000_c10m4` sha 54f12e31…: 70.2 % cut, 1,870 pieces, 12,083 points, median 5 |
| **V2 `c15m4`** | `… --thr 0.15 --min-pts 4 …` | `p4z7_c15m4` sha b4747e01…: 37.1 % cut, 4,121 pieces, 35,273 points, median 6 | `p4z9000_c15m4` sha 45ae8de7…: 50.7 % cut, 3,054 pieces, 22,536 points, median 6 |

- Runs: z 7000 = `data/scratch/f6_screen/confirm/windings/p4z7_auto` (S_pack 17.25), z 9000 =
  `data/scratch/fourth_test/dev/p4z9000_auto` (S_pack 18.5). Sets in `data/scratch/f7_switch/windings/`.
- **Choice of numbers.** 0.10 and 0.15 and min 4 were picked on the dev step screen above (dev labels). 4 points:
  round 2 found 4-point oracle pieces still link; 2-3-point pieces are the lowest-precision constraints.
- **PHerc1667.** `postproc --run data/scratch/fourth_test/repro/p1667_auto` with either variant reads decision 'off'
  and writes relative_windings.json sha 85dcbfb46a66b192… = M2, byte for byte (`data/scratch/f7_switch/repro/`). So the
  PHerc1667 dev fit is skipped (as in f6).
- **Reproduction.** With the new option at its default, postproc re-creates the test-6 sets byte for byte:
  z 9000 `p4z9000_cut263` 0954c72a… (chains bdbc7bce…), z 7000 `p4z7_f6` **03c014bd…** (chains b5d20c1d…).
  tools/auto_windings.py (8f097312…) and eval/p0139_graph.py (fb7553b1…) are unchanged; the autowind chain code is not
  touched, so the AUTOWIND.md table is unaffected.
- Known risk, stated now: V1 keeps only 12,083 points at z 9000 (a quarter of the no-raise set), so it may fail the
  primary items there; V2 may fail the z 7000 switch item.

## 3. GATE (fixed now)
Scorer `data/scratch/f7_switch/dev_table_f7.py` (sha in G0): the stored dumps of each band's 3 base seeds
(z 7000: f6z_base / f6z_bases2 / f6z_bases3; z 9000: p4_base / p4_bases2 / p4_bases3), same endpoint code as test 6
(own-offset seam wrong area on the band's PRIMARY_K, paired BootV, switches per 1000 adjacent pairs, all vertices).
PRIMARY_K: z 7000 `13,16-32,34-44,47-61,63-108`; z 9000 `21-32,34-60,62-87,90-101,106-108` (both as frozen before).

**A variant GOES iff on BOTH burned bands (z 9000 and z 7000), against EACH of the band's 3 base seeds (6 comparisons):**
1. primary own-offset upper 95 % CI < 0;
2. primary estimate ≤ −4.0 pp;
3. switch Δ ≤ +1.0 per 1000 (point estimate);
4. non-primary windings: own-offset seam wrong area Δ ≤ +3.0 pp (point estimate; `subset_wrong_area` `nonprimary`
   subset, own offsets). Test 6's z 7000 value was +5.47.

Scorer check (before G0, on the stored test-6 fits): it reproduces the test-6 numbers exactly (z 7000 f6z_ms1 − base
−7.85 [−9.01, −6.68], switch +6.51, non-primary +5.47; z 9000 f6_cut − p4_base −5.41 [−7.05, −3.94], switch +3.79), and the
test-6 cut fails this gate on both bands (switch item; non-primary item at z 7000); `dev_table_check.json`.

Reported for every comparison whatever the outcome: global seam Δ, arm switch rate, the switch decomposition of
section 1 for the new fit, and the same rows for the test-6 cut fit as reference.

**Cap and order (≤ 4 dev fits in total, about $3-7 each).**
- Fits are seed 1, weight 10, spacing 0, villa 0fb38c45, env files `spiral/arms/f7_*.env` (copies of
  f6z_ms1.env / f6_cut.env with ARM and REL_WINDINGS changed). Arms: `f7_c10z7`, `f7_c15z7`, `f7_c10z9`, `f7_c15z9`.
- **Stage A (screen):** V1 and V2 on z 7000 (`f7_c10z7`, `f7_c15z7`), in parallel. z 7000 first because section 1 shows it
  is the band where switch cost is visible.
- **Early drop:** a variant that fails any of items 1-4 against any z 7000 base seed is dropped; no z 9000 fit for it.
- **Stage B:** the z 9000 fit (`f7_c10z9`, `f7_c15z9`) for each variant that passed every z 7000 item.
- If no variant passes stage A, stop: 2 fits used, no confirmation.

**Selection if both go:** lowest worst-case switch Δ over the 6 comparisons; tie (within 0.1) -> lowest worst-case
primary estimate. Frozen with its numbers before any confirmation step.

## 4. If exactly one variant goes: confirmation (stage 1 of notes/2026-10-24-f7-confirm-prereg.md)
- Region: PHercParis4 z 10800-11800 (stage 0). Two-stage freeze: stage 1 (method, arms, analysis, primary set) before
  the method chains exist; chains built blind from the label-free crop solve with `autowind chains --raise auto
  --no-solve` + `autowind.postproc` (the selected variant); their sha frozen (stage 2) before any method pod.
- Arms: 2 method seeds, 3 base seeds, the oracle. **PASS iff every one of the 6 combos has upper CI < 0 AND switch
  Δ ≤ +5 per 1000** (same margin as before), p_arm < 0.05, and the oracle is informative.
- No label-based chain metric on z 10800-11800 until all fits are scored. Target: results by Oct 28 (Oct 29 latest).

## 5. Stop rule, budget, rules
- If nothing goes: stop and write up. z 10800-11800 then stays untouched except the label-free prep and label-only
  audits already listed in its stage 0.
- Budget: $50 total for dev fits and the confirmation; finish early and report before exceeding it.
- Pods `vss-f7-*`, terminated right after each pull, logged in data/scratch/p0139_spiral/pods_arms.txt; never touch
  other pods; kill by listed PID only.
- Frozen files never modified; every published chain set still reproduces (AUTOWIND.md table, plus the test-6 cut set).
- Eval lock for jobs over ~6 GB, or > 20 GB available and logged (`data/scratch/f7_switch/eval_lock_exceptions.log`).

---

## Log (append-only)
### G0 (2026-10-24T18:39:52Z): gates frozen
sha256 of this file up to and including the "## Log (append-only)" line: d9dcf415b9dae1aa4e9dbe9dd34011ea53b95cc1dc75c42f6ef1f5b934a74f5b (snapshot
`data/scratch/f7_switch/gate_note_snapshot.md`). dev_table_f7.py sha256 cc7786b7f247de6f83c39be80b3b12cd0bd537218d5e3ae9de69c9b504006292; tools/autowind/postproc.py sha256 af57310120d64205c9076338cae6c91601f2bc2d77a67b05c24aa1b82752f1e2
(`data/scratch/f7_switch/gate_freeze.sha256`). No f7 dev fit exists; no pod has been created.

### L1 (2026-10-24T18:51:32Z): stage A pods created
Two pods, one per arm, envs spiral/arms/f7_c10z7.env / f7_c15z7.env (= f6z_ms1.env with ARM and REL_WINDINGS
changed: Z0 7000, Z1 8500, weight 10, spacing 0, seed as f6z_ms1):
- `f7_c10z7` (chains p4z7_c10m4, rw sha 6903f93d…): pod <pod-id> vss-f7-f7_c10z7, 1xL40S datacrunch, $1.81/h, 18:48:51Z.
- `f7_c15z7` (chains p4z7_c15m4, rw sha b4747e01…): pod <pod-id> vss-f7-f7_c15z7, 1xRTX6000Ada datacrunch, $1.48/h,
  18:50:39Z (the create call timed out and the grabber adopted the pod it had made; the pod list shows no duplicate).
AUTOWIND.md now documents `postproc --mode cut` and `--min-pts`, with the test-6 cut sets and the PHerc1667 identity in the
reproduction table (documentation only; no code change after G0).

### L2 (2026-10-24T22:24:36Z): V2 (c15m4) fails stage A on z 7000 -> dropped
`f7_c15z7` pulled 22:10:03Z (240 windings), pod <pod-id> terminated 22:10:05Z (3.32 h x $1.48 = about $4.9).
`data/scratch/f7_switch/dev_table_f7.py f7_c15z7` -> dev_table_c15z7.json/.log:

| vs base | primary own-offset Δ [95 % CI] | switch Δ /1000 | nonprimary own-offset Δ | items failed |
|---|---|---|---|---|
| f6z_base | −5.19 [−6.12, −4.32] | +6.43 | +6.60 | switch, nonprimary |
| f6z_bases2 | −5.27 [−6.22, −4.39] | +7.03 | +4.90 | switch, nonprimary |
| f6z_bases3 | −5.29 [−6.23, −4.42] | +6.64 | +4.92 | switch, nonprimary |

(test-6 cut f6z_ms1 for comparison, same code: −7.85, switch +6.5, nonprimary +3.8 to +5.5.) Switches did not fall
and the primary gain shrank. Per section 3, V2 gets no z 9000 fit. V1 (`f7_c10z7`) is still fitting.

### L3 (2026-10-24T23:07:28Z): V1 (c10m4) fails stage A on z 7000 -> NO-GO, f7 stops
`f7_c10z7` pulled 22:50:54Z (240 windings), pod <pod-id> terminated 22:50:54Z (4.03 h x $1.81 = about $7.3).
`dev_table_f7.py f7_c10z7` -> dev_table_c10z7.json/.log:

| vs base | primary own-offset Δ [95 % CI] | switch Δ /1000 | nonprimary own-offset Δ | items failed |
|---|---|---|---|---|
| f6z_base | −1.40 [−2.10, −0.71] | +6.17 | +2.96 | estimate, switch |
| f6z_bases2 | −1.48 [−2.17, −0.83] | +6.76 | +1.26 | estimate, switch |
| f6z_bases3 | −1.50 [−2.18, −0.85] | +6.38 | +1.28 | estimate, switch |

**Decision (section 3): neither variant passes stage A, so there is no stage B, no selection and no confirmation.**
2 of the 4 allowed dev fits were used. Spend about $12.2 of $50. No vss-f7 pod is left (pod list at 23:07Z shows only
other projects' pods). z 10800-11800 keeps only its label-free prep and label-only audits; no chain set was built on it.

Post-hoc (descriptive, dev labels only; `switch_diag.py --band z7000_c10 / z7000_c15`, near/far split around each
fit's own chain set). Switch excess vs mean base: test-6 cut (0.263) +6.78, c15m4 +6.70, c10m4 +6.44. All three are
about the same, while the primary gain falls with fewer kept points (−7.9 / −5.2 / −1.5). Very close to kept chain
points (< 16 vox) the switch rate is 177-212 per 1000 against 63-72 for base under every variant. With fewer points the
excess moves to bin edges and far from the chains, instead of going away. So the excess was not carried by the
label-wrong pieces, as the diagnosis in section 1 proposed: cutting more of them did not remove it. It looks like a
cost of the chain radius term itself (weight 10) on this band, which chain selection cannot fix. That points at the
fit side (weight, or how the radius term is spread around each anchor), which was out of scope here. Not tested.
