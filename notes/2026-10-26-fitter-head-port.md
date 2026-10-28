# 2026-10-26: our chain input on the CURRENT villa spiral fitter (HEAD port), and the anneal option rebased for upstream

Sections 0-4 are fixed before the HEAD fit (freeze line G0 in the log). Results, deviations and the PR draft go below
the "Log" line, timestamped. Test 13 (notes/2026-10-26-anneal-prereg.md) is running on the pinned fitter on pods
vss-an-*; nothing here touches those pods, the pinned checkouts (data/ext/villa-0fb38c45, data/ext/villa) or any file
they use. Scripts for this note live in `data/scratch/head_port/` (new files only).

## 0. Question
All our fits pin villa 0fb38c45 (2026-08-18; `config/recipe.toml`, `env/pod_setup.sh` fetches it on
each pod into ~/ext/villa). Villa main has since moved the fitter to `spiral-fitting/` (#1548) and changed it a lot (68
commits). Two questions:
1. Does our autowind chain input (relative_windings.json, unattached strips) still load and fit on HEAD?
2. With HEAD and default settings, is the score on an already-used dev band equivalent to the pinned score?

## 1. Design
- **HEAD:** ScrollPrize/villa main `322cb0899ac9efe2588d5a3395dce6dc2dea3c89` (2026-10-24), separate checkout
  `data/external/villa_head/` (local; read-only use) and fresh clones on the pod. The fit runs on the UNPATCHED tree.
- **Band:** PHerc1667 vol 20251217075048 L2 (9.596 um), z 6500-8000: the PHerc1667 confirmation band (2026-10-22,
  notes/2026-10-22-confirmation-region.md). Already burned; not Paris4 z10800-11800, no Paris4 z8500-9000 labels used.
- **Arm (one fit), `c1667_m2s1_head`:** the same arm as the pinned `c1667_m2s1` (M2 chains `p1667_m7_ray_c`, sha256
  85dcbfb4…0804b5, weight 10, spacing 0, seed 1 = default). It is the chain arm, not the chain-free base, because
  question 1 needs the chains in the fit; "baseline" here means no new options (no patch, no anneal, HEAD defaults).
- **Settings = HEAD defaults, except** (a) what defines the arm and (b) what our inputs force:
  - `{"input_disable_patches": true, "loss_weight_shell_outer": 0, "dense_spacing_mode": "grad_mag",
    "optimizer_num_training_steps": 30000, "z_begin": 6500, "z_end": 8000, "pcl_unattached_pcl_min_point_spacing": 0.0,
    "loss_weight_unattached_pcl_radius": 10.0, "output_step_size": 4}` (`data/scratch/head_port/pod/overrides_head.json`).
  - vs the pinned overrides: `loss_weight_shell_patch_radius` dropped (key removed on HEAD; HEAD rejects unknown
    override keys). `dense_spacing_mode` "phase" no longer exists. The HEAD default "winding_model" needs a
    winding-inference store and an outer shell that this band does not have, so "grad_mag" is the only usable mode.
    30000 steps equals the HEAD default.
  - `output_step_size 4` only changes the exported mesh grid. It matches the pinned arm's step-4 re-export
    (export_fine.sh), which is what the stored dump was scored on.
  - spiral-scroll.json is as in run_arm.sh minus `surf_sdt` (HEAD has no surf-SDT input; make_surf_sdt.py is gone).
    No `num_windings` (as pinned: generic default).
- **Inputs held fixed:** the same lasagna nx/ny/grad_mag (data/p1667/inputs), m7 L2 mask and umbilicus. Tracks are
  built with the PINNED extractor (`spiral/extract_tracks.py`, the same arguments as run_arm.sh), so the tracks
  match the pinned fits'. Lasagna pools are packed with the HEAD packer (same flags). The only planned differences are
  the fitter code and the forced settings above.
- **Pod pipeline:** `env/pod_setup.sh` (FIT_ONLY=1, pinned env, used only for the track extractor), then
  `data/scratch/head_port/pod/head_setup.sh` (HEAD clone + `uv sync`; a second patched clone with its own env runs the
  upstream unit tests for the anneal patch on CPU), then `head_arm.sh` (pack, tracks, dataset, HEAD headless
  `fit_spiral.py`). Meshes are pulled and dumped locally with `release/fit_vertex_dump.py --scroll p1667 --z0 6500
  --z1 8000` under `flock /tmp/vss_eval.lock`.

## 2. What is compared, and what "equivalent" means
**Scorer:** `data/scratch/head_port/head_compare.py`. It imports the endpoint code of eval/method_table.py,
spiral_eval.py and subset_wrong_area.py unchanged: inner (k < 34) own-offset seam wrong area, seam switches per 1000
adjacent pairs, and paired BootV over the same reference vertices.

**Primary comparison:** HEAD fit − pinned `c1667_m2s1` (stored dump), paired, on inner own-offset wrong area (pp).
**Secondary:** switches per 1000 (HEAD − pinned).

**Noise floor, measured before the HEAD fit** (`data/scratch/head_port/reference.json`; pinned dumps only):

| pinned pair (vs c1667_m2s1) | primary Δ (pp) [95 % CI] | switch Δ /1000 [95 % CI] |
|---|---|---|
| same arm and seed rerun on another GPU (c1667_m2s1_fig, RTX6000Ada vs L40S) | +0.16 [+0.04, +0.27] | +0.67 [+0.11, +1.38] |
| seed 2 (c1667_m2s2) | +0.17 [+0.01, +0.35] | +0.41 [−0.87, +1.49] |

Levels: pinned M2 s1 50.74 % inner wrong area, 73.86 switches per 1000. M2 vs base primary −27.6 to −29.3 pp.

**Verdict (fixed now):**
- **EQUIVALENT** iff the 95 % CI of the primary Δ lies inside [−2.0, +2.0] pp **and** the switch Δ estimate lies
  inside [−5, +5] per 1000.
  - 2.0 pp is about 12x the rerun noise and 7 % of M2's gain (about 28 pp).
  - It is well inside the 80 % keep rule the non-regression gates use (a loss of about 5.6 pp).
  - ±5 per 1000 is the switch margin of our confirmations.
- **HEAD BETTER:** upper CI < −2.0 and switch Δ ≤ +5.
- **HEAD WORSE:** lower CI > +2.0, or switch Δ > +5.
- **Otherwise: NOT SHOWN EQUIVALENT** (report the numbers, no claim).
- **Load check (question 1), pass iff all hold:**
  - the HEAD log reports 577 unattached collections and 18,670 points, as the pinned log did;
  - the unattached radius loss is non-zero in the step log;
  - the fit completes 30,000 steps with no traceback;
  - it exports tifxyz windings that the dumper reads.
- **Always reported:**
  - levels;
  - HEAD vs each pinned base seed (cross-fitter, primary and switch), against test 13's non-regression thresholds (keep
    ≥ 80 % of the M2 gain; switch ≤ M2 mean + 2) as information only;
  - fit wall time and GPU;
  - matched share (the unmatched fraction).
- **Interpretation limits:**
  - HEAD − pinned bundles every upstream change: flow stages 1 → 2, phase → grad_mag spacing, no surf SDT, track DT
    start 10000 → 25000, the sweeps' changes. A difference cannot be attributed to one of them.
  - No HEAD chain-free base is run, so HEAD's own chain gain is not measured.
  - One seed.

**Prediction (before the fit):**
- The chains load: P ≈ 0.85; the risks are track-store or crossing-cache format changes.
- If it runs, P(EQUIVALENT) ≈ 0.35. Too much default behaviour changed to expect ±2 pp; more likely within ±5 pp, sign
  unknown.

## 3. Anneal option for upstream (code only; no fit)
- `spiral/patches/chain_anneal_head.patch` (new; `chain_loss_tol.patch` is unchanged, sha c0657b76…).
  - Only the anneal schedule of test 12/13, rebased onto HEAD `spiral-fitting/`.
  - Dropped: the margin/median options (test 12: harmful or inert) and the robust-cut and per-point-weight code from
    robust_unattached (failed test).
- Keys follow the HEAD precedent `model_flow_field_high_res_lr_{scale_final, ramp_start_step, ramp_steps}`:
  - `loss_weight_unattached_pcl_scale_final` (default 1.0 = off);
  - `loss_weight_unattached_pcl_ramp_start_step` (0);
  - `loss_weight_unattached_pcl_ramp_steps` (1).
  - Mapping from test 13's keys: final_scale → scale_final; decay_start → ramp_start_step; decay_end − decay_start →
    ramp_steps.
  - The schedule arithmetic is the same expression as chain_loss_tol.patch, `1 + (final − 1)·(it − start)/(end −
    start)`, so the same schedule gives the same floats.
- At defaults, the multiplier is exactly 1.0 and the weights are `w * 1.0 == w`. The loss is computed as before.
- Adds:
  - descriptions.json entries (the upstream test requires a description for every key);
  - a unit test in tests/test_spiral_helpers.py.
- Run on the pod (CPU): upstream tests/test_config.py, test_spiral_helpers.py and test_lr_schedule.py, on the patched
  tree and on the unpatched tree for comparison.

## 4. Budget and rules
- **Cap: $40.**
  - Expected: 1 pod, A6000 at about $0.54/h or L40S at about $0.8/h. Setup about 0.5 h, tracks about 0.1 h, fit about
    3 h (pinned: 10,102 s on L40S). That is about $2-4.
  - Offers above $1.0/h are not used. One relaunch is allowed if the pod dies before writing meshes.
- **Pod:** `vss-head-m2s1`, created and deleted only through `data/scratch/head_port/pods/prime_rest.py`, whose delete
  refuses any name not starting with vss-head-.
  - One watchdog with a deadline of launch + 9 h.
  - Terminate right after the pull.
  - Re-list after any failed create.
  - Never touch other pods (sm86-speed, kv4-*, flint-*, nb-gen3, vss-an-*, vss-full-*, vss-surf-*).
  - Log: `data/scratch/head_port/pods/pods_head.txt`.
- **Nothing frozen is modified.** No commit, push, PR or post. The PR text below is a draft only: the option goes
  upstream only if test 13 passes, and the user decides.

---

## Log (append-only)
### G0 (2026-10-26T04:27:16Z): sections 0-4 frozen before any HEAD fit
- sha256 of this file up to and including the "## Log (append-only)" line:
  155df07f7950785233f0025930ec386241ea90a6cd7ae21283a8d924ba23787d (snapshot `data/scratch/head_port/freeze/prereg_snapshot.md`).
- `data/scratch/head_port/freeze/freeze.sha256`: head_compare.py 4c23968b…, head_arm.sh 39c1c59f…, head_setup.sh
  9d83f121…, overrides_head.json 31a0aecc…, chain_anneal_head.patch db79cd0a…, reference.json faab22cf….
- Unchanged code used: eval/method_table.py 6cfc15df…, spiral_eval.py 813860d7…, subset_wrong_area.py 8890d16d…,
  release/fit_vertex_dump.py 80e8b55c… (the current version used by tests 12/13; the 2026-10-22 freeze listed
  cecb5513…, before the --scroll paris4 change). Pinned dumps c1667_m2s1.npz 2859a1d4…, c1667_m2s1_fig.npz 7dfc4786….
### D1 (2026-10-26T04:30Z): deviation before any HEAD fit. HEAD's default has no tracks; the pod setup was fixed
The full read of the upstream diff, which finished after G0, found three things that change §1. No fit had started.
1. **`input_use_tracks` now defaults to false** (#1651, "Spiral hyperparams from recent sweeps", 2026-08-30). At the
   pin, tracks were used whenever `paths.tracks_dbm` existed.
   - "HEAD and default settings" therefore means **no tracks**. The pinned arm had tracks (track_radius weight 50, the
     largest term at step 0).
   - Overrides unchanged (overrides_head.json 31a0aecc…; `input_use_tracks` is left at its default).
   - head_arm.sh no longer extracts tracks, and spiral-scroll.json has no `tracks_dbm`.
   - This is now the largest known difference between the two fits. The equivalence rule (§2) is unchanged.
   - If the fit is NOT equivalent, the obvious next fit is the same arm with `input_use_tracks: true` and the
     pinned-extractor tracks. That is not run here (one fit).
2. **The native build compiles `vc_object_store/src/TrackCrossings.cpp`**, so the sparse checkout must include
   `/vc_object_store/` (added on the pod and in data/external/villa_head).
3. **`paths.surf_sdt` is now a hard error in spiral-scroll.json.** It had already been dropped in §1; this confirms it.

Because tracks are off, the pinned pod_setup.sh (pinned env, needed only for the track extractor) is no longer run.
- head_setup.sh installs its own base tools (apt git/curl/build-essential/cmake, uv, Python 3.14).
- If the native build fails with g++ 11, it retries once with g++-13.
- New shas (`data/scratch/head_port/freeze/d1.sha256`): head_setup.sh 668d421f…, head_arm.sh 5747a5ca….
- head_compare.py, overrides_head.json, chain_anneal_head.patch and reference.json are unchanged.

### F1 (2026-10-26T04:31Z): what changed upstream since 0fb38c45 that matters to us (code read only)
Read from `data/external/villa_head` at HEAD 322cb089, with the old path at the pin. The fitter moved to `spiral-fitting/`
(#1548, 2026-08-21). 68 commits touch it since then, plus 4 on the old path before the move.

**Breaks our pinned recipe on HEAD (each would stop the run):**
- `dense_spacing_mode: "phase"` was removed together with the surf-SDT store, `make_surf_sdt.py`, `sdt_losses.py` and
  `phase_tuning.py` (#1871 "Spiral simplification"). The enum is now grad_mag | winding_model.
  - The new default `winding_model` (#1613) needs `<ds>/winding_inference/` (made by an external export tool) and an
    `outer_shell/`.
  - With our inputs only `grad_mag` runs. Upstream's own headless test spec also uses grad_mag.
- `loss_weight_shell_patch_radius` was removed (#1871). The headless CLI still raises KeyError on unknown override keys
  (#1793 only filters them in the service/runtime).
- `paths.surf_sdt` in spiral-scroll.json is now an error, because `parse_scroll_spec` rejects unknown `paths` keys.
  Unknown top-level keys are now ignored.
- Checkpoints from before September (schema v1, exponential gap) cannot be resumed (#1871). Pinned checkpoints are dead.
- The native helpers are now `vc_spiral.{spiral_sampling, track_crossings, track_store, surface_index}`.
  - `uv sync` in spiral-fitting builds them (scikit-build-core, nanobind, C++23, cmake ≥ 3.22).
  - The build also needs `vc_object_store/` from the repo root.
  - `grid_raycast` is gone. Our `env/build_vc_native.sh` does not apply.
- Python ≥ 3.14 and torch 2.11-2.12 from the cu128 index; that needs a driver for CUDA ≥ 12.8. The pinned setup used
  lasagna's cu130 venv.

**Defaults that change a fit even with identical overrides** (AST diff of `Config.__init__`):
- `input_use_tracks`: tracks used whenever the path existed → **false** (#1651). This is the biggest one for us; see D1.
- `loss_start_track_dt`: 10000 → 25000 (#1651).
- `model_num_flow_stages`: 1 → 2 (#1693).
- `track_max_track_crossing_per_step`: 1 → 2 (#1562).
- winding-model pair counts: 12k → 128k.
- Removed (#1871): `track_crossing_mode` (count grouping only), `dt_progressive_*`, dense attachment, unverified
  patches, influence/anchor, `model_num_flow_timesteps`.
- Added, off or neutral by default:
  - `input_use_*` toggles;
  - `optimizer_flow_*` gradient conditioning (#1790);
  - fiber supervision (#1788);
  - `pcl_link_*` / `pcl_fiber_link_*`;
  - the native `min_spacing` barrier (weight 2, on in every mode);
  - softplus gap parameterisation (#1625, model change).
- The sweep PRs changed defaults only through the rows above. Loss weights are unchanged:
  - `loss_weight_unattached_pcl_radius` 2;
  - `_dt` 4;
  - `pcl_unattached_pcl_min_point_spacing` 16;
  - `patch_radius_loss_margin` 0.025;
  - `loss_start_patch_dt` 25000;
  - 30000 steps.

**Unattached strips / relative windings (our chain input):**
- `relative_windings.json` is still the conventional RELATIVE-role file, with the same JSON format
  (`vc_pointcollections_json_version "1"`, `p`, `wind_a`).
  - New: an optional collection `uuid` (#2077).
  - New: a `drawn_control_points.json` role.
  - Still **no per-point or per-chain weight field.**
- With patches disabled, every collection with ≥ 2 points in the z-ROI becomes an unattached strip, as before.
- The radius loss is still the hinge on |normalised radius − row mean| with margin `patch_radius_loss_margin·dr`, a
  masked mean, weighted in `FitContext.step`.
- New:
  - unwrapping goes through the cached `ThetaCrossingMap` along a sampled chain walk (#1529/#1634), sampled without
    replacement with masks;
  - vertical fibers get a radial offset (#1788);
  - `loss_start_unattached_pcl_dt` decouples the strip DT start (default follows `loss_start_patch_dt`);
  - the progressive-DT mask is gone.
- Open PR #2269 (ChaseHendrick, open, mergeable) adds a separate radius margin for unattached collections. It edits the
  same hunks as our patch (config.py ~l.588, descriptions.json ~l.213, tests/test_spiral_helpers.py), so whichever
  merges second needs a trivial rebase. Test 12 found a dead zone on our chains harmful. Worth a comment from us only
  if asked.

**DT and radius terms:**
- **DT:**
  - families are `patch_dt` (4), `unattached_pcl_dt` (4) and `track_dt` (10, only with tracks);
  - `dt_target_mode` strip_median by default, so no whole-object cache;
  - a family's cache is built only when its weight is > 0, so weight 0 still skips it;
  - #1792 lowers the cache point cap from 20M to 5M;
  - #1787 run-DT scheduling is interactive only (headless leaves it None);
  - with 30000 steps, DT acts only in the last 5000.
- **Radius:**
  - `patch_radius` (8, needs patches), `unattached_pcl_radius`, `track_radius` (50, needs input_use_tracks) and
    `shell_outer` (1);
  - unverified-patch radius and `shell_patch_radius` are gone.

**Winding count / repair:**
- #2048: optional `num_windings` (int ≥ 2) in spiral-scroll.json.
  - It sets `shell_outer_winding_idx` and `model_gap_expander_num_windings` (default 130 = Scroll 1), and raises the gap
    capacity to N+3 if needed.
  - Overrides still win. It is worth setting for real runs, because the exporter writes windings up to that index.
  - Not used here (the pin also ran on 130).
- #2023/#1855 (minimal winding repair keeps all measured equations) and #1864 (count the point-to-attachment step):
  `find_inconsistent_windings.py` is a standalone audit/repair tool over verified patches and absolute-winding
  collections. It is not called by fit_spiral and not usable patch-free, so not for our chain-only fits as they are.
- #2001 orients winding-inference rays (winding_model mode only). #2028 builds spiral angles in float64 (always on).
  #2218 is faster track-satisfaction scoring (identical results).
- **Exported grids (#1899):**
  - columns are reversed (outermost wrap in column 0);
  - rows are reversed only when z is known to run bottom-to-top;
  - `meta.json` gets `grid_orientation`.
  - Our dumper is unaffected: nearest-surface winding from `w###` names, sign and offset refitted.

**Data requirements:**
- A headless fit needs:
  - `umbilicus.json` and spiral-scroll.json (schema 1);
  - the normals pool (dense_normals 100);
  - the grad_mag pool in grad_mag mode, packed first with `pack_resident_pools.py` (our flags still exist; `--what`
    takes only normals,grad_mag);
  - patches, unless `input_disable_patches`.
- Optional: tracks (off by default), fibers, PCLs.
- The track DBM is still dbm.sqlite3. #2055 (Python < 3.13 fallback) and #2020 (cache mtime) don't affect us.
- Headless resume: built-in `checkpoint_fitted.ckpt` autosave every 1000 steps plus `FIT_SPIRAL_RESUME_PATH`.
  `FIT_SPIRAL_RESUME_STEP` is removed. Our fit_ckpt.py wrapper would still work but is not needed.
- **PR #2251** (NovaCorpAI, "spiral-fitting: fit from the published dataset as shipped"):
  - **OPEN, not merged.**
  - GitHub says MERGEABLE (no conflicts) but mergeStateStatus BLOCKED (needs review; the only check is a Vercel
    deploy-authorization failure, not tests).
  - No formal reviews. CVasilopoulos reproduced it at 857bc4b, measured 15.3 GB, and called it ready from their side.
    The author asked for review from the fitter maintainers on 2026-10-25.
  - Head 2dd56d7, +790/−23 in fit_session.py, fit_spiral.py, pack_resident_pools.py, spiral_service.py, README and a new
    test.
  - What it does: download only what a default fit opens (about 15 GB instead of the 177 GB rclone of the published
    dataset).
  - `fit_spiral.py --dataset … --check` reports the inputs as required/optional/off without torch or a GPU, prints a
    spiral-scroll.json template when it is missing, and exits 0/1. It checks entries and sidecar meta.json, not file
    completeness.
  - For us, `--check` is a cheap pre-flight for pod datasets once merged. We stream our own band crops, so the
    download change does not matter to us.

### F2 (2026-10-26T04:31Z): the anneal option on HEAD (`spiral/patches/chain_anneal_head.patch`, db79cd0a…)
- 4 files, +56/−4:
  - config.py: 3 keys;
  - fit_spiral.py: `get_unattached_pcl_weight_scale()` next to `get_flow_field_high_res_lr_scale()`, and the scaled
    weights in `FitContext.step`;
  - descriptions.json: 3 entries;
  - tests/test_spiral_helpers.py: 2 tests.
- Made with `git diff` in data/external/villa_head (branch chain-anneal-head, uncommitted). It applies with `git apply`
  to a clean 322cb089. chain_loss_tol.patch is unchanged (c0657b76…).
- The interactive preview path (unweighted diagnostics) is untouched. The DT target-cache gate still uses the unscaled
  weight, as in the test-13 code.
- Test-13 schedules in HEAD keys:
  - an_f03: `{"loss_weight_unattached_pcl_scale_final": 0.03, "loss_weight_unattached_pcl_ramp_start_step": 15000,
    "loss_weight_unattached_pcl_ramp_steps": 7500}`;
  - an_e03: the same with start 12500;
  - cl_ann: 0.1 / 15000 / 7500.

### P1 (2026-10-26T04:31Z): PR DRAFT. Not opened, pushed or posted
Goes upstream only if test 13 passes, and only when the user decides. The numbers in [brackets] are filled from test 13
and the HEAD-port fit at that point. Patch: `spiral/patches/chain_anneal_head.patch`. Branch name if opened:
`spiral-fitting-unattached-weight-ramp`.

> **spiral-fitting: optional ramp of the unattached-strip weights over the fit**
>
> **What it adds.** Three config keys, modelled on `model_flow_field_high_res_lr_{scale_final,ramp_start_step,ramp_steps}`:
>
> - `loss_weight_unattached_pcl_scale_final` (default `1.0`)
> - `loss_weight_unattached_pcl_ramp_start_step` (default `0`)
> - `loss_weight_unattached_pcl_ramp_steps` (default `1`)
>
> Both unattached-strip weights (`loss_weight_unattached_pcl_radius`, `loss_weight_unattached_pcl_dt`) are multiplied
> by a factor that is 1 until the ramp starts, moves linearly to `scale_final` over `ramp_steps` optimisation steps,
> and stays at `scale_final` afterwards. A factor of 0 skips the strip losses once the ramp ends.
>
> **Defaults are a no-op.** With the default `scale_final = 1` the factor is exactly 1.0, so the weights and the loss
> are bit-for-bit what they were. All three keys are run-boundary settings (no new prefix class) and have entries in
> `descriptions.json`.
>
> **Why.** Relative-winding strips (point collections not on any verified patch, e.g. automatically traced chains with
> relative winding steps) are good at fixing the winding count. Held at full weight to the end, they also pin the
> surface where the chain points sit, and that adds sheet switches around the chains. Annealing the strip weight
> late in the fit keeps the count, which is set early, and lets the other losses place the surface:
>
> - On PHercParis4 z 7000-8500 (organiser winding labels, pre-registered, 3 base seeds), chain weight 10 → ×0.1 over
>   steps 15000-22500:
>   - kept the full winding-count gain: own-offset wrong area −7.9 pp vs no chains, the same as without the ramp;
>   - cut the extra seam switches from +6.5…+7.1 to +2.2…+2.8 per 1000 adjacent pairs.
> - [test 13 result for the selected schedule: primary …, switches …; PHerc1667 non-regression …]
> - These runs used the same schedule on the 2026-08 fitter. On current main, the chain input loads and fits
>   [HEAD-port result].
>
> Example (the schedule we use):
> ```json
> {"loss_weight_unattached_pcl_radius": 10.0,
>  "loss_weight_unattached_pcl_scale_final": 0.03,
>  "loss_weight_unattached_pcl_ramp_start_step": 15000,
>  "loss_weight_unattached_pcl_ramp_steps": 7500}
> ```
>
> **Scope.**
> - No change to how strips are loaded, sampled or unwrapped.
> - The interactive preview diagnostics stay unweighted.
> - The DT target cache is still gated on the configured (unscaled) DT weight.
> - Iterations count from the start of the fit, as for the other `loss_start_*` and ramp keys.
>
> **Tests.**
> - `tests/test_spiral_helpers.py::UnattachedPclWeightScaleTests`: default is 1 everywhere; ramp endpoints and midpoint.
> - `tests/test_config.py` (every key described) passes. [pytest summary from ~/work/head/patch_tests.log]
>
> **Overlap.** #2269 (separate radius margin for point collections not on a verified patch) edits neighbouring lines in
> `config.py`, `descriptions.json` and `tests/test_spiral_helpers.py`. Whichever lands second needs a small rebase;
> the two options are independent.

Not in the PR: our chain builder, labels and scorer; the margin/median options of chain_loss_tol.patch (no gain in test
12); the robust-cut and per-point-weight code (failed test).

### D2 (2026-10-26T04:34Z): bootstrap retry
bootstrap_pod.sh ends with `tail -2 ~/pod_setup.log`. Because the pinned setup was skipped (D1), that file did not
exist, and the tail failed attempt 1 after all inputs and windings had been copied (sha ok). I wrote a one-line
placeholder `~/pod_setup.log` on the pod, and attempt 2 passed. Data and design are unchanged.

### D3 (2026-10-26T04:38Z): head_setup clone fix
head_setup.sh only ran `git checkout` when HEAD differed from 322cb089. After `clone --no-checkout`, main was already
at 322cb089, so the checkout was skipped and spiral-fitting/ was never written (`uv sync` failed with "No such file
or directory"). Fix: an unconditional `git reset --hard 322cb089` (tested locally on a sparse no-checkout clone) plus a
file-existence check. I stopped setup, applied the fix (head_setup.sh sha b8b62eac…) and reran boot_head.sh, which is
idempotent. The fit was not affected: it had not started.

### D4 (2026-10-26T04:41Z): missing HEAD dependency, setup typo
- HEAD's `spiral_helpers.py` imports `vc3d_fiber_format_adapter` at module level. That module loads the
  stdlib-only package `vesuvius/src/vc3d_fiber_format` from the same repo, which was outside my sparse checkout. The fit
  would have crashed on import, and the upstream tests failed at collection on both trees. I added that one directory to
  the sparse checkout, set on every run, and the env check now imports `spiral_helpers`.
  - **Data requirement for the summary:** running spiral-fitting from a checkout needs
    `spiral-fitting/ + vc_object_store/ + vesuvius/src/vc3d_fiber_format/`.
- `tail -3 a b` is invalid with two files. It ended setup before the done marker; now `tail -n 3`.
- Earlier, killing setup mid-run (D3) interrupted an apt install of g++-13. I repaired it with `dpkg --configure -a`.
  The native build succeeded with the image's g++ 11 anyway.
- head_setup.sh sha e00a6576…

### R1 (2026-10-26T06:10Z): RESULTS
**Run.**
- Pod vss-head-m2s1, 1x RTX A6000 48 GB (massedcompute, $0.54/h, driver 580.126.09).
- villa 322cb089, unpatched tree; torch 2.11.0+cu128, Python 3.14.9.
- Fit 04:43:00 → 05:53:10Z: **4,210 s**, of which about 13 min was one-off Triton kernel compilation at steps 0-4.
  The pinned arm took 10,102 s on an L40S.
- Pod created 04:30:14Z and terminated 05:54:50Z (TERMINATED; re-listed; watchdog stopped).
- **Spend 1.41 h × $0.54 = $0.76** (cap $40).
- Pulled to `data/scratch/head_port/c1667_m2s1_head/`: meshes, fit.log, timings, setup and test logs, satisfaction
  json. The 1.3 GB checkpoint was not kept.

**Load check (question 1): PASS.** It is the same fit as pinned on all four points:
- `Loaded point collection with 577 collections (18670 points)`, `pcls: 0 cross-patch, 577 unattached`, sampling
  group relative_windings: 577.
- `unattached_pcl_radius` is non-zero throughout: 1251.1 at step 0, 8.7 at 10k, 2.6 at 29.8k. Pinned: 1222.6 → 1.6.
- All 30,000 steps completed. No traceback, no non-finite steps (150 step lines).
- Exported 55 windings (w010-w064, raw + spliced) at scale 0.25 = step 4. The dumper read all 110 dirs.
- Chain satisfaction at the end: 501/577 collections (86.8 %), 18,452/18,670 points (98.8 %).
  Pinned: 457/577 (79.2 %), 17,901 (95.9 %).

**Score (question 2): NOT SHOWN EQUIVALENT** (rule of section 2). `data/scratch/head_port/result.json` sha 87f34f36…;
dump `vertex/c1667_m2s1_head.npz` sha 520381e6….

| arm | inner wrong area | switches /1000 | Δ primary vs pinned m2s1 [95 % CI] | Δ switch [95 % CI] |
|---|---|---|---|---|
| pinned c1667_m2s1 (ref) | 50.74 % | 73.86 | | |
| **HEAD c1667_m2s1_head** | **46.62 %** | **68.37** | **−4.11 [−8.09, −0.31]** | **−5.49 [−12.50, +1.92]** |
| pinned same-seed rerun (fig) | 50.90 % | 74.53 | +0.16 [+0.04, +0.27] | +0.67 [+0.11, +1.38] |
| pinned seed 2 | 50.91 % | 74.27 | +0.17 [+0.01, +0.35] | +0.41 [−0.87, +1.49] |

- **Verdict:**
  - Not EQUIVALENT: the CI is not inside ±2 and the switch Δ is outside ±5.
  - Not BETTER: the upper CI of −0.31 is not below −2.
  - Not WORSE.
  - The point estimates favour HEAD, but the difference from pinned is about 25x the rerun noise in the primary.
    HEAD's fit is materially different from the pinned one, not a reproduction of it.
- **Cross-fitter vs the pinned bases** (information only): primary −32.27 / −31.72 / −33.44, switch −68.8 / −68.7 /
  −71.3. Against test 13's non-regression lines:
  - keep ≥ 80 % of the M2 gain (≤ about −22.5 pp): met;
  - switch ≤ M2 mean + 2 (76.1): met.
  - But see coverage below.

**Matched share. Read this before reading the numbers above as "HEAD is better".** The wrong-area ratio is computed
over *matched* vertices, which lie within tol 4 vox of a fitted winding.

| inner (k < 34) vertices | pinned m2s1 | HEAD |
|---|---|---|
| matched at tol 4 | 71.65 % | **48.78 %** |
| right winding, as a share of ALL inner vertices | 35.26 % | **25.29 %** |
| wrong area on the vertices both match (35.5 % of inner) | 53.76 % | 46.75 % |
| diagnostic, tol 8: matched / wrong among matched / right share | 91.24 / 52.93 / 42.94 % | 79.81 / 49.91 / 39.98 % |

(The diagnostic tol-8 dumps are in `data/scratch/head_port/vertex_tol8/` and are not part of the pre-registered score.
The pinned column there is the same-seed fig rerun.)
- HEAD's surfaces sit 4-8 vox off the reference sheets far more often: the inner matched share falls in k 10-20
  (71.7 → 45.4 %) and k 20-34 (71.6 → 50.1 %). Outer windings fall less (60.1 → 57.1 %).
- On the vertices it does match, HEAD assigns the right winding more often (−7 pp on common support), consistent with
  its higher chain satisfaction.
- **Net: HEAD gets fewer inner vertices right in absolute terms** (25 % vs 35 % at tol 4; 40 % vs 43 % at tol 8). The
  primary's improvement is partly a coverage effect. The primary alone would have overstated HEAD.

**Interpretation (limits as in section 2: everything is bundled, one seed, no HEAD base).** The most plausible
contributor to the positional loss is D1:
- HEAD has no tracks by default (#1651). The pinned fit's `track_radius`/`track_dt` terms (75.0 / 55.9 at step 29.8k)
  pulled the surface onto traced sheet points; HEAD's only data terms here are dense normals/spacing and the chains.
- grad_mag spacing in place of phase, 2 flow stages and no surf SDT are the other candidates.
- This is a hypothesis, not a measurement.

**For porting our pipeline to HEAD:**
1. The chain input works unchanged (relative_windings.json + spiral-scroll.json schema 1).
2. Before trusting HEAD numbers, re-enable tracks (`input_use_tracks: true` plus a HEAD-built track store) or run a
   HEAD chain-free base, so that chain gain and positional accuracy can be separated.
3. Running from a checkout needs `spiral-fitting/ + vc_object_store/ + vesuvius/src/vc3d_fiber_format/`. Set
   `PYTHONUNBUFFERED=1`, or the step-loss lines only appear at exit.
4. Expect about 13 min of Triton compilation on first use per pod.

**Prediction check:**
- Load: P 0.85 → it loaded.
- EQUIVALENT: P 0.35 → it was not. The difference was within ±5 pp as guessed, but the sign of the primary hides a
  coverage loss.

**Anneal patch on the pod** (`chain_anneal_head.patch` db79cd0a…, on 322cb089, CPU): upstream tests/test_config.py +
test_spiral_helpers.py + test_lr_schedule.py gave **56 passed** on the patched tree (54 upstream + the 2 new
`UnattachedPclWeightScaleTests`). The unpatched tree gave **54 passed**.

**Fill-ins for the P1 PR draft:**
- "[HEAD-port result]" → "the chain input (577 strips, 18,670 points) loads and fits for 30,000 steps on 322cb089 with
  no other change".
- "[pytest summary]" → "56 passed (54 existing + 2 new)".
- Do NOT quote the HEAD-vs-pinned score in the PR. It is not a like-for-like comparison of the option, and the
  coverage caveat applies.
- The test 13 fields stay open until test 13 reports.

---

## ADDENDUM A1 (2026-10-26T06:18Z): does the chain gain hold on HEAD? Fixed before any A1 fit
Asked by the coordinator after R1. Same rules, the remaining cap ($40 − $0.76), pods vss-head-*, the same lock and lease
rules. Same band (PHerc1667 z 6500-8000) and the same scorer endpoints as section 2.

### A1.1 Arms (HEAD 322cb089, unpatched tree, seed 1 = default; one pod each)
| arm | chains (p1667_m7_ray_c, w 10, spacing 0) | tracks | overrides file (`data/scratch/head_port/pod/`) |
|---|---|---|---|
| **c1667_m2s1_trk** (a) | yes | ON | overrides_m2s1_trk.json |
| **c1667_base_trk** (b) | no | ON | overrides_base_trk.json |
| **c1667_base_notrk** (c) | no | off | overrides_base_notrk.json |
| c1667_m2s1_head (R1, exists) | yes | off | overrides_head.json |

- **The pairs are (a)−(b)** (primary question: tracks ON, configured like pinned) **and R1−(c)** (secondary: HEAD
  defaults, tracks off). Within each pair the arms differ only by the chains: relative_windings.json plus the two
  chain keys, exactly as pinned c1667_m2s1 vs c1667_base.
- **"As close to the pinned arm as HEAD allows"** (a, b). I AST-diffed the config defaults of 0fb38c45 and 322cb089:
  - Of the 121 keys both have, 6 differ.
  - Set back to the pinned values: `loss_start_track_dt` 10000 (HEAD 25000), `model_num_flow_stages` 1 (HEAD 2;
    HEAD's transforms.py says 1 "is exactly the original"), `track_max_track_crossing_per_step` 1 (HEAD 2).
  - Plus `input_use_tracks: true`.
  - Left as they are: `dense_spacing_mode` (phase no longer exists, so grad_mag as in R1); the two
    `sample_count_winding_model_*` keys (unused in grad_mag mode).
  - The 81 pinned-only keys (phase spacing, influence, progressive DT, track walks, unverified patches) cannot be set.
  - The unattached-strip DT term starts at loss_start_patch_dt = 25000 in both versions; no change is needed.
  - Every other override is as in R1.
- **(c)** is R1's configuration minus the chains, so R1−(c) is a clean HEAD-defaults pair. The two pairs differ by the
  tracks and the 3 restored keys together; the A1 gains are compared, not attributed.
- **Tracks** (a, b): built once, on the (a) pod.
  - Extracted with the PINNED extractor, exactly as run_arm.sh did for the pinned c1667 arms (`spiral/extract_tracks.py`,
    mask m7 L2 th0.2, z 6244-8256, 12 workers, pinned env from env/pod_setup.sh FIT_ONLY=1 on that pod only).
  - The packed store and crossings are then rebuilt with HEAD's `convert_track_store.py --force` and
    `build_track_crossings.py --force` (the store version is 1 in both).
  - The same files are pulled to `data/scratch/head_port/tracks_m7L2/` and pushed to the (b) pod, so (a) and (b) use byte-identical
    tracks.
  - Load check: HEAD reports the packed store's track count in the ROI. Pinned: 201,288 tracks, 16,761,300 points.
    Report it; a large difference is a deviation to explain.
- **Scripts (new):**
  - pod/head_arm2.sh 8a4bccf624c08e43: head_arm.sh generalised, plus PYTHONUNBUFFERED;
  - pod/head_tracks.sh 203bf83fcf705aa2;
  - pods/boot_head2.sh 20ce3a721d00f3f8;
  - the overrides a568a7f8868a96f0 / 0fdb64844a23ecf3 / e178bd6cf42c2a63;
  - scorer data/scratch/head_port/head_gain.py fc8058c0a507bea8.
  - head_setup.sh is unchanged from D4.
- **Pods:** vss-head-trk (a), vss-head-btrk (b), vss-head-bnot (c).
  - Created through grab_rest.py; A6000/L40S/RTX6000Ada at ≤ $1.0/h.
  - One watchdog each, deadline launch + 9 h. Terminated after the pull.
  - Expected spend: about 3 × 1.6 h × $0.54 ≈ $3. Hard stop: if spend reaches $30, stop launching and report.

### A1.2 Endpoints (scorer `head_gain.py`; vertex dumps tol 4 as in R1)
On inner (k < 34) vertices, own-offset, paired BootV, for each arm and each pair (chain − base), with 95 % CI:
- **primary:** wrong area among matched vertices;
- **switch:** seam switches per 1000;
- **matched:** matched fraction of ALL inner vertices;
- **correct:** absolute-correct fraction, i.e. matched AND right winding, over ALL inner vertices. Coverage cannot game
  this one.

**Pinned reference, measured now from stored dumps** (`data/scratch/head_port/gain_pinned_ref.json` a8a63331bfbd5941).
It is the chain gain on 0fb38c45, m2s1/m2s2 vs the 3 base seeds:

| pinned | primary Δ (pp) | switch Δ /1000 | matched Δ (pp) | correct Δ (pp) |
|---|---|---|---|---|
| m2s1 − base / bases2 / bases3 | −28.16 / −27.60 / −29.32 | −63.3 / −63.2 / −65.9 | −0.13 / −0.21 / +1.04 | +20.15 / +19.73 / +21.22 |
| m2s2 − the same | −27.98 / −27.43 / −29.15 | −62.9 / −62.8 / −65.5 | −0.08 / −0.16 / +1.09 | +20.05 / +19.64 / +21.12 |
| **mean of the 6** | **−28.27** | **−63.9** | **+0.26** | **+20.32** |
| base seed noise (bases2 − base) | −0.55 [−0.74, −0.37] | −0.06 | +0.07 | +0.41 |

Pinned levels: m2s1 50.74 % wrong, 71.65 % matched, 35.30 % correct; base 78.89 % / 71.78 % / 15.15 %. The CI on
each pinned pair is about ±1.3 pp (primary) and ±0.9 pp (correct).

### A1.3 Verdict, per pair: (a)−(b) is the headline, R1−(c) secondary
- **HOLDS** iff all three:
  1. primary Δ ≤ −22.6 pp (80 % of the pinned mean −28.27, test 13's keep rule) with CI upper < 0;
  2. correct Δ ≥ +16.3 pp (80 % of +20.32) with CI lower > 0;
  3. matched Δ ≥ −5 pp (the chains do not buy the gain by losing coverage).
- **HOLDS, REDUCED:** primary CI upper < 0 and correct CI lower > 0, but the HOLDS conditions fail.
- **NOT HELD:** otherwise.
- **Always reported:**
  - levels of all 4 HEAD arms next to pinned;
  - switch Δ next to pinned's −63.9;
  - coverage with tracks: HEAD base_trk vs pinned base (matched and correct levels), and m2s1_trk vs pinned m2s1
    (primary Δ, paired, as in R1, with head_compare.py) as information;
  - tol-8 diagnostic dumps if the matched fraction is again ≥ 10 pp below pinned.
- **Limits:** one seed per HEAD arm (pinned seed noise on the gain is about ±0.9 pp); the HEAD-vs-pinned differences
  bundle all upstream changes.

### A1.4 Predictions (before the fits)
- (a)−(b) HOLDS: P ≈ 0.6. REDUCED: 0.3. NOT HELD: 0.1.
- Tracks ON restore coverage, i.e. base_trk matched within 5 pp of pinned base (71.8 %): P ≈ 0.6.
- R1−(c) HOLDS: P ≈ 0.4. At low coverage the correct Δ is capped by the matched share.

### D5 (2026-10-26T06:30Z): correction to D3/D4 (process bookkeeping, no effect on R1)
The boot relaunch after D3 had not died. `setsid` forks, so the pid I recorded was its parent's. That run carried on as
an orphan, and attempt 3 started while it was still running, giving two overlapping head_setup runs. The likely cause of
the interrupted dpkg in D4 is that overlap, not the kill. R1 is unaffected: attempt 4 redid the setup from a clean state
(env check plus 54/56 tests) before the fit. From A1 on, boot scripts run as tracked background tasks, and watchdog
pids are read from `ps`.

### D6 (2026-10-26T06:42Z): one pod instead of three
grab_rest found 0 offers of any type for vss-head-btrk / vss-head-bnot for 20 min; only vss-head-trk (A6000 48 GB)
was created. In R1 one fit used about 2.5 GB of GPU memory and about 50 % utilisation, so all three A1 arms run
concurrently on the trk pod:
- (c) starts now, during track extraction;
- (a) and (b) start once the tracks are built; (b) reads the same track files in place.
This changes wall time, not the arms. Settings, inputs and seeds are as frozen in A1.1. The tracks are still pulled to
`data/scratch/head_port/tracks_m7L2/` for the record. No other pods are created unless this one fails.

### D7 (2026-10-26T06:50Z): `input_use_outer_shell: false` for the tracks-ON arms
- **What happened:** (a) and (b) failed at load with no outer_shell/meta.json.
- **Cause:** HEAD's `input_use_outer_shell` defaults to true and resolves the conventional `<dataset>/outer_shell` path
  even when the folder does not exist. With tracks on, `filter_tracks_by_shell` then loads the shell to clip tracks to
  it.
- **The pinned fits had no shell for this band either.** Their tracks were not shell-filtered: the ROI held 201,288
  tracks, and the shell weights were 0. So `input_use_outer_shell: false` is the HEAD setting that reproduces pinned.
  In (c) and R1 the shell was never loaded (no tracks, shell weight 0), so their settings are equivalent without the
  key.
- **Action:** the key was added to overrides_m2s1_trk.json and overrides_base_trk.json (new shas appended to
  `freeze/a1.sha256`). The failed fit dirs were renamed `fit.failed_shell`, and both arms relaunched at about 06:50Z.
- **No fitting happened before the crash:** load stage only, so nothing was seen.

### A1 load checks (2026-10-26T06:55Z, no scores seen)
- **Tracks:** the pinned extractor gave 277,464 tracks over z 6244-8256 in 465 s (`tracks_m7L2/*.extract.json`; the
  published parameters as pinned). HEAD rebuilt the packed store and crossings (1,659,939 crossing events).
- **(a) and (b)** both report `201296 tracks, 16761424 points in ROI`. Pinned: 201,288 / 16,761,300, a difference of
  +8 tracks (+0.004 %). Extraction with the pool is not bit-exact, but it is the same track set to 4 decimals. (a) and
  (b) read the same files.
- **Chains:** (a) has 577 collections / 18,670 points; (b) and (c) have no unattached groups, as intended.
- **Track files kept** in `data/scratch/head_port/tracks_m7L2/` (689 MB).

### D8 (2026-10-26T08:05Z): arm (c) exported no windings; re-export with HEAD's own winding cap
- **What happened:** (c) c1667_base_notrk finished cleanly (4,478 s, no traceback) but wrote **0 windings**
  (`save_mesh fitted: winding range [10, 10)`).
- **Cause:** HEAD derives the output winding range only from patches, unattached strips and tracks
  (`spiral_helpers.compute_winding_range_and_input_extents`). A fit with none of the three gets the empty range by
  design. R1 had the chains, so its range was [10, 65); pinned bases had tracks.
- **Fix, export only, model unchanged:** `pod/reexport_range.py` (sha e2420bce…) runs HEAD fit_spiral.py unchanged.
  - It resumes from (c)'s own checkpoint_fitted.ckpt: completed_iterations 30,000, so zero optimisation steps run.
  - The only substitution: when the data-derived range is empty, use [10, 130). 130 is HEAD's export cap
    (`shell_outer_winding_idx`; save_mesh clips to it) and equals the pinned base exports: 120 windings, 10-129.
  - Output goes to `c1667_base_notrk/fit_reexport/`. The scored meshes are this re-export.
- **What it means for the R1−(c) pair:** R1's export range [10, 65) is narrower than (c)'s [10, 130). Vertices that only
  a winding ≥ 65 could match count as unmatched for R1 but can match for (c), which favours (c) on matched/correct.
  - Reported as is. A check restricted to the label windings R1's range covers is added if matched differs by > 5 pp.
  - (a) and (b) are unaffected: both have tracks, so their range comes from data.

### R2 (2026-10-26T10:15Z): A1 RESULTS. The chain gain HOLDS on HEAD with tracks ON
**Run.**
- One pod, vss-head-trk: A6000 48 GB, massedcompute, $0.54/h (D6). Created 06:20:44Z, terminated 09:59:34Z (re-listed
  clean; watchdog stopped). **A1 spend 3.65 h × $0.54 = $1.97. Total for the note, R1 + A1: $2.73** of $40.
- **Fit times** (3 fits sharing the GPU, so not comparable to R1's single-fit time):
  - (a) 11,469 s; (b) 11,427 s;
  - (c) 4,478 s, plus the 106 s re-export of D8.
- **No traceback in any arm.** 150 step lines each; track terms non-zero in (a) and (b) (track_radius 52.6 / 87.0 at
  29.8k; pinned base 89.4).
- **Exports:**
  - (a) winding range [10, 77), 67 windings;
  - (b) [10, 95), 85 windings;
  - (c) [10, 130) by D8, 120 windings.
- **Satisfaction:**
  - (a): chains 483/577 (83.7 %), 17,940 points (96.1 %); tracks 45.1 %, track points 76.3 %;
  - (b): tracks 46.3 %, track points 76.3 %;
  - pinned base: tracks 31.2 %, track points 66.7 %.
- **Files:**
  - meshes and logs in `data/scratch/head_port/<arm>/`;
  - dumps in `vertex/` (shas: m2s1_trk 53d73598…, base_trk ea6329b5…, base_notrk 4b39e3f9…);
  - scores in `gain_a1.json` (4c1c1ca2…), `gain_notrk.json` (a06ce33b…) and `result_trk.json` (833c05ad…).

**Levels** (inner k < 34, tol 4):

| arm | wrong area (among matched) | switches /1000 | matched | absolute-correct |
|---|---|---|---|---|
| pinned c1667_m2s1 | 50.74 % | 73.86 | 71.65 % | 35.30 % |
| pinned c1667_base / bases2 / bases3 | 78.89 / 78.34 / 80.06 % | 137.2 / 137.1 / 139.7 | 71.78 / 71.86 / 70.61 % | 15.15 / 15.56 / 14.08 % |
| **HEAD (a) m2s1_trk** | **52.52 %** | **55.02** | **73.51 %** | **34.90 %** |
| **HEAD (b) base_trk** | **80.24 %** | **88.60** | **78.22 %** | **15.46 %** |
| HEAD R1 m2s1_head (tracks off) | 46.62 % | 68.37 | 48.78 % | 26.04 % |
| HEAD (c) base_notrk (tracks off) | 86.06 % | 100.35 | 66.09 % | 9.21 % |

**Chain gain (chain − base, paired, 95 % CI) next to pinned:**

| pair | primary Δ (pp) | switch Δ /1000 | matched Δ (pp) | correct Δ (pp) | verdict (A1.3) |
|---|---|---|---|---|---|
| pinned (mean of 6 pairs; range) | −28.27 (−27.4…−29.3) | −63.9 | +0.26 (−0.2…+1.1) | +20.32 (19.6…21.2) | |
| **HEAD tracks ON, (a) − (b)** | **−27.71 [−28.94, −26.40]** | **−33.58 [−39.11, −27.59]** | **−4.70 [−5.35, −4.04]** | **+19.44 [+18.40, +20.47]** | **HOLDS** |
| HEAD tracks off, R1 − (c) | −39.44 [−43.33, −35.84] | −31.98 [−40.81, −23.60] | −17.31 [−18.51, −15.84] | +16.82 [+14.88, +19.09] | **HOLDS, REDUCED** (fails matched ≥ −5) |

- **(a) − (b) HOLDS: all three conditions pass.**
  - Primary −27.71 ≤ −22.6, upper CI < 0.
  - Correct +19.44 ≥ +16.3, lower CI > 0.
  - Matched −4.70 ≥ −5, but only just. The matched gap comes from the HEAD base *gaining* coverage (78.2 % vs pinned
    71.8 %), not from the chain arm losing it (73.5 % vs pinned 71.7 %).
  - **In size the HEAD gain is 98 % of the pinned gain on the primary and 96 % on absolute-correct.** Both lie at the
    edge of the pinned seed range.
- **The switch gain is smaller in size (−33.6 vs −63.9) because HEAD's base switches far less** (88.6 vs about 138).
  The HEAD chain arm itself switches less than pinned: 55.0 vs 73.9.
- **R1 − (c), tracks off: REDUCED.** The chains still add +16.8 pp absolute-correct, but they cost 17 pp of coverage.
  The D8 export-range asymmetry does not explain the coverage loss: at R1's own offsets (5-15), inner labels k 10-33
  sit at fit windings 15-48, inside R1's [10, 65). The −39 pp primary is inflated by that coverage loss and should not
  be quoted as a gain.

**HEAD tracks ON vs pinned, same arm** (information; R1's equivalence rule applied for comparison):
- (a) − pinned m2s1: primary **+1.79 [+0.95, +2.61]**, switch −18.84 [−23.37, −14.33], matched +1.86, correct
  **−0.40 [−1.09, +0.35]**.
  - Strictly "not shown equivalent": the upper CI of 2.61 is above 2.0, and the switch is outside ±5, in HEAD's favour.
  - On absolute-correct, the coverage-proof measure, HEAD-with-tracks reproduces pinned within ±1 pp. The global offset
    is the same (15) and unmatched is 32.2 % vs 32.1 %.
- **(b) − pinned bases:** primary +1.34 / +1.90 / +0.17, matched +6.4 to +7.6, correct +0.31 / −0.10 / +1.38, switch
  about −49.
- **Tracks restore coverage.** Matched: (a) 73.5 % vs R1 48.8 %, a gain of +24.7 [22.9, 26.4]; (b) 78.2 % vs (c)
  66.1 %, a gain of +12.1.
  - The pre-registered prediction "base_trk within 5 pp of pinned base" missed on the high side: +6.4 pp above it.
- **So the R1 coverage loss was the missing tracks** (with the 3 restored keys; not separated).

**Prediction check:**
- (a) − (b) HOLDS: P 0.6 → held.
- Coverage restored: P 0.6 → yes, overshooting.
- R1 − (c) HOLDS: P 0.4 → REDUCED.

**For the submission:**
1. The chain gain carries over to current villa main **when tracks are on** (`input_use_tracks: true`, pinned-extracted
   tracks, `input_use_outer_shell: false` when there is no shell). It is about the same size as pinned on primary and
   absolute-correct, with fewer switches.
2. Do not run our pipeline on HEAD with its default tracks-off setting.
3. HEAD with no patches, chains or tracks exports nothing (D8).
4. One seed per HEAD arm. The 3 restored keys and the tracks are bundled.
