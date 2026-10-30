# 2026-10-29: KP chains: per-ray linked chains from Kelly-numbered Stevens patches on PHerc1667. Pre-registration

Status: **PREPARED, NOT RUN.** Chain sets, envs, scorer, stages and decision rules are frozen at the sha in log entry
F0. **No fit runs before the user's go-ahead** (logged as an entry below). Run directory: `data/scratch/kp_chains/`.

## 0. Question
The structure ablation (notes/2026-10-28-structure-ablation-prereg.md, R1) found the oracle chains' fit gain is carried
by linkage (rule R2: Link 0.558; thinned + linked A2 G 1.04). Its next step was patch-derived chains on PHerc1667 with
Stevens' patches. Kelly's wrap index (community/wrap_index.csv) numbers those patches. Pooled over all 7 bands it scores
C/A/Q 71.9/95.2/68.5, against our M2 fits' 74.2/65.6/48.7 (notes/2026-10-27-kelly-headtohead-prereg.md).

Two questions:
- **C1:** does our fitter, fed per-ray linked chains built from Kelly's index (KP), or KP plus our M2 chains (KPM),
  beat Kelly's own index on Q?
- **C2:** short of that, does it beat our M2 fits, on Q and on inner own-offset wrong area?

## 1. What exists before this freeze (disclosed)
- **Labels already viewed.** All 7 whole-scroll bands' label metrics for base, M2 (s1, s2) and Kelly (k_all_comp,
  k_good_comp) are published in the h2h and whole-scroll notes. The B6 (z 6500-8000) oracle-chain fit is known: inner
  own-offset wrong 46.3 % vs base 78.9 % and M2 50.7 %. Nothing was viewed for Paris4 z 8500-9000.
- **No KP label metric.** No fit with a KP or KPM set exists, and no label, mesh or label centre was read to build or
  describe them. The only scorer run was a smoke test with stand-ins (kp := the M2 dump, kpm := the base dump). It
  reproduced the published pooled rows exactly (M2 74.243/65.622/48.72; Kelly 71.918/95.246/68.499; ΔQ M2 − Kelly
  −19.78 [−21.15, −18.44]). Outputs: `data/scratch/kp_chains/smoke/`.
- **Iteration on the construction, label-free only.** The first transport rule referenced each ray to a circular-mean
  Kelly angle. Its transport deviation p99 was 60-150°, because Kelly's axis sits 50-516 L2 vox from our umbilicus.
  It was replaced by the far-field reference plus the parallax rule R2b (§2). This was decided from those label-free
  diagnostics and the R3 conflict counts. KP-vs-M2 step agreement (an inter-method number, §5) was seen while building.
- **Circularity check (brief item 1): not circular at the input level; design-level exposure disclosed.**
  - Kelly's solve inputs are the patches, the CT, a 134 µm radial pitch, the Stevens same-wrap rule, and an axis made
    of patch centroids. In Kelly's words: "The solve is reference-free: no published segment or reference mesh enters
    it" (community/README.md:101; docs/CLAIMS.md A1; phase/tools/CONTRACT.md:14). Repo commit c55161d5. Our
    wrap_index.csv sha 1c0b7d30… equals the SRC_SHA in Kelly's build_s4_index.py.
  - Kelly's axis file reproduces the thN column label-free (median 0.11°, p99 2.8°). It is a centroid line, not a mesh
    product.
  - Kelly's development *reference* is the same human mesh as our labels: 20260612121456-w011_20260108140509268_
    merged_v4_flatboi_straightened_v4 (CONTRACT.md:38; data/segments_1667.txt:6; fixture provenance.json:1763; our
    eval/p1667_labels.py:6, rendered on the 2.399 µm volume). Kelly scored and tuned against it on:
    - slab 2 (7.91 µm z 4096-4864 ≈ L2 z 5380-6010, inside B4 and touching B5);
    - HO-A (location not in the release);
    - the switch-risk checker (CLAIMS B4/B5);
    - the fit anchors (CLAIMS B1, taken from the reference's evaluated point pairs).
    So method choices were exposed to our label source, but no label values enter the index.
  - Stevens' patches are grown from a surface-prediction OME zarr and the CT (pipeline9 readme, steps 2-3). The bad
    set comes from badpatchfinder ('c') plus an optional manualBadPatch.csv ('v', 'n'). No mesh is an input.
    Residual unknown: the training data of the surface-prediction model.
  - Ours: the umbilicus (the ray origin) is derived from the human mesh, as for every earlier chain set.
  - Verdict: proceed, with the exposure disclosed. B4 holds most of slab 2 and is a confirmation band, which favours
    Kelly, not KP, on C1.

## 2. Chain sets (frozen; `data/scratch/kp_chains/{crossings,build_sets,check_load}.py`)
**Rays.** Exactly the M2 chain rays (tools/auto_windings.py probes(), M2 recipe):
- z = zc + {−64, 0, +64} in [z0, z1), zc = 192c + 96;
- θ = 0, 15, …, 345°;
- origin = np.interp (clamped) of umbilicus_p1667_L2.json; point (cx + r cos θ, cy + r sin θ, z).

The grid matches each band's M2 set, except that B1's M2 set has no z 480 rays (KP B1 has no z 416 or z 480 piece).

**Crossings** (`crossings.py`; the zips are read in memory with tifffile, and nothing is executed):
- Only Kelly component-0 patches are used (56,926; good + bad zips).
- Patches are mapped to L2 by the patch-numbering affine (0599873e…), then /4.
- Surface: valid 2×2 cells, with spiral_eval's degenerate-quad filter (radius ≤ 4 × the median). Each cell is split
  into 2 triangles.
- A crossing is the intersection of the ray with a triangle cut at plane z. For each (ray, patch) the script records
  the median r, the spread and the hit count.
- Crossing points lie at median 1.39 vox (max 2.30) from the nearest patch vertex. A rerun is byte-identical.

**Rules per ray** (`build_sets.py`), TAU = 0.59 × 134 µm / 9.596 µm = 8.239 vox (the Stevens/Kelly same-wrap
separation):
- **R1 self-consistency:** a patch whose hits on the ray spread > TAU is dropped on that ray.
- **R2a transport:** θ_K(c) is Kelly's angle about Kelly's axis at the crossing (mapped back to 7.91 µm), and
  d = wrap(θ_K(c) − thN). Drop the crossing if |d| > 90°.
- **R2b parallax:** φ is Kelly's angle of the ray direction (far field, from the affine's linear part). Drop the
  crossing if |wrap(θ_K(c) − φ)| > 45°. Winding: t = k + (thN + d)/2π, then w = rint(t − φ/2π).
- **R3 same winding:** crossings with equal w and radial spread ≤ TAU become one point at their median r. If the
  spread is > TAU, every crossing of that w on the ray is dropped.
- **R4 collision:** two points with different w closer than TAU in r: both are dropped.
- **R5 monotone:** every adjacent pair whose w does not rise is dropped (both points), repeated until none is left.
- **R6:** a ray with ≥ 2 points becomes one collection kp_z<z>_t<θ>_1. Point ids ascend outward, wind_a = w − min w,
  and gaps keep Kelly's step count.

**KPM** = KP + the band's M2 set (sha-checked). On a ray with a KP piece spanning [r_lo, r_hi], M2 points with r in
[r_lo − TAU, r_hi + TAU] are removed and the M2 piece is split there. Sub-pieces with ≥ 2 points are kept as
m2_z<z>_t<θ>_<i> (wind_a re-based). Rays without a KP piece keep their M2 pieces whole.

**Load check** (`check_load.py`, output check_load.json): all 14 sets pass. Each set loads through villa 0fb38c45's
load_point_collection / normalise_pcl_winding_annotations, the z-roi rule (margin 0) and check_doc (role relative).
Counts match the manifest; winding annotations are all present, finite and non-decreasing. The only warning is
metadata.winding_is_absolute missing, the same as for every M2 and SA set.

| band | z | M2 set (sha) | KP set | KPM set |
|---|---|---|---|---|
| B1 | 400-1500 | wholescroll p1667z400_m7_ray_c (867e7e0d…) | kp_z400 | kpm_z400 |
| B2 | 1500-3000 | wholescroll p1667z1500_m7_ray_c (de6b44e1…) | kp_z1500 | kpm_z1500 |
| B3 | 3000-4500 | wholescroll p1667z3000_m7_ray_c (0bd2cc58…) | kp_z3000 | kpm_z3000 |
| B4 | 4500-6000 | ink_lowband p1667z45_m7_ray_c (0fc2b0cb…) | kp_z4500 | kpm_z4500 |
| B5 | 6000-6500 | wholescroll p1667z6000_m7_ray_c (fca951d4…) | kp_z6000 | kpm_z6000 |
| B6 | 6500-8000 | confirm_region p1667_m7_ray_c (85dcbfb4…) | kp_z6500 | kpm_z6500 |
| B7 | 8000-9000 | wholescroll p1667z8000_m7_ray_c (82dbf46f…) | kp_z8000 | kpm_z8000 |

Sets: `data/scratch/kp_chains/sets/<name>/relative_windings.json`; shas in sets/manifest.json and freeze/.

## 3. Arms (`spiral/arms/{kp,kpm}_z<z0>.env`)
Each env is a copy of the band's M2 s1 env (wsf<z0>_m2s1; ink2_m2s1 for B4; c1667_m2s1 for B6). Only ARM and
REL_WINDINGS (`$HOME/data/PHerc1667/windings/<set>/relative_windings.json`) change, plus the comment line. Recipe:
weight 10, spacing 0, seed 1, mask m7L2, same villa, recipe and export as M2.

## 4. Scorer, stages and decision rules (fixed now; `data/scratch/kp_chains/score_kp.py`)
The two frozen scorers run unchanged; only their dump look-ups are redirected through module attributes:
- eval/kelly_h2h.py (bdfb5641…, variant primary; lenient and tol2 as sensitivity): C, A, Q and paired BootV CIs for
  each arm vs k_all_comp and vs M2, on the stage's bands (all flagged fresh, so the "all7" pool is the stage pool).
- eval/wholescroll_pool.py (dfdb1f57…): the arm vs M2 s1 on inner own-offset seam wrong area, switches, outer and
  unmatched, pooled over the stage's bands.

Dumps: `data/scratch/kp_chains/vertex/{kp,kpm}_z<z0>.npz`. Every scorer run is
`flock /tmp/vss_eval.lock uv run python data/scratch/kp_chains/score_kp.py --stage <dev|conf> --arms <…> --out …`.

**Dev stage (B3 z 3000-4500, B5 z 6000-6500; 4 fits: KP and KPM on each).** Both bands are already burned for M2
and Kelly. Both have full KP ray coverage and are outside Kelly's slab 2, and B5 is cheap (500 slices).
- Choose the arm with the higher pooled Q. If |Q_KP − Q_KPM| < 0.5 pp, choose KP (the simpler arm).
- **Go to confirmation** only if Q_chosen > Q_M2 on the dev pool (point estimate). Otherwise stop. The dev label
  metrics are then reported, and nothing is fitted on the other bands.
- A fit that crashes gets one rerun. If it fails again, the arm is missing: if both arms are missing, stop; if one is,
  the other is chosen.

**Confirmation stage (the chosen arm on B1, B2, B4, B6, B7; 5 fits)**, pooled over those 5 bands:
- **C1, beats Kelly:** lower 95 % bound of ΔQ (arm − k_all_comp) > 0. The reverse (upper bound < 0) is reported as
  "Kelly's index beats the fitted arm".
- **C2, improves the fitter:** lower bound of ΔQ (arm − M2) > 0, **and** upper bound of Δ inner own-offset wrong area
  (arm − M2) < 0, **and** switch Δ ≤ +5 pp (point estimate).
- **Reported, not gated:** co-covered ΔA vs Kelly (S1: lower bound > −1 pp), union Q with Kelly, per-band rows,
  outer and unmatched Δ, the lenient and tol2 variants. After confirmation, a descriptive all-7 pool of the chosen
  arm (`--stage all`); no rule rests on it.
- A rule that passes on primary but fails on lenient or tol2 is flagged as fragile, not reversed.

## 5. Label-free stats (`stats.py` → stats.json; full counts in sets/manifest.json)
| band | rays | ray cov KP | KP pieces / pts | med len | med wind span | dw 1/2/3+ | KP–M2 step agree | KPM cols / pts | M2 pts |
|---|---|---|---|---|---|---|---|---|---|
| B1 z400 | 408 | 0.735 | 300 / 4,186 | 15 | 18 | 0.82/0.12/0.06 | 0.838 | 622 / 5,837 | 6,728 |
| B2 z1500 | 576 | 1.0 | 576 / 11,648 | 20 | 22 | 0.88/0.08/0.04 | 0.914 | 1,302 / 15,049 | 17,116 |
| B3 z3000 | 552 | 1.0 | 552 / 11,877 | 21 | 24 | 0.91/0.05/0.04 | 0.937 | 1,197 / 14,805 | 15,798 |
| B4 z4500 | 576 | 1.0 | 576 / 14,683 | 25 | 27 | 0.92/0.05/0.03 | 0.954 | 1,308 / 17,131 | 18,531 |
| B5 z6000 | 192 | 1.0 | 192 / 5,150 | 27 | 28 | 0.92/0.06/0.03 | 0.950 | 445 / 5,934 | 6,523 |
| B6 z6500 | 552 | 1.0 | 552 / 14,146 | 26 | 28 | 0.92/0.05/0.03 | 0.926 | 1,259 / 16,713 | 18,670 |
| B7 z8000 | 384 | 0.662 | 254 / 4,133 | 16.5 | 18 | 0.90/0.06/0.03 | 0.859 | 642 / 7,287 | 7,962 |

- Conflict drops per band (B1…B7):
  - R2b parallax crossings: 1,627 / 968 / 2,556 / 1,733 / 341 / 1,912 / 2,792 (23 % of B7's crossings);
  - R3 groups: 563 / 698 / 375 / 516 / 254 / 510 / 61;
  - R4 points: 107 / 561 / 665 / 620 / 143 / 331 / 57;
  - R5 points: 10 / 18 / 2 / 16 / 14 / 54 / 0;
  - R1: 5-85 per band.
- 99.8-100 % of adjacent pairs rise outward before R5.
- About 20 % of points rest on a single patch. Points resting only on bad-zip patches: 140-861 per band.
- Gap-ratio share outside [0.5, 2]: 15-26 %. Spacing varies too much along a ray for this to be diagnostic.
- The KP–M2 step agreement is inter-method (no labels): among consecutive KP points whose M2 partners
  (|Δr| ≤ TAU/2) sit in one M2 piece, the share where the step counts agree.

## 6. Prediction (written before any fit)
- **Dev:** go to confirmation (P ≈ 0.75). KP is chosen over KPM (P ≈ 0.6). On B3/B5, KP covers every ray, so KPM
  adds only 2,928 and 784 M2 points, and the tie rule likely applies.
- **Confirmation:** the arm beats M2 on Q by +5 to +15 pp and on inner wrong area. **P(C2) ≈ 0.55.** It stays below
  Kelly on Q by 5-15 pp: **P(C1 beats Kelly) ≈ 0.10**, P(Kelly beats the arm) ≈ 0.7.
- Reasons:
  - the fitter's ceiling on B6 (oracle chains still leave 46.3 % inner wrong);
  - Kelly's own report that a fit copies index errors instead of repairing them (CLAIMS D1);
  - KP coverage gaps on B1 and B7 (0.66-0.74 of rays).
- Low confidence: the structure ablation came in above prediction (A2 G 1.04 vs a predicted 0.7-0.9).

## 7. Cost and run plan (after the go-ahead only)
- **Local (default, $0):** one fit at a time on the RTX 3090, only once vss-ws-local has finished (its unit and lease
  are not touched).
  - Each fit runs under its own GPU lease (session vss-kp-local, max_mem_gb 21) with a 19,000 MiB guard.
  - Fit times, from earlier runs: 500 slices ≈ 3.3 h; 1,500 slices 4.2-5.3 h.
  - Dev: 4 fits ≈ 15-17 h. Confirmation: 5 fits ≈ 21-26 h. **Total ≈ 36-43 GPU-h**, plus ~15 min of scoring per
    stage.
- **Runner:** data/scratch/wholescroll/local_ws.sh (per-band HOME, set sha check, lease, guard, vertex dump),
  changed only in ARMS, env and set shas, HOME (data/scratch/kp_chains/local/home_z<z0>), dump dir and lease name.
  It is written at go-ahead, its sha is logged before the first fit, and it is launched with
  `systemd-run --user --unit=vss-kp-local …`. Fit order: dev B5 KP, B5 KPM, B3 KP, B3 KPM; then score dev; then
  confirmation B4, B6, B2, B1, B7.
- **Pods (only with the user's approval):** ≤ $2.50/h, names vss-kp-*, terminated right after pull. At earlier rates
  ($0.54-1.81/h, 4-6 h per fit) that is ≈ $2-11 per fit. **Cap $40** for all 9; check in before exceeding it.

## 8. Disclosures and limits
- **Fitter ceiling.** On B6 the oracle chains reach 46.3 % inner wrong (M2 50.7 %). If B6 is typical, chain quality
  cannot carry the fit to Kelly's A ≈ 95 %. C1 is a long shot by design.
- **Design-level exposure of Kelly's method to merged_v4** (§1), concentrated on B4. It is not separable here.
  Per-band rows are reported.
- **KP inherits Kelly's errors.** The fitter sees them as linked "truth". KPM does not resolve conflicts between KP
  and M2; it only fills outside the KP span.
- **Dev cannot test KPM's main use** (rays and bands with KP gaps: B1, B7). KPM's dev result is therefore weak
  evidence about KPM on those bands.
- **One seed per arm.** On Paris4, M2 seeds differed by 0.5 pp of wrong area.
- **Thin support.** ~20 % of points rest on one patch; bad-zip patches are included, as in Kelly's k_all_comp.
- **Only Kelly component 0** (relative numbering holds only within a component).
- **Data licence.** Patches and the Kelly index are CC BY-NC. Every derived file stays local.

## 9. Deviations from the brief (decided before freeze)
- KPM env files are named `kpm_z<z0>.env`, not `kp_*`. Arms are named kp_z<z0> / kpm_z<z0>.
- R2b (parallax ≤ 45°) was added, because Kelly's seam angle is measured about a centroid axis, not our umbilicus.
- Dev pass is a point-estimate gate (Q_chosen > Q_M2). The CI rules apply only at confirmation.
- The h2h wrapper drops m2s2 from its method list (stage pools use M2 s1 only, matching the M2 env each KP env copies).

## 10. Frozen files
- crossings.py, build_sets.py, stats.py, check_load.py, score_kp.py; the 7 crossings npz; the 14 sets, the manifest,
  stats.json and check_load.json; the 14 KP/KPM envs and their 7 M2 templates; the M2 sets, base and M2 dumps;
  Kelly inputs; the frozen scorers and fit stack; and this note up to the Log line. Full list and shas:
  `data/scratch/kp_chains/freeze/freeze.sha256`.
- Never modified: eval/kelly_h2h.py, eval/wholescroll_pool.py, data/scratch/patch_numbering/score.py,
  tools/auto_windings.py, eval/p0139_graph.py, tools/autowind/postproc.py, tools/check_viewer*,
  data/scratch/surf_gate/measure.py, data/scratch/anneal/conf_table_an.py, data/scratch/structure_ablation/score_sa.py,
  and every file frozen in an earlier prereg note.
- No commits, pushes, posts or uploads.

---

## Log (append-only)
### F0 (2026-10-28T19:49:57Z): frozen
- **sha256 of this file** up to and including the "## Log (append-only)" line (lines 1-218):
  972a12abc7d8441d51d8c9cf2897b6bc2ed92ed6e57ad5727d048a22da770b53.
- Snapshot and the full sha list (104 files: 14 sets, manifest, stats, load check, 14 KP/KPM envs + 7 M2 templates,
  5 scripts, 7 crossings npz, 7 M2 sets, 14 base/M2 dumps, Kelly h2h tables, Kelly/Stevens/affine/umbilicus inputs,
  labels npz, frozen scorers, fit stack, frozen tools): `data/scratch/kp_chains/freeze/` (prereg_snapshot.md,
  freeze.sha256 sha 044659fc24d15639535a1be7a8f83cb0e01ae763714b682b364c54a4627cdf80). `sha256sum -c` passes.
- **State at freeze:** sets built; load check OK on all 14; stand-in smoke test reproduces the published h2h rows.
  No kp_*/kpm_* fit, pod or dump exists, and no KP label metric has been computed. vss-ws-local holds the local GPU.
  **Fits wait for the user's go-ahead.**

### G1 (2026-10-28T19:55Z): go-ahead, dev stage only
- The user approved **"Dev stage only"**: the 4 dev fits (kp_z6000, kpm_z6000, kp_z3000, kpm_z3000), the dev scoring
  and the dev decision. No confirmation fit is started even if the dev rule says go; that needs a new go-ahead.
  No pods (pods were approved only for the full test, which was not chosen).

### R1 (2026-10-28T19:56Z): dev runner written and launched
- Runner `data/scratch/kp_chains/kp_local.sh`, sha256 d771c4722bdf889f043491d0d3c96d51c7faffa857234f4b4a80236810bf6a0e.
  Adapted from data/scratch/wholescroll/local_ws.sh (per-band HOME, set sha check, lease, guard, local pull, vertex
  dump); changed in ARMS (fit order kp_z6000, kpm_z6000, kp_z3000, kpm_z3000), env and set shas (read from
  freeze/freeze.sha256), HOME (data/scratch/kp_chains/local/home_z<z0>, inputs and m7 mask copied from
  data/p1667_ws/z<z0>), dump dir data/scratch/kp_chains/vertex, lease ~/.gpu-leases/vss-kp-local.json (session
  vss-kp-local, max_mem_gb 21, one lease per fit). Patterns taken from data/scratch/structure_ablation/sa_local.sh:
  GPU-busy = fitter-venv process or > 1 GiB only; villa check by absolute patch paths; guard 19,000 MiB; claims
  (data/scratch/kp_chains/claims/); resumable.
- **Villa tree** (as the wsf*_m2s1 arms: villa 0fb38c45 + fit_spiral_shell_optional.patch only, none of
  spiral/patches applied; wholescroll prereg lines 338-364): data/scratch/anneal/local_probe/ext/villa,
  checked before and after every fit. At launch: HEAD 0fb38c45b488…, the only modified file is
  volume-cartographer/scripts/spiral/fit_spiral.py, diff sha 75c27ecb… (= shell-optional line), no patch reverse-applies,
  chain_loss_tol.patch forward-applies. No env sets VILLA_PATCH.
- **Gate** (re-checked before each fit, every 120 s): all arms in local_ws.sh's ARMS line have dumps in
  data/scratch/wholescroll/vertex/, vss-ws-local inactive, no lease other than ours, no fitter-venv / > 1 GiB GPU process.
  At launch vss-ws-local runs wsf6000_m2s2; missing ws dumps: wsf6000_m2s2 wsf400_bases2 wsf400_m2s2 wsf1500_m2s2
  wsf3000_m2s2. vss-ws-local and the ws pod units are not touched.
- Fits run without the eval lock (as local_ws.sh); dumps (`release/fit_vertex_dump.py --scroll p1667 --fit
  data/scratch/p0139_spiral/<arm>/meshes_step4 --name <arm> --z0 <z0> --z1 <z1> --out-dir data/scratch/kp_chains/vertex`)
  and the scorer run under flock /tmp/vss_eval.lock. Meshes are kept (score_kp.py reads them for the pool scorer).
- Crash handling per section 4: one rerun of the same command (run_arm.sh resumes from its autosave); a second crash
  marks the arm missing. Guard stops and any check failure halt the runner.
- After the 4 dumps: freeze check, then once: `flock /tmp/vss_eval.lock uv run python
  data/scratch/kp_chains/score_kp.py --stage dev --arms kp,kpm --out data/scratch/kp_chains/dev/dev_table.json`
  (log dev/dev_table.log, decision dev/dev_table_decision.json).
- Launched as `systemd-run --user --unit=vss-kp-local --collect …`, log data/scratch/kp_chains/local/kp_local.log.

### L1 (2026-10-30T01:55Z): dev stage result. Rule says GO; confirmation waits for the user
- Fits (local RTX 3090, $0): kp_z6000 130 min, kpm_z6000 125 min, kp_z3000 and kpm_z3000 (241 min for kpm_z3000), peak
  process VRAM 6.9-9.1 GB, all attempt 1, no deviations. Scorer run once at 00:13:51Z
  (dev/dev_table.json, dev_table_decision.json, dev_table.log). Freeze check passed (lines 1-218 still 972a12ab…).
- Pooled dev (B3 + B5), primary variant, C / A / Q (%):

  | arm | C | A | Q |
  |---|---|---|---|
  | base (no chains) | 77.1 | 20.4 | 15.8 |
  | M2 s1 | | | 61.7 |
  | **KP** | 76.9 | 84.7 | **65.2** |
  | KPM | 75.1 | 72.8 | 54.7 |
  | Kelly k_all_comp | 73.7 | 98.0 | 72.3 |

- KP vs M2: ΔQ +3.47 [2.86, 4.06]; co-covered ΔA +4.0 [3.3, 4.7]; inner own-offset wrong area −4.81 [−5.26, −4.39];
  switches −0.6 [−3.7, +2.9]; outer +2.0 [0.8, 3.2]; unmatched −0.1.
- KP vs Kelly: ΔQ −7.10 [−8.69, −5.75]; co-covered ΔA −3.87 [−4.58, −3.20] (fails the reported S1 line, > −1).
  Union Q with Kelly 80.8 (reported only).
- KPM vs M2: ΔQ worse; inner wrong area +8.47 [7.62, 9.27]. Adding the M2 chains to the patch chains hurts.
- **Dev rule:** chosen KP (Q 65.2 vs 54.7). Q_KP 65.2 > Q_M2 61.7, so **go to confirmation**. As frozen (line 232),
  no confirmation fit starts without a new go-ahead from the user.
- Against the predictions: P(dev pass) 0.75 → passed; P(beats M2) 0.55 → dev points that way; P(beats Kelly) 0.10 →
  dev says no (−7.1 pp).

### G2 (2026-10-30T02:00Z): go-ahead, confirmation stage on pods
- The user approved the **confirmation stage on pods: "Pods in parallel"** (2026-10-30 ~02:00Z): the chosen arm KP on
  B1 z400, B2 z1500, B4 z4500, B6 z6500, B7 z8000 (5 fits, frozen envs kp_z<z0>.env and frozen sets), then the frozen
  scorer once (`--stage conf --arms kp`), then the descriptive all-7 pool (`--stage all`).
- Pod rules given with it: Prime Intellect, single GPU, >= 48 GB, x86_64, driver >= 580, <= $2.50/h, not spot, no
  GH200/GB200, no Lambda Labs; names vss-kp-<band>; terminated right after pull; aim <= $30, stop and report before
  projected spend passes $45. The local RTX 3090 may take at most one band (same stack, lease, guard).
- Rule update from the user (2026-10-30 ~01:55Z): a single-GPU, non-spot **H200 (<= ~$6/h) is allowed only as a
  fallback**, for a band that has no acceptable normal-priced offer when it is ready to launch; do not wait hours for
  cheap cards. Bands run on H200 are logged with the reason. GPU type per band is reported; mixed types are flagged as
  in the whole-scroll test.

### R2 (2026-10-30T02:00Z): confirmation run plan (pods + local), written before any confirmation fit
- **Bands → hardware.** B4 kp_z4500 on the local RTX 3090 (the one local band allowed; 1500 slices, the dev fits
  showed 6.9-9.1 GB peak). B6 kp_z6500, B2 kp_z1500, B1 kp_z400, B7 kp_z8000 on Prime pods vss-kp-b6/b2/b1/b7,
  one pod per band where offers allow; otherwise a finished pod takes the next band sequentially.
- **Market at 01:53Z:** zero normal single-GPU offers (no A6000 / L40S / A100 / RTX 6000 Ada / H100 at any price);
  the only single GPU on offer is a nebius H200 SXM (16 vCPU, 200 GB, $5.40/h + 1,500 GB minimum disk ≈ $5.55/h).
- **Offer rules (kp_coord.py):** normal = single GPU, >= 48 GB, <= $2.50/h, not spot, >= 6 vCPU, >= 40 GB RAM, not
  GH200/GB200/H200, not lambdalabs, not blacklisted; cheapest first. **H200 fallback:** single, non-spot, <= $6/h,
  only after 20 min of polling with no normal offer, at most one live H200. Budget gate before every create or reuse:
  closed spend + live pods' price x remaining estimate (5 h per band) + new price x 5 h <= $45. With one H200 band
  (~$28 projected) the gate leaves room only for normal cards for the rest; no second H200 band.
- **Stack on every pod** (checked by kp_boot.sh before launch and by kp_pull.sh after the fit): env/pod_setup.sh
  (FIT_ONLY=1) builds villa 0fb38c45 + pherc343 fit_spiral_shell_optional.patch (as the wsf*_m2s1 pods); HEAD
  0fb38c45b488…; no spiral/patches/*.patch reverse-applies; chain_loss_tol.patch forward-applies; the only
  modified file is fit_spiral.py; normalised diff sha (`git diff --full-index --src-prefix=a/ --dst-prefix=b/`)
  e7afc8c5… = the local tree's (the local tree's plain `git diff` sha 75c27ecb… depends on its mnemonic-prefix config,
  so the normalised form is compared). torch.cuda check; x86_64 and driver >= 580 host check. On the pod, the set,
  the env, run_arm.sh, export_fine.sh, fit_ckpt.py and recipe.toml are sha-checked against freeze/freeze.sha256.
  Overrides are checked after the pull (recipe.toml + band + env FIT_OVERRIDES; step 4 adds only output_step_size).
- **Inputs:** B1/B2/B7 from data/p1667_ws/z<z0> (as wsf*_m2s1), B4 from data/p1667_z4500 (as ink2_m2s1), B6 from
  data/p1667 (as c1667_m2s1); no file in them is newer than the matching M2 s1 fit's ds/band.json.
- **Deviation (layout only):** on pods each band lives under a per-band HOME ~/b<z0> (inputs, mask, set, work), as
  kp_local.sh did locally, so a pod can take a second band without mixing inputs. Recipe, env files, scripts and
  stack are unchanged.
- **Helpers (new files, data/scratch/kp_chains/pods/):**
  - kp_coord.py 289887843939001edb0ec556817428d797f857ace220b9e8ce8ffc8ad64ca73f: pod coordinator (offers, H200
    fallback, budget gate, create / adopt after a failed create, boot, poll, pull, vertex dump, reuse or terminate;
    crash → one rerun of the same command, second crash → missing). Uses data/scratch/wholescroll/pods/prime.py
    (6599588a…) read-only.
  - kp_boot.sh 12766fb70ec125b2e0dea1938688e8ee541affef2aeb24035e7956fa61addd4d: host check, copies, sha and villa
    checks, pod_setup, launch of `run_arm.sh <env> && export_fine.sh <env> 4`.
  - kp_pull.sh bfee0258a2dee64337c82176f3d27fe770aa07efa47d727593aad20534bd1463: pull_arm.sh's file selection from
    ~/b<z0>, winding count, overrides check, villa re-check, gpu.txt, pulled.ok.
  - kp_conf_local.sh de17002e76ce0363c5d76816bcdd1a67744157f53c99d3ac8e99f529e4d87ef9: kp_local.sh changed only in
    ARMS (kp_z4500), band check, input source (data/p1667_z4500), lease note and names, scorer section removed.
- **Dumps:** release/fit_vertex_dump.py → data/scratch/kp_chains/vertex/kp_z<z0>.npz under flock /tmp/vss_eval.lock
  (pod dumps with CUDA_VISIBLE_DEVICES=, numpy only). Meshes kept in data/scratch/p0139_spiral/kp_z<z0>/.
- **Scoring (after all 5 dumps, freeze check first), once:** `flock /tmp/vss_eval.lock uv run python
  data/scratch/kp_chains/score_kp.py --stage conf --arms kp --out data/scratch/kp_chains/conf/conf_table.json`; then
  `--stage all --out data/scratch/kp_chains/all/all_table.json` (descriptive). Both with CUDA_VISIBLE_DEVICES=.
- Launched as systemd user units vss-kp-conf-local and vss-kp-coord; freeze check passed at 01:59Z.

### R2a (2026-10-30T02:40Z): H200 fallback pod billed above the cap; terminated; coordinator patched
- 02:20:53Z kp_coord.py created **vss-kp-b6** (pod <pod-id>, 1x H200 141GB SXM, nebius eu-north1, image
  cuda_12_1_pytorch_2_2) for kp_z6500 as the H200 fallback (no normal offer for 20 min). The offer list showed $5.40/h
  (+$0.15 disk); the pod list then showed **priceHr $6.69/h**: nebius bills vCPU (16 x $0.015) and RAM (200 x $0.0045)
  on top, and kp_coord.py had not counted those unit prices. $6.69/h is above the user's H200 cap (~$6/h), and the
  pod still had no ssh after 11 min. **Terminated at 02:32:30Z by hand, before any data copy or fit. Cost ≈ $1.31**
  (0.196 h x $6.69).
- kp_coord.py patched while stopped (new sha bd3104cb664b833541c3f2e4760e6c59933c7ae3fb90642ac0a1dd69e0a50ef2):
  the offer price adds vCPU/RAM unit prices when they are not included; after a create the billed priceHr is
  recorded, and a pod billed over its cap ($2.50 normal, $6 H200) is terminated at once and its offer blacklisted.
  No rule, threshold or recipe change. With correct pricing the nebius H200 ($6.69/h) is not an acceptable fallback,
  and **no acceptable offer exists at 02:40Z**. The coordinator keeps polling for normal cards (and an H200 <= $6/h).
- kp_z4500 (B4) has been fitting on the local RTX 3090 since 02:00:23Z.

### R2b (2026-10-30T03:06Z): deviation (coordinator decision ~02:45Z): local 3090 queue after B4; no $6.69 H200
- **Correction to R2a:** R2a was written at ≈ 02:33Z, not 02:40Z as its header says (the coordinator restart it
  describes is at 02:33:20Z).
- **Decision (coordinator, 2026-10-30 ~02:45Z):** (1) do not take the $6.69/h nebius H200; H200 cap stays <= $6/h
  effective (vCPU/RAM included). (2) The "at most one band locally" limit is lifted: after B4, bands without a pod run
  on the local RTX 3090 one at a time, order **B7 kp_z8000, B1 kp_z400, B2 kp_z1500, B6 kp_z6500**; same stack,
  checks, lease (vss-kp-local, max_mem_gb 21, guard 19,000 MiB) and flock rules, under systemd-run units. The pod
  coordinator keeps polling; an acceptable pod (<= $2.50/h, or H200 <= $6/h effective) takes the largest
  not-yet-started band. A band never runs in two places; on a race the path that started first keeps it.
  (3) Logged here before the next band starts. (4) Target: all 5 dumps scored by 2026-10-31 18:00Z at the latest;
  report early if the projection passes 2026-10-31 20:00Z.
- **Mechanism:** one atomic claim file per band, data/scratch/kp_chains/claims/claimed_<arm> (O_EXCL / noclobber),
  first word "local" or "pod". The local queue claims a band just before its fit (after the gate and lease) and skips
  any band claimed "pod"; kp_coord.py claims before a create or reuse, never gives a pod to a band claimed "local",
  and releases its claim only if the pod fails before the fit is launched (or the pod vanishes, losing the fit).
- **Helpers (new / changed while stopped):**
  - pods/kp_conf_local2.sh 29a417a9106088c8f2972dd58dc6355e22fbe49461a55ed3ffd3e113fbb37d99 (new): kp_conf_local.sh
    changed only in ARMS (the queue above), a wait for vss-kp-conf-local (B4) to be inactive (before its signal trap,
    since it shares local/run.pgid) plus the same line in the gate, claim-skip instead of halt, a per-band missing
    marker (local/missing_<arm>; crash rule unchanged: one rerun, then missing), band check and input source per band
    (B7/B1/B2 data/p1667_ws/z<z0>, B6 data/p1667, as kp_boot.sh), names. Unit vss-kp-conf-local2.
  - pods/kp_coord.py d01c40bb00a1f1a72347559698647ff11b38ee1ec6e052be46899e346f85ef8c (stopped 03:00Z, no pod live,
    patched, restarted): claims as above; done() also counts local/missing_<arm>. Offer rules, caps and gate as in
    R2/R2a. kp_conf_local.sh (B4) is untouched and still running.
- **Projection (local only, from the dev and earlier fit times):** B4 done ≈ 05:45Z; B7 ≈ 09:15Z; B1 ≈ 12:45Z;
  B2 ≈ 17:15Z; B6 ≈ 21:45Z (2026-10-30); conf scorer ≈ 22:15Z 2026-10-30. Within the 2026-10-31 18:00Z target.
  Any pod that appears shortens this.
- **Hardware reporting:** the GPU type per band (local RTX 3090 vs pod type) is reported with the conf table;
  mixed types are flagged as in the whole-scroll test.

### R2c (2026-10-30T03:07:12Z): finisher for the confirmation scorer (coordinator)
- Neither runner (kp_conf_local2.sh, kp_coord.py) runs the scorer. Added data/scratch/kp_chains/pods/kp_finish.sh
  (sha b957288445ef8c83…), unit vss-kp-finish. It waits until vss-kp-conf-local, vss-kp-conf-local2 and vss-kp-coord have all
  exited and all 5 confirmation dumps exist, re-checks freeze.sha256, then runs the frozen scorer exactly once:
  `score_kp.py --stage conf --arms kp --out data/scratch/kp_chains/conf/conf_table.json`, then the descriptive
  `--stage all --arms kp` pool (out data/scratch/kp_chains/all/). Both under flock, GPU hidden. If any dump is missing
  it scores nothing and logs why. No recipe, band or rule changes.

### L2 (2026-10-30T19:58Z): confirmation result. C1 not met (Kelly's index beats KP); C2 not met
- **Fits** (all 5 on the local RTX 3090, driver 615.78.08, attempt 1, no crash, no guard stop; villa / env / set /
  overrides checks passed before and after each; meshes kept in data/scratch/p0139_spiral/kp_z<z0>/):

  | band | arm | runner | fit start → dumped (Z) | fit + export | peak VRAM | windings | dump sha |
  |---|---|---|---|---|---|---|---|
  | B4 z4500-6000 | kp_z4500 | kp_conf_local.sh | 10-30 02:00:23 → 06:24:05 | 261 min | 9,186 MiB | 156 | b887a54a… |
  | B7 z8000-9000 | kp_z8000 | kp_conf_local2.sh | 06:25:04 → 08:58:29 | 152 min | 7,534 MiB | 150 | d7685422… |
  | B1 z400-1500 | kp_z400 | kp_conf_local2.sh | 08:58:31 → 11:33:57 | 153 min | 8,482 MiB | 220 | 15ff47f5… |
  | B2 z1500-3000 | kp_z1500 | kp_conf_local2.sh | 11:34:00 → 15:22:23 | 226 min | 9,468 MiB | 152 | 2aee2d8c… |
  | B6 z6500-8000 | kp_z6500 | kp_conf_local2.sh | 15:22:27 → 19:52:28 | 266 min | 9,558 MiB | 146 | 00aeeb5e… |

- **Pods:** no acceptable offer appeared all day (kp_coord.py polled 02:00-19:52Z; zero normal single-GPU offers;
  only the nebius H200 at $6.69/h effective). The only pod was vss-kp-b6 (H200, R2a), terminated before any fit.
  **Spend: $1.31 total.** 0 vss-kp-* pods at 19:55Z; no lease left; units vss-kp-conf-local, -conf-local2, -coord,
  -finish all exited.
- **GPU types (flag, as in the whole-scroll test):** all 5 KP confirmation fits (and the 2 dev KP fits) ran on the
  RTX 3090. Their M2 s1 comparators ran on pods: wsf400_m2s1 A6000, wsf1500_m2s1 A100, ink2_m2s1 A6000,
  c1667_m2s1 L40S, wsf8000_m2s1 A6000. Every KP-vs-M2 pair is therefore mixed-GPU (same stack, seed 1).
- **Scoring:** the conf scorer ran exactly once (vss-kp-finish / pods/kp_finish.sh, R2c; 19:53:18-19:54:19Z, exit 0;
  freeze check passed before it): conf/conf_table.json (2d1f2e00…), conf_table_decision.json (1f440d21…),
  conf_table.log. Then the descriptive all-7 pool once (exit 19:55:34Z): all/all_table.json (124df3ba…). After both:
  freeze.sha256 passes, lines 1-218 still 972a12ab…. The RuntimeWarnings in the logs come from B1 having no outer
  vertices (n_outer 0, outer per-band row null); the pooled outer uses the other bands.
- **Pooled confirmation (B1, B2, B4, B6, B7), primary, C / A / Q (%):**

  | arm | C | A | Q |
  |---|---|---|---|
  | base | 75.0 | 20.0 | 15.0 |
  | M2 s1 | 73.5 | 60.5 | 44.5 |
  | **KP** | 72.7 | 60.1 | **43.7** |
  | Kelly k_all_comp | 71.3 | 94.3 | 67.3 |

- **C1 (beats Kelly): not met; the reverse holds.** ΔQ KP − Kelly −23.58 [−25.13, −21.88]: Kelly's index beats the
  fitted arm.
- **C2 (improves the fitter): not met.** ΔQ KP − M2 −0.81 [−1.43, −0.22] (lower bound < 0); inner own-offset wrong
  area KP − M2 +0.38 [−0.26, +0.97] (upper bound > 0); switch Δ −1.9 [−3.75, −0.08] (≤ +5, the only part met).
- **Reported:** co-covered ΔA vs Kelly −28.43 [−30.09, −26.77] (S1 fails); co-covered ΔA vs M2 −0.67 [−1.43, +0.13];
  outer wrong Δ +0.66 [−0.25, +1.61]; unmatched Δ +1.02 [+0.75, +1.29]; **union Q with Kelly 74.8 [73.0, 76.5]**.
- **Variants:** lenient: ΔQ vs Kelly −20.05 [−21.58, −18.36], vs M2 +0.28 [−0.24, +0.79]; tol2: vs Kelly −21.25
  [−22.68, −19.79], vs M2 −0.43 [−0.87, −0.01]. No rule passes on primary, so there is no fragility flag; the
  verdicts agree on all three variants (lenient leaves ΔQ vs M2 inside zero).
- **Per band (primary Q; KP / M2 / Kelly) and inner wrong Δ KP − M2:** B1 47.7 / 51.6 / 48.4, +2.8 [1.5, 4.0];
  B2 57.8 / 49.5 / 71.0, −14.9 [−16.4, −13.5]; B4 46.7 / 60.8 / 80.3, +19.6 [18.4, 20.6]; B6 41.4 / 37.4 / 74.4,
  −5.4 [−6.6, −4.1]; B7 14.5 / 14.2 / 46.0, −1.5 [−2.2, −0.9]. KP helps on B2 and B6 and hurts on B4 (the band
  holding most of Kelly's slab 2) and B1.
- **Descriptive all-7 pool (B1-B7, `--stage all --arms kp`; no rule):** KP C/A/Q 73.7 / 66.4 / 49.0, M2 74.2 / 65.6 /
  48.7, Kelly 71.9 / 95.2 / 68.5. ΔQ vs M2 +0.24 [−0.22, +0.68]; vs Kelly −19.54 [−20.93, −18.18]; inner wrong Δ vs
  M2 −0.99 [−1.46, −0.52]; switch Δ −1.5 [−3.1, +0.2]; union Q with Kelly 76.2 [74.7, 77.7].
- **Against the predictions (§6):** P(C2) 0.55 → not met (KP ≈ M2 on the confirmation pool, −0.8 pp Q, after +3.5 pp
  on dev); P(C1) 0.10 → not met; P(Kelly beats the arm) 0.7 → yes (−23.6 pp, beyond the predicted 5-15 pp).
- **Deviations in this stage:** R2a (H200 pod billed $6.69/h, terminated before use, $1.31), R2b (all bands on
  the local 3090 after the coordinator lifted the one-band limit; per-band HOME layout), R2c (finisher ran the
  scorer). No recipe, env, set, scorer or rule change; no rerun; no missing arm.
