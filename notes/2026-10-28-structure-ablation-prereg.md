# 2026-10-28: structure ablation (A1-A3): which part of the oracle chains' structure carries the fit gain. Pre-registration

Status: **PREPARED, NOT RUN.** Chain sets, envs, scorer and decision rules are frozen at the sha in log entry F0. **No fit
runs before the user's go-ahead** (logged as an entry below). Run directory: `data/scratch/structure_ablation/`.

## 0. Question
On Paris4 z 7000-8500, oracle chains cut own-offset primary wrong area by 31.3-31.4 pp vs the 3 base seeds; our cut
chains (p4z7_f6, f6z_ms1) cut 7.9-8.0 (about 25 % of that gap). The hypothesis tested here is that the oracle's value is structure, not step counts:
- (a) **completeness**: a point on every sheet along each ray;
- (b) **linkage**: long linked pieces (oracle 654 pieces, median 120 points; p4z7_f6 5,768 pieces, median 6).

Which of (a) and (b) carries the gain? The answer picks the next lever: segment/patch-derived chains (linkage), per-ray
sheet recovery at contacts (completeness), or human sheet tracing (both). A3 also tests the 10-25 finding directly:
our structure with oracle-correct steps, against f6z_ms1 (our structure, our steps).

## 1. What exists before this freeze (disclosed)
- The six f6 confirmation fits on this band, dumped and scored (notes/2026-10-24-f6-confirm-prereg.md, Results), and
  every later dev number on this band (tests 12 and 13, the label audit z7000). This band is burned dev; nothing
  here is a confirmation.
- The A sets' only label input is the oracle chain set band_oracle_p4z7 (from labels_w0701), the same file the
  f6z_oracle arm used. Building them read oracle point positions and wind_a, and the radius column of its rays.json.
  The rays.json label-winding column was not read.
- No fit with any A set exists. No label metric was computed for this note, and none for Paris4 z 8500-9000 or
  z 10800-11800.

## 2. Chain sets (frozen; `data/scratch/structure_ablation/build_sets.py`)
Inputs (sha-checked by the script): band_oracle_p4z7 relative_windings.json fc8a00a5…, its rays.json 5acbdc3f…;
p4z7_f6 relative_windings.json 03c014bd…, chains.json b5d20c1d…. Both sets lie on the same 576 rays (24 z levels 7008-8480
every 64 × 24 angles every 15°, from the collection names). The radius r along the ray comes from rays.json (oracle) and
chains.json (ours). Every piece is radius-increasing in point-id order.

- **A1 complete + fragmented.** Every oracle point is kept. Each oracle piece, in collection order, is cut into
  consecutive sub-pieces. Each length is drawn i.i.d. from p4z7_f6's 5,768 piece lengths with numpy
  `default_rng(20261028)`. A draw that would leave a 1-point remainder takes that point too, and the last sub-piece is
  the remainder. Sub-pieces share no point, so the oracle step across each cut is lost, as in the f6 cut.
- **A2 thinned + linked.** Oracle pieces are kept whole, minus each oracle point with no p4z7_f6 partner on its ray.
  - Partners: a **one-to-one** greedy match per ray, by |Δr| ascending (stable sort), with |Δr| ≤ R_MATCH.
  - **R_MATCH = 7.658 vox** (chain frame) = 0.5 × the median radial spacing of oracle single steps (15.315 vox,
    n 67,582). Per-ray medians range 10.9-23.4 vox.
  - Kept points keep their oracle wind_a, so a gap carries the known step count.
- **A3 thinned + fragmented.** A2's points, split wherever two consecutive kept points have partners in different
  p4z7_f6 pieces. The pieces are ours and the steps are the oracle's.
- All sets: pieces with < 2 points are dropped (the fitter drops them too). wind_a is re-based to 0 at each piece's
  first point (differences unchanged). Format as the inputs (name `saA<n>_z<z>_t<θ>_<i>`, metadata {}, colour 0,
  file-global ascending point ids). A rerun into a second directory is byte-identical.

| set | sha256 |
|---|---|
| sa_A1 | 5d7ead93fb2d28a1a9f441db114505e4ed0195f36eef407859710c17afd6f9ee |
| sa_A2 | 338329f6b3f32a8701329fd8553dd97fac7405fcc612003aad8e3c7aaaf09bf2 |
| sa_A3 | 00f165917047bc5436c654c6fdd640d8cb90409bdf4c97425b6ae26978b14046 |

Paths: `data/scratch/structure_ablation/sets/sa_A<n>/relative_windings.json`; stats in `sets/manifest.json`.

**Structure stats (label-free; 576 rays in every set):**

| set | pieces | points | median / mean len | q10-q90 len | pts/ray | linked steps | steps > 1 | oracle pts kept |
|---|---|---|---|---|---|---|---|---|
| oracle | 654 | 68,325 | 120 / 104.5 | 47-120 | 118.6 | 67,671 | 89 | 100 % |
| p4z7_f6 | 5,768 | 53,749 | 6 / 9.3 | 2-21 | 93.3 | 47,981 | 3,705 | n/a |
| A1 | 7,972 | 68,325 | 6 / 8.6 | 2-19 | 118.6 | 60,353 | 84 | 100 % |
| A2 | 643 | 47,013 | 81 / 73.1 | 32-93 | 81.6 | 46,370 | 10,355 | 68.8 % |
| A3 | 5,340 | 46,673 | 5 / 8.7 | 2-20 | 81.0 | 41,333 | 7,143 | 68.3 % |

- 47,014 matched pairs, so 6,735 p4z7_f6 points (12.5 %) have no oracle partner within R_MATCH.
- A1 vs p4z7_f6 piece lengths: KS distance 0.028.

**Load check** (`check_load.py` → `check_load.json`). It runs `load_point_collection` and
`normalise_pcl_winding_annotations`, taken verbatim from villa 0fb38c45 spiral/point_collection.py, plus fit_spiral's
unattached z-roi rule (z 7000-8500, margin 0) and the < 2-point drop. CPU only, no torch.
- Every set loads in full: A1 7,972 strips / 68,325 points, A2 643 / 47,013, A3 5,340 / 46,673. All points are
  annotated, all wind_a are finite and non-decreasing, and the counts match the manifest.
- `autowind.export check_doc` (role relative): 0 errors. Its only warning is the missing winding_is_absolute key, the
  same as on the oracle and p4z7_f6 files (VC3D-only; the fitter takes the role from the file name).

## 3. Arms (`spiral/arms/sa_A<n>.env` = f6z_oracle.env with only ARM, REL_WINDINGS and the comment line changed)

| arm (ARM) | env file | chains | FIT_OVERRIDES | seed | env sha |
|---|---|---|---|---|---|
| sa_a1z7 | sa_A1.env | sa_A1 | spacing 0, weight 10 | 1 | 6a337eb3… |
| sa_a2z7 | sa_A2.env | sa_A2 | spacing 0, weight 10 | 1 | 83c22259… |
| sa_a3z7 | sa_A3.env | sa_A3 | spacing 0, weight 10 | 1 | b0ede466… |

REL_WINDINGS = `$HOME/data/PHercParis4/windings/sa_A<n>/relative_windings.json`. Before each fit, the set is copied
there from `sets/` and its sha checked against section 2.

**Anchors (existing; no refit).** Envs and dumps (`data/scratch/f6_screen/confirm/vertex/`):

| arm | role | env sha | dump sha |
|---|---|---|---|
| f6z_base / f6z_bases2 / f6z_bases3 | bases, seeds 1/2/3 | d010f0b0… / d8781fe3… / dc2622dc… | b3987225… / 3c32bda5… / 4ca9c66e… |
| f6z_oracle | complete + linked, oracle steps | 96017e35… | 303e456a… |
| f6z_ms1 | our structure, our steps (seed 1) | a98ac1e3… | 0022d9f2… |
| f6z_ms2 | the same, seed 2 (seed-noise reference) | c2e63224… | a6000b7f… |

Fit stack unchanged: villa 0fb38c45 fit_spiral without patches, config/recipe.toml 9ba8494e…, run_arm.sh
0310894f…, fit_ckpt.py 0fb1bb76…, export_fine.sh 779d504b… step 4. Inputs: data/paris4_z7000/{inputs,masks}, the
same files the f6z pods got. No reruns, except a checkpoint resume with the same env.

## 4. Scorer (fixed now)
Dump, per arm, after the step-4 export and pull to data/scratch/p0139_spiral/<arm>/ (as collect.sh / local_conf.sh):
```
flock /tmp/vss_eval.lock uv run python release/fit_vertex_dump.py --scroll paris4 \
  --fit data/scratch/p0139_spiral/<arm>/meshes_step4 --name <arm> --z0 7000 --z1 8500 \
  --out-dir data/scratch/structure_ablation/vertex
```
Table, once all 3 dumps exist:
```
flock /tmp/vss_eval.lock uv run python data/scratch/structure_ablation/score_sa.py \
  --out data/scratch/structure_ablation/sa_table.json
```
- `score_sa.py` imports the test-13 dev scorer `data/scratch/anneal/dev_table_an.py` (f2c785a6…, unchanged). It puts
  `structure_ablation/vertex` first in that scorer's dump path and runs it as
  `dev_table_an.py sa_a1z7 sa_a2z7 sa_a3z7 --refs`. --refs adds f6z_ms1, f6z_ms2 and f6z_oracle. The test-13 gate
  columns it prints are ignored.
- Unchanged underneath: dev_table_f7.py cc7786b7…, eval/paris4_label_audit.py 76081274… (S1 trusted,
  audit z7000/audit.json 5195362d…), spiral_eval.py 813860d7…, subset_wrong_area.py 8890d16d…, fit_vertex_dump.py
  80e8b55c…, subsets_bandparis4z7.json 864b3984….
- **Metrics,** each arm vs each of f6z_base / f6z_bases2 / f6z_bases3:
  - primary (original labels): own-offset seam wrong area on PRIMARY_K 13,16-32,34-44,47-61,63-108, paired BootV
    (30° × 250 z, 1000 resamples, seed 1);
  - switches per 1000 adjacent pairs, all vertices;
  - nonprimary own-offset on the S1 trusted vertices and on the original labels;
  - S1 primary and S1 switches; global seam wrong area.
- **Gap share** per base seed b: G(arm, b) = (arm − b) / (f6z_oracle − b) on the primary. G(arm) = mean over the 3
  seeds. Reference: f6z_ms1 ≈ 0.25.
- **Contrasts** (Y − X primary, pp, paired BootV, same code path as the primary):
  - linkage given complete, A1 − oracle; linkage given thinned, A3 − A2;
  - completeness given linked, A2 − oracle; completeness given fragmented, A3 − A1;
  - total structure, A3 − oracle; step correctness on our structure, f6z_ms1 − A3.
- Output: sa_table.json (+ _orig, _audit) and sa_table_decision.json.
- **Smoke test (2026-10-28, before freeze; stand-in dumps, no new information).** Command: `score_sa.py --smoke
  cl_annz7 cl_dt0z7 an_e03z7 --out smoke/smoke_sa.json`, under the eval lock: 36 min wall, ~6.5 GB RSS.
  - It reproduces the published rows: cl_annz7 primary −7.85 / −7.93 / −7.95, switch +2.23 / +2.82 / +2.44; f6z_ms1
    −7.85 / −7.93 / −7.95, +6.51 / +7.10 / +6.72; oracle −31.34 / −31.42 / −31.44.
  - Each contrast equals the difference of the two rows' arm values (e.g. cl_annz7 − oracle +23.49 = 64.6 − 41.1).
  - f6z_ms1 G = 0.252, f6z_ms2 G = 0.269.
  - The rule code ran end to end. With stand-ins all at G ≈ 0.25 it gives R4, which is the correct reading: each
    property alone recovers nothing.

## 5. How each comparison is read, and the decision rules
2×2: oracle = complete + linked (G = 1 by definition), A1 = complete + fragmented, A2 = thinned + linked,
A3 = thinned + fragmented. In units of the base → oracle gap:
- linkage loss Link = ½[(1 − G_A1) + (G_A2 − G_A3)];
- completeness loss Comp = ½[(1 − G_A2) + (G_A1 − G_A3)];
- interaction = (1 − G_A1) − (G_A2 − G_A3);
- correctness gain Corr = G_A3 − G_ms1.

Materiality threshold δ = 0.15 of the gap (≈ 4.7 pp). One seed per arm: on this band ms1 vs ms2 differ by 0.5 pp
(0.02 G), and a primary BootV half-width is ≈ 1.2 pp (0.04 G).

Rules, applied in order on original-label G:
- **R1, neither** (G_A3 ≥ 0.85): our structure with correct steps nearly matches the oracle. The structure is not the
  bottleneck; step correctness is, and the 10-25 finding is overturned for this fitter. Next: step correctness (step
  classifier, or revisit HITL step checking).
- **R2, linkage-dominant** (Link ≥ 0.15 and Link − Comp ≥ 0.15). Next: patch / segment-derived chains. On Paris4 the
  segments are the labels (circular), so the test goes to PHerc1667 with Stevens' patches (backlog 10-28).
- **R3, completeness-dominant** (Comp ≥ 0.15 and Comp − Link ≥ 0.15). Next: per-ray sheet recovery at contacts (m7
  misses one sheet at contacts ~52 % of the time).
- **R4, both** (Link ≥ 0.15, Comp ≥ 0.15, |Link − Comp| < 0.15). Next: a source that gives both, i.e. human sheet
  tracing, or segment patches that span contacts.
- **R5, either suffices** (otherwise; a large negative interaction: A1 and A2 both near the oracle, A3 not). Next: the
  cheaper single lever, by default patch / segment-derived chains.
- **Always reported, whatever the rule:**
  - Corr. Corr < 0.15 confirms "steps don't matter on our structure". Corr ≥ 0.15 means step correctness matters on
    our structure even if structure dominates.
  - The same rules on S1-trusted G (secondary; a disagreement is reported, not resolved).
  - Switch Δ and S1 nonprimary per arm (not gated).
  - Contrast CIs. A rule that rests on a contrast whose CI spans 0 is flagged as weak.

**Prediction** (written before any fit):
- Linkage-dominant (R2) is the most likely outcome: A1 G ≈ 0.4-0.6, A2 G ≈ 0.7-0.9, A3 G ≈ 0.3.
- Corr ≈ 0.05-0.1, i.e. not material.
- Low confidence: the fitter samples 13 strips per step whatever their length (f6 note §9), and that couples
  fragmentation with per-point weight.

## 6. Cost and run plan (after the go-ahead only)
- **Local (default, $0):** one fit at a time on the RTX 3090, only after the vss-ws-local unit has finished (its units
  and lease are not touched).
  - Each fit takes ~2.8 h, peak ~12-13 GB, under its own GPU lease (session `vss-sa-local`, max_mem_gb 21) with a
    19,000 MiB guard.
  - The runner is local_conf.sh with ARMS, env shas, HOME, the dump dir and the lease name changed. It is written at
    go-ahead and its sha is logged before the first fit. HOME = data/scratch/structure_ablation/home, with
    data/PHercParis4/{inputs,masks} linked to data/paris4_z7000/{inputs,masks}.
  - The first arm builds the z7000 track/SDT cache. 3 arms ≈ 8.5-9 h wall.
- **Pods (only if the local GPU stays busy):** ≤ $2.50/h, names vss-sa-*, terminated right after pull. At the f6z
  rates ($0.54-1.81/h, 4-6 h each) that is ≈ $3-11 per arm. **Cap $20** for all 3; check in before exceeding it.
- **Fit order:**
  1. A3 first: it alone decides R1 and Corr.
  2. A1: linkage, the largest structural difference.
  3. A2.
  - Scoring waits for all 3 (no interim label metric).

## 7. Disclosures and limits
- **Burned dev band, one seed per arm.** A result is a dev-band attribution, not a confirmed effect.
- **A1 matches piece length, not pieces per ray.** It has 7,972 pieces (13.8 per ray) vs p4z7_f6's 5,768 (10.0),
  because it fragments 27 % more points.
- **A3 vs f6z_ms1 is not only step correctness.** A3 also lacks the 6,735 p4z7_f6 points with no oracle partner
  (spurious detections, points outside the oracle's labelled windings, or points > 7.66 vox off). Its points sit at
  oracle positions (≤ 7.66 vox from ours). Corr is therefore an upper bound on the value of correct steps alone.
- **A2/A3 thinning uses one-to-one matching.** At a contact, m7 often finds one sheet where there are two. Any-match
  within R_MATCH would let one of our points keep both oracle points there and refill exactly the completeness gap
  under test. Kept fractions, one-to-one vs any-match:
  - R_MATCH 6: 66.4 % vs 69.5 %;
  - 7.66: 68.8 % vs 74.8 %;
  - 10: 70.3 % vs 80.1 %;
  - 12: 71.0 % vs 83.2 %.
  - A per-ray R_MATCH (half the ray's median spacing) keeps 69.1 %.
- **Fragmentation also changes sampling** (13 strips per step, whatever the length). That is part of "linkage" as
  this fitter sees it.
- The oracle stops at its labelled windings (k 10-129, at most 120 points per ray). "Complete" means complete within
  that range.

## 8. Deviations from the brief (decided before freeze)
- One-to-one partner matching instead of "any p4z7_f6 point within r" (reason in §7).
- One global R_MATCH, not per ray (per-ray keeps 69.1 % vs 68.8 %).
- A3 breaks are taken from the partner's p4z7_f6 piece id, not from break radii. This is equivalent except when
  partners cross.
- Arm names end in z7 (`sa_a1z7` …), because the frozen dev scorer asserts it. The env file names are sa_A1-3.env as
  requested.
- wind_a is re-based to 0 in every A set, A2 included (step differences unchanged).

## 9. Frozen files
- build_sets.py 0e177d2e…, check_load.py 358f9734…, score_sa.py b30000ae…, the three sets, the three envs, and
  this note up to the Log line.
- Never modified: tools/auto_windings.py, eval/p0139_graph.py, tools/autowind/postproc.py, dev_table_an.py,
  conf_table_an.py, and every file frozen in an earlier prereg note.
- No commits, pushes, posts or uploads.

---

## Log (append-only)
### F0 (2026-10-28T03:00:30Z): frozen
- **sha256 of this file** up to and including the "## Log (append-only)" line: 8abc4525fc78f71854b1aef7b689e95a679b37a599af6f860277ca763c7b0f08.
  Snapshot and the full sha list (sets, envs, anchors, dumps, scorers, fit stack, frozen tools):
  `data/scratch/structure_ablation/freeze/` (prereg_snapshot.md, freeze.sha256).
- **State at freeze:** the sets are built and the load check passed. No sa_* fit, pod or dump exists. vss-ws-local
  holds the local GPU. Fits wait for the user's go-ahead.
### G1 (2026-10-28T03:05Z): go-ahead
- The user approved the 3 fits ("Use pods too"), with this plan: local RTX 3090 primary, order A3, A1, A2, once
  vss-ws-local's current fit wsf6000_bases2 has dumped (vss-ws-local is stopped for the 3 fits and restarted after,
  exactly as launched); pods in parallel only for not-yet-started arms (single GPU ≥ 48 GB, x86_64, driver ≥ 580,
  ≤ $2.50/h, not spot, not GH200/GB200, not Lambda Labs, never H200; cap $20 total); claim markers so an arm runs in
  exactly one place; scorer once all 3 dumps exist.
- Pre-run check 03:05Z: every sha in freeze/freeze.sha256 verifies; this note's lines 1-233 still hash to 8abc4525….
### E1 (2026-10-28T03:10:00Z): local runner written and started (unit vss-sa-local)
- `data/scratch/structure_ablation/sa_local.sh` sha256 70f5bf0672f85bb984f7974a278d9d18b4b70f234192a74c3f2b8ea205781d66,
  copied from local_conf.sh (8e3ddcc3…). Changes beyond §6's list (ARMS, env shas, HOME, dump dir, lease name):
  - **Villa check fixed.** local_conf.sh ran `git -C <villa> apply --reverse --check spiral/chain_loss_tol.patch`
    with a relative patch path, which resolves inside the villa tree, so the check could never fire. sa_local.sh uses
    absolute paths and checks, before every fit: HEAD 0fb38c45; none of spiral/patches/*.patch reverse-applies;
    chain_loss_tol.patch forward-applies (positive control); `git diff` sha 75c27ecb… (the one flpipe shell-optional
    line in fit_spiral.py, as pod_setup.sh) and no other modified file. Tested before launch: passes on
    local_probe/ext/villa, fails on villa_cl ("chain_loss_tol.patch found applied").
  - Frozen set copied to `$HOME/data/PHercParis4/windings/sa_A<n>/` and sha-checked (source and copy) per arm; env band
    and REL_WINDINGS path asserted.
  - **HOME inputs are per-entry symlinks**, not directory links: data/PHercParis4/inputs/* → data/paris4_z7000/inputs/*,
    masks/* → data/paris4_z7000/masks/*. Reason: run_arm.sh writes the packed normal pools (`*.respool_g2*`,
    packed_g2.done) into the inputs dir, and this keeps them out of the shared data/paris4_z7000. Same files read.
  - Claim markers `data/scratch/structure_ablation/claims/claimed_<arm>` (noclobber; "local …" or "pod …"); an arm
    claimed by a pod is skipped. A failed arm keeps its claim (no automatic retry anywhere).
  - Lease `~/.gpu-leases/vss-sa-local.json` (max_mem_gb 21, noclobber), held from the GPU hand-over to the end of the
    last local arm; guard 19,000 MiB; overrides check also asserts seed 1.
  - **GPU hand-over:** waits until vss-ws-local logs "wsf6000_bases2: run exit 0" with no lease on the box (local_ws.sh
    drops its lease before that line), takes our lease (so local_ws.sh's gate parks before it claims its next arm),
    waits for "wsf6000_bases2: dumped" plus the park line, then `systemctl --user stop vss-ws-local`.
  - Fit under `flock /tmp/vss_eval.lock` unless MemAvailable > 20 GB at fit start (exception logged in
    structure_ablation/lock_exceptions.log). Dump: the §4 command verbatim, always under the lock.
  - At the end (or on a halt): lease removed, vss-ws-local restarted with the user's original systemd-run command. If all
    3 dumps exist then, it runs the §4 scorer once (marker scored.started).
- Launched as `systemd-run --user --unit=vss-sa-local --collect …` (not nohup; see anneal D4). Log:
  structure_ablation/local/sa_local.log. At 03:10Z wsf6000_bases2 was at 61 % (optimiser ETA ~04:30Z, then step-4
  export and dump).
### E2 (2026-10-28T03:13:28Z): pod path started (unit vss-sa-grab)
- `structure_ablation/pods/sa_grab.py` (12d64d86…) polls Prime every 60 s with prime.py (read-only reuse of the
  whole-scroll helper). Offer filter: single GPU, ≥ 48 GB, ≤ $2.50/h, not spot, ≥ 6 vCPU, ≥ 40 GB RAM, not GH200/GB200,
  never H200, not lambdalabs, not a known-bad (provider, gpuType) pair. Arm eligible: no dump, no claim, and not the
  arm the local runner takes next; pod order A2, A1, A3; one pod per arm ever. Claims as the local runner; a failed
  create drops our fresh claim and re-lists pods. Budget: no create if spend + (burn + price) × 6 h > $20; all live
  vss-sa-* pods terminated if spend reaches $19.
- Each pod gets its own unit vss-sa-boot-<arm> running `pods/sa_boot.sh` (ffae7a39…): host check (x86_64, driver ≥ 580,
  else delete), bootstrap_pod.sh with data/paris4_z7000/{inputs,masks} and the frozen set, pod_setup, torch CUDA, set +
  env sha on the pod, villa check on the pod (absolute paths, as sa_local.sh), pod_arms.sh sa_A<n>.env; then poll,
  pull_arm.sh, count + overrides check, terminate, §4 dump under flock. Infra failure before the fit: pod deleted,
  claim released. Fit failure: pod deleted, claim kept, halt.
- First poll 03:13Z: 0 usable offers (grab_coord.log shows 0 usable since 10-26 16:45Z); eligible A2, A1.
### E3 (2026-10-28T04:21-04:32Z): GPU hand-over, runner bug D1, A3 started
- 04:21:45Z local_ws.sh logged "wsf6000_bases2: run exit 0" (fit + step-4 export done), pulled 162 windings and began
  its dump. 04:21:47Z sa_local.sh took lease vss-sa-local.json. 04:23:14Z "wsf6000_bases2: dumped (meshes kept)";
  local_ws.sh then parked at its gate ("waiting for GPU: leases: vss-sa-local.json"). It had not claimed
  wsf6000_m2s2. 04:23:17Z `systemctl --user stop vss-ws-local` ok. No ws claim is left without a dump, so the
  later restart starts wsf6000_m2s2 fresh.
- **D1 (runner bug, operational; no fit affected).** After the stop, sa_local.sh's "GPU free" check counted every
  compute process. The desktop apps (voxtype, t3code, slack GPU processes; 10-400 MiB each, ~2.3 GB total) never
  leave, so the runner would have halted after 10 min and restarted vss-ws-local. At 04:31:41Z I stopped it with
  `systemctl --user stop vss-sa-local` (no fit had started; it removed its lease and did not restart ws). Fix: the GPU
  counts as busy only if a compute process belongs to the fitter venv (local_probe/ext/venv, as local_ws.sh's gate) or
  uses > 1024 MiB. v1 kept as local/sa_local.v1_70f5bf06.sh. **sa_local.sh is now sha256
  195f3d2556e9b0ffcc1ed497d59f31d353b3cd61a998e8f9a68bada0bb02fee7.** Nothing else changed. Relaunched 04:32:09Z with
  the same systemd-run command. The GPU sat idle 04:23-04:32Z, with no lease on the box (vss-ws-local was stopped).
- 04:32:10Z: lease taken (vss-ws-local inactive; 2,363 MiB desktop use); sa_a3z7 claimed (local); env and set sha ok;
  villa check ok (HEAD 0fb38c45b488; no patch in spiral/patches reverse-applies with absolute paths;
  chain_loss_tol.patch forward-applies; diff = the shell-optional line only). The fit runs without the eval lock
  (MemAvailable 41 GB > 20; logged). **sa_a3z7 fit start 04:32:10Z** on the local RTX 3090.
### E4 (2026-10-28T09:48-10:00Z): A3 done and dumped; A1 started; pod grabber D2
- sa_a3z7: optimiser start 04:59:23Z (the first arm built the z7000 tracks and SDT cache, 04:32-04:59Z). The fit log
  shows "Loaded point collection with 5340 collections (46673 points)", and ds/relative_windings.json is 00f16591…
  (= sa_A3). **Run exit 0 at 09:48:44Z after 316 min; peak process VRAM 17,152 MiB** (above §6's ~12-13 GB estimate,
  below the 19,000 MiB guard and the 21 GB lease). Overrides check ok (recipe + band + spacing 0, weight 10; seed
  default 1). Pulled 240 windings. Dump 09:48:47-09:59:34Z under the eval lock (CPU shared with other jobs):
  vertex/sa_a3z7.npz sha256 605b8f62…. Pulled meshes deleted (work copy kept under HOME). No label metric viewed.
- sa_a1z7: claimed local; env sha ok, set sa_A1 sha ok (5d7ead93…), villa check ok (same as E3); fit without the eval
  lock (MemAvailable 44 GB > 20; logged). **Fit start 09:59:34Z** (tracks/SDT cache reused).
- **D2 (pod path, operational).** sa_grab v1 excluded "the arm the local runner takes next" (first unclaimed arm in local
  order). Once A1 was claimed locally, that made A2 permanently ineligible, and sa_grab exited at 09:59:35Z (spend $0,
  no pod ever created). That was stricter than the approved plan (pods run not-yet-started arms). v2: the local runner's
  next arm is excluded only while the runner is between arms (no local claim without a dump). While it is busy, every
  unclaimed arm is open to a pod, and claims still keep each arm in one place. v1 kept as
  pods/sa_grab.v1_12d64d86.py. **sa_grab.py v2 sha256 c5534607e4ea60b8144d3907c674d5922aefe3c7083b5b54383e3f8637672c6b**;
  restarted as unit vss-sa-grab at 09:59:59Z, eligible [sa_a2z7], 0 usable offers.
### E5 (2026-10-28T14:12-14:26Z): A1 done and dumped; A2 started locally; pod path closed with no pod
- sa_a1z7: the fit log shows "Loaded point collection with 7972 collections (68325 points)" (= sa_A1). **Run exit 0 at
  14:12:54Z after 253 min; peak process VRAM 15,892 MiB.** Overrides ok; pulled 240 windings. Dump
  14:12:57-14:25:00Z under the eval lock: vertex/sa_a1z7.npz sha256 9dc26a40…. Pulled meshes deleted. No label metric
  viewed.
- sa_a2z7: claimed local 14:25:00Z (no pod offer ever met the filter, so it was still unclaimed). Env sha ok, set sa_A2
  sha ok (338329f6…), villa check ok; fit without the eval lock (MemAvailable 40 GB > 20; logged). **Fit start
  14:25:00Z.** The fit log shows "Loaded point collection with 643 collections (47013 points)" (= sa_A2).
- Pod path: sa_grab polled 03:13-09:59Z and 09:59-14:25Z and saw 0 usable offers on every poll (offers listed were
  CPU nodes, H200, or other GPUs that failed the filter). It exited at 14:25:52Z with no eligible arm. **No pod was
  created; pod spend $0.00.** No vss-sa-* pod exists (prime list empty for our names).
### E6 (2026-10-28T18:46-18:57Z): A2 done and dumped; GPU returned; scorer started
- sa_a2z7: **run exit 0 at 18:46:24Z after 261 min; peak process VRAM 17,412 MiB.** Overrides ok; pulled 240 windings.
  Dump 18:46:28-18:57:10Z under the eval lock: vertex/sa_a2z7.npz sha256 eca63dbb…. Pulled meshes deleted.
- Dumps: sa_a1z7 9dc26a407f13fcc4ff3029dfbee91d7cdf652d1680234c49abe4cb39b3e41512, sa_a2z7
  eca63dbb591d3dd55920d055f2c3e5cac3776f6627b18a3ef01ed79f728d7cbd, sa_a3z7
  605b8f624324fd3c9059da8e87329c0c5903671e922204bc0587db08e5d32ee3. All three ran on the local RTX 3090
  (driver 615.78.08); none on a pod.
- 18:57:10Z: lease vss-sa-local.json removed; vss-ws-local restarted with the user's original systemd-run command
  (ExecStart verified). It resumed: "wsf6000_bases2: dump exists, skip", claimed wsf6000_m2s2, fit start 18:57:11Z,
  under its own lease vss-full-local.json (the only lease on the box).
- Frozen-file check at 18:57Z: every sha in freeze/freeze.sha256 verifies.
- 18:57:15Z: the §4 scorer started, unchanged (`flock /tmp/vss_eval.lock uv run python
  data/scratch/structure_ablation/score_sa.py --out data/scratch/structure_ablation/sa_table.json`, log sa_table.log;
  marker scored.started). No label metric was viewed before this point.
- Correction to E5: the offer types named there ("CPU nodes, H200 …") come from one manual listing at 03:12Z (CPU nodes
  and one nebius H200 at $5.40/h). sa_grab logged only offer counts, so the later offers' types are not recorded.
### R1 (2026-10-28T19:12:07Z): frozen scorer output and verdict
- Scorer: §4 command unchanged, 18:57:15-19:12:07Z (15 min) under the eval lock; outputs sa_table.json (+ _orig,
  _audit) and sa_table_decision.json (smoke false). Full stdout: structure_ablation/sa_table.log.
- **Primary own-offset wrong area, original labels** (arm %; Δ pp vs f6z_base / bases2 / bases3, 95 % paired BootV
  for the first):

| arm | arm % | Δ primary | switch Δ | nonprim S1 (orig) | G | G_S1 |
|---|---|---|---|---|---|---|
| A1 sa_a1z7 (complete + fragmented) | 60.57 | −11.85 [−13.58, −10.24] / −11.93 / −11.95 | +7.62 / +8.22 / +7.84 | −3.91 (−2.23) | 0.379 | 0.374 |
| A2 sa_a2z7 (thinned + linked) | 39.89 | −32.53 [−34.94, −30.11] / −32.61 / −32.63 | +0.14 / +0.74 / +0.36 | −8.07 (−2.70) | 1.038 | 1.062 |
| A3 sa_a3z7 (thinned + fragmented) | 55.44 | −16.99 [−19.28, −14.93] / −17.07 / −17.09 | +2.67 / +3.27 / +2.89 | −4.66 (−1.70) | 0.543 | 0.584 |
| f6z_ms1 (ref) | 64.57 | −7.85 / −7.93 / −7.95 | +6.51 / +7.10 / +6.72 | −1.69 (+5.47) | 0.252 | 0.266 |
| f6z_ms2 (ref) | 64.04 | −8.39 / −8.47 / −8.48 | +7.63 / +8.23 / +7.85 | −1.71 (+5.57) | 0.269 | 0.283 |
| f6z_oracle (ref) | 41.09 | −31.34 / −31.42 / −31.44 | +1.83 / +2.43 / +2.05 | −7.40 (−2.59) | 1 | 1 |

  The reference rows reproduce the smoke test and the published f6 numbers exactly.
- **Contrasts** (Y − X primary, pp, 95 % CI):
  - linkage given complete, A1 − oracle: +19.48 [+16.75, +21.78];
  - linkage given thinned, A3 − A2: +15.54 [+13.27, +17.65];
  - completeness given linked, A2 − oracle: −1.20 [−2.78, +0.10] (CI spans 0);
  - completeness given fragmented, A3 − A1: −5.13 [−6.36, −3.91];
  - total structure, A3 − oracle: +14.35 [+10.83, +17.36];
  - step correctness on our structure, f6z_ms1 − A3: +9.13 [+7.68, +10.62].
- **Decision (original labels): R2, linkage-dominant.** Link 0.558, Comp −0.101, interaction +0.125. R1 does not fire
  (G_A3 0.543 < 0.85). R2's conditions (Link ≥ 0.15, Link − Comp ≥ 0.15) rest on the two linkage contrasts, and both
  CIs exclude 0, so R2 is **not weak**. Next per §5: patch / segment-derived chains, tested on PHerc1667 with Stevens'
  patches (backlog 10-28).
- **Corr = G_A3 − G_ms1 = +0.291 ≥ 0.15: material.** Step correctness matters on our structure as well (an upper
  bound, §7: A3 also drops the 12.5 % of our points with no oracle partner). This contradicts the prediction
  (Corr ≈ 0.05-0.1) and the 10-25 finding for this fitter.
- **S1-trusted (secondary): R2 as well** (Link 0.552, Comp −0.136, interaction +0.148, Corr +0.318). No disagreement.
- Against the prediction: R2 as predicted. A2 G 1.04 (predicted 0.7-0.9) and A3 G 0.54 (predicted ≈ 0.3) are both
  higher than predicted, and A1 G 0.38 is in range. Completeness loss is ≤ 0: the thinned sets do as well as or better
  than the complete ones (A2 ≈ oracle, CI spans 0; A3 better than A1).
- The test-13 gate columns in the output are ignored, as frozen.
- Dev-band attribution on the burned band, one seed per arm (§7); not a confirmed effect.
### E7 (2026-10-28T19:13Z): wrap-up state
- vss-ws-local: active (restarted 18:57:10Z; fitting wsf6000_m2s2 under its lease vss-full-local.json, the only lease
  on the box). vss-sa-local exited normally ("sa_local done"); vss-sa-grab exited (no eligible arm); no vss-sa-* unit
  remains. vss-ws-grab was never touched.
- Pods: none created (spend $0.00); the Prime pod list has no vss-sa-* pod.
- Lock exceptions: the 3 fits ran without the eval lock (MemAvailable 40-44 GB > 20; structure_ablation/
  lock_exceptions.log). Every dump and the scorer ran under the lock.
- Frozen files: this note's lines 1-233 still hash to 8abc4525…; every sha in freeze/freeze.sha256 verified at 18:57Z.
  No commits, posts or uploads.
