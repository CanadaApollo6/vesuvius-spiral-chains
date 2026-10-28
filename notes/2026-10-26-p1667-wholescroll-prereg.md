# 2026-10-26 PHerc1667 (Scroll 4) whole scroll with the frozen M2 recipe: pre-registration

Status: pre-registered, before any fit, chain set or label metric on a new band. Protocol as in test 8
(notes/2026-10-24-ink-lowband-prereg.md): choices are frozen here with UTC timestamps before the outputs they govern
are looked at, and any change after that goes into a dated addendum below the Log line. Goal: take the band-level
PHerc1667 result (test 2, z 6500-8000, -28 pp; test 8, z 4500-6000, -53 pp) to the whole scroll with the same recipe,
unchanged.

## 1. Feasibility (no spend; done 2026-10-26T04:00-04:25Z)

### 1a. What exists
- **Volume:** 20251217075048 (2.399 um), level 2 (9.596 um, the fit frame) has shape (9269, 3808, 3808). Scroll
  content (a streamed L5 slab through the umbilicus, `data/scratch/wholescroll/ct_zprofile.py`, non-zero share > 2 %)
  runs over **L2 z 344-9216**. The ends taper: below z ~750 and above z ~8500 the cross-section shrinks fast.
- **Reference labels: one source, merged_v4.** `data/eval/p1667/labels_merged_v4.npz` (sha 8ee42ee1...) are winding
  numbers on the organisers' merged human mesh `merged_v4_2p399` (data/dl_p1667/), assigned by eval/p1667_labels.py by
  unwrapping the mesh (one connected component holds 99.999 % of the vertices; seam at 144 deg; k0 39). The
  single-wrap/multi-wrap segment meshes (w011-w041) agree 81 % with it and are the same human work. The merged mesh is
  **one object over the whole scroll**, so test 8 and test 2 used the same source as any other band would.
- **Label coverage** (`data/scratch/wholescroll/label_cov.py` -> label_cov_merged_v4.json; deg = angular coverage
  around the umbilicus per 250-z slab):
  - labelled vertices span **z 420-9342** and windings **k 11-41** only. The inner 10 turns (k < 11, the core) and any
    turns outside k 41 are **never labelled anywhere in the scroll**.
  - z 2000-7500: 21-29 windings at full 360 deg; inner k < 34 has 22 of 23 windings full (k 11 lacks a sector
    everywhere).
  - z 500-2000: thinner, 14-19 full windings (k 12-30 at most); z 250-500 only scraps (~4 turns' worth).
  - z 7500-9250: inner full windings fall from 20 to 14 (k 28-34 drop out above z 7750); z 9250-9342 scraps.
  - k >= 37 is not CT-supported (test 2 audit); this is why the primary set is k < 34.
- **Umbilicus:** `umbilicus_p1667_L2.json` (sha e6a862b8...; derived from the same human mesh) has control points at
  z 672-8800 only. auto_windings (chain rays) and spiral_eval use np.interp, which **clamps** outside that range;
  the pinned fitter (villa umbilicus.py json_umbilicus_z_to_yx, scipy interp1d fill_value='extrapolate', then a
  sigma-75 z smoothing) **extrapolates** the end segment linearly: at z 400 that centre is (dx, dy) = (-123, -81) vox
  from the clamped one, at z 9000 (-179, +71). So in B1 below z 672 and in B7 above z 8800 the chains and the fit use
  different centres. Not changed (frozen input and code); flagged as a risk for the end bands.

**Coverage, honestly:** labels exist over z 420-9342, i.e. over essentially the whole height, but only for windings
k 11-41 (31 of the scroll's turns), and thinly at both ends. The plan is therefore **"numbered everywhere, scored
where labels exist"**: every band is fitted and every fitted surface carries a winding number; the label metrics
cover only k 11-41 and the reported fit areas say how much of the numbered surface is unscored.

### 1b. Band tiling (L2 z, half-open)
| band | z | slices | status | labelled verts (mesh) |
|---|---|---|---|---|
| B1 | 400-1500 | 1100 | **fresh** (scroll bottom; umbilicus clamped below 672) | 3.5 M |
| B2 | 1500-3000 | 1500 | **fresh** | 7.6 M |
| B3 | 3000-4500 | 1500 | **fresh** | 8.5 M |
| B4 | 4500-6000 | 1500 | **not fresh** (test 8; fits ink2_*) | 8.9 M |
| B5 | 6000-6500 | 500 | **fresh** (the gap between the two used bands) | 2.9 M |
| B6 | 6500-8000 | 1500 | **not fresh** (test 2 confirmation; fits c1667_*) | 9.1 M |
| B7 | 8000-9000 | 1000 | **fresh** (scroll top; umbilicus clamped above 8800) | 5.5 M |

Numbered: z 400-9000 = 8600 of the 8872 content slices (97 %). Not numbered: z 344-400 and z 9000-9216, the two end
caps, because the frozen pipeline extracts tracks and masks over [z0 - 256, z1 + 256) and the volume ends at 9269
(z1 = 9000 keeps z1 + 256 inside it). Bands are fitted independently (the recipe is a band recipe); band edges are
not stitched, so winding numbers are consistent within a band, not across bands. Cross-band stitching is not
claimed.

### 1c. Cost and time estimate (from test 8's actual spend)
- Test 8 fits (30k steps, z 4500-6000): A6000 $0.54/h 5.3-5.5 h fit, 5.75-7.1 h pod time (incl. setup, M2 idle
  ~1.5 h waiting for chains); L40S $0.82/h 2.4 h fit, 2.5 h pod. Per arm per 1500-z band about **$2.0-3.9**; test 8's
  four arms cost **$12.84**. Test 2's L40S fits took 2.6-2.8 h.
- This run: 5 fresh bands x 4 arms (base s1, base s2, M2 s1, M2 s2) = 20 single-GPU pods. Band size in 1500-z
  equivalents 0.73 + 1 + 1 + 0.33 + 0.67 = 3.7; with a floor of ~1.5 h setup+fit overhead per pod, estimate
  **20 pods, ~110-130 pod-hours on A6000, $55-90**, $120 with re-runs. M2 pods are only created once their chain set
  is frozen (no idle). **Cap $250 (task); internal stop at $200** (no new pods once committed spend + running pods'
  projected cost passes $200).
- Local, $0: mask ~30-45 min per band on the 3090 (GPU lease vss-full, 10 GB), inputs ~5 min download per band, chains
  ~60-85 min per band on CPU (6 workers, 2 bands at a time), reference samples ~2 min per band, vertex dumps ~2 min
  per arm.
- Not run: the oracle arm (human-label chains) of test 2. Its cost would be another 5 pods; without it there is no
  per-band informativeness check (see 2f).

## 2. Pre-registration (frozen at the timestamp in the Log, before any chain set, fit or label metric on B1, B2, B3, B5, B7)

### 2a. Recipe (exactly test 8's M2 and base arms; no change other than the band)
- **Mask** per band: `spiral/predict_band.py run --level 2 --um 1 --ref-um 1 --threshold 0.2` with
  `data/models/surface_m7_nnunet` (fold_0 checkpoint_best.pth 17465b77...) over [floor128(z0 - 256), min(z1 + 256,
  9269)), then `data/scratch/confirm_region/mask_add_levels.py` -> `data/p1667_ws/z<z0>/masks/m7_L2_th0.2.zarr`
  (`data/scratch/wholescroll/run_band_prep.sh`, the test-8 run_mask.sh with the band as a parameter).
- **Fit inputs:** `spiral/prep_inputs.py all --scroll PHerc1667 --z0 <z0> --z1 <z1> --margin 256` ->
  `data/p1667_ws/z<z0>/inputs`.
- **M2 chains:** `uv run python -m autowind chains --mask data/p1667_ws/z<z0>/masks/m7_L2_th0.2.zarr --umbilicus
  data/eval/p1667/umbilicus_p1667_L2.json --z0 <z0> --z1 <z1> --voxel-um 9.596 --raise off --workers 6 --out
  data/scratch/wholescroll/autowind_z<z0>` (run_chains.sh), all recipe flags at their defaults (dtheta 15, zoffs
  -64,0,64, stride 96, seg 144, rmin 48, conf 1.0, require-agree 1, max-step 2, bridge 1, clean conf model). The
  `relative_windings.json` is copied unchanged to `data/scratch/wholescroll/windings/p1667z<z0>_m7_ray_c/` and its
  sha256 is logged in an addendum **before that band's M2 pods are created**.
- **Fitter:** pinned villa 0fb38c45 fit_spiral through `spiral/run_arm.sh` + `export_fine.sh 4`, recipe
  `config/recipe.toml`, 30k steps; M2 adds FIT_OVERRIDES
  `{"pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}`.
- **Arm files** `spiral/arms/wsf<z0>_{base,bases2,m2s1,m2s2}.env` (20 files): copies of ink2_base / ink2_bases2 /
  ink2_m2s1 / ink2_m2s2.env with only ARM, Z0, Z1 and REL_WINDINGS changed (s2 = `"optimizer_random_seed": 2`, as in
  test 8). Concatenated sha256 of the 20 files (ls order): f3cb3da4...5e45e (full value in the shas file).
- **Not-fresh bands:** B4 uses the test-8 fits (ink2_base / ink2_m2s1, ink2_bases2 / ink2_m2s2) and dumps
  (data/scratch/ink_lowband/vertex/); B6 uses the test-2 fits (c1667_base / c1667_m2s1, c1667_bases2 / c1667_m2s2;
  dumps data/scratch/confirm_region/p1667/vertex/). No re-fit.
- **Shas** (full list in `data/scratch/wholescroll/prereg_shas.txt`, written with this section): tools/auto_windings.py
  8f097312...ddafa44edb, eval/p0139_graph.py fb7553b1..., tools/autowind/cli.py 3da5957f..., core.py 88bcd81f...,
  packing.py 3ea55390..., presets.py 64ba80d6..., conf model da3515a8..., run_arm.sh 0310894f..., export_fine.sh
  779d504b..., recipe.toml 9ba8494e..., predict_band.py 374fde50..., prep_inputs.py 7deac631..., mask_add_levels.py
  a4b4b261..., labels 8ee42ee1..., umbilicus e6a862b8..., spiral_eval.py 813860d7..., subset_wrong_area.py 8890d16d...,
  fit_vertex_dump.py 80e8b55c..., **eval/wholescroll_pool.py dfdb1f57...** (new scorer, below). All recipe shas equal
  test 8's. Repo HEAD aba7d02, working tree uncommitted; the shas are the operative record.

### 2b. Scoring (all frozen)
- Per band: reference sample `data/scratch/wholescroll/build_ref.py <z0> <z1>` (= spiral_eval.reference, stride 2,
  labels only), per-arm vertex dumps `release/fit_vertex_dump.py --scroll p1667 --z0 --z1 --tol 4`
  (pods/collect.sh), then **`eval/wholescroll_pool.py`** (new file). It reproduces, per band, exactly
  `subset_wrong_area.py --split 34 --own-offsets` (inner/outer) and method_table's dump-based seam switches; checked
  before this freeze on the test-8 band, where it returns -53.333 [-54.271, -52.378] inner and -59.14 switches,
  identical to subset_s1.json / spiral_s1.json.
- **Pooling (scroll-wide):** per band, offsets are refitted per band (each band is its own fit). Pooled wrong area =
  sum of wrong matched vertices / sum of matched vertices over the bands (the vertex sample is uniform on the mesh, so
  this is area-weighted). Bootstrap: each band's 30 deg x 250 z blocks resampled independently (rng 1000 + band
  index, 1000 resamples), paired between arms; 95 % percentile CI.
- **Areas:** labelled human area (merged_v4 quads whose four corners share one k, mean z in the band; cm^2 at
  9.596 um); **correctly numbered area** = labelled area x share of labelled vertices that are matched and on the right
  winding (own offsets), inner and all-k; **numbered fit area** = sum of `area_cm2` over the fit's w*_spliced
  surfaces (B6 has no meshes left, so its fit area is n/a and the pooled fit area is given for the bands that have it).

### 2c. Primary claim and pass rule
**Primary endpoint:** pooled over the five fresh bands (B1, B2, B3, B5, B7), seed pair s1 (base s1 vs M2 s1): inner
(k < 34) own-offset seam wrong area, M2 - base.

**PASS** if all three hold:
1. pooled s1-s1 95 % CI upper bound < 0;
2. pooled seam switches per 1000 (all vertices), M2 - base, point estimate <= +5 (the test-2 switch margin);
3. pooled s2-s2 (base s2 vs M2 s2) inner difference point estimate < 0.

Otherwise **FAIL**. If a fresh band has no completed s1 pair after the allowed resumes (2e), the primary is computed
over the remaining fresh bands and reported as "PASS/FAIL (incomplete: band missing)"; the missing band's labelled
area counts as not correctly numbered for both arms in the area tally.

### 2d. Secondary read-outs (reported, not part of the claim)
1. **Per band** (all 7, fresh flagged): inner own-offset base / M2 / diff [CI], switches, outer own-offset diff,
   unmatched share, labelled cm^2, correctly numbered cm^2 base vs M2, numbered fit cm^2; the test-2 pair rule
   (CI upper < 0 and switch diff <= +5) per band and pair; **count of fresh bands passing** (s1 and s2).
2. **Scroll-wide, all 7 bands** (s1-s1 and s2-s2): same pooled metrics; **flagged as not fresh** because B4 and B6
   were seen before (B4 results and B6 results are already published in the drafts).
3. **Outer (k >= 34) own-offset**, pooled fresh: reported; "outer regression" flagged if its CI lower bound > 0. The
   outer labels are not CT-supported (k >= 37).
4. **Scale:** scroll-wide correctly numbered labelled area (cm^2), M2 vs base, and the numbered fit area (cm^2) with
   the share of it that has labels (labelled area / fit area).
5. **Figure:** `release/figures/p1667_wholescroll_overview.png`: per band inner wrong area base vs M2 (both seeds)
   along z with CIs and the fresh/used flag, plus a panel of correctly numbered area per band.

### 2e. Running rules
- No label metric (spiral_eval, subset_wrong_area, wholescroll_pool, fit_vertex_dump summaries) on a fresh band
  before this section is frozen. Reference samples (labels only) and coverage counts are fine.
- No reruns except resuming a failed pod from its checkpoint with the same env (a new pod may be created for that; it
  keeps the arm name). A band whose arm still fails after 2 resumes is reported as not fitted.
- The base pods of a band may start before its chain set is frozen (they do not read it); the M2 pods are created
  after the band's chain sha is logged here.
- Pods vss-full-<arm>, 1 x 48 GB GPU (A6000 / L40S / RTX6000Ada / L40, <= $1.3/h, non-spot). Each pod is terminated by
  pods/collect.sh once its arm is pulled; a final pod list confirms none is left.

### 2f. Predictions (honest, written before any output)
- B2 and B3 (mid-scroll, well labelled, like B4/B6): inner diff -25 to -55 pp each; both pass the per-band rule.
- B5 (500 slices): like its neighbours, -25 to -55 pp; a short band leaves the fitter less z context, small risk.
- B1 and B7 (ends): uncertain. Fewer labelled turns, a clamped umbilicus over 270 / 200 slices, tapering sheets.
  P(per-band pass) ~ 0.6 each; one of them could be worse with chains.
- Pooled fresh s1-s1: -25 to -45 pp. **P(PASS) ~ 0.85.** Switches: M2 lower than base (test 2 -63, test 8 -59).
- Outer: small (-6 to +3 pp), as in tests 2 and 8.
- No band has an oracle arm, so a band where the base is already right (little room) is not detected as
  uninformative; its diff would just be small.

## Log (append-only)

### 2026-10-26T04:28:02Z: section 2 frozen
- sha256 of this note up to and including the "## Log (append-only)" line: 061f26a6137af1d1e68ff453bac4749a6bf01629adec61d6ebda06a7919833cc.
- data/scratch/wholescroll/prereg_shas.txt (all recipe, scorer, pod-script and arm-file shas) sha256 bea457351b2acaa7a81c107ee1232806c9bebf6c1b9c5c99483db9b2801f3523.
- At this time: no chain set, fit, vertex dump or label metric exists for B1, B2, B3, B5, B7. Running: B2 fit inputs
  (prep_inputs) since 04:24:47Z; no mask yet. The only label-side outputs are the coverage counts in section 1a.

### 2026-10-26T04:31:06Z: operational note (no recipe change)
- At 04:28Z our B2 mask started while another agent's GPU lease (14 GB, from 04:26:33Z) was already held, putting the
  combined leases at 24 > 21 GB. Stopped it at 04:31Z (killed by PID, 9 of 242 blocks done; predict_band resumes
  from its block markers) and released our lease. data/scratch/wholescroll/prep_all2.sh replaces prep_all.sh: same
  prep_inputs and run_band_prep.sh commands, but each mask waits until the other leases + 10 GB fit in 21 GB.
  The reference samples for the five fresh bands were built at 04:30Z (labels only; B1 has k 11-32, so no outer
  vertices there).

### 2026-10-26T08:37:58Z: deviation (before any output it governs): mask start rule
- Section 2a wrote the mask range as [floor128(z0 - 256), ...). predict_band.py asserts that the start is a multiple
  of 384 (level-1 chunks); tests 2 and 8 started at 6144 and 4224, both multiples of 384, so their actual rule is
  floor384(z0 - 256). floor128 gives the same start for B2 (1152), B3 (2688) and B7 (7680), but 128 for B1 and 5632
  for B5, which predict_band refused (B1 failed at 08:30:13Z on the assertion, nothing written). B1 and B5 masks are
  made with data/scratch/wholescroll/run_band_prep2.sh (sha256 1aa0001871a892ad14e33f92180f6f0d0bae6699e6831572cda2b00e822a9031; = run_band_prep.sh with floor384), i.e. over
  z 0-1756 (B1) and 5376-6756 (B5), queued by prep_rest.sh (c07e98415e9f8ce197a743f9c6f227b556fb166f71322327fe5f51b9b395d93b). No chain set or fit of B1/B5 existed.
- Operational: the first wsf1500_base pod (<pod-id>) sat in PROVISIONING for 31 min without ssh; deleted at 08:27Z and
  the arm relaunched on a new pod (<pod-id>), same env. pods/watchdog.sh (3aff0c28de48a9eda44e8de83cc2ffadc2c09994dead96f483c7231f495b472c) now does this automatically (delete,
  re-list, relaunch; at most 3 tries per arm).

### 2026-10-26T09:12:38Z: B z1500-3000 M2 chain set frozen (before its M2 pods are created)
- data/scratch/wholescroll/windings/p1667z1500_m7_ray_c/relative_windings.json sha256 de6b44e1a92ec75df17303545f060e3fb17dd7b04dc7233c9888f84ecb2ab9a0
  (unchanged copy of autowind_z1500/relative_windings.json; chains log logs/chains_z1500.log; CHAINS_RC 0 2026-10-26T09:12:10Z).

### 2026-10-26T09:55:39Z: B z3000-4500 M2 chain set frozen (before its M2 pods are created)
- data/scratch/wholescroll/windings/p1667z3000_m7_ray_c/relative_windings.json sha256 0bd2cc58407be28adcef67572532f012265dfda6304f0caff06db680c1f9d909
  (unchanged copy of autowind_z3000/relative_windings.json; chains log logs/chains_z3000.log; CHAINS_RC 0 2026-10-26T09:55:31Z).

### 2026-10-26T09:58:39Z: B z8000-9000 M2 chain set frozen (before its M2 pods are created)
- data/scratch/wholescroll/windings/p1667z8000_m7_ray_c/relative_windings.json sha256 82dbf46f5ec4f1037bbbd28de05a7846d599788ee61eeb5e9d0964669149d435
  (unchanged copy of autowind_z8000/relative_windings.json; chains log logs/chains_z8000.log; CHAINS_RC 0 2026-10-26T09:57:57Z).

### 2026-10-26T10:27:40Z: B z6000-6500 M2 chain set frozen (before its M2 pods are created)
- data/scratch/wholescroll/windings/p1667z6000_m7_ray_c/relative_windings.json sha256 fca951d472c5b64706563276e40a4e456ef6bf627aaa962f51b55c3174baf27d
  (unchanged copy of autowind_z6000/relative_windings.json; chains log logs/chains_z6000.log; CHAINS_RC 0 2026-10-26T10:26:41Z).

### 2026-10-26T10:33:40Z: B z400-1500 M2 chain set frozen (before its M2 pods are created)
- data/scratch/wholescroll/windings/p1667z400_m7_ray_c/relative_windings.json sha256 867e7e0da6cbb1e012d3f73b636ffd2f45594a50597c51971989f0aa4659efd1
  (unchanged copy of autowind_z400/relative_windings.json; chains log logs/chains_z400.log; CHAINS_RC 0 2026-10-26T10:33:18Z).

### 2026-10-26T11:08:06Z: deviation (coordinator decision; before any whole-scroll fit result is pulled): GPU types
- Why: since ~08:40Z Prime has offered almost no 48 GB cards (11:00Z: one A100 40GB, one A10 24GB, two H200 at
  $5.40/h, CPU nodes); 16 of 20 arms were waiting on the 2e type list. State at this entry: 4 fits running on A6000
  (wsf1500_base, wsf1500_bases2, wsf3000_base, wsf8000_m2s2); no arm has been pulled (no pulled.ok, vertex dir
  empty), no label metric on any fresh band.
- Change to 2e: an arm may run on any single GPU with >= 48 GB VRAM (e.g. A6000, L40S, L40, RTX 6000 Ada, A100 80GB,
  H100 80GB, H200 if <= cap) at <= $2.50/h, non-spot. Same fitter bundle, env files and settings; nothing else changes.
- Concurrency cap: at most 8 live vss-full-* pods at a time (leaves cards for the separate vss-an-* run).
- Budget unchanged: $250 total cap; internal stop at $200 (no new pod if spend so far plus the live pods' burn over a
  further 6 h would pass $200).
- Reporting: each arm's GPU type goes in the results table. GPU-type check (secondary, descriptive): per band, any arm
  pair whose two arms ran on different GPU types is flagged, and the s1 vs s2 spread of A6000-only pairs is compared
  with mixed-type pairs. The primary rule (2c) is unchanged.
- Implementation (new files; frozen ones untouched): pods/grab_coord.py replaces the 16 per-arm grab processes and
  pods/regrab.sh; pods/watchdog2.sh replaces watchdog.sh (deletes stuck pods; the coordinator re-creates them).
  Shas in the next entry.

### 2026-10-26T11:10:10Z: operational: process swap for the GPU-type deviation (no recipe change)
- 11:09Z: stopped by PID regrab.sh (a3c5c13b..., ran 11:06-11:09, relaunched nothing), watchdog.sh, and the 16
  pending launch.sh + grab_pods.py processes (none had created a pod; re-listed: only the 4 A6000 pods live).
- Started data/scratch/wholescroll/pods/grab_coord.py (sha256 46c9778e2d0305f7265106d2e61a2eb901701900c2fc7946e5756f7ca06fdf69),
  boot2.sh (829b84b8544cba42af5b789bea5a125dcc59f55df075cfac8f4d606c03c1fa03, = the boot half of launch.sh) and
  watchdog2.sh (16f6ce49bfed3b5cbdba46fef355b79167cc8540c7b9b4653d809b4fd4358c59, = watchdog.sh without the relaunch).
  grab_coord: >= 48 GB VRAM, <= $2.50/h, non-spot, >= 6 vCPU, >= 40 GB RAM; <= 8 live vss-full-* pods; budget gate
  as in the entry above; s1 pairs first; at most 4 pods per arm. Running pods and collect.sh are untouched.

### 2026-10-26T11:19:08Z: operational: host rule for the GPU-type deviation (no recipe change)
- 11:11:59Z grab_coord created wsf8000_base on a GH200 96GB (lambdalabs, $2.29/h, pod <pod-id>). Host: aarch64,
  driver 570.148.08; the fitter env (torch 2.13 cu130) needs driver >= 580, so pod_setup reported cuda_available false.
  Stopped its boot and grab_coord.py by PID, deleted the pod at 11:17:51Z (~6 min, ~$0.22). No fit ran on it.
- Replaced by grab_coord2.py (sha256 58371e261dc8234bde31c99b59d53f65c50c76f7c61ed9a67ce931c708875716) and boot3.sh
  (abb664dd612e3c49f18acdb2d3c3157fa93c1130a456ad02c23f3eb746c01cd7): same rules as grab_coord.py, plus skip aarch64
  types (GH200/GB200), provider lambdalabs (old drivers, as on 2026-10-19), and pairs in pods/bad_offers.txt; boot3
  checks x86_64 and driver >= 580 over ssh before setup, and deletes the pod (adding its provider/type to
  bad_offers.txt) if the check fails or setup ends "no CUDA". The fitter bundle and settings are untouched.

### 2026-10-26T20:18:16Z: operational: all s1 pairs in; commitment before the first pooled read-out
- Pulled and terminated: all 10 s1 arms (5 fresh bands x base/M2) plus wsf1500_bases2, wsf3000_bases2,
  wsf8000_bases2, wsf8000_m2s2. Still to run: wsf6000_bases2, wsf400_bases2, wsf1500_m2s2, wsf3000_m2s2,
  wsf6000_m2s2, wsf400_m2s2 (no >= 48 GB offer on Prime since 16:45Z; 0 live pods, burn $0/h).
- Before computing anything pooled: all six remaining s2 arms will be run under the same rules whatever the s1
  read-out says (grab_coord2.py keeps polling; budget gate unchanged). The s1 pooled numbers computed now are
  provisional for the verdict, which needs condition 3 (s2-s2); nothing in the recipe, scoring or pass rule changes.

### 2026-10-27T02:00:13Z: scheduling-only change (coordinator decision): test-13 priority
- The test-13 confirmation (vss-an-* pods: f7z_base, f7z_bases2, f7z_bases3, f7z_oracle, an_e03_c11s1, an_e03_c11s2)
  uses the same >= 48 GB / <= $2.50/h rule and has priority for cards. From 01:59Z a vss-full-* pod is created only
  when data/scratch/anneal/pods/CONF_DONE exists or each of those 6 arms has a live vss-an-<arm> pod or a pulled.ok
  (i.e. all 6 have been served).
- grab_coord2.py (no create in flight, 0 live vss-full pods) stopped by PID; grab_coord3.py started (sha256
  480a52b1ff338fe811d684d89657ae611ca591fd032d48c28c800b8f7848dd0d; = grab_coord2.py + this gate). Recipe, GPU rule,
  cap, budget, scoring and pass rule unchanged. Remaining: the 6 s2 arms listed in the previous entry.

### 2026-10-27T03:58:35Z: operational: test-13 priority gate open since 03:38Z
- data/scratch/anneal/pods/CONF_DONE written 03:37:29Z by the test-13 run (its 6 confirmation fits moved to the local
  RTX 3090; no vss-an pod needed). grab_coord3.py's gate opened at 03:38:30Z; still no qualifying offer on Prime.

## Results (PROVISIONAL, written 2026-10-27 ~06:10Z: seed pair 1 complete; 6 seed-2 arms still queued for GPUs)

Verdict: **pending** (conditions 1 and 2 met on s1; condition 3 needs the s2 pairs of the fresh bands). Read-outs:
`data/scratch/wholescroll/eval/pool_s1_fresh.json`, `pool_s1_all.json` (eval/wholescroll_pool.py, sha dfdb1f57...).

### Primary (pooled over the 5 fresh bands, seed pair s1)
| quantity | base | M2 | M2 - base [95 % CI] | rule |
|---|---|---|---|---|
| inner (k < 34) own-offset seam wrong area, % | 77.6 | 35.0 | **-42.6 [-44.0, -41.3]** | 1: CI upper < 0, **met** |
| seam switches per 1000 (all vertices) | | | **-45.5 [-49.7, -41.7]** | 2: <= +5, **met** |
| outer (k >= 34) wrong area, % | | | -2.8 [-3.8, -1.7] | 2d.3: no outer regression |
| s2-s2 inner diff, point | | | pending | 3: < 0, **pending** |

### Per band (seed pair s1; inner own-offset wrong area %, base -> M2)
| band (L2 z) | fresh | base | M2 | diff [95 % CI] | switches diff | outer diff [CI] | pair rule | labelled inner cm^2 | correct inner cm^2 base / M2 |
|---|---|---|---|---|---|---|---|---|---|
| 400-1500 | yes | 75.3 | 25.6 | -49.7 [-54.5, -44.3] | -52.6 | n/a (no k >= 34 labels) | pass | 81 | 14 / 41 |
| 1500-3000 | yes | 76.6 | 33.5 | -43.2 [-45.0, -41.3] | -55.4 | -3.2 [-5.8, -0.6] | pass | 165 | 28 / 80 |
| 3000-4500 | yes | 79.2 | 19.6 | -59.6 [-62.1, -57.4] | -62.9 | -1.9 [-3.1, -0.7] | pass | 166 | 26 / 100 |
| 4500-6000 (test 8) | no | 78.9 | 25.5 | -53.3 [-54.3, -52.4] | -59.1 | -2.9 [-3.8, -2.0] | pass | 167 | 29 / 98 |
| 6000-6500 | yes | 77.8 | 25.0 | -52.8 [-54.2, -51.1] | -47.6 | -6.0 [-9.9, -2.2] | pass | 55 | 10 / 33 |
| 6500-8000 (test 2) | no | 78.9 | 50.7 | -28.2 [-29.4, -26.9] | -63.3 | -6.4 [-8.4, -4.4] | pass | 166 | 25 / 59 |
| 8000-9000 | yes | 78.5 | 75.1 | -3.4 [-4.6, -2.5] | +3.8 | -1.2 [-2.4, 0.0] | pass (marginal) | 91 | 15 / 17 |

Fresh bands passing the per-band pair rule (s1): 5 / 5. B7 (z 8000-9000) passes only formally: M2 removes 3.4 pp
there. That band is where the pinned fitter extrapolates the umbilicus (control points end at z 8800; section 1a
risk), and its M2 fit is almost as wrong as base.

### Scroll-wide, all 7 bands (s1; not fresh: includes test-8 and test-2 bands)
- inner 78.1 -> 35.9 %, diff -42.2 [-43.1, -41.3]; switches 131.9 -> 80.3 per 1000 (-51.6 [-55.1, -48.6]);
  outer -3.8 [-4.5, -3.1]; unmatched 27.2 -> 28.7 % (+1.5 [1.1, 1.8]).
- Areas: labelled human area 1183 cm^2 (inner 891). Correctly numbered inner area 148 cm^2 (base) -> **428 cm^2 (M2)**;
  all-k 195 -> 482 cm^2. Numbered fit area (6 bands with meshes; test 2's band has none left): base 3218, M2 2677 cm^2.

### Figure
`release/figures/p1667_wholescroll_overview.png` (release/fig_p1667_wholescroll.py): s1 only for now; s2 points are
added when pool_s2_all.json exists.

### Cost (to 2026-10-27 06:00Z): $60.80 over 16 pods (estimate $55-90; cap $250)
| arm | GPU | $/h | h | $ |
|---|---|---|---|---|
| wsf1500_base (stuck, deleted) | A6000 | 0.54 | 0.53 | 0.28 |
| wsf1500_base | A6000 | 0.54 | 6.25 | 3.38 |
| wsf1500_bases2 | A6000 | 0.54 | 5.42 | 2.93 |
| wsf3000_base | A6000 | 0.54 | 6.66 | 3.60 |
| wsf8000_m2s2 | A6000 | 0.54 | 3.89 | 2.10 |
| wsf8000_base (GH200, deleted: driver 570) | GH200 96GB | 2.29 | 0.10 | 0.22 |
| wsf8000_base | A100 80GB | 2.16 | 2.95 | 6.37 |
| wsf6000_base | A100 80GB | 2.16 | 2.73 | 5.91 |
| wsf400_base | A100 80GB | 2.16 | 2.98 | 6.43 |
| wsf1500_m2s1 | A100 80GB | 2.16 | 3.54 | 7.66 |
| wsf3000_m2s1 | RTX 6000 Ada | 1.439 | 2.88 | 4.14 |
| wsf8000_m2s1 | A6000 | 0.54 | 5.04 | 2.72 |
| wsf6000_m2s1 | RTX 6000 Ada | 0.75 | 2.32 | 1.74 |
| wsf400_m2s1 | A6000 | 0.54 | 5.07 | 2.74 |
| wsf3000_bases2 | A100 80GB | 2.16 | 3.39 | 7.33 |
| wsf8000_bases2 | A100 80GB (PCIe) | 1.20 | 2.72 | 3.26 |
Hours are create -> terminate at the offer price logged at create (Prime bills the datacrunch A100 at $2.2148/h,
so those rows are ~2.5 % low). Live vss-full pods at this writing: 0.

### GPU-type check (deviation 11:08Z), s1 pairs
Mixed-type s1 pairs: B1 (A100 / A6000), B2 (A6000 / A100), B3 (A6000 / RTX 6000 Ada), B5 (A100 / RTX 6000 Ada),
B7 (A100 / A6000); none is A6000-only, so the planned A6000-only vs mixed spread comparison waits for s2.

### 2026-10-27T09:55:57Z: deviation (coordinator decision; hardware and scheduling only), logged before any local fit
- **What:** the 6 remaining s2 arms (wsf6000_bases2, wsf6000_m2s2, wsf400_bases2, wsf400_m2s2, wsf1500_m2s2,
  wsf3000_m2s2) may also run on the local RTX 3090, with the pod-identical stack test 13 built
  (data/scratch/anneal/local_probe/ext: villa 0fb38c45 + fit_spiral_shell_optional.patch, Python 3.14, torch cu130,
  spiral deps, vcnative), used read-only (no VILLA_PATCH in these envs; PYTHONDONTWRITEBYTECODE=1). grab_coord keeps
  polling Prime; each arm is taken by whichever path claims it first (atomic claim file pods/claimed_<arm>), never both.
- **Unchanged:** recipe (frozen env files, sha-checked against prereg_shas.txt before each fit; unchanged
  spiral/run_arm.sh 0310894f... and export_fine.sh 4 779d504b...; overrides.json checked after each fit
  = recipe.toml + band + the env's FIT_OVERRIDES), chain sets (shas in this log; the M2 ones are sha-checked), inputs
  and masks (local copies of data/p1667_ws/z<z0>/{inputs,masks}, the same files the pods receive), scorer, pass rule,
  budget (local compute $0).
- **When:** local fits start only after test 13 is finished with the GPU: data/scratch/anneal/conf_table_an.json exists,
  no test-13 lease in ~/.gpu-leases/, and no test-13 fit process on the GPU. No preemption or sharing.
- **How:** one fit at a time under lease ~/.gpu-leases/vss-full-local.json (max_mem_gb 21; only taken when no other
  lease exists); nvidia-smi logged every second; the fit is stopped if its process VRAM reaches 19,500 MiB, and a stop
  is reported, not retried with changed settings. Order: wsf6000_bases2, wsf6000_m2s2 (500 slices, memory measured
  first), wsf400_bases2, wsf400_m2s2 (1100), wsf1500_m2s2, wsf3000_m2s2 (1500). Outputs in the pod format
  (data/scratch/p0139_spiral/<arm>/ + pulled.ok + gpu.txt) and the same release/fit_vertex_dump.py call as collect.sh
  (data/scratch/wholescroll/vertex/<arm>.npz; meshes kept for the fit areas).
- **Reporting:** GPU type recorded per arm ("local RTX 3090"); mixed-GPU pairs flagged as planned (11:08Z entry).
- **Memory risk:** test 13's 1000-slice Paris4 band peaked at 13.0 GB process VRAM; these bands go up to 1500 slices.
  If a 1500-slice arm hits the guard it is reported and left to the pod path.

### 2026-10-27T10:01:19Z: operational (local-3090 deviation above): process swap and local runner
- grab_coord3.py stopped (PID 724156) and replaced by data/scratch/wholescroll/pods/grab_coord4.py sha a0ae79f6d67b24cd2d7765daa59b2582df4b250dfc06fbbec822c75a0269d4ab
  (= grab_coord3 + claim files pods/claimed_<arm>: written atomically before any create; arms claimed "local" are
  skipped; s2 order largest band first, the reverse of the local order). 0 vss-full pods live at the swap.
- Local runner data/scratch/wholescroll/local_ws.sh sha 3cf55b558696333259d44d7127de2576c2b45afdda2ea3dff24c82da6f2d0092 (new file; modelled on test 13's local_conf.sh).
  It waits for the gate in the entry above, claims each arm just before its fit (so the pod path can still take later
  arms), checks the frozen env sha (prereg_shas.txt), code shas, villa 0fb38c45 with none of spiral/patches
  applied, and the M2 chain-set sha; copies data/p1667_ws/z<z0>/{inputs, masks/m7_L2_th0.2.zarr} into a per-band
  HOME (data/scratch/wholescroll/local/home_z<z0>); guard 19,500 MiB; halts and releases the arm's claim on a guard
  stop or any failure.
