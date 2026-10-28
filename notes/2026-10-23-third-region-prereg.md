# 2026-10-23: third confirmation region, PHercParis4 z 9000-10500 (L2): FINAL pre-registration

**FROZEN (stage 1) at 2026-10-23T08:09:26Z.** The sha256 of every input is in section 10. The stage-2 addendum (section 11)
adds the method chain-set sha. It is appended before any method pod is created. Nothing in sections 1-10 changes after
the freeze. Any deviation goes in a dated addendum at the end, with its reason.

The draft of this note (packing rule, S_pack threshold 21.0) is kept at
`data/scratch/paris4_run/prereg_draft_snapshot.md` (sha 208dce44…). The coordinator replaced it on 2026-10-23
(section 1).

## 1. Method: one fixed recipe, no gm raise, for any scroll (the packing rule is dropped)
- **Method** = the packaged auto-winding chains with `--raise off`. Fit them with the villa 0fb38c45 fit_spiral recipe
  plus the chain overrides of PHerc1667 M2. No recipe choice depends on the region.
- **Why the rule was dropped.** On the pre-registered endpoint type (own-offset seam wrong area, method − base), no
  raise was at least as good as raise on all three development and confirmation regions:
  - PHerc0139 band 1: tie (+1.0, n.s. vs arm23);
  - PHerc0139 band 2: better than the gm raise, −9.8 [−13.7, −6.2], and tied with the oracle;
  - PHerc1667: no raise (M2) passed.
  Evidence: notes/2026-10-23-noraise-p0139.md and notes/2026-10-22-confirmation-region.md §11. The draft's
  contingency 1.3 applies: the rule and threshold are dropped, not refitted.
- **S_pack** is reported as a descriptive number only:
  - PHerc0139 18.5 / 18.5, PHerc1667 24.0;
  - Paris4 S_mask 19.0 [19.0, 20.0], with S_pack computed from the autowind crops after the chain run if their format
    allows it.
  It plays no role in any decision.

## 2. Region and inputs
- **Volume:** PHercParis4 (Scroll 1), volume 20260411134726 (2.400 um).
- **Working frame:** L2 = 9.6 um, shape 18946 × 8174 × 8174 (zyx), p_L2 = p_2400 / 4.
- **Band:** z 9000-10500 (L2), full slice.
- **m7 mask:** the public bucket `PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr`.
  - Only the chunk rows covering the band ±256 were byte-copied: L0 rows 45-56, L1 23-27, L2 11-14, L3 5-7; 8289
    chunks, 2.5 GB.
  - Local copy: `data/paris4/masks/m7_L2_th0.2.zarr`, with the bucket metadata saved next to it.
  - No GPU was used.
- **Lasagna nx / ny / grad_mag:** the public run `20260411134726-lasagna-20260419180421-L2`, group 4 (scale 16), which
  maps 1:1 onto target group 2 with an identity affine.
  - Built by `spiral/prep_inputs.py all --scroll PHercParis4 --z0 9000 --z1 10500 --margin 256` into
    `data/paris4/inputs/`.
  - grad_mag is an input of the fitter only. The method chains do not use it (`--raise off`).
- **CT occupancy for the SDT:** 2.400 um level 5, max-pooled into `data/paris4/inputs/ct.zarr/4` (1185 × 511 × 511,
  28.3 % occupied).
- **Umbilicus:** the public `20260411134726-umbilicus-20260524235033.json` / 4, saved as
  `data/eval/paris4/umbilicus_paris4_L2.json`. It is not derived from any mesh.
- **Spiral sense: CW.** The outward d-theta sign is +1 in all 7 sampled reference groups (sense_check method), which is
  the opposite of PHerc0139 / PHerc1667 (ACW). This corrects the draft's §3 note.

## 3. Reference and scorer
- **Labels:**
  - Source: the organiser set 20260701, 28 winding-group meshes w010-w129 (tifxyz_original, L2 xyz).
  - Built with `data/scratch/third_region/paris4_labels.py` into `data/eval/paris4/labels_w0701.npz`, one k per
    vertex.
  - Provenance caveat: this is an organiser spiral-fit output, not a raw human trace. See draft §4 and the
    disclosures in section 9.
- **Scorer hook:** `eval/spiral_eval.py --scroll paris4`. It builds one surface per k, keeping k in its group's named
  range (the spill is dropped).
  - The raw grid (≈ 18.5 vox per step, rows not constant-z) is **upsampled 4x bilinearly**, to ≈ 5-vox spacing, before
    sampling. A new vertex is valid iff all its cell corners are valid.
  - The reference is sampled at **stride 2** (≈ 10 vox): 12,507,086 vertices and 24,885,164 adjacent pairs; 5.8 % are
    contacts.
  - Cache: `ref_paris4_z9000-10500_s2_m0_v2.npz`.
- **Scoring tolerance:** tol 4, the seam-aware residual and BootV (30° × 250 z, 1000 resamples, seed 1), all as
  PHerc1667.
- **Primary-endpoint input:** `release/fit_vertex_dump.py --scroll paris4`. It streams surfaces one at a time and keeps
  the full step-4 fit resolution, with no decimation.
- **Single-fit spiral_eval runs** (ray pairs, secondary) use `--min-edge 8` decimation for memory. Paired PHerc1667
  scoring peaked at 13 GB with 1.1e8 quads, and the Paris4 slice is about 4.6x larger in area.
- **Oracle ceiling** (merged-mesh oracle from the labels, `--min-edge 8`): ray pairs acc_matched 99.94 [99.91, 99.97] %, coverage 99.69 %; vertex wrong area (seam) 0.18 [0.12, 0.24] %, with open sheets 0.01 % and contacts 2.89 % (decimation); switches 2.31 / 1000; unmatched 0.23 %. Runtime 880 s, peak 7.0 GB (`data/scratch/paris4_run/score_oracle_z9000.json`). The scorer machinery is sound on Paris4.
- **PHerc1667 reproduction with the hooked scorers:** all **identical** (`data/scratch/paris4_run/repro_compare.py`, outputs in `data/scratch/paris4_run/repro/`):
  - spiral_eval oracle vs segments z 6500, compared with the stored json;
  - fit_vertex_dump of c1667_base_fig;
  - subset_wrong_area, new vs frozen copy;
  - label_subsets p1667 (sha 37a7f179) and ct_gap_audit count p1667 (sha 236bbe01);
  - pairs_ksplit, new vs frozen copy;
  - **PHerc1667 base vs oracle** subset scores from the stored dumps;
  - the full PHerc1667 method table (M1 fail, M2 PASS) recomputed with the hooked spiral_eval, subset_wrong_area
    and pairs_ksplit.
  The base and oracle meshes were deleted earlier, so the paired mesh-level base-vs-oracle run cannot be redone.
  The dump-level reproduction plus the oracle-vs-segments mesh run cover both code paths.
  - Frozen copies of every PHerc1667-frozen eval file are in `release/frozen/*_p1667_freeze_<sha8>.*`.
  - The hooks are backward compatible: the `--scroll` choices are extended, and `--primary-k` is optional.

## 4. Label audit and the primary winding set (from audits only; no fit exists)
### 4.1 Label consistency
`eval/label_subsets.py --scroll paris4` on the upsampled reference wrote `subsets_bandparis4.json`.
- 111 of 120 windings are consistent.
- Inconsistent: 33, 61, 123-129.
- The draft's scratch run on the raw grid found 87 consistent, because the sparse grid fails the spacing test more
  often. The hooked, frozen run is the one used.

### 4.2 m7 / grad_mag audit
Command: `eval/ct_gap_audit.py --scroll paris4`. Outputs: `ct_counts_bandparis4.json`, `ct_profiles_bandparis4.npz`.

Setup:
- 6 z × 24 angles, avoiding the seam; 437 ray segments.
- CT is not sampled: whole rays over an 8174² slice would stream about 100 GB, and the PHerc1667 verdict rested on m7
  and grad_mag. So n_ct = 0, and the inputs are m7 and grad_mag.
- grad_mag is calibrated so that the median dk = 1 pair = 1 turn (0.690 per turn).

Results:

| span | rays | m7 runs between (expected) | gm turns (expected) |
|---|---|---|---|
| all dk = 1 | 11,410 | 0.11 (0); P(0) 0.90 | 1.00 |
| all dk = 2 | 5,654 | 0.87 (1) | 2.08 |
| all dk = 3 | 3,676 | 1.66 (2) | 3.20 |
| all dk = 5 | 2,101 | 3.36 (4) | 5.46 |
| all dk = 10 | 1,054 | 7.61 (9) | 11.04 |

- **m7 undercounts uniformly**, by about 0.14 per intervening sheet at every span size (−0.13 / −0.34 / −0.64 /
  −1.39). Contact merges at tight packing explain this; label drift does not.
- **grad_mag's per-label scale varies about 3x along k:**
  - 0.6-1.0 for k 10-60;
  - 1.1-1.4 for k 60-99;
  - 0.55 at k 102-105;
  - 1.6-1.8 at k 109-118.
- **Outer shell k 125-129 looks over-labelled**, the PHerc1667 k ≥ 37 signature: gaps 6.6-10 vox and gm 0.12-0.21 per
  label.

### 4.3 Primary winding set (rule, applied by `data/scratch/paris4_run/primary_kset.py`)
- A winding k is **CT-reliable** iff every dk = 1 pair touching it has:
  - mean extra m7 runs ≤ 0.3, and
  - median gm turns in [0.6, 1.5].
- In addition, no dk = 10 span containing k may show **drift**. A span drifts when m7 runs **and** gm turns both
  deviate from their expected values (9 and 10) by more than 1, in the same direction.
- **Primary = CT-reliable ∩ label-consistent**, the union of the contiguous ranges (the draft required ≥ 15
  windings).
- Failing dk = 1 pairs:
  - gm: 11-13, 88-89, 102-105, 109-118, 120-121, 125-129;
  - m7: 110-112, 118-119 (0.302), 122-125.
- Drifting span: w10-w20 (m7 −1.27, gm −2.17).
- **PRIMARY_K = 21-32, 34-60, 62-87, 90-101, 106-108: 80 windings.** This is hard-coded in
  `eval/method_table_paris4.py`.
- **Disclosed refinement:** the draft said only "dk = 10 spans drift by at most ±1". After reading the audit, and
  before any fit or chain exists, the drift reading was fixed to "both signals agree", as in the PHerc1667 §4b verdict.
  m7 alone was not used because of the uniform contact-merge undercount above, and grad_mag alone because of its
  non-stationary scale.
  - With m7 alone, 8 of 12 spans would fail and the primary set would shrink to about 30 windings.
  - The all-consistent set is reported as a secondary (section 6.4), so the reader can see the effect of this choice.

## 5. Arms (env files `spiral/arms/p4_*.env`; shas in section 10)
Shared lines:
```
SCROLL=PHercParis4
VOL=20260411134726
UM=9.6
SENSE=CW
Z0=9000
Z1=10500
MASK_TAG=m7L2
MASK_LOCAL=$HOME/data/PHercParis4/masks/m7_L2_th0.2.zarr
```

| arm | chains | FIT_OVERRIDES | seed |
|---|---|---|---|
| p4_base, p4_bases2, p4_bases3 | none | none | 1 / 2 / 3 |
| p4_oracle | `band_oracle_paris4`: auto_windings oracle from the labels, unchanged tool, loader patched to the raw group grids (`paris4_oracle_chains.py`) | chain overrides | 1 |
| p4_ms1, p4_ms2 (method) | `paris4_m7_ray_c` | chain overrides | 1 / 2 |

- The chain overrides are `{"pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}`.
- **Method chain command** (frozen; tool shas in section 10). It is deterministic, uses no labels, and is run blind:
  ```
  uv run python -m autowind chains --mask data/paris4/masks/m7_L2_th0.2.zarr \
    --umbilicus data/eval/paris4/umbilicus_paris4_L2.json --z0 9000 --z1 10500 --voxel-um 9.6 --raise off \
    --workers 7 --out data/scratch/paris4_run/windings/paris4_m7_ray_c
  ```
  `tools/auto_windings.py` is untouched.
- **Fit:** villa 0fb38c45 fit_spiral, config/recipe.toml, run through run_arm.sh, then export_fine.sh at step 4.
  No reruns, except that a failed pod resumes from its checkpoint with the same env.
- **Disclosure:** the villa defaults shell_outer_winding_idx = 130 and model_gap_expander_num_windings = 130 are
  unchanged. Paris4 may have about 135 windings (the organisers used 135), so the outermost windings may be squeezed.
  This applies equally to every arm, and the primary set ends at k 108.

## 6. Analysis plan (`eval/method_table_paris4.py`; all six fits are scored before any label-based chain diagnostic)
1. **Oracle informativeness, checked first.** Compute the primary endpoint with the oracle in place of the method,
   against each base seed.
   - If any oracle − base upper CI is ≥ 0, the region is **uninformative**: no claim is made, and every number is still
     reported.
2. **Primary endpoint**, per (method seed, base seed), 2 × 3 = 6 combos:
   - the quantity is method − base own-offset seam wrong area on the PRIMARY_K windings. The offsets are refitted on
     those vertices alone (`subset_wrong_area.sub_scores`);
   - the CI is the paired BootV (30° × 250 z, 1000 resamples, seed 1);
   - the two-sided bootstrap p is 2 min(P(d ≥ 0), P(d ≤ 0)).
3. **Switch margin:** method − base seam switches per 1000 adjacent pairs, all vertices, point estimate ≤ +5.
4. **Decision:**
   - A combo passes iff its upper CI is < 0 and the switch margin holds.
   - **PASS iff all 6 combos pass, p_arm = max p < 0.05, and the region is informative.**
   - There is a single method arm, so **no Holm step is applied**. This is stated here, and was not chosen after the
     fact.
   - Every combination is reported, whatever the outcome.
5. **Pre-registered secondaries**, reported whatever the outcome:
   - **global (shared-offset) seam wrong area on all vertices**, method − base. It **may undercount at tight packing**:
     a single global offset can absorb a coherent ±1 shift, as seen on PHerc0139;
   - **ray pairs** (`pairs_ksplit.py --primary-k`; acc_matched, coverage, acc_all, seam variants), all and primary;
   - **subsets**: own- and shared-offset wrong area on consistent / inconsistent / primary / nonprimary;
   - method − oracle, and the oracle's pair read-out vs base seed 1;
   - **generation-19 control:** the organisers' spiral-output-generation-19, an m7-independent fit (L2 spiral-fit
     geometry, winding IDs 10-135), used as an alternative reference.
     - Per arm: own-offset seam wrong area on the generation-19 vertices in the band with ID in PRIMARY_K, method −
       base, BootV.
     - Also the agreement of labels_w0701 with generation-19 (the fraction of vertices whose k − ID equals the modal
       offset).
   - **2023 human-segment control:** the GP segments 20231005123336, 20231007101619, 20231012184424 and 20231210121321
     (`*-on-20260411134726-2.4um.tifxyz` / 4) are single-sheet human traces without winding numbers. The metric is
     **trace pair accuracy**:
     - take pairs of trace vertices on one radial ray (Δangle ≤ 1.5°, |Δz| ≤ 12) whose unwrapped trace angles differ
       by n full turns (|Δφ − 2πn| < 0.5 rad, n ≥ 1);
     - a pair is correct iff the fit's nearest windings within tol 4 differ by exactly n;
     - report per arm, method − base with a paired block bootstrap over 30° × 250 z blocks, and the labels_w0701
       reference's own accuracy on the same pairs (the noise floor).
   - Both controls use labels or traces only and no fit. They are built after launch and before the fits are scored.
     If either cannot be built (for example, an unmappable geometry), it is reported as "not done", with the reason.
6. **Blinding:**
   - No label-based chain or step metric was computed on Paris4 before this freeze: no dry-run step precision or drift
     on any chain set except the oracle, which is built from the labels.
   - After the freeze the fits are scored first. Label-based chain diagnostics run only after the method table is
     written.

## 7. What was computed before the freeze (disclosed)
- S_mask on five Paris4 bands (draft §1.4).
- Label coverage, label subsets and the m7 / gm audit (section 4).
- The oracle ceiling, and the oracle chains, which are label-derived by construction.
- The method chain run (section 5 command, exact flags) started at 2026-10-23T08:08:18Z, before this freeze.
  - It is deterministic and label-free.
  - No label-based metric has been or will be computed on it before the fits are scored.
  - Only its output sha is appended in stage 2.

## 8. Pods and cost
- **Order:** p4_base (smoke test: time, VRAM, RAM), then p4_oracle, p4_bases2, p4_bases3, then p4_ms1 and p4_ms2.
  The method pods start only after the stage-2 addendum.
- **GPU:** 1x L40S preferred; A6000 / RTX6000Ada fallback. The GPU, nproc and RAM are recorded per pod in
  `data/scratch/p0139_spiral/pods_arms.txt`.
- **Cap: $70 total.**
- **Pod rules:**
  - never touch pods we did not create; re-list after a failed create;
  - terminate after the pull; log everything;
  - score under `flock /tmp/vss_eval.lock`;
  - after dumping, delete the pulled meshes, keeping p4_base and p4_ms1 for figures.
- Commit nothing; publish nothing.

## 9. Disclosures
- The reference is an organiser spiral-fit output, built with a newer villa fitter on m7 tracks. Where it followed m7
  merges that no annotation overrode, our m7-based chains may agree with its errors. That would inflate the method's
  apparent gain; the generation-19 and 2023 controls address this.
- m7's training data is undocumented and may include Scroll 1. The band avoids the GP label z-range (L2 11832-17625).
- The rule threshold and the drop decision come from PHerc0139 and PHerc1667.
- PHerc1667's M2 pass was informed by its dry run, as disclosed there.
- Set 20260623 and generation-19 exist and were not used for the primary labels.
- The Paris4 seam is about 3-10°; the seam-aware metrics use per-bin offsets.
- Reference upsampling (4x bilinear) and `--min-edge 8` for the single-fit runs are Paris4-specific. The primary
  endpoint does not use the decimation.
- The drift reading in 4.3 was refined after the audit and before any fit.
- Two-stage freeze, as PHerc1667 §8/§11. The method chain sha is appended in stage 2, because the chain build (about 3.5 h) runs during the stage-1 launch.

## 10. sha256 of every input (stage 1)
Directories are tree hashes: sha256 over the sorted lines `<relpath> <sha256>` (`data/scratch/paris4_run/sha_manifest.py`). The full list is in `data/scratch/paris4_run/freeze_stage1_manifest.tsv`.

| file | sha256 | size |
|---|---|---|
| `data/eval/paris4/labels_w0701.npz` | b0a5989e1446561b8342c52f79eacc2f89219295a4b2e78722a78fed2e87fdcc | 358513 B |
| `data/eval/paris4/labels_w0701.json` | d0f2c9416c48d0ee2063640131a6384b8fdbdba46ba41ddfce3729de086e3a35 | 14228 B |
| `data/eval/paris4/umbilicus_paris4_L2.json` | bb3c351d7f6fbc5fbc2ee8554a1a021a53bd881e8baf90783e3032e62019c390 | 10929 B |
| `data/eval/paris4/cover_z9000-10500.json` | 5f6e9f0921213a62f187df04bdf1b7db85374635f129f436b25797cf28357f0f | 28367 B |
| `data/scratch/label_audit/subsets_bandparis4.json` | 14417eb3d4df3845bc7dc6673bfa2212a858506c3149d882a187919d64c243ec | 95490 B |
| `data/scratch/label_audit/ct_counts_bandparis4.json` | 3973f29d5680d33bb53c6a989e8cee93486e7f18c9d98e34c4ec19aaa4e830a9 | 6112848 B |
| `data/scratch/label_audit/ct_profiles_bandparis4.npz` | 5da7b802a6d73688b71908d5972547f99122c288e1a83f409136147cb99e7cd5 | 2848442 B |
| `data/scratch/spiral_eval/cache/ref_paris4_z9000-10500_s2_m0_v2.npz` | 1fcc352fe2d7718a1fa136edadead9099b4cad11257aa5d22347085d28326121 | 1373717264 B |
| `data/scratch/paris4_run/score_oracle_z9000.json` | 3a5700d9f9af411ade356ff36e84adcad631cbdec32ba4560fbda55bf2cb45ca | 27297 B |
| `data/paris4/masks/m7_L2_th0.2.zarr.bucket_metadata.json` | 09b37dd002c29fb1b40e245d372ec48da233741dce16276965989fcdc26593f3 | 4000 B |
| `data/paris4/masks/m7_L2_th0.2.zarr` | edd86a832913fcb404612f0c45009c6997f78753c63f9da0e60e82363f3afeb4 | tree:9555 files |
| `data/paris4/inputs/umbilicus.json` | 4f121f4e443e72394d24ff6360bfd7f3ccb43e86acee94f8e13129d65de3c400 | 11156 B |
| `data/paris4/inputs/PHercParis4.lasagna.json` | 78e931689edddd76f1d7a987d765cdfc30ff8179c64537a42403fac3d0bac111 | 4726 B |
| `data/paris4/inputs/PHercParis4_nx.ome.zarr` | 8bc9a45fb2865b78200b7d9f319965241840ae7db854d2537b3c35b787fb3ed4 | tree:22803 files |
| `data/paris4/inputs/PHercParis4_ny.ome.zarr` | 989d04135f4bdfb11be9e86b403034a1f6e53e8f7e95393991e1ffc68cd44565 | tree:22803 files |
| `data/paris4/inputs/PHercParis4_grad_mag.ome.zarr` | 5e4f01393838c07381d6cb0a9fa458c21b34169df732dae040d3ef96bb61dbe8 | tree:22803 files |
| `data/paris4/inputs/ct.zarr` | 4f15a59b48b7711fd3361ef4ff5880dd8c57b5d640aa447e8bb5a4b949b1a9b0 | tree:26 files |
| `data/scratch/paris4_run/windings/band_oracle_paris4/relative_windings.json` | fadf496f86168975f8298da6bb971fc96a5f3847aeaa75ddf3c5f2babe6509b8 | 5221376 B |
| `spiral/arms/p4_base.env` | 2c7d5fecc34e32af1eb3c7c1ab2abfe0a2c3e7e81b3717dd3d78cfc92d4aaaed | 661 B |
| `spiral/arms/p4_bases2.env` | f07b392d3cd49a324e2e077973dd9e5c59d2554385bcca467f5334bfbfe23c86 | 693 B |
| `spiral/arms/p4_bases3.env` | d302fd94d055f19476fe27566066388472933e3ae58cc290a787a286ddd77aaa | 693 B |
| `spiral/arms/p4_oracle.env` | 217a74c29b7d78fdf8fe8c249da70c4ac76ab639d94566bf15b275be631129ec | 914 B |
| `spiral/arms/p4_ms1.env` | 350ac8355f47a74ce7d27b00c13c5ff3d0c7a848a4db24166837a3530bd68b8b | 871 B |
| `spiral/arms/p4_ms2.env` | 6151fdc2b71aac0f043a0e242865c213c0e3c9f647b723ae254f58dfae7ecb17 | 899 B |
| `eval/spiral_eval.py` | 813860d7a033b93e462d5736aceaf50cc9a2f69a8ccfcdea48258eda6cea07b0 | 50588 B |
| `eval/subset_wrong_area.py` | 8890d16d6b4b9d7aeb6044e02a1342bd3cd26a11c7da7da60a8bbc02418b3917 | 8002 B |
| `eval/label_subsets.py` | 5a63f5a30eb840c84a552a14b89c7a5cbeab66162d50a55be05c0c1d85101942 | 13374 B |
| `eval/pairs_ksplit.py` | f23e4728897c3119205cbe4b088e91da41479fe07353debadd377ae8a4ee7b34 | 4824 B |
| `eval/ct_gap_audit.py` | baeb21b3f63b0f22b19d1bf02adbd11c4f80a1fd21e8f98d72bb3d72b08af354 | 21114 B |
| `eval/method_table_paris4.py` | c1bcf3b04c6c7d6a02b0c2ebcafc527e2fd6c3a67efeeb7f3c331355f91cb659 | 7488 B |
| `eval/method_table.py` | 6cfc15dfaf2c4a140b50a99238259180bcc40b1fed3000d5300606b17ad7609e | 7516 B |
| `release/fit_vertex_dump.py` | 80e8b55cdedbfdb42da07ea58884b5ec64f42508a8f5d1c36c527e3cc734131b | 3466 B |
| `tools/autowind/__init__.py` | ae2a786a0170464942bb03d68cd81c745dd5d96bd09c8b43f9ed7317e1e77849 | 651 B |
| `tools/autowind/__main__.py` | b15178c5a330d418d81bec7e7c77f9fca8dc70494c15a2ee44a1c8b59aa21471 | 146 B |
| `tools/autowind/cli.py` | 46331cf626c786f55674ea52a9c3c9008465f65c9cbebab56c8d6f95cbccb824 | 5136 B |
| `tools/autowind/core.py` | a42a9c2ee5fc9ca2950f88a83a26a3d2adbd1a678ed8c40c3cc14f092446d775 | 15797 B |
| `tools/autowind/presets.py` | 64ba80d698ae772e0f81eb66e96c57b4787fbd12b7a39bf750bfc6779055e88d | 2291 B |
| `tools/autowind/test_smoke.py` | e43460522b6b7a060a350b5ef7bbd0831bd65e07f405e1191a6746af499fd0f5 | 5930 B |
| `tools/auto_windings.py` | 8f0973120146e10f30811f28dab6e01c921be96f39273724869b01ddafa44edb | 79864 B |
| `release/AUTOWIND.md` | 05c179619e3d34319c8b1ffd000b4b195f5681c48ef70656b79414c2e40917a9 | 13313 B |
| `spiral/prep_inputs.py` | 7deac6311d8f4b444cfc2e40cee03ae04764b080a360c8b1a95c5bd2658e1af1 | 26721 B |
| `spiral/run_arm.sh` | 0310894f63fd5a2c04f08de79af7d8df664d51a8dc27a322fe690f2bda4366dc | 10585 B |
| `spiral/export_fine.sh` | 779d504b3c8e891d9cfc0214bef92eb9712a3afb73575c3f11adb892caa3b92f | 3109 B |
| `spiral/bootstrap_pod.sh` | 1d29ec4e1193be04dcfde4005384e34f6fd41950967867308d79ac46caa4cb87 | 3039 B |
| `spiral/pod_arms.sh` | 81c5117b1908911644eb4c67bc0bad1c9288ad0de86c7b9c611f15ab617a4249 | 1711 B |
| `spiral/pull_arm.sh` | f7737bcaba8c32a0ea5736433531f3935de679462e2ef69880f8d20e3ef2ee57 | 1786 B |
| `config/recipe.toml` | 9ba8494ef25ba7fd6d60f492f51827015130f8ae55fd7d0a9d63285d762db1f1 | 4004 B |
| `data/scratch/third_region/paris4_labels.py` | 478177309683123af44677cc2b57caddca7734de3f81f84bd81034899d092c48 | 9665 B |
| `data/scratch/third_region/packing_stat.py` | 594863ce4e4727765df221fff67c76305f9ae375e2afe21f31998abf79ea6585 | 5854 B |
| `data/scratch/third_region/packing_stat_mask.py` | 08e6cf517662bb422e4a9f3d97897251e9fc3eecb7efec9be5c4237746721f71 | 8118 B |
| `data/scratch/paris4_run/primary_kset.py` | c51413b5b4958a726fbdae9231cb228ce5e409d6fff240d840a6516421a37ad3 | 2233 B |
| `data/scratch/paris4_run/paris4_oracle_chains.py` | fdf1f6795219706ab566437bb8a894fcecfe54f785b11261937a33bacfe8d33c | 1152 B |
| `data/scratch/paris4_run/run_q5_chains.sh` | 68785f2dc1f78aec4ba6a9b455dbe458d47acf81192528aee03350c669b4289d | 1444 B |
| `data/scratch/paris4_run/sha_manifest.py` | 916dd8c28819499cbf08cc2c58b57d5713e44f31d1ba991f1a2de3e834320cd1 | 1027 B |
| `data/scratch/paris4_run/repro_compare.py` | aba19d39b14580759a9123895437fa947ed340b5d2a03b4623a69c2a18a502c9 | 2753 B |
| `data/scratch/paris4_run/pods/grab_pods.py` | 55117c5e1f627351e3ac98b0c381837483bc2da93c1d9a928c038882d011ca4a | 4405 B |
| `data/scratch/paris4_run/pods/boot_and_launch.sh` | 950737e27e85c078531d05d9473852e8d67b308d06db43dec10054adcfb94bcb | 4340 B |
| `data/scratch/paris4_run/pods/auto_boot.sh` | d948594546b6175f4c9dc95d0a5a8fe515fe6eb61e27fbab530792b86692af69 | 611 B |
| `data/scratch/paris4_run/pods/collect.sh` | 3c0d648a497a72cdce0d227a3753b5865c7b692a610d3a3a35e5f7f71bb93b0d | 3993 B |

Other pins:
- villa fit_spiral 0fb38c45;
- the draft snapshot 208dce44…;
- the method chain command in section 5 (`tools/autowind/*` shas above, `--raise off`).

## 11. Stage-2 addendum (method chain set; appended before any method pod)
**Stage 2 frozen at 2026-10-23T11:03:13Z, before any method pod was created.**
- Command: section 5, exactly. It ran 2026-10-23T08:08:18Z to 11:02:43Z (10,464 s), exit 0, under the lock.
- `data/scratch/paris4_run/windings/paris4_m7_ray_c/relative_windings.json`: **485029131145fcfe67b9497f1fc277ccc53711cb91c3b6e5ba0ab60f7c400d63**
- `autowind_run.json` 21e230f55ce0b01fce850a94c82a70264474cfe65effe78600be1ad33febaa54; `chains.json` 066b01c4043abbe9a3095097bdbad7ef2490249bd5e3df0e7985aaeac5dcf30f; `steps.csv` e854c93171351af9a0ce214cc33be1fd0dd1c9c57c8470f191e32c1913aa94ae; `summary.json` 8414b4363f53b5a79eef0eb1b6e0ec88d5761d673ffcfcb454efde1ea5e469ee; `umbilicus_maskframe.json` b46dde22b9e9f3c51167de6d4b5e63d612df75e6ff74fa45c8be54d330826311
- conf model `tools/auto_windings_conf_clean.json` da3515a85b9bb31efb3cb677b7930fbd97e8369cff6ad8571a8e0ffe8ddf3d2d.
- `tools/autowind/*` is unchanged from section 10.
- Label-free counts: 552 rays, 759 chains, 48,938 points, median 67 points per chain.
- No label-based metric was computed on these chains. They are compared with labels only after all fits are scored.
- Method pods p4_ms1 and p4_ms2 boot with this set. bootstrap verifies the windings sha on the pod.

## Addendum A1 (2026-10-23T08:18:54Z; operational, no analysis change)
- `data/scratch/paris4_run/pods/collect.sh` was edited before any arm was collected. It now deletes a non-figure
  arm's meshes only after the control dump marker `data/scratch/paris4_run/controls/<arm>.done` exists. The reason:
  the generation-19 and 2023-segment controls (6.5) need each fit's meshes at their own vertices, which the reference
  dumps do not cover.
- New sha: 69cbf84f9019b1b010ffb7d190b9c6f448e429e0775a629f36e54a7030ef4129.
- The control builders live in `data/scratch/paris4_run/controls/`. They use labels and traces only and are built
  before any fit is scored.
- Smoke pod p4_base: 1x L40S massedcompute, $0.82/h, pod <pod-id>, created at 08:17:18Z.

## Addendum A2 (2026-10-23T08:58:02Z; control construction, fixed before any fit is scored)
Built from labels and traces only. No arm output was read, nor the method chains.

**2023 trace control**
- 20231005123336 lies outside the band (L2 z 10932-17923). Three segments contribute.
- 237,104 trace pairs from 153,675 trace vertices, in 72 blocks; n = 1-4 turns.
- The trace grid is sampled at stride 2. Rays run every 25 z in 1.5° bins with |z − level| ≤ 6, and one vertex is
  kept per crossing.
- The headline metric counts unmatched pairs as wrong. Matched-only accuracy and coverage are reported alongside.
- The traces meet only reference windings 28-43, so this control speaks to the inner range only.
- Floor (labels_w0701 on the same pairs): acc_all 59.6 % [56.7, 62.8], acc_matched 89.5 %, coverage 66.6 %.

**Generation-19 control**
- Source: `intermediate/tifxyz_normalized` plus `windings.tif`, already in the L2 frame. Only the band rows were
  fetched by byte range.
- 4,936,911 vertices with ID 10-135 in the band; 2,649,596 of them have an ID in PRIMARY_K.
- labels_w0701 agrees with gen-19 IDs only for about IDs 20-39 (87-94 %). Beyond about ID 60, k − ID drifts to −1 to
  −7: gen-19 counts about 6 more windings.
- The labels_w0701 floor on the gen-19 endpoint is 64.2 % [62.2, 66.2] own-offset wrong area.
- So this control measures agreement with a different winding count. It is reported as is, with no claim.

**Scripts and collection**
- Scripts: `data/scratch/paris4_run/controls/{common,build_traces,gen19_fetch,build_gen19,control_dump,control_table}.py`.
- Oracle test: 99.97 % of trace labels are identical to the reference's.
- `collect.sh` now runs `control_dump.py` per arm, under the lock, before mesh deletion. New sha:
  93d6639a590619d5bad076b22972eaa80eb6913c0fb76bb768115950ad0603f0.

**sha256 of the control files**
- common.py 09810568cc2ff57dd77d6c4e14af13d3868953b0038a9235c288d0e637907f4f
- build_traces.py b8dbe13cd8b8e30c29dafe02506889f80989a42cd4692d5a0759835202a784b2
- gen19_fetch.py f3ff9adf4bd424b237989bc1ea82972003cf5643a5d9bd5e175262454d9d045f
- build_gen19.py 4e97c4f573789f79743babecaa61c864ad07396f76df9173d64a3826233d49bf
- control_dump.py eb93f72637b19b180e2551fd997bec13080d41207d28fa3846c53a4999af0439
- control_table.py d7104f856002c8d36b9580e5e90369255eac551994bfe7b9682da5a373c6a868
- traces_2023.npz afe593ad058ba56c5993964da8eea56bdf8d7b59eb9e16e3b1a833d4f3bb961d
- gen19_vertices.npz fd2809bdc4675a3a3aefa6c6ca19712b812d573d17b2e88c1725096b4acf8164

## Results (2026-10-23T17:55Z; eval/method_table_paris4.py -> data/scratch/paris4_run/method_table.json)
**Oracle check: informative.** Oracle − base on the primary windings: −24.7 [−27.9, −21.7], −24.8 and −24.7 for the
three base seeds.

**Primary: FAIL.** All 6 combos are worse than base, by +10.4 to +10.6 points:
- base 69.5-69.6 %, method 80.0 / 80.1 %, CIs about [8.4, 12.6];
- switches +15.6 to +18.4 per 1000, above the +5 margin;
- no combo passes (one arm, no Holm).

**Secondaries**, method − base, seed 1 (seed 2 is the same within 0.5):

| read-out | difference | note |
|---|---|---|
| global seam wrong area | +2.4 [0.6, 4.2] | |
| own-offset, consistent | +1.5 [−0.5, 3.5] | |
| own-offset, inconsistent | +4.9 | |
| own-offset, nonprimary | +1.5 n.s. | |
| ray pairs, acc_matched (primary) | −5.5 [−7.5, −3.4] | |
| ray pairs, coverage (primary) | −13.7 | |
| method − oracle, primary | +35.2 [33.1, 37.4] | |
| 2023 traces, acc_all | **+9.5 [7.0, 12.5]** | method 49 % vs base 40 %, oracle 57 %, reference floor 59.6 %; inner windings 28-43 only |
| generation-19 own-offset wrong area | +10.4 [9.1, 11.7] | floor 64 %; different winding count |

The 2023 traces are the only read-out in the method's favour.

**Label-based chain diagnostics**, run only after all six fits were scored (`data/scratch/paris4_run/chain_diag_compare.txt`):
- step precision 78.9 % (dr = 1: 81.2 %, dr = 2: 47.2 %);
- errors are mostly under-counts at contacts: under_contact 3,569, under_spaced 1,301, over 348;
- drift over 10-20 windings: mean −1.87, only 45.5 % within ±1;
- precision is lowest at θ 315-15° (60-69 %).
- **Reading:** m7 merges contact sheets at Paris4's tight packing (S_mask 19.0, PHerc0139-like), so the no-raise chains
  under-count windings. This is the failure mode the gm raise was built for. The dropped packing rule would have picked
  raise here (expected S_pack 18-18.5 < 21). This is post hoc and not a claim.

**Cost:** six pods, all terminated after their pulls; pods list empty; about $16.9 of $70.
| arm | GPU | hours | $/h |
|---|---|---|---|
| p4_base | L40S | 2.9 | 0.82 |
| p4_oracle | L40S | 3.0 | 0.82 |
| p4_bases2 | A6000 | 6.6 | 0.54 |
| p4_bases3 | L40S | 3.7 | 0.82 |
| p4_ms1 | L40S | 3.0 | 0.82 |
| p4_ms2 | A6000 | 5.6 | 0.54 |

Meshes are kept for p4_base and p4_ms1 only.
