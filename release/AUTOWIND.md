# autowind: automatic relative-winding chains for the villa spiral fitter

`autowind` turns a binary surface-mask OME-Zarr and an umbilicus into `relative_windings.json`. Each chain is one
radial ray of sheet crossings with cumulative `wind_a`, and the pinned villa spiral fitter (0fb38c45) loads it as
unattached strips. On PHerc1667, chains from this recipe passed a pre-registered confirmation in all 6 seed combos
(inner wrong-winding area -27.4 to -29.3 pp, about 86 % of the human-label oracle's gain; notes/2026-10-22-confirmation-region.md). **On tightly packed scrolls the same recipe makes fits worse.** On PHercParis4 (pre-registered, sheet
spacing about 19 vox) it was +10.5 pp wrong area vs no chains, because merged contacts make the chains undercount
(notes/2026-10-23-third-region-prereg.md). Use it where the mask's sheet spacing is about 24 vox or more, or check step
precision on a labelled patch first.

The package (`tools/autowind/`) is a thin driver. The algorithm is in `tools/auto_windings.py` and
`eval/p0139_graph.py`, which are imported unchanged. Their sha256 values (8f097312…, fb7553b1…) are frozen by the
pre-registration, and copies are in `release/frozen/`.

## Install

```sh
git clone <this repo> && cd <repo>
uv sync                         # Python 3.12; numpy, scipy, zarr 3, connected-components-3d, tifffile, s3fs
uv run python -m autowind --help
```
`pyproject.toml` installs the repo in editable mode with uv_build (`module-root = "tools"`), so
`python -m autowind` works from any directory. A side effect is that `tools/` is on `sys.path`.
Without that install, use `PYTHONPATH=tools uv run python -m autowind ...`.

A standalone copy needs:
- `tools/autowind/`;
- `tools/auto_windings.py` and the two models `tools/auto_windings_conf_clean.json` and
  `tools/auto_windings_gm_z4500_clean.json`;
- from `eval/`: `p0139_graph.py`, `p0139_crossings.py`, `p0139_sites.py`, `winding_sites.py` and `spiral_eval.py`.
  These are imported at module level only.

## Inputs

| flag | what |
|---|---|
| `--mask` | Surface-mask OME-Zarr group, as a local path or an `s3://` URL (read anonymously). Level `"0"` is the working frame and values > 0 are sheet. Level `"3"` (8x) is used for the radial extent of each ray. If the store has no 8x level `"3"`, autowind builds it from level 0 once (8^3 any-pooling) and caches it. |
| `--umbilicus` | Villa format: `umbilicus.json` (`{"control_points": [{"x","y","z"}]}`), or a Thaumato `.txt` (`y,z,x` per line, +500 offset), read as villa's `umbilicus.py` reads it. Interpolation is linear in z and clamps outside the control points; villa extrapolates. |
| `--umbilicus-transform` | Optional. A JSON `transformation_matrix` (3x4) that maps the umbilicus xyz into the mask frame, e.g. the published 2.399 um umbilicus onto a 9.362 um mask. |
| `--z0 --z1` | Band in mask level-0 voxels. Rays sit at z = 192k + 96 + {-64, 0, 64} inside [z0, z1). |
| `--voxel-um` | Voxel size of mask level 0. It is recorded in the run file, and autowind warns outside 8.5-10.5 um (see Limitations). |
| `--raise {off,gm,auto}` | Default `off`. `gm` needs `--grad-mag`, a lasagna grad_mag OME-Zarr in the mask frame whose level `"2"` is sampled every 4 voxels, u8 density as written by villa lasagna / `spiral/prep_inputs.py`. `auto` (needs `--grad-mag` too) computes the label-free packing statistic S_pack from the solved crops and applies `gm` at threshold `--auto-gm-thr` (0.263) iff S_pack < `--spack-thr` (21.0 vox), else no raise; see [`--raise auto`](#--raise-auto-spacing-aware-rule). |
| `--out` | Output directory. |
| `--preset {p0139_band1,p1667}` | Fills the inputs of our published chain sets. Explicit flags win. |

The recipe flags default to the confirmed setting, and changing them leaves the validated setting:
- `--dtheta 15 --zoffs -64,0,64 --stride 96 --seg 144 --rmin 48`;
- `--conf 1.0 --require-agree 1 --max-step 2 --bridge 1 --max-len 0`;
- `--conf-json tools/auto_windings_conf_clean.json`;
- `--gm-model tools/auto_windings_gm_z4500_clean.json --gm-thr 0.18`.
- `--raise auto`: `--spack-thr 21.0 --auto-gm-thr 0.263` (`--gm-thr` is ignored in `auto` mode).

Caches:
- `--work` (default `<out>/work`) holds mask chunks (`m7_L0/`, packed bits), extent slices and grad_mag slices.
  The grad_mag slice is keyed by z range only, so use a separate `--work` per grad_mag store.
- `--crops` (default `<out>/crops`) holds the per-crop solve checkpoints. A rerun resumes from them, and changing
  `--conf`, `--raise` etc. only reassembles.
- `--crops-ro DIR` adds read-only crop sources. Presets use it, so the frozen crop sets are never written to.
- autowind refuses cached crops that were solved for a different umbilicus or band.

## Outputs

- `relative_windings.json`: `vc_pointcollections_json_version` 1. One collection per chain, with `p = [x, y, z]`
  (mask frame) and `wind_a` 0, 1, 2, ... outward. Collection names are `auto_z<z>_t<theta>_<n>` (`autogm_...` after
  the raise).
- `chains.json`: per chain:
  - `z`, `theta`, `pts` ([z, y, x]), `wind`, `r` (radius) and `rl` (mask run length) per point;
  - `perr` per step: certifier error probability, 1.0 = bridged;
  - `gm_p` per step: raise probability, with `--raise gm` only.
- `steps.csv`: every step on every ray, with `dr`, `p_err`, `agree` (second crop), `kept` and `bridged`.
  With `--raise gm` it is in `noraise/`.
- `summary.json`: counts. `autowind_run.json`: options, crops planned/reused/solved, timing, peak RSS, and the
  sha256 of the outputs, models and frozen scripts.
- `umbilicus_maskframe.json`: the umbilicus used, in the mask frame, in villa format. Give the same file to the
  fitter.
- With `--raise gm`, `noraise/` holds the chains before the raise.
- With `--raise auto`: `packing.json` (S_pack, its block-bootstrap 95 % interval, slab medians, the decision and the
  threshold used). `autowind_run.json` records `raise_applied` = {mode, gm_thr, packing}. If the rule picks `gm`, the
  layout is that of `--raise gm` (`noraise/` + raised set); if it picks `off`, that of `--raise off`.

## End-to-end example (PHerc1667, z 6500-8000, 9.596 um)

```sh
# 1. chains (61 min on 6 workers from scratch; seconds when crops are cached)
uv run python -m autowind chains --mask data/p1667/masks/m7_L2_th0.2.zarr \
    --umbilicus data/eval/p1667/umbilicus_p1667_L2.json --z0 6500 --z1 8000 --voxel-um 9.596 \
    --raise off --workers 6 --out runs/p1667_chains
#    the same inputs as the published set:  uv run python -m autowind chains --preset p1667 --out runs/p1667_chains

# 2. check with the pinned loader (villa env; release/villa_patch adds check_relative_windings.py)
cd $VILLA/volume-cartographer/scripts/spiral
python check_relative_windings.py runs/p1667_chains/relative_windings.json --umbilicus runs/p1667_chains/umbilicus_maskframe.json
#    expect chains_with_decreasing_wind_a 0; frac_dropped ~0.14 at spacing 16 (which is why spacing must be 0)

# 3. fit: villa 0fb38c45 fit_spiral, 343 recipe (config/recipe.toml) + pherc343 shell-optional patch.
#    Dataset root = umbilicus.json, spiral-scroll.json (lasagna normals / grad_mag, surf-SDT), and the chains:
cp runs/p1667_chains/relative_windings.json $DS/relative_windings.json
FIT_SPIRAL_CONFIG_OVERRIDES='{"input_disable_patches": true, "loss_weight_shell_outer": 0,
  "loss_weight_shell_patch_radius": 0, "dense_spacing_mode": "phase", "optimizer_num_training_steps": 30000,
  "z_begin": 6500, "z_end": 8000,
  "pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}' \
  FIT_SPIRAL_OUT_DIR=$OUT python fit_spiral.py --dataset $DS
#    (we run it through spiral/run_arm.sh with spiral/arms/c1667_m2s1.env, which builds $DS, then
#     export_fine.sh <arm.env> 4 to re-export the meshes on a 4-vox grid -> meshes_step4)

# 4. score against the human labels (PHerc0139 / PHerc1667 only), chains arm vs no-chain baseline
uv run python eval/spiral_eval.py --scroll p1667 --fit <base>/meshes_step4 --fit-b <arm>/meshes_step4 \
    --z0 6500 --z1 8000 --tol 4 --out runs/p1667_score.json
uv run python eval/subset_wrong_area.py --scroll p1667 --z0 6500 --z1 8000 --subsets subsets_bandp1667.json \
    --split 34 --own-offsets --dump-a <base dump> --dump-b <arm dump>        # the pre-registered primary endpoint
```
The only fitter settings the chains need are the last two overrides. Villa reads `<dataset>/relative_windings.json`
as a relative point collection (role RELATIVE). Without patches, every chain point is an unattached strip point. See
`release/villa_patch/`.

## Reproducibility (checked 2026-10-23)

The CLI reproduces our chain sets byte-for-byte from the cached crops (`--no-solve`):

| chain set | command | relative_windings.json sha256 | chains.json |
|---|---|---|---|
| p1667 `band_m7_p1667_ray_c` (M2, confirmed) | `chains --preset p1667 --raise off` | 85dcbfb46a66b192… = recorded | identical (and steps.csv) |
| p1667 `band_m7_p1667_ray_gmxc` (M1) | `chains --preset p1667 --raise gm` | 74b4edacda757ae3… = recorded | identical |
| p0139 band 1 `band_m7_ray_gmxc` | `chains --preset p0139_band1 --raise gm` | b012c8839efe397c… = recorded | identical |
| p0139 band 1 `band_m7_ray_c` (its `noraise/`) | same run | cb82b51d5315a2ab… | identical (and steps.csv) |
| PHercParis4 `paris4_m7_ray_c` (built from scratch with the generic flags, blind, 2026-10-23) | `chains --mask … --voxel-um 9.6 --raise off` | 485029131145fcfe… (frozen before any fit) | n/a |
| p1667, `--raise auto` (checked 2026-10-23) | `chains --preset p1667 --raise auto --grad-mag …` | 85dcbfb46a66b192… = M2 (S_pack 24.0 -> off) | identical to M2 |
| p0139 band 1, `--raise auto` | `chains --preset p0139_band1 --raise auto` | 9e404b9fbf58f6fa… (S_pack 18.5 -> gm 0.263) | 662/664 chains as the arm29 set `hitl_b1_recal0` (2 chains differ by a `gm_p` tie at the threshold) |
| PHercParis4 z 9000-10500, `--raise auto` | `chains --mask … --raise auto --grad-mag …` | 77949c5fb12619f1… (S_pack 18.5 -> gm 0.263) | identical to an explicit `--raise gm --gm-thr 0.263` |
| PHercParis4 z 9000-10500, `postproc --mode cut` (f6 dev, checked 2026-10-24) | `python -m autowind.postproc --run <auto run> --mode cut` | 0954c72a8ed7df88… = recorded | identical (bdbc7bce…) |
| PHercParis4 z 7000-8500, `postproc --mode cut` (test 6 confirmation set `p4z7_f6`) | same, on the z 7000 `--raise auto` run | 03c014bdf5e28d0e… = frozen | identical (b5d20c1d…) |
| p1667, `postproc --mode cut` (default, `--thr 0.10 --min-pts 4`, `--thr 0.15 --min-pts 4`) | same, on the p1667 `--raise auto` run | 85dcbfb46a66b192… = M2 (decision off -> copy) | identical to M2 |

The crop solve itself went through the generic inputs: the mask path, the umbilicus file (for p0139, the 2.399 um
file plus `--umbilicus-transform`) and fresh worker processes. A random 12 cached crops per scroll were re-solved,
and 12/12 checkpoints were identical on both scrolls. Bit-identity needs single-threaded BLAS. autowind sets this
before numpy loads and starts its solve workers with `spawn`.

## Runtime and memory

Measured on an i5-13600K (20 threads), with 32 GB RAM.

| stage | cost |
|---|---|
| crop solve | about 10 core-seconds per 192^3 crop (44 PHerc1667 crops in 73 s on 6 workers). PHerc1667 band (1500 z, 2,275 crops): 61 min on 6 workers. PHerc0139 band 1 (4,929 crops, denser and wider): about 2 h on 11 workers. |
| memory | Peak RSS is 1.9 GB per solve worker and about 0.3 GB for the main process. Budget about 2 GB x `--workers`. |
| reassembly + certifier (+ raise) | 1-4 s from cached crops. `--raise gm` loads a grad_mag slab: 324 MB for 1500 z of PHerc1667, memory-mapped. |
| disk | Mask chunk cache: about 0.5 GB per 1500-z band on PHerc0139 and 0.15 GB on PHerc1667. Crop checkpoints: 9-20 MB per band. |
| fit (villa, not autowind) | 30k steps on one 48 GB GPU: 2.7-3.8 h per arm, about $2-5. |

Smoke test, which builds a synthetic spiral OME-Zarr, solves 9 crops, runs the raise, and re-solves one cached
PHerc1667 crop: `uv run --with pytest pytest -q tools/autowind/test_smoke.py`. It passes 2/2 in about 20 s.

## `--raise auto` (spacing-aware rule)

Registered for the fourth pre-registered test (notes/2026-10-24-fourth-test-prereg.md). The rule uses only the mask
(and, if it picks the raise, grad_mag), never labels:
1. **S_pack** = the median radial gap between consecutive mask crossings on the probe rays of the solved crops
   (`tools/autowind/packing.py`; same definition as `data/scratch/third_region/packing_stat.py`). Crossings are
   deduplicated across crops as in reassembly; a gap counts only if one crop segment holds both crossings.
2. **S_pack < 21.0 vox -> `gm` at threshold 0.263, else no raise.** 21.0 is the log-midpoint of PHerc0139 (18.5) and
   PHerc1667 (24.0). 0.263 is the PHerc0139 zero-drift threshold against CT-supported counts (fixed 2026-10-22).

| region (z) | S_pack [95 %] | decision |
|---|---|---|
| PHerc0139 band 1 (3000-4500) | 18.5 [18.0, 19.0] | gm 0.263 |
| PHerc1667 (6500-8000) | 24.0 [23.5, 25.0] | off (= the confirmed M2 set) |
| PHercParis4 (9000-10500) | 18.5 [18.0, 19.0] | gm 0.263 |

S_pack is in mask voxels; it was checked at 9.4-9.6 um only.

**Status: experimental. It failed its development go criterion (2026-10-23)**, so the fourth pre-registered test was
stopped before confirmation (notes/2026-10-24-fourth-test-prereg.md):
- On PHercParis4 z 9000-10500, the `auto` set (gm 0.263) erased no raise's +10.5 pp loss but only tied the no-chain
  base fit (−0.3 to −0.5 pp, n.s.).
- It also had +40 seam switches per 1000 pairs against base.
- The S_pack branch works: it keeps PHerc1667 at no raise. The raise itself is the limit: only about 35 % of raised
  steps are true merged contacts on every tight region.
Default stays `off`.

## `postproc --mode cut` (experimental, tight packing only)

`uv run python -m autowind.postproc --run <autowind chains --raise auto dir> --mode cut [--thr T] [--min-pts N] --out <dir>`
post-processes a finished `--raise auto` run. If its S_pack decision is `off`, it copies `relative_windings.json`
and `chains.json` byte for byte, so PHerc1667 stays the confirmed M2 set for every option. If the decision is `gm`,
it takes the no-raise chains and cuts every single step with grad_mag `gm_p >= T` (default 0.263). There is no
shared point, wind is re-zeroed per piece and no step value is changed. Pieces with fewer than N points are dropped.

- `--thr` (default 0.263): the f6 / test-6 recipe.
- `--min-pts` (added 2026-10-24 for f7; default 2 = the test-6 recipe, byte for byte): the minimum points per kept
  piece. The f7 switch-aware variants are `--thr 0.10 --min-pts 4` and `--thr 0.15 --min-pts 4`
  (notes/2026-10-24-f7-switch-aware-cut.md). Their status is in that note.

## When to use `--raise gm` (open question)

The raise turns a chain step of 1 into 2 where a logistic on the integrated lasagna grad_mag gives p >= 0.18. It is
meant for **tightly packed sheets, where the mask merges touching sheets** into one crossing, so the chain misses a
winding. It was fitted on PHerc0139 z 4500-6000 only.

| | PHerc0139 band 1 (z 3000-4500) | PHerc1667 (z 6500-8000) |
|---|---|---|
| sheet pitch (median, p10-p90) | ~16 vox (8-32) | 26.4 vox (13-54) |
| steps raised | 18 % | 13.8 % (2,385 / 17,291), 86 % of them wrongly |
| step precision vs labels | no raise 88.8 / 87.4, raise 85.1 / 84.0 (even / odd angles) | no raise 95.8 %, raise 86.9 % (k < 34) |
| long-range drift (10-20 turns) | the raise removes the under-count; at thr 0.18 the drift is +0.19 | no raise +0.95 (74.7 % within ±1), raise +2.74 (30.6 %) |
| fit result | raise needed: inner wrong area -32.5 pp at thr 0.18, against -15.4 / -17.2 at 0.229 / 0.263 | **raise fails** (M1: 2 of 6 combos, ~0 pp); no raise passes (M2: -27 to -29 pp) |

- Default `off`. Use `gm` only if the sheets are packed at about 16 vox or less at ~9.5 um, or if a labelled check
  shows that the chains under-count.
- Check the no-raise chains first. If the drift over 10-20 turns is clearly negative (chains count fewer turns than
  the labels), the mask is merging contacts and `gm` may help.
- The threshold is also packing-dependent. On PHerc0139, the zero-drift threshold varies from 0.15 to 0.26 between
  slabs and bands.
- We have **no validated label-free rule** that picks raise or no raise, or a threshold. Local pitch did not
  separate the zones, a context-dependent raise rule was a NO-GO (notes/2026-10-22-context-raise-rule.md), and the
  scroll-level S_pack rule (`--raise auto`) failed its dev go criterion on Paris4 (above).

## Limitations

- **Validated at two voxel sizes only.** The frozen parameters are in voxels: 192^3 crops, patch-graph block 12,
  radius 40, k 30, and run-merge gap 2.5 vox. The step-confidence model was fitted on PHerc0139 at 9.362 um. Masks
  far from about 9.5 um should be resampled; `--voxel-um` warns but does not rescale.
- **Mask quality bounds the result.** The tool was developed on m7 masks with threshold 0.2. Merged contacts give
  under-counts (see `--raise`); missed sheets are bridged as +1 and therefore also under-count; spurious sheets
  over-count. On PHerc1667, the precision of the confirmed set is 95.8 % per step on inner turns. Outer turns
  (k >= 37) were not CT-supported, so they are unverified.
- **Long chains carry their errors.** One wrong step shifts every later point of its chain by one turn, and the
  fitter has only global strip weights, so per-step `perr` cannot be used. Pair precision on PHerc1667 falls from
  90 % (1-2 turns apart) to 39 % (11-20 turns apart). The fit gain comes from local consistency.
- **Scoring is scroll-specific.** `eval/spiral_eval.py` and `subset_wrong_area.py` need human winding labels, and
  exist only for PHerc0139 and PHerc1667.
- **Coverage.** Rays are horizontal from the umbilicus, every 15 deg, 3 z levels per 192 z. Each chain is one ray,
  and chains do not link to each other. If the umbilicus is wrong by much more than a sheet pitch, the rays cross
  sheets obliquely; crops whose ray grazes the sheets are skipped.
- **One pass in three pre-registered tests.** The method was confirmed on one scroll (PHerc1667), and the no-raise
  arm was selected after a step-level dry run on that scroll's labels (disclosed, TECHNICAL_REPORT.md section 3.3).
  It failed on PHerc0139 band 2 and on PHercParis4. Both failures are at tight packing: step precision there was
  78.9 % on Paris4, with mostly under-counts at contacts.
- **Presets:** `p0139_band1` streams m7 from the open-data bucket (anonymous S3). `p1667` uses our local m7 mask,
  which is not public yet.
