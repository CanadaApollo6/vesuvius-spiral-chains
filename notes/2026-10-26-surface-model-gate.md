# 2026-10-26: Do newer public surface models keep touching sheets apart better than m7? Pre-set gate

**Gate frozen: 2026-10-26T04:25Z (UTC).** Sections 0-4 are written after scouting the candidates (what exists, licence,
resolution; weights of one candidate downloaded and converted, no inference run) and after reading the label-only
spacing distribution of the dev rays (no mask of any model sampled). They are written before any candidate or m7 mask
is sampled along a ray. Nothing in sections 0-4 changes after this time; the log and results are append-only.
Run directory `data/scratch/surf_gate/`. Untrusted downloads under `data/untrusted/<name>/`.

## 0. Question
autowind counts windings along rays as consecutive m7 mask crossings. Where two sheets touch, m7 gives one crossing, a
2-winding step reads as 1, and the chain undercounts (release/SUBMISSION_DRAFT.md section 3; Paris4 has 3,569
undercounts at contacts). Every fix so far worked downstream of the mask. We never tried another mask. Question: does
any newer public surface model resolve touching sheets as separate crossings more often than m7, without losing
sheets elsewhere?

## 1. Candidates (scouted before the gate)
| id | model / source | resolution, frame | licence | how obtained | training overlap with dev regions |
|---|---|---|---|---|---|
| m7 | `scrollprize/surface_m7_nnunet` (Kaggle 1st-place nnU-Net component), the pipeline mask | Paris4 bucket `...-surface-m7-L2-th0.2.zarr` (CT L2, 9.6 µm); PHerc0139 bucket `...-m7-L0-th0.2.zarr` (9.362 µm); PHerc1667 local `data/p1667/masks/m7_L2_th0.2.zarr` (bucket recipe) | Apache-2.0 | published / existing | unknown (Kaggle Dataset100) |
| F0 | Socratic Method "C3-F0" 200k (ScrollFiesta team, Ben Kyles et al.; github.com/Hob3rMallow/socratic_method `releases/c3-f0-200k-20260907`, sha256 54db9a59…a175). The "retrained 9 µm surface model distilled from 2.4 µm predictions" of the Aug 2026 prize. m7 architecture, initialised from m7, trained on 2.4 µm teacher predictions | 7.91-9.37 µm; run here at the pipeline grid (CT L2 for Paris4 / PHerc1667, L0 for PHerc0139) | Apache-2.0 (code and weights; upstream m7 / teacher keep their own terms) | inference (local 3090) | **PHerc0139, PHerc1667 and PHercParis4 are all in its 12 training scrolls** (teacher pseudo-labels, regions unknown) |
| P4-2um | bucket `PHercParis4/.../20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr`, metadata model `ps256_trainpy/checkpoint_inference_ready.pth` = HF `scrollprize/surface_recto_3dunet` (Angelotti et al., 3D ResEnc U-Net, 256³, medial-surface recall loss) | 2.4 µm native (CT L0); L2 = nearest downsample | MIT (model card); data CC BY-NC 4.0 | published | card lists PHerc0139, 1667, 0343P, 0500P2, MANBp; **Paris4 not listed** |
| 0139-2um | bucket `PHerc0139/.../20260102150214-surface-20260203205514-surface-recto-2um-ps192-L0-th0.5.zarr`, on the 2.399 µm scan; same model family (no metadata; not confirmed) | 2.399 µm native; mapped to the 9.362 frame by `transform_0139_2p399_to_9p362.json` (the affine our labels already use) | not stated (organiser bucket) | published | likely in-sample |
| 0139-090 | bucket `PHerc0139/.../20250728140407-surface-20250701154204-surface-recto-090.zarr`, older (2025-07) organiser model on the 9.362 µm volume, same grid as m7-L0 | 9.362 µm | not stated | published | unknown |

Not available: the Kaggle 2nd-10th placings released no weights that could be found (writeups only); no community
surface zarrs exist for our three scrolls (yupeng's uploads cover PHerc0814 / PHerc1447 only); no PHerc1667 surface
prediction is in the bucket. `surface_recto_3dunet` is not run separately: its Paris4 output is P4-2um, and on
PHerc0139 / 1667 it is in-sample.

## 2. Measurement (label-anchored, ray level; identical code for every mask)
Dev rays only, from the existing oracle ray files (human-mesh crossings, `[r, k, zyx]` per ray):
- **Paris4 z 7000-8500** (`data/scratch/fourth_test/windings/band_oracle_p4z7/rays.json`, burned by tests 6, 7, 11), k
  in PRIMARY_K `13,16-32,34-44,47-61,63-108`;
- **PHerc0139 band 1** z 3000-4500 (`auto_windings/band_oracle`) and **band 2** z 1750-3000 (`band_oracle_z1750`), k 23-50
  (dev_steps zones);
- **PHerc1667 z 6500-8000** (`band_oracle_p1667`, wide-spacing control), k < 34.
Not used: Paris4 z 8500-10500 (HITL blinding), z 10800-11800 (reserved), and anything else.

**Ray subset (fixed now):** every 4th z level of each ray file starting at the 2nd (indices 1, 5, 9, ...): Paris4 7072,
7328, 7584, 7840, 8096, 8352; b1 3104 … 4384 (6 levels); b2 1824 … 2848 (5 levels); PHerc1667 6624 … 7904 (6 levels);
all 24 angles. The same rays for every mask. m7 is also scored on all levels as a check that the subset is typical.

**Crossings.** As `auto_windings.ray_crossings` does for m7: the mask (value > 0) is dilated by one voxel (6-neighbour
cross) in its own grid and sampled along the ray every 0.5 voxel of its own grid; samples closer than 2.5 own-grid
voxels join one run; runs touching the sampled segment's ends are dropped; the run centre is the mean of the undilated
samples. Positions are in the region's ray frame (9.6 µm Paris4 / PHerc1667, 9.362 µm PHerc0139). Native 2.4 µm masks are
sampled at their own resolution (0.125 coarse-vox steps, 2.5 fine-vox gap), and also at their bucket L2 level.

**Matching.** A label crossing is detected if a run's interval [start, end] lies within 3 vox of it; it is assigned the
nearest run by interval distance.

**Metrics** (per region and mask; 95 % CIs by cluster bootstrap over rays, 2,000 resamples, seed 0; candidate − m7 paired
over the same rays):
1. **Contacts** = consecutive label crossings on a ray with Δk = 1, both k in the region's set, spacing < 8 vox
   (the label-only 5-10th percentile in the tight bands; recto-face to recto-face less than about half a pitch).
   Each contact pair is **resolved** (both detected, by different runs), **fused** (both detected, same run) or
   **missed** (at least one not detected). Fused and missed both make a chain undercount.
   - fused rate = fused / all contacts (the question asked);
   - **resolved rate = resolved / all contacts (primary);**
   - the same three rates for spacing 8-12 and >= 12 vox (reported).
2. **Sheet recall** = detected share of labelled crossings with k in the set.
3. **Extra runs** = runs whose centre lies between the first and last labelled crossing of the ray and that match no
   label crossing (within 3 vox), per labelled crossing. Real but untraced sheets also count, so this is an upper bound
   on false sheets; it guards against a model that "resolves" by shattering sheets.
4. **Face offset** = run centre − label r (outward +) for crossings that are alone in their run, median [CI].
   Reported only (every fitter input comes from the same mask, so a constant offset moves together; test 11).
5. Secondary: resolved rate on the pairs m7 fused (selection-biased towards m7's failures; reported only).

## 3. GATE (fixed now): what justifies swapping a candidate into the chain pipeline
A candidate **passes** iff, against m7 on the same rays, in **both tight regions Paris4 z7000 and PHerc0139 b1**:
1. resolved rate at contacts **+10 pp or more**, paired 95 % CI lower bound > 0;
2. fused rate at contacts lower (point estimate);
3. sheet recall no lower than m7 − 2 pp;
4. extra runs no more than m7 + 0.05 per labelled crossing;
and on **PHerc1667** (control): sheet recall and resolved rate over all pairs (any spacing) no lower than m7 − 2 pp.
PHerc0139 b2 is reported, not gated.

For the native 2.4 µm masks the gate is read at native resolution (the model's ability), and the L2 reading says
whether the gain survives at the pipeline grid. A native-only pass means a swap needs native-resolution crops in
`autowind` (64x the voxels per crop), which is stated as the cost, not hidden.

Reading:
- PASS -> the candidate is a mask worth a pre-registered autowind fit (a separate test; no fit here). The note says
  what the swap needs (mask store, autowind `--mask`, kimimaro tracks, surf-SDT, grad_mag unchanged).
- A pass on Paris4 only by a model whose PHerc0139 / 1667 scores are in-sample is still reported as such.
- FAIL -> no surface model tried here beats m7 at contacts by a margin that matters; stop.
- Leakage caveat fixed now: F0 trained on all three scrolls (teacher labels), 0139-2um / 1667 overlap is likely; only
  P4-2um on Paris4 is cleanly out-of-sample. Labels could also have been traced on one of these predictions.

## 4. Budget and rules
$40 cap; local 3090 first (lease in ~/.gpu-leases/, inference under `flock /tmp/vss_eval.lock`), pods `vss-surf-*`
only if local is too slow. Crops streamed, never whole volumes. Untrusted weights: own dir, `python -I`,
`torch.load(weights_only=True)`, paths as args. No fits. Frozen files untouched (tools/auto_windings.py is only
imported for its affine and constants, as dev_steps.py does).

F0 inference recipe (fixed now; the authors' shipped point): 8-way mirror TTA, foreground probability >= 0.30, on the
pipeline grid without resampling (spacing 1, as the bucket m7-L2 recipe), via nnU-Net's predictor with m7's plans (same
architecture and CT normalisation, checked: 956/956 tensors match in name and shape). Inference per 192³ block centred
on the ray plane, keeping the central 128 x 128 columns (32-voxel halo, as the authors') and z0 ± 8. Check: m7 run
through the same block procedure (TTA, P >= 0.2) on one Paris4 level must reproduce the bucket m7 (Dice >= 0.9 on the
kept voxels); otherwise the procedure is fixed before F0 is scored.

---

## Log (append-only)
### L1 (2026-10-26T05:00Z): m7 block-procedure check missed its Dice bar; procedure kept, with the reason (before any F0 score)
- m7 through `infer_rays.py` (TTA, P >= 0.2) at Paris4 z 7072, 558 tiles, against the bucket m7-L2 on the kept voxels:
  Dice **0.848** (bar 0.9: missed). Voxel counts match (30.55 M vs 30.33 M).
- Diagnosis (`check_m7chk2.py`, 120 random tiles): Dice is the same in the tile cores (0.845, >= 16 px from the edge) and
  at the edges (0.825), so the halo and block layout are not the cause. With a 1-voxel tolerance Dice is 0.931: the
  difference is 1-voxel jitter at band boundaries. Our own full-band m7 run on PHerc0332 reached Dice 0.815 against the
  bucket (notes/2026-10-22-confirmation-region.md), so this is the reproducibility level of the bucket recipe.
- What the gate measures agrees on those 24 rays (`cmp_level.py p4z7 7072 m7 m7chk`): resolved at contacts 17.6 vs
  16.1 %, fused 27.1 vs 27.1 %, recall 79.5 vs 79.2 %, extra runs 0.110 vs 0.112, offset -0.16 vs -0.26.
- **Deviation:** the procedure is not changed, and F0 is scored against the bucket / pipeline m7 as frozen. F0-vs-m7
  differences carry about ±1.5 pp of procedure uncertainty at contacts, and that is stated next to them.
- Disclosure: the published-mask rows (section 1: P4-2um, 0139-2um, 0139-090) had already been sampled and tabled
  (gate rules, subset levels) before this check. Those masks need no procedure. No F0 output had been sampled.
- F0 inference started at about 04:57Z (`run_f0.sh`, under the eval lock; Paris4, then PHerc0139 b1, then PHerc1667).
  Speed: 1.4 s per 192³ block with TTA, 558 blocks per Paris4 level.
### L2 (2026-10-26T05:03Z): published masks scored (gate subset; all-level rerun as a secondary)
`measure.py table` (subset, 2,000 resamples later in R1) and `measure.py table --all` (every level, 300 resamples).
Numbers are in out/summary.json and out/summary_all.json and in the results table below. In short: P4-2um gains
+15 pp (native) and +12 pp (L2) resolved at Paris4 contacts. Natively it has twice m7's extra runs; at L2 its fused rate
is higher than m7's. The PHerc0139 masks (2um at native and L2, and 090) all lose at b1 contacts. The all-level rerun
reproduces the subset within about 1-2 pp everywhere. m7's subset rates match its all-level rates (`subset_check.py`).

## Results (2026-10-26T07:26Z): GATE FAILED for every candidate. m7 stays.
Gate subset, 2,000 cluster-bootstrap resamples over rays (`measure.py table`; out/summary.json). Δ = candidate − m7
on the same rays, with the paired 95 % CI. "Contacts" = Δk 1 label pairs < 8 vox apart. Resolved + fused + missed = 100 %.
"res all" = resolved share over every Δk 1 pair at any spacing.

| region | mask | contacts | resolved % (Δ [CI]) | fused % | missed % | recall % (Δ) | extra / label (Δ) | offset vs label | res all % |
|---|---|---|---|---|---|---|---|---|---|
| Paris4 z7000 | m7 | 1088 | 17.6 | 30.8 | 51.7 | 78.9 | 0.092 | -0.26 | 58.5 |
| | P4-2um native | | 32.7 (+15.2 [+11.9, +18.4]) | 25.6 | 41.7 | 80.6 (+1.7) | 0.191 (**+0.099**) | +0.10 | 63.2 |
| | P4-2um L2 | | 30.1 (+12.5 [+9.6, +15.6]) | **47.0** | 23.0 | 88.2 (+9.3) | 0.101 (+0.008) | +0.11 | 67.1 |
| | F0 | | 26.7 (**+9.2** [+6.5, +12.0]) | **37.2** | 36.0 | 82.6 (+3.7) | 0.110 (+0.017) | -0.00 | 63.1 |
| PHerc0139 b1 | m7 | 261 | 24.1 | 51.0 | 24.9 | 88.3 | 0.193 | +0.09 | 74.0 |
| | 0139-2um native | | 21.1 (**-3.1** [-7.8, +1.8]) | 37.9 | 41.0 | 83.2 (**-5.1**) | 0.240 (+0.047) | -0.45 | 67.2 |
| | 0139-2um L2 | | 19.9 (**-4.2** [-9.0, +0.8]) | 45.2 | 34.9 | 86.1 (**-2.2**) | 0.214 (+0.021) | -0.52 | 69.5 |
| | 0139-090 | | 8.8 (**-15.3** [-20.5, -10.3]) | **78.2** | 13.0 | 92.8 (+4.5) | 0.152 (-0.042) | +0.01 | 72.0 |
| | F0 | | 26.4 (**+2.3** [-2.2, +6.9]) | **53.3** | 20.3 | 90.1 (+1.8) | 0.202 (+0.009) | +0.27 | 76.3 |
| PHerc0139 b2 (not gated) | m7 | 64 | 14.1 | 50.0 | 35.9 | 82.9 | 0.180 | +0.12 | 67.5 |
| | 0139-2um native | | 12.5 (-1.6 [-11.1, +8.1]) | 37.5 | 50.0 | 79.8 (-3.1) | 0.215 | -0.40 | 64.0 |
| | 0139-2um L2 | | 15.6 (+1.6 [-8.6, +11.3]) | 43.8 | 40.6 | 83.2 (+0.3) | 0.189 | -0.35 | 67.0 |
| | 0139-090 | | 10.9 (-3.1 [-9.7, +2.8]) | 73.4 | 15.6 | 92.0 (+9.1) | 0.136 | +0.16 | 70.5 |
| PHerc1667 (control) | m7 | 42 | 54.8 | 33.3 | 11.9 | 91.9 | 0.091 | -0.06 | 86.8 |
| | F0 | | 23.8 (-31.0 [-46.9, -17.9]) | 61.9 | 14.3 | 88.3 (**-3.6**) | 0.080 (-0.011) | +0.19 | **79.7** (-7.1) |

Gate items failed (bold above):
- **P4-2um** exists only on Paris4, so it cannot meet the two-region rule.
  - Native: Paris4 item 4 fails (extra runs +0.099 > +0.05).
  - L2: Paris4 item 2 fails (fused 47.0 vs 30.8).
  - Its PHerc0139 sibling (0139-2um, same family, in-sample) fails item 1 at b1, natively and at L2.
- **0139-090** fails item 1 at b1 (-15.3 pp).
- **F0**:
  - Paris4: item 1 (+9.2 < +10) and item 2 (fused 37.2 vs 30.8) fail.
  - b1: item 1 fails (+2.3, CI includes 0).
  - Control fails: PHerc1667 recall -3.6 pp and all-pair resolved -7.1 pp.
  - All this despite all three scrolls being in its training set. Procedure uncertainty is about ±1.5 pp (L1).

Secondary, all levels (`--all`, out/summary_all.json): within 1-2 pp of the subset for every published mask. Paris4
P4-2um native +15.3 [+13.6, +16.8], L2 +12.0 [+10.4, +13.8]; b1 0139-2um native -2.4 [-5.3, -0.1], L2 -4.9; 090 -12.4.

### What the numbers say (post hoc, no verdict attached)
- **No candidate keeps touching sheets apart better in a way that holds across scrolls.** The two models that help on
  Paris4 (P4-2um +12-15 pp, F0 +9 pp) do not help on PHerc0139 b1: -3 to -4 pp for the 2um family, +2 pp (n.s.) for F0.
  F0 also loses sheets and pairs on the wide-spacing PHerc1667 control.
- **Most of the gain comes from finding sheets, not splitting them.** At Paris4 contacts m7 *misses* one of the
  two sheets in 52 % of pairs and fuses them in only 31 %. P4-2um L2 and F0 cut misses (to 23 % and 36 %) but fuse more
  (47 % and 37 %). Only P4-2um native lowers both, and it pays with twice m7's extra runs. For a chain, fused and
  missed both undercount, which is why the resolved rate is the primary.
- **Resolution alone does not fix contacts.** Natively at 2.4 µm, the 0139-2um mask is no better than 9 µm m7 at b1
  contacts. That is in-sample, on the scan its labels were traced on.
- **The one possible lead is post hoc and single-region:** P4-2um at the pipeline grid (L2) on Paris4, with recall
  +9.3 pp, all-pair resolved +8.6 pp and unchanged extra runs. It is out-of-sample for that model, but it was not
  replicated on a second scroll (no 2um prediction exists for PHerc1667, and 0139-2um did not show it). Note also
  that Paris4's human meshes may have been traced on this prediction. Testing it would need its own
  pre-registration, a swap of all m7-derived fitter inputs, and a fresh band; it is not proposed here.
- **Face offset:** every candidate sits within ±0.5 vox of the labels (m7 -0.26 / +0.09). The 0139-2um mask sits
  about 0.5 vox further inward. No candidate changes the offset picture of test 11.

### What a swap would have needed (none is proposed)
`autowind chains --mask <store>` with the store's level 0 in the ray frame, and the same store for the kimimaro
tracks (`extract_tracks.py`) and surf-SDT (`make_surf_sdt.py`), so every positional input moves together (test 11's
lesson). A native 2.4 µm mask would need native-resolution crops (64x the voxels) or a defined downsample. Its L2
nearest level is what the table scores.

### Spend and housekeeping
- **/usr/bin/bash.** No pods created (`prime pods list`: no vss-surf-*). Local 3090: lease `~/.gpu-leases/vss-surf-gate.json`
  (14 GB) was released after the run. GPU free 23.0 GB at start (mem_log.txt); inference ran under the eval lock.
  About 2.65 h of GPU time: F0 on 18 levels (5,864 blocks with TTA, 2.42 h) plus the m7 check on 1 level (0.23 h).
- **Downloads:**
  - `data/untrusted/socratic_f0/checkpoint_f0_00200000.pt` (sha256 54db9a59…a175, matches the authors' pointer).
    Loaded with `weights_only=True` under `python -I`. Converted to `data/scratch/surf_gate/f0_model/`, which reuses
    m7's plans and dataset json. None of the authors' code was downloaded or run.
  - Published masks and CT were streamed by chunk; nothing was stored except the sparse F0 / m7chk probability
    zarrs in `pred/`.
- **Frozen files untouched.** `eval/spiral_eval.affine` was imported read-only for the PHerc0139 affine.
- **Scripts:** `data/scratch/surf_gate/{infer_rays,measure,convert_f0,inspect_ckpt,check_m7chk,check_m7chk2,cmp_level,subset_check}.py`,
  `run_f0.sh`. Outputs: `out/`. Logs: `logs/`.
