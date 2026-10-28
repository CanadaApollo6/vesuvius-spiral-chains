# 2026-10-24: f7 confirmation, switch-aware cut on PHercParis4 z 10800-11800 (L2): pre-registration

Status: **STAGE 0 (region choice) frozen at the timestamp in addendum S0.** Stage 1 (method, arms, analysis) is
appended only if a variant goes at the dev gate (notes/2026-10-24-f7-switch-aware-cut.md); stage 2 (method chain sha)
before any method pod. Template: notes/2026-10-24-f6-confirm-prereg.md and notes/2026-10-24-fourth-test-prereg.md.
Run directory: `data/scratch/f7_switch/`.

## 0. Stage 0: region choice (frozen before any chain, crop, fit or label-based chain/fit metric on it)
**Region: PHercParis4 (Scroll 1), volume 20260411134726, L2 (9.6 um), z 10800-11800, full slice.**

Scouting (2026-10-24 18:05-18:32Z; all label-free except coverage / consistency, which use the labels only):

| Paris4 band (L2) | S_mask [95 %] (layers / 2) | slabs | label cover (k with >= 95 % angle cover in every 250-z slab) | note |
|---|---|---|---|---|
| 3000-4500 | 20.0 [19.5, 21.0] | 19.0-22.0 | 120 (10-129) | lower edge of the label set (z 3000) |
| 3500-5000 | 21.5 [20.5, 22.0] | 20.0-22.5 | 120 | not tight |
| 5000-6500 | 21.0 [20.0, 22.0] | 19.5-22.5 | 120 | at the S_pack threshold |
| 5200-6700 | 20.0 [19.5, 21.0] | 19.0-22.0 | 120 | |
| **10800-11800** | **19.0 [18.0, 19.5]** | 18.5-19.0 | **120 (10-129)** | **chosen** |
| 10800-12300 | 18.5 [18.0, 19.0] | 18.0-19.0 | 120 | 468 z inside the 2023 GP label range |
| 11000-12500 | 18.5 [18.0, 19.0] | 17.5-19.5 | 120 | 668 z inside the 2023 GP label range |

(`data/scratch/f7_switch/scout/packing_stats_mask_f7.json` sha 9b5d2b21…, same script and command as
`data/scratch/third_region/run_paris4_more.sh`; coverage `data/eval/paris4/cover_z<band>.json` from
`data/scratch/third_region/paris4_labels.py cover`, log `data/scratch/f7_switch/scout/cover.log`.)

Choice by the brief's criteria:
- **Fresh.** Before this freeze the only computations on z 10800-11800 are S_mask, label coverage, label consistency
  and the scorer check below. No chain, crop, fit, step metric or fit metric exists. No chain or fit of any earlier
  test touched it.
- **Tight.** S_mask 19.0 [18.0, 19.5] = Paris4 z 9000-10500 (19.0); expected S_pack about 18-18.5, below 21.0, so the
  S_pack branch is active.
- **Separation from dev data.** ≥ 300 z from both burned Paris4 bands: 300 z above z 9000-10500 and 2300 z above
  z 7000-8500. The autowind crops of z 9000-10500 end at z 10560 (last zc 10464 + 96); those of z 10800-11800 start
  at z 10752 (first zc 10848 − 96), so no crop overlaps. (The fitter's ±256 input margins overlap in CT inputs only,
  as for z 7000 / z 9000.)
- **Outside the 2023 GP label zarr range (L2 11832-17625),** where m7's training data may include Scroll 1. This is the
  rule the fourth test's stage 0 used to reject z 12500-14000 / 15500-17000. It is why z 11000-12500 (the brief's
  example, 668 z inside that range) and z 10800-12300 were not chosen, and why the band is 1000 z, not 1500.
  Disclosed: the top crops (zc 11616, reaching z 11712) stay below 11832.
- **Not chosen:** z 5000-6500 (S_mask 21.0, at the branch threshold); z 3000-4500 and 5200-6700 (S_mask 20.0, looser
  than both dev bands; z 3000 is the edge of the label set); z 3500-5000 (21.5).
- **Labels.** The organiser 20260701 winding-group meshes (`labels_w0701.npz`, the same source as both dev bands) cover
  all 120 windings k 10-129 with ≥ 95 % angle cover in every 250-z slab. Adjacent-k radial order: median 93.3 %
  increasing (z 9000: 92.3 %, z 7000: 92.8 %).
- **Label consistency** (`eval/label_subsets.py --scroll paris4z11` -> `data/scratch/label_audit/subsets_bandparis4z11.json`,
  sha d5ef8cef…): **113 of 120 consistent**; inconsistent 48, 61, 62, 126-129 (z 9000: 111; z 7000: 105).
- **Scorer ready.** `eval/spiral_eval.py --scroll paris4 --z0 10800 --z1 11800` (sha 813860d7…, unchanged): reference
  8,201,759 vertices (stride 2), 16,285,018 adjacent pairs, 3.2 % contact. Merged-mesh oracle ceiling (`--min-edge 8`,
  `data/scratch/f7_switch/stage0/score_oracle_z10800.json` sha b2c8a560…): vertex seam wrong area 0.05 [0.03, 0.08] %,
  switches 0.64 / 1000, unmatched 0.16 %; ray pairs acc_matched 99.99 %, coverage 99.81 %. Sound.
- **Scorer hooks.** `eval/label_subsets.py` and `eval/ct_gap_audit.py` get a backward-compatible `--scroll paris4z11`
  choice (band 10800-11800; m7 / grad_mag paths of the z 10800 rows, `data/paris4_z10800/`). Existing choices unchanged
  (originals and diffs in `data/scratch/f7_switch/orig/`).
- **Disclosed:** same scroll, same lasagna and m7 runs and same label source as both dev bands. The test is transfer
  to a fresh band of a scroll whose two other bands shaped the method, not a new-scroll test. The band is smaller than
  the dev bands (1000 vs 1500 z: 8.2 M vs 12.4-12.5 M reference vertices), so CIs will be wider.

Next (label-free / label-only, any time after this freeze): fetch the m7 rows and lasagna inputs; m7 / grad_mag audit
and primary winding set (Paris4 rule, unchanged); oracle chains; controls; the label-free crop solve. No method chain
set is built before stage 1, and no label-based chain metric is computed until all fits are scored.

## Addenda (append-only)
### S0 (2026-10-24T18:32:29Z): stage 0 frozen
sha256 of this file up to and including the "## Addenda (append-only)" line: 10655e54929676845f6181a2ad8fcfc4380b1ef8bdb19a68db793428b800bd68
(`data/scratch/f7_switch/freeze_stage0.txt`, snapshot `data/scratch/f7_switch/stage0/stage0_snapshot.md`). Nothing has been built on
z 10800-11800. Next: label-free input fetch and label-only audits, in parallel with the dev stage.
### STOP (2026-10-24T23:07:28Z): no variant passed the f7 dev gate; stage 1 not written, no confirmation
See notes/2026-10-24-f7-switch-aware-cut.md L2-L3. What exists on z 10800-11800 (label-free / label-only, as stage 0
allowed): m7 rows and lasagna inputs (data/paris4_z10800/); m7 / grad_mag audit (ct_counts_bandparis4z11.json); oracle
chains (windings/band_oracle_p4z11, sha ba26d8a4…); the label-free crop solve (cropsolve_p4z11, 4,237 crops, exit 0
20:45:08Z; its `--raise off` chain file was not opened); the primary set by the unchanged Paris4 rule
(primary_kset.py -> 21-47, 49-60, 63-99, 111-115, 119-122, 85 windings; not frozen). Also prepared but unused:
method_table_f7z11.py and spiral/arms/f7z_{base,bases2,bases3,oracle}.env. No method chain set and no fit exist
on this band. It stays fresh for a later test.
