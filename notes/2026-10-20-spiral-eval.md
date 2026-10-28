# 2026-10-20: spiral-fit evaluation for PHerc0139 (`eval/spiral_eval.py`)

This scores a villa spiral fit against the human meshes and the frozen p0139 pairs. The fit is given as per-winding
tifxyz in the 9.362 um frame of 20250728140407. Everything here was validated on controlled inputs; the real fit
(`data/scratch/p0139_spiral/arm0/`) has not been scored yet.

```
uv run python eval/spiral_eval.py --fit <dir> [--fit-b <dir>] --z0 3000 --z1 4500 --tol 4 --out <json> [--dump-pairs <csv>]
uv run python eval/spiral_eval.py build-oracle --out <dir> [--source 2um|9um]
uv run python eval/spiral_eval.py corrupt --src <dir> --out <dir> --kind shift|relabel|merge|drop --winding W --sector A0 A1
```

## Inputs
- **Fit format.** The fit directory is searched recursively for x/y/z.tif, xyz in 9.362 L0 voxels. Invalid vertices
  are -1, nan or mask 0.
  - The winding comes from meta `winding`, or from a villa combined surface (`winding_column_ranges` +
    `component_winding_ids`), or from `w<int>` in the directory name.
  - Several surfaces may share one winding.
  - Grids finer than 2.5 vox are decimated.
- **Nearest surface.** Distance is exact point-to-triangle (each quad = 2 triangles). The candidates are the 8 quads
  with the nearest centroids, searched per surface. A point is matched if its nearest surface is within `--tol` (4 vox).
- **Pairs.** `scoring_pairs.csv` kind `mix` is the primary set (24,972 pairs). Kind `legacy` (195k adjacent normal-ray
  pairs) is a secondary block.
  - Global zyx = site `origin_zyx` + local voxel. This is the CT frame, because the per-site NCC shift is already in
    the labels.
  - A pair is in the band if both ends have z0 <= z < z1.
  - The 172 pairs that skip a winding absent from their site are dropped (77 of them fall in z 3000-4500).
- **Human vertices.** From the 2.399 tifxyz, mapped with the published affine, stride 2 on the 48 um grid (about
  10 vox spacing), z in the band.
  - Blacklisted w041/42/45/46 are left out of the vertex sample but kept as oracle surfaces.
  - z 3000-4500 gives 3,634,771 vertices and 7.24M adjacent pairs.
  - A vertex is "contact" if another human mesh is within 4 vox.
- **Umbilicus.** The public `PHerc0139/representations/umbilicus/20260102150214-umbilicus-20260826112529.json`
  (2.399 xyz), mapped by p9 = A p + t from the transform json. A local copy is in `data/eval/p0139_umbilicus/`.
  - At z 3000 / 3750 / 4500 it maps to x 2762 / 2782 / 2868 and y 3210 / 3187 / 3373.
  - On the level-3 and level-2 CT slices it sits at the centre of the innermost turns
    (`data/scratch/spiral_eval/umb_z*.png`).

## Metrics
All metrics use a paired bootstrap: 1000 resamples, seed 1. The resampling units are the test sites for pairs, and
blocks of 30 deg x 250 z for vertices. A/B differences B - A use the same resamples.

Strata:
- pairs: all, contact/open, site class, and step 0/1/>=2;
- vertices: all, contact/open, and site class for vertices inside a site crop.

1. **Pairs.**
   - correct = both ends matched and |j_B - j_A| == expected.
   - Reported as `acc_matched` (over pairs with both ends matched), `coverage` (share of pairs with both ends
     matched) and `acc_all` (unmatched counted wrong).
   - `*_seam` variants map j into the human label frame with the seam-aware offset below.
2. **Vertices.**
   - d = j - s k, where s = +-1 is the orientation with the larger mode share.
   - Two offsets are removed:
     - **global**: the mode of d, as specified;
     - **seam**: the mode of d per (10 deg x 250 z) bin.
   - Reported: switches per 1000 adjacent pairs (both matched) where d - offset changes, `wrong_area` (share of
     matched vertices with d != offset), and the unmatched share.
   - Why the seam variant exists: human segments are not cut on one seam. w<=34 start at about 120-140 deg and
     w>=36 at about 225-260 deg, each spanning 365-400 deg. A fit has its own seam, so even a correct fit has d
     piecewise constant *in angle*, and the global offset would mark whole arcs as wrong. For fits labelled like the
     humans (all validation inputs below) the two variants agree.
3. **#2167.**
   - For each 50-z slab, the gap sector is the largest empty run of 1-deg bins of w035 around the umbilicus.
   - In z 3000-4500 (30 slabs) it starts at 144-152 deg and is 105-119 deg wide, median 112.5. #2167 reports a
     median of 95-98 deg over its whole z range.
   - Flagged = a vertex, or either end of a pair, inside the gap.
   - Every metric is reported three ways:
     - all;
     - `excl_flag_touch`: flagged items touching w033-w037 dropped. For a pair, touching means its winding interval
       meets 33-37; for a vertex, k in 33-37.
     - `excl_flag`: all flagged items dropped.
   - In z 3000-4500, **no test site lies in the gap** (the in-band sites sit at 287-55 deg), so 0 pairs are flagged.
     902k vertices are flagged, 99k of them on w033-37.

## Validation (z 3000-4500, tol 4)
Pairs are `mix` (n 7,471 after dropping 77 skip pairs) / `legacy` (n 64,148). Vertex numbers use the global offset.

| fit | pair acc_matched (mix) | contact / open | coverage | legacy acc | switch /1000 | wrong area % | unmatched % |
|---|---|---|---|---|---|---|---|
| oracle (2.4 um meshes as fit) | 99.53 [99.23, 99.77] | 95.2 / 100 | 100 | 99.57 | 0.00 | 0.00 | 0.00 |
| oracle, pair ends un-shifted (affine frame) | 100.00 | 100 / 100 | 100 | 100.00 | 0.00 | 0.00 | 0.00 |
| oracle9 (public 9.362 meshes, 187 um grid) | 99.56 [99.31, 99.77] | 95.6 / 99.98 | 99.97 | 99.54 | 16.7 (contact 120, open 0.01) | 1.60 (contact 10.8) | 0.17 |
| (a) all windings +1 | 99.53 (identical) | 95.2 / 100 | 100 | 99.57 | 0.00 | 0.00 | 0.00 |
| (b1) w032 -> 33 in 300-340 deg | 94.23 [90.76, 97.62] | 93.7 / 94.3 | 100 | 92.21 | 0.05 [0.025, 0.073] | 0.36 | 0.00 |
| (b2) merge w032+w033 in 300-340 deg (outer -1) | 94.43 [91.35, 97.47] | 93.6 / 94.5 | 100 | 95.75 | 1.07 (seam 0.48) | 9.51 (seam 3.36) | 0.00 |
| (c) drop w032 over 300-120 deg | 98.48 (acc_all 85.24) | 87.1 / 99.8 | **86.55** [81.2, 91.7] | 98.42 | 0.00 | 0.04 | 1.42 |

How to read the table:
- **GT ceiling.** The GT plumbing check is 100%. The oracle's 0.47-pt shortfall is entirely the per-site registration
  of the human meshes: the published affine is off by 0.8-2.3 vox, and with the NCC shift removed the oracle scores
  100.00. The loss falls only on contact pairs. So a perfectly CT-aligned fit can reach 100, and the oracle is the
  floor for a fit carrying the published affine's error.
- **Corruptions hit exactly the predicted pairs.** Per pair, against the oracle, from the `--dump-pairs` CSVs:
  - (b1): 396 / 396 predicted mix pairs went wrong (pairs with exactly one end on w032 inside the sector); 0 changed
    elsewhere.
  - (b2): 381 of 383 predicted (the other 2 were already wrong); 0 changed elsewhere.
  - (c): 1,069 of 1,072 pairs with an end on the dropped half; 0 changed elsewhere.
  - (a): bit-identical to the oracle.
- **Switch-rate size.** Under (b1) the rise is small in absolute terms: only the two sector edges of one winding
  switch, about 300 of 7.2M pairs. The CI still excludes 0.
- **A/B check.** `--fit oracle --fit-b c_relabel` gives B - A pair acc -5.30 [-8.85, -1.99] and switch
  +0.05 [0.025, 0.073].
- **Contact-zone floor (oracle9).** The same human surfaces, sampled on 20-vox quads, switch 120 per 1000 in
  contact zones (another mesh within 4 vox) and about 0 in open zones. Inside the gated test-site crops it is about 0.
  The errors are spread over all adjacent winding pairs, so this is geometric fidelity, not label error. Read
  contact-stratum vertex metrics against this floor, and export fits at a grid step of 5 vox or less. Pairs are
  unaffected (99.56).

**Runtime and memory** (20 CPU threads; the machine was shared with other jobs, about 26 GB in use):
- The first run builds the human vertex cache in 50 s (`data/scratch/spiral_eval/cache/`).
- After that, each fit takes 50-150 s, and an A/B run takes 191 s.
- A whole oracle run took 100 s; peak RSS was 3.1 GB with the 16.6M-quad oracle and 1.4-2.4 GB otherwise.

## Caveats
- All in-band test sites sit outside the #2167 gap. The pair exclusions have no effect in this band; other bands were not
  checked. The vertex exclusion does have an effect.
- A real spiral fit has its own seam. Use the `*_seam` numbers, or check `offset_bins_off_global` in the JSON. If it
  is greater than 0, the global offset is mixing a seam with errors.
- The per-site shift means human meshes and pair labels disagree by about 1.4 vox. Tolerances below about 2 vox will
  under-match the oracle.

Files are in `data/scratch/spiral_eval/`:
- oracle fits: `oracle/`, `oracle9/`;
- corruptions: `c_*/`;
- results: `*.json`, `*.log`, `*_pairs.csv`.
