# Real ~9 µm surface training set (Dataset103_Real9), 2026-10-18

Real ESRF crops at 9.362 µm and 8.64 µm, labelled from the public segment meshes of four scrolls other than PHerc0139. They are for fine-tuning the surface model at the test-set resolution. PHerc0139 is excluded: `EXCLUDE` in the script and an assert in `main()`.

- Script: `scripts/real9_trainset.py`. It imports `read_box`, `gates`, `cast_pairs`, `band_from_surface`, `grid_normals`, `shift_search` and others from `scripts/p0139_testset.py`. Steps: `fetch`, `mine`, `build`, `finalize`, plus `cfg` to print the transforms.
- Per-scroll output: `data/train/real9/<scroll>/{imagesTr,labelsTr,ignoreTr}/`.
  - The files are hardlinks into `data/train/real9/_work/<scroll>/all/`, which also holds `instancesTr`.
  - Manifest: `data/train/real9/manifest.json`. It has the crop origin, scroll, voxel size, gate stats, transforms, the segment frame check, registration stats and the failures.
- nnU-Net: `data/nnunet/nnUNet_raw/Dataset103_Real9/`. 405 cases, same conventions as Dataset101: Tiff3DIO, `.tif`, no spacing sidecars. Labels: background 0, surface 1, ignore 2, with the ignore mask merged into the label as 2. Images are symlinks.
- Preview: `data/scratch/real9_preview.png`. It shows two crops per scroll, three orthogonal slices each: CT, label contour in red, ignore in green.
- Licence: CC BY-NC / ash2txt data. It stays local. Nothing is uploaded, published or committed.
- Streaming: only 192³ crops plus margin are read, through anon s3fs. Upper bound on bytes read: 12.3 / 20.1 / 2.8 / 2.7 GB. The frame check also reads one L4 overview per scroll.

## Counts

| scroll | target volume | µm | candidates | built | passed | segments usable / total | segments in passed crops |
|---|---|---|---|---|---|---|---|
| PHerc0814 | 20250804134230 | 9.362 | 150 | 150 | **148** | 19 / 19 | 19 |
| PHerc0500P2 | 20250820143440 | 9.362 | 150 | 147 (3 errors) | **124** | 46 / 46 | 45 |
| PHerc0009B | 20250521125136 | 8.64 | 120 | 120 | **103** | 18 / 19 | 18 |
| PHerc1447 | 20250521151220 | 8.64 | 120 | 120 | **30** | 5 / 15 | 2 |
| **total** | | | 540 | 537 | **405** | | |

Failed crops by gate (a crop can fail several):

| scroll | ct | label | papyrus | coverage |
|---|---|---|---|---|
| PHerc0814 | 0 | 0 | 2 | 0 |
| PHerc0500P2 | 5 | 19 | 19 | 20 |
| PHerc0009B | 3 | 6 | 6 | 17 |
| PHerc1447 | 0 | 33 | 33 | 90 |

The 3 errors are 0500P2 crops at z ≈ 6720. They read "no overlap between render and CT" because they lie beyond the coverage of the 4.317 µm source volume. They were logged and dropped.

Median stats of the passed crops (0814 / 0500P2 / 0009B / 1447):

| stat | 0814 | 0500P2 | 0009B | 1447 |
|---|---|---|---|---|
| pooled on-papyrus | 0.82 | 0.81 | 0.85 | 0.87 |
| unlabelled material | 0.28 | 0.20 | 0.48 | 0.74 |
| label fraction | 0.094 | 0.13 | 0.061 | 0.035 |
| ignore fraction | 0.31 | 0.29 | 0.43 | 0.60 |
| instances per crop | 11 | 17 | 5 | 2 |
| pitch (vox) | 15.6 | 12.1 | 16.4 | 10 (default) |

## Transforms (mesh to target volume)

All transforms come from the bucket `metadata.json`:

- `samples.<scroll>.sample.properties.volume_transforms[from].transforms[to].matrix` (3x4, xyz);
- each segment's `original_volume_id`;
- the `tifxyz-transformed` `<id>-on-<vol>-<um>.tifxyz` paths.

The rule is to always load the mesh on its original tracing volume. Use identity if that volume is the target. Otherwise use the published affine, or skip. No registration was invented. The NCC search below only refines a published affine.

- **PHerc0814**:
  - 12 segments were traced on the 9.362 µm target, so identity.
  - 7 segments are on 20260309142202-2.399um and use the published affine 20260309142202 → 20250804134230.
  - On top of that affine, a per-crop NCC translation refinement: NCC 0.94 / 0.956 / 0.966 (p10/p50/p90) against 0.87 median with the published affine alone. Median shift 1.07 vox (p90 2.35).
- **PHerc0500P2**:
  - All 46 segments are on 20250528085330-4.317um. They use the published composite 20250528085330 → 20250526151718 → 20250820143440.
  - The 2.215 → 9.362 step comes from only 4 landmarks, so the per-crop NCC refinement matters here. Search radius 10. NCC 0.79 / 0.919 / 0.93 against 0.69 published-only. Median shift 2.5 vox (p90 4.1).
  - Instances whose group registration failed went to ignore (430).
- **PHerc0009B**: 18 segments are native to the 8.64 target, so identity.
- **PHerc1447**: 15 segments carry `-on-20250521151220-8.640um` (identity), but only 5 fit (see the frame check below).

## Skipped segments

- PHerc0009B `20250910185200`: obj mesh only, on 2.401 µm. There is no tifxyz on its original volume.
- PHerc1447: ten `20250502*-auto_grown_*` segments.
  - Their meshes are labelled as on the target volume but do not fit it: 0–19% of points lie outside the volume (z up to 24973 against a shape of 24297), and 38–91% fall on masked (zero) CT at L4.
  - Their true tracing frame has no published transform, so they were skipped.
  - The 5 that fit are 20250502205333, 20250702235910, 20250703025628, 20250703034159 and 20251105093211: 0% outside and ≤1.5% masked.
  - Only 2 of these 5 appear in passed crops, so 1447 contributes low diversity.
- Frame check results elsewhere (every segment's mesh against the target L4):
  - 0814 ≤0.2% masked;
  - 0500P2 ≤14.8% (mask edge / 4-landmark affine);
  - 0009B ≤12%;
  - ≤0.3% outside everywhere.

This was found the hard way. The first build passed 51 PHerc1447 crops whose CT was entirely zero: Otsu on empty CT calls every voxel "papyrus". Two gates now prevent this:

1. A per-segment frame check in `mine`: skip if more than 1% of points are outside the volume or more than 25% are on masked CT.
2. A per-crop CT gate: at least 30% non-zero CT, at least 95% of label voxels on non-zero CT, and core material of at least 5%.

All 30 PHerc1447 crops that pass now are 100% non-zero CT.

## Labels and gates

Labels follow the p0139 test-set conventions.

**Label band.** A binary ~3-vox recto band from `band_from_surface`. Meshes are rasterised with sub-quad sampling at ≤0.5 vox. Grid steps are coarse: about 20 vox (187 µm) for most 0814/0009B/1447 meshes, 9.2 vox for 0500P2, and 5.1 vox for one 0814 mesh.

**Instances.** Connected pieces of each segment's raster.
- Pieces under 50 vox go to ignore.
- Instances under 500 vox are rejected.

**Local dedupe.** Voxels of a smaller segment within 4 vox of a larger segment are dropped and sent to ignore (dilated 1). Mining counts area only for non-duplicate points; candidates are disjoint 128³ cores chosen round-robin over dominant segments.

**Instance gates** (from the p0139 `gates()`): on-papyrus ≥ 0.5, frac_off ≤ 0.3, normal agreement ≥ 0.7, deep tiles ≤ 2, crossing ≤ 0.05, registration (NCC ≥ 0.80, shift spread ≤ 2.5). A failing instance goes to ignore (dilated 3) and does not fail the crop.
- Rejections across all scrolls: fragment 1568, on-papyrus 553, registration 430, normal 246, crosses 228, frac_off about 110.

**Crop gates:**
- ct: as above;
- label: ≥ 3000 kept core voxels;
- papyrus: pooled on-papyrus ≥ 0.65;
- coverage: unlabelled material ≤ 0.80.

**Ignore mask.** The union of:
- unlabelled CT material further than dmax = clip(0.7·pitch, 4, 10) from any label (dilated 1);
- off-papyrus label voxels (dilated 2);
- rejected instances (dilated 3);
- dedupe voxels (dilated 1).

It is restricted to non-label voxels.

**Band offset check.** This compares the band ridge with the CT half-max edge (negative means inside the sheet).

| | 0814 | 0500P2 | 0009B | 1447 | jeong200 reference | p0139 test set |
|---|---|---|---|---|---|---|
| edge_t (vox) | −0.69 | −1.33 | −0.45 | −0.78 | −0.93 | −1.78 |

Band thickness is about 3 vox. This matches the recto-face convention (`data/train/real9/_work/band_offset.json`).

## Limitations

- Mesh grids are coarse (187 µm), so labels between grid nodes are bilinear and can cut corners on tight curvature. The NCC refinement is translation-only, per crop.
- Sheets of different segments closer than 4 vox are merged into one label (the smaller one becomes ignore). Tight contact is under-represented.
- The 8.64 µm crops (0009B, 1447) are kept at native resolution, not resampled to 9.362 µm. nnU-Net spacing is 1,1,1 as in Dataset101.
- Labelling is sparse. Unlabelled material, a median 20–74% of the material, is set to ignore rather than background. PHerc1447 is the sparsest: 2 segments, 2 instances per crop.
- 0500P2 depends on a 4-landmark affine. Crops whose registration failed were dropped or set to ignore. Residual misregistration below the NCC gate is possible, with octant deviation 0.82 vox median.

## Re-running

```
uv run python scripts/real9_trainset.py fetch
uv run python scripts/real9_trainset.py mine --scroll PHerc0814,PHerc0500P2 --n-max 150
uv run python scripts/real9_trainset.py mine --scroll PHerc0009B,PHerc1447 --n-max 120
uv run python scripts/real9_trainset.py build --workers 8
uv run python scripts/real9_trainset.py finalize
```

Wall time for the whole effort (recon, debugging and builds) was about 1 h. A clean `build` takes about 8 s per crop per worker.

## Dataset104_Real9Thin (thin-label variant)

`data/nnunet/nnUNet_raw/Dataset104_Real9Thin/` has the same 405 crops and the same images as Dataset103. Build it with `uv run python scripts/real9_trainset.py thin --workers 8`.

**Rebuild without streaming.** The cleaned 1-vox raster is the p0139 "surfacesTr": off-material voxels (> 2 vox from material) and rejected instances are removed. The `thin` step re-derives it from the saved CT crop and the saved per-group registration shifts, so nothing is streamed. The result is checked against Dataset103: band and ignore masks match to 0 voxels in all 405 crops. The rasters are saved in `_work/<scroll>/all/surfacesTr/`.

**Labels:**
- 1 = the raster itself.
- 2 = the Dataset103 ignore mask, plus the shell within 1.5 vox (EDT) of the raster, excluding the raster.
- 0 = background, which includes "gap" shell voxels within 2.5 vox of a *different* instance's raster, where the shells of two sheets touch or overlap.

**Files.** Images and labels are hardlinks, i.e. real files for tar. Dataset103's images were converted from symlinks to hardlinks too. On disk the two datasets total 2.8 GB because the images are shared. A separate tar of each dataset is 2.6 GB.

**Mean label fractions per crop, Dataset103 → Dataset104:**

| class | Dataset103 | Dataset104 |
|---|---|---|
| surface | 0.087 | 0.033 |
| ignore | 0.363 | 0.434 |
| background | 0.550 | 0.533 |

**Gap voxels.** Only 6 crops (all PHerc0814) have any gap voxels, 140 voxels in total, and all of them are between connected pieces of the *same* segment. No crop has gap voxels between different segments. This follows from the build: the local dedupe deletes any smaller-segment raster within 4 vox of a larger segment. Kept rasters of different segments are therefore always more than 4 vox apart, and the median crop pitch is at least 6.4 vox. The < 4 vox compressed case is not in this training data.

**Dedupe at 1.5 vox instead of 4 (diagnostic only, nothing written to a dataset).** Stats are in `data/train/real9/thin_stats_dedup1.5.json`.
- 336 of 405 crops would have gaps between segments.
- Of the extra raster voxels:
  - about 60% lie within 2 vox of the kept raster, i.e. duplicate traces of the same face;
  - about 25% lie 2–4 vox away;
  - only about 40% of the 1.5–4 vox pairs have below-median CT on the line between them.
- Taught gaps would mostly fall between duplicate traces of one sheet. A CT-aware dedupe would be needed to keep genuinely separate touching sheets: keep a trace only if a dark gap separates it from the kept one along the normal.

Preview: `data/scratch/real9_thin_preview.png`. It shows the four lowest-pitch crops: CT, Dataset103 and Dataset104, with yellow for taught gaps. A 1-vox raster looks dashed in 2D slices where the surface is oblique.

## Dataset105_Real9Gap and Dataset106_Real9ThinGap (CT-checked dedupe)

Build both with `uv run python scripts/real9_trainset.py gap --workers 8`. Dataset106 is the "104b" variant; it is renamed because nnU-Net needs a numeric `DatasetXXX_` id.

Both use the same 405 crops and the same images (hardlinks) as Dataset103. The rasters are re-derived locally with no streaming and saved in `_work/<scroll>/all/surfacesTr_ctdedup/`. All 405 crops still pass the crop gates.

**CT-checked dedupe.** A smaller segment's raster voxel within 4 vox of a kept raster is removed, as before, unless the CT shows a distinct sheet:
- The voxel is at least 1.5 vox from the kept raster.
- On the straight line to its nearest kept raster voxel, extended by 1.5 vox beyond both ends, the blurred CT (σ 0.7) dips below "dark".
- The dip must be strictly between the two rasters (≥ 0.5 vox from each).
- Above-"bright" material must lie on both sides of the dip.
- The crop's levels set the thresholds. Air and sheet are the medians below and above Otsu on unmasked CT; dark = air + 0.4·(sheet − air), bright = air + 0.6·(sheet − air).
- The per-voxel decision is smoothed by majority over a 5³ neighbourhood.

A duplicate offset into air (no material beyond it) or into the same sheet (no dip) is removed. Result: 0.38 M raster voxels kept as distinct against 35.3 M removed as duplicates, so about 99% of close tracings are judged duplicates.

**Labels:**
- "Gap" = non-raster voxels within 2.5 vox of two different kept instances, i.e. between distinct sheets up to about 5 vox apart.
- Gap voxels are background (0), overriding ignore.
- Dataset105: 3-vox band = 1 (Dataset103 convention).
- Dataset106: raster = 1, with the 1.5-vox shell set to ignore (Dataset104 convention).

**Counts:**

| | 103 | 104 | 105 | 106 |
|---|---|---|---|---|
| crops with gap voxels | 0 | 6 | 331 | 331 |
| gap voxels | 0 | 140 | 2.64 M | 2.87 M (0.43 M of them in the shell) |
| ignore → background vs 103/104 | – | – | 2.68 M (285 crops ≥ 500 vox) | 2.95 M (293 crops) |
| new surface voxels | – | – | 0.77 M (+0.3%) | 0.30 M (+0.3%) |
| surface → ignore (instances rejected by the crossing gates once close sheets are kept) | – | – | 0.32 M | 0.13 M |
| mean surface / ignore / background fraction | .087/.363/.550 | .033/.434/.533 | .087/.363/.550 | .033/.434/.533 |

**Close pairs (< 4 vox):**
- Raster voxels within 4 vox of another segment's raster: 0 crops before (by construction). Now 296 crops, of which 267 have ≥ 100 such voxels and 125 have ≥ 1000.
- First-hit spacings along the normal under 4 vox: before, 100 crops and 0.01% of hits (same-segment pieces). Now 238 crops and 0.6% of hits.
- Per-crop median spacing is unchanged: p10/p25/p50/p75/p90 = 10.0/11.9/14.1/17.1/19.6 vox, against 10.0/12.0/14.1/17.1/19.6.
- Compressed sheets are now present in most crops, but they are a small share of the labelled area.

Preview: `data/scratch/real9_gap_preview.png`. It shows the five crops with the most gap voxels, zoomed: CT, 103, 105 and 106, with yellow for taught gaps. Some yellow lies between pairs 4–5 vox apart that already existed in 103; there the gap rule only overrides ignore.

**Limitations:**
- Two wraps of the same segment that form one connected piece in the crop are one instance. No gap is taught between them.
- The CT test is a single straight-line profile per voxel. Partial-volume gaps that never dip below 0.4 of the air-to-sheet contrast at 9 µm are treated as duplicates.
