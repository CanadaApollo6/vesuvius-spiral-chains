# 2026-10-27 Head-to-head: our whole-scroll winding numbers vs Kelly's index on Stevens' patches (PHerc1667): pre-registration

Status: pre-registered before any comparison number was computed. The question it answers: "Kelly's index already gets
~98 % of patches right on Scroll 4; how does your automatic numbering compare?" The two existing numbers measure
different things:
- Kelly, 97.7 / 98.6 %: share of human-labelled **patches** (that exist) with the right global winding, one offset per
  patch component (notes/2026-10-26-patch-numbering-prereg.md).
- Ours, inner seam wrong area: share of **matched label vertices** with the wrong winding under local offsets, one per
  250 z x 10 deg bin (notes/2026-10-26-p1667-wholescroll-prereg.md). That convention is lenient: it only checks local
  consistency.

This note defines one metric set, computed on the same human-label vertices for both, with the same offset freedom.

Already looked at before this freeze (label-free, or ours-only and already published):
- our per-band numbers in data/scratch/wholescroll/eval/pool_s1_all.json (lenient wrong area, unmatched share, cm²);
- Kelly's patch-level numbers for z 4500-6000 and z 6500-8000 from the patch-numbering note;
- geometry-only checks (2026-10-26T21:00-21:35Z, scripts in /tmp, no method-vs-label number printed):
  - the fitter's seam angle, from the fit meshes (first/last valid column of each w*_spliced row), in the
    `umbilicus_p1667_L2.json` frame: per-arm median 175-186 deg; 91-97 % of row ends fall in [160, 200); B6 measured on
    the c1667_*_fig reruns, because the c1667 meshes were deleted;
  - Kelly's thN against the human-frame angle θ_H of patch medoids: thN ≈ 284 deg − θ_H (79 % within ±15 deg). So Kelly's
    angle runs **opposite** to θ_H;
  - continuity of the human turn coordinate T_H on mesh-adjacent reference vertices: 0.08-0.12 % of pairs jump.

No coverage of label vertices by Stevens' patches and no Kelly or fit residual on label vertices has been computed.

## 1. Data (fixed)

- **Bands:** the 7 whole-scroll bands, L2 z half-open: B1 400-1500, B2 1500-3000, B3 3000-4500, B4 4500-6000,
  B5 6000-6500, B6 6500-8000, B7 8000-9000.
  - Stevens' patches span z 281-9004, so Kelly has patches in all 7 bands.
  - **For our method, B4 and B6 are not fresh** (test 8 and test 2 bands; our recipe was confirmed on them).
  - For Kelly, every band is fresh: the index was built without these labels. Kelly's patch-level result on B4/B6 has
    already been computed and published by us.
- **Evaluation vertices:** the cached spiral_eval reference sample per band
  (data/scratch/spiral_eval/cache/ref_p1667_z<z0>-<z1>_s2_m0_v2.npz: merged_v4 human mesh, stride 2, labels
  merged_v4).
  - It is restricted to **inner k < 34**, as in our primary metric. The outer windings k >= 37 are not CT-supported, and
    B1 has no k >= 33.
  - It is further restricted to vertices **outside the seam wedge** [160, 200) deg of `ang` (the umbilicus-frame angle
    stored in the sample). This removes 7-13 % of inner vertices per band, for both methods. It is called E below.
  - Why the wedge: the dumps store only the fit's winding j per vertex, not the in-surface position. The fitter cuts
    each winding at ~180 deg in this frame, with ±5-10 deg scatter. Inside the wedge, a turn coordinate built with a
    fixed 180 deg seam would mark some correct vertices wrong for ours only. The wedge is a property of the fitter's
    export, not of the numbering, so it is removed for both methods.
- **Ours:** the seed-1 dumps (release/fit_vertex_dump.py, tol 4: nearest fitted surface by exact point-triangle
  distance, its winding j, distance):
  - M2 arms: wsf<z0>_m2s1, ink2_m2s1 (B4), c1667_m2s1 (B6);
  - base arms: wsf<z0>_base, ink2_base, c1667_base (the same fitter without our chains; reference row only).
  - shas: data/scratch/kelly_h2h/input_shas.txt.
- **Kelly:**
  - community/wrap_index.csv (k_q3c, component, thN). Of 56,968 patches, 56,934 are numbered; 56,926 are in component 0.
  - Stevens' s4_good_patches.zip and s4_bad_patches.zip, read in memory with `python -I` and mapped to L2 by the affine
    of the patch-numbering note (data/scratch/patch_scout/affine_s4_7p91_to_2p399.json, then /4).
  - The 34 unnumbered patches are dropped from Kelly's surface set.
- **Human turn coordinate:** θ_H = atan2 around the label `centre`, T_H = k − ((θ_H − 144) mod 360)/360. This is
  continuous along a sheet (0.1 % adjacent-pair jumps, above).

## 2. Per-vertex assignment

- **Covered by ours:** the dump has the vertex matched, i.e. the nearest fitted surface is within **4.0 L2 vox**
  (point-triangle). Then j = that surface's winding.
- **Covered by Kelly:** the nearest numbered Stevens patch surface is within **4.0 L2 vox**. The search runs over all
  patches with any vertex in [z0 − 6, z1 + 6). It uses the same code as our dumps (spiral_eval.Surfaces, exact
  point-triangle distance on the patch quads, nearest surface wins), with label = patch id.
  - Patch vertex spacing is ~3.3 L2 vox, below spiral_eval's 2.5-vox decimation threshold, so there is no decimation.
  - Overlapping patches: the nearest one wins, as with our overlapping fit surfaces.
- **Turn coordinates:**
  - ours: T_O = j + s·φ/360, with φ = (ang − 180) mod 360 (fit seam at 180 deg, umbilicus frame);
  - Kelly: T_K = k_p + s·(thN_p[deg] − Δθ_H(medoid_p → v))/360. The patch value is transported along the sheet to the
    vertex. Δθ_H is the short-arc human-frame angle from the patch medoid (patches_all.tsv) to the vertex, with a
    minus sign because Kelly's angle runs opposite to θ_H. It is not wrapped, so it stays continuous across Kelly's own
    cut.
- **Residual:** R = T_H − σ·T_m. A covered vertex is **correct** if |R − o| < 0.5, i.e. it lies within half a turn of
  the method's offset o.

## 3. Offset conventions (the same freedom for both, fitted on that method's covered vertices of E)

- **Per band (primary for ours; "apples to apples" for Kelly):** one (σ, s) ∈ {±1}² and one continuous offset o per
  band and method, chosen to maximise the number of correct vertices. o is found by a window search: the width-1 window
  [r_i, r_i + 1) holding the most residuals.
- **Per band per component (Kelly native; primary for Kelly):** (σ, s) per band as above, then one o per Kelly
  component within the band. Component 0 holds 99.99 % of the patches, so this should barely differ from per band.
- (σ, s) are fitted for both methods (not fixed in advance). For Kelly, agreement with the label-free choice of the
  patch prereg is reported (s_K = +1, σ_K = +1, in that note's T = σ(k + s·thN/360) form). Here the transport sign is
  folded into T_K, so s = +1 means the same thing.
- Offsets are fitted once per band on the point estimate and held fixed in the bootstrap (as in spiral_eval/BootV).

## 4. Metrics (per band, per method; E = inner, non-wedge reference vertices)

1. **Coverage** C = covered / |E|.
2. **Accuracy where covered** A = correct / covered.
3. **Correct share** Q = correct / |E| = C × A. **Correctly numbered area** (cm²) = Q × inner labelled area of the
   band.
   - The inner labelled area is wholescroll_pool.labelled_area()["inner"]: merged_v4 quads with one k, cm² at
     9.596 µm.
   - This assumes the wedge behaves like the rest of E (the vertex sample is uniform on the mesh).
4. **Co-covered accuracy:** on vertices both methods cover, the accuracy of each method with its own offsets (from
   step 3, not refitted on the overlap). This is the head-to-head on identical vertices, with no coverage selection.
5. **Overlap table** (shares of E, then cm²):
   - both cover: both right / only ours right / only Kelly right / both wrong;
   - only ours covers: right / wrong;
   - only Kelly covers: right / wrong;
   - neither covers.
6. **Union ceiling** = share of E where at least one method is right. This is an oracle picker, not a method.

Method rows:
- **ours = M2 s1** (per-band offset);
- base s1 (per-band offset; reference row);
- **Kelly = all numbered patches (good + bad), per-component offset**: the primary Kelly row, Kelly's full output and the
  set behind the 98 % figure;
- Kelly all, per-band offset;
- Kelly good-only, per-component offset.

The primary comparison is **M2 s1 vs Kelly-all per-component**.

**Pooling:** vertex-pooled ratios over bands (Σ numerators / Σ denominators). The vertex sample is uniform on the mesh,
so this is area-weighted. Both pools are reported:
- **all 7 bands (primary pool)**;
- **fresh 5** (B1, B2, B3, B5, B7). These are fresh for ours; Kelly is fresh everywhere.

**Bootstrap:**
- blocks are spiral_eval.BootV's (250 z × 30 deg of `ang`), 1,000 resamples;
- rng np.random.default_rng(2700 + band index 0..6);
- the same resample is used for every method and metric of a band (paired);
- pooled CIs come from independent per-band resamples, summed;
- 95 % percentile intervals; differences are taken per resample.

### Sensitivity read-outs (descriptive, no claims)
- **S-a no wedge:** E includes the wedge (fixed 180-deg seam for ours, which penalises ours near the seam).
- **S-b lenient local offsets:** for both methods, one offset per 250 z × 10 deg bin of `ang` (window search; bins with
  < 50 covered vertices fall back to the band offset). This is our existing "seam" convention, now applied to Kelly too.
- **S-c tolerance 2 vox:** covered = distance < 2.0 for both. Our dumps store the distance and Kelly's query returns
  it. Same offsets as the primary, refitted on the new covered set.
- **S-d Kelly whole-scroll:** one (σ, s, o) for Kelly over all 7 bands together. Kelly's index is meant to be consistent
  across the scroll; our bands are fitted independently and are not stitched, so we have no counterpart.
- **S-e seed 2:** for each band where an M2 s2 dump exists when the s1 run finishes, the M2 s2 row (per-band offset).
  There is no pooled s2 claim unless all 7 exist.

## 5. What each outcome supports (pooled; a claim in our favour must also hold in the fresh-5 pool)

| Statement | Required |
|---|---|
| "Kelly numbers more of the labelled inner surface correctly" | Q_Kelly − Q_M2: 95 % CI lower bound > 0 (all 7) |
| "Ours numbers more of it correctly" | Q_M2 − Q_Kelly lower bound > 0 in all 7 **and** fresh 5 |
| "Kelly is more accurate on the same vertices" | co-covered A_Kelly − A_M2 lower bound > 0 |
| "Ours is more accurate on the same vertices" | co-covered A_M2 − A_Kelly lower bound > 0, all 7 and fresh 5 |
| "Ours covers more" / "Kelly covers more" | C difference lower bound > 0 (ours: also fresh 5) |
| "No detectable difference" on a metric | its CI contains 0 |

- If accuracy favours one method and coverage the other, the note says so in one plain sentence. "More correct area"
  is decided by Q alone.
- Per band: the same differences with CIs, and the count of bands where each statement holds.
- The union ceiling and overlap table are descriptive. "The two are complementary" may be said only if the union
  ceiling exceeds the better single method's Q by at least 5 pp pooled. The complementary area is then quantified in
  cm².

## 6. Predictions (written before any number)

- Kelly's vertex-level accuracy where covered: 93-99 %, close to the patch-level 98 %.
- Ours (M2) under the strict per-band offset: 40-75 %. Under our lenient convention it is ~50-80 % of matched vertices
  (25-50 % inner wrong in five bands, 75 % in B7), and per band is stricter.
- Coverage: ours ~70 % (unmatched ~26-32 % all-k). Kelly: unknown, guess 60-90 % (Stevens' patch area is several times
  the labelled area, with heavy overlap).
- **Expected outcome: Kelly wins on co-covered accuracy and on Q by a wide margin (P ~ 0.9). Coverage could go either
  way.** B7 (our worst band) is the largest gap. The union ceiling may exceed Kelly's Q by a few pp where Stevens'
  patches have holes.
- Caveats, stated in advance:
  - Our fitter's umbilicus was derived from the human mesh. This favours ours slightly. Kelly's axis comes from patch
    centroids.
  - The labels are one merged human read (81 % single-wrap agreement at build).
  - Kelly's index is whole-scroll. Here it receives a per-band offset that it does not need; S-d reports the strict
    whole-scroll version.

## 7. Files and running rules

- Code (new files only):
  - data/scratch/kelly_h2h/kelly_cover.py: per band, nearest numbered patch id and distance for every reference vertex;
    `python -I`; reads the zips in memory and runs nothing from the download dirs;
  - eval/kelly_h2h.py: the scorer, frozen by sha in the Log before it is run on real data;
  - release/figures/p1667_kelly_h2h.png: the figure.
- Smoke tests before the freeze print only counts and shapes, no coverage, accuracy or residual on label vertices.
- CPU only, $0. Jobs over 6 GB RAM run under `flock /tmp/vss_eval.lock` unless free memory is above 20 GB (logged).
- Read only: data/scratch/wholescroll/ (seed 2 is still running there), the frozen scripts, and the download dirs.

## Log (append-only)

### 2026-10-26T21:39:48Z: sections 1-7 frozen
- sha256 of this note up to and including the "## Log (append-only)" line: 9ffc55fed6fae83f19bc99dd832d807ca8f20ed434e00e8fc0a6390275860ac4.
- data/scratch/kelly_h2h/input_shas.txt sha256 893fa836432767f944f5fe6618c2a4379f44588c146535f8539c197734e71248.
- At this time no comparison number exists; no Kelly coverage or residual on label vertices has been computed.

### 2026-10-26T21:42:34Z: correction to a pre-freeze diagnostic (no definition changes)
- The "0.08-0.12 % of adjacent pairs jump" T_H continuity figure in the header came from a /tmp check that swapped the
  label `centre` columns (they are z, x, y). Redone with the correct order, on the same reference samples, labels only:
  0.0 % of adjacent pairs jump in every band, and the umbilicus-frame and label-centre-frame angles differ by a median
  of 2.4-6.0 deg per band (not 7-33). The scorer uses the correct order (as data/scratch/patch_numbering/score.py
  th_hum). The Kelly direction check (thN ≈ 284 − θ_H) used score.py's correct form and stands.

### 2026-10-26T21:45:35Z: code frozen (before the scorer is run on real data)
- eval/kelly_h2h.py sha256 bdfb56413873c05c97181aad098a613095e48ef5f3d7e8f07002aa0c58a2fe56.
- data/scratch/kelly_h2h/kelly_cover.py sha256 618e91574987e88bf9a6e353d883d3331b6ea7bdaad64f585781022b1c93dad8. Ran 21:42-21:50Z on all 7 bands (rc 0, ~1 min per band, 14
  workers; 43 GB free, so no lock), writing kelly_z<z0>.npz. It printed only patch counts: in-band numbered patches
  good/bad 4864/1064 (B1), 10086/2036 (B2), 8867/1740 (B3), 11296/2162 (B4), 4820/941 (B5), 11684/2112 (B6),
  3677/586 (B7).
- All shas (scorer, cover script, the 7 kelly_z*.npz): data/scratch/kelly_h2h/scripts_sha256.txt sha256 d02c5b1265c2d02003c4489fd2607c59dd82643b5405ed6d5c6054944541c872.
- Pre-run checks, with no comparison number printed:
  - `--smoke`: inputs load. |E| per band is 786,920 / 1,498,934 / 1,466,005 / 1,392,300 / 447,025 / 1,351,195 /
    869,500.
  - M2 s2 dumps exist for B4, B6 and B7 (S-e rows).
  - A synthetic check of the offset code (T = −T_H + 7 ± 0.3 gives 100 %; 10 % off-by-one gives 90.0 %; two groups
    with separate offsets give 100 %).

## Results (appended 2026-10-26T21:50Z; everything above the Log is unchanged since the freeze)

Run: `uv run python -I eval/kelly_h2h.py --out data/scratch/kelly_h2h/h2h_s1.json`, with the frozen scorer bdfb5641….
It took 69 s on CPU at $0. Output sha 8c60f19b…; log data/scratch/kelly_h2h/logs/h2h_s1.log. The figure is
release/figures/p1667_kelly_h2h.png (92d8b952…, from data/scratch/kelly_h2h/make_fig.py). There were no deviations
from sections 1-7.

### Per band (primary: E = inner k < 34 outside the 160-200° wedge; ours = M2 s1 per-band offset; Kelly = all patches, per-component offset)

C = coverage, A = accuracy where covered, Q = correct share of E (all %). cm² = Q × inner labelled area. Differences are
ours − Kelly, with 95 % CIs.

| band | E verts | ours C / A / Q | Kelly C / A / Q | ΔC | ΔQ | co-covered A ours / Kelly (Δ) | correct cm² ours / Kelly (of labelled) |
|---|---|---|---|---|---|---|---|
| B1 400-1500 | 786,920 | 68.0 / 75.9 / **51.6** | 61.5 / 78.7 / 48.4 | +6.5 [1.3, 12.4] | **+3.3 [0.3, 6.4]** | 88.3 / 80.0 (**+8.3 [5.6, 11.0]**) | 42.0 / 39.4 (81.4) |
| B2 1500-3000 | 1,498,934 | 72.9 / 68.0 / 49.5 | 75.8 / 93.7 / **71.0** | −2.9 [−4.6, −1.2] | −21.4 [−23.7, −18.9] | 71.5 / 94.2 (−22.7) | 81.8 / 117.2 (165.1) |
| B3 3000-4500 | 1,466,005 | 75.0 / 81.5 / 61.2 | 70.8 / 97.9 / **69.3** | +4.2 [1.1, 7.8] | −8.1 [−9.9, −6.3] | 93.6 / 98.2 (−4.6) | 101.7 / 115.2 (166.3) |
| B4 4500-6000 (used) | 1,392,300 | 78.7 / 77.3 / 60.8 | 81.6 / 98.4 / **80.3** | −3.0 [−4.5, −1.3] | −19.5 [−21.0, −17.8] | 79.2 / 98.5 (−19.4) | 101.5 / 133.9 (166.8) |
| B5 6000-6500 | 447,025 | 80.9 / 78.5 / 63.5 | 83.5 / 98.5 / **82.2** | −2.5 [−4.1, −1.1] | −18.6 [−20.9, −16.2] | 80.5 / 98.7 (−18.3) | 35.0 / 45.3 (55.1) |
| B6 6500-8000 (used) | 1,351,195 | 71.7 / 52.2 / 37.4 | 77.0 / 96.7 / **74.4** | −5.2 [−7.1, −3.3] | −37.0 [−39.2, −34.8] | 55.8 / 97.5 (−41.6) | 62.2 / 123.6 (166.1) |
| B7 8000-9000 | 869,500 | 74.2 / 19.1 / 14.2 | 47.3 / 97.4 / **46.0** | +27.0 [18.4, 35.2] | −31.8 [−40.4, −23.5] | 27.2 / 97.7 (−70.5) | 12.8 / 41.6 (90.5) |
| **pooled, all 7** | 7,811,879 | 74.2 / 65.6 / 48.7 | 71.9 / 95.2 / **68.5** | **+2.3 [1.0, 3.8]** | **−19.8 [−21.2, −18.4]** | 73.2 / 95.9 (**−22.7 [−24.0, −21.3]**) | **437 / 616** (891) |
| pooled, fresh 5 | 5,068,384 | 73.7 / 65.7 / 48.4 | 67.9 / 93.8 / 63.7 | +5.8 [3.8, 8.0] | −15.3 [−17.2, −13.5] | 76.1 / 94.5 (−18.4 [−20.3, −16.4]) | 273 / 359 (558) |

Other method rows (pooled, all 7; C / A / Q):
- Kelly all, per-band offset: 71.9 / 95.2 / 68.5. This is identical to per-component, because component 0 is
  practically everything.
- Kelly good-only: 67.9 / 95.5 / 64.9. Ours − Kelly-good ΔQ is −16.2 [−17.5, −14.9] and ΔC is +6.3.
- Our base fit without chains: 75.5 / 20.1 / 15.2.

The fitted orientations equal the patch prereg's label-free choice in every band (Kelly σ = +1, s = +1). Ours is
σ = +1, s = −1 in every band.

**Overlap, pooled over all 7 bands** (share of E, then cm² of the 891 cm² inner labelled surface):

| | both cover | only ours covers | only Kelly covers | neither |
|---|---|---|---|---|
| both / the one right | 40.6 % (366 cm²) | right 6.9 % (62 cm²) | right 13.8 % (124 cm²) | 10.9 % (96 cm²) |
| only ours right | 1.2 % (10 cm²) | wrong 10.2 % (88 cm²) | wrong 1.1 % (9 cm²) | |
| only Kelly right | 14.1 % (127 cm²) | | | |
| both wrong | 1.2 % (10 cm²) | | | |

The union ceiling (an oracle picker) is 76.6 % [75.0, 78.2], which is **+8.1 pp [7.7, 8.5]** over Kelly's Q (fresh 5:
+8.9).

### Sensitivity (pooled all 7, ours − Kelly)

| variant | ours A / Q | Kelly A / Q | ΔC | ΔQ | Δ co-covered A |
|---|---|---|---|---|---|
| primary | 65.6 / 48.7 | 95.2 / 68.5 | +2.3 | −19.8 | −22.7 |
| S-a no wedge (fixed 180° seam) | 64.6 / 48.0 | 95.2 / 69.4 | +1.4 [0.1, 2.8] | −21.5 | −24.4 |
| S-b lenient (250 z × 10° offsets, both) | 68.6 / 50.9 | 95.5 / 68.7 | +2.3 | −17.8 | −20.5 |
| S-c tolerance 2 vox | 67.2 / 30.2 | 96.0 / 49.5 | **−6.6 [−7.8, −5.4]** | −19.2 | −22.3 |
| S-d Kelly, one offset for the whole scroll | | 95.21 / 68.47 | | | |
| S-e seed 2 (B4 / B6 / B7 only) | Q 60.8 / 37.4 / 17.3 (s1: 60.8 / 37.4 / 14.2) | | | | |

- S-c: at 2 vox, Kelly covers more in 5 of 7 bands. Stevens' patches lie closer to the human surface than our fitted
  surfaces do, so **our coverage edge exists only at the 4-vox tolerance**.
- S-d: Kelly's single offset over all 7 bands loses nothing (A 95.21 vs 95.24, Q 68.47 vs 68.50). Kelly's numbers are
  globally consistent over the whole scroll. Ours are per band and not stitched.

### Verdict under section 5

| Statement | Result |
|---|---|
| Kelly numbers more of the labelled inner surface correctly | **Yes.** ΔQ −19.8 pp [−21.2, −18.4] (fresh 5: −15.3). 616 vs 437 cm² of 891 |
| Ours numbers more of it correctly | No (pooled). It holds in 1 of 7 bands: B1, +3.3 [0.3, 6.4] |
| Kelly is more accurate on the same vertices | **Yes.** 95.9 vs 73.2 %, Δ −22.7 pp [−24.0, −21.3]. Kelly wins 6 of 7 bands; ours wins B1 (+8.3) |
| Ours covers more | **Yes at 4 vox:** +2.3 pp [1.0, 3.8], fresh 5 +5.8. Bands: ours B1, B3, B7; Kelly B2, B4, B5, B6. It reverses at 2 vox (S-c) |
| Complementary (union ≥ best + 5 pp) | **Yes.** +8.1 pp = 72 cm²: 62 cm² where only our surfaces exist and are right, plus 10 cm² where both cover and only ours is right |

- **Accuracy and coverage split, as section 5 anticipated.** Kelly is far more accurate. We cover slightly more at the
  4-vox tolerance, but not at 2 vox.
- **On "more correct area", Q decides, and Kelly wins.**
- **Predictions held, except B1.** Kelly is 95.2 % vertex-level, against a patch-level 98 %. On the same bands, B4
  is 98.4 vs 98.6 and B6 is 96.7 vs 97.7, so the vertex-level metric reproduces the patch-level numbers. Ours is
  65.6 %. Kelly wins Q and accuracy.
- **Not predicted:** at the scroll bottom (B1, fresh), Kelly's index is weak (78.7 %, the same with a whole-scroll
  offset), and ours wins on both Q and co-covered accuracy there.

Derived from the overlap table (not pre-registered; arithmetic only):
- A practical non-oracle combination, "Kelly where a patch exists, our fit elsewhere", would reach Q = 68.5 + 6.9 =
  75.4 %, about +62 cm².
- It would also add 88 cm² of wrong numbers. Our fill is right on only 40 % of the hole area, so the combined
  accuracy drops to ~85 %.

### Answer to "Kelly already gets 98 %; how do you compare?"

Scored the same way on the same human-label vertices (inner windings, seven bands, z 400-9000), Kelly's index on
Stevens' patches is clearly better at numbering. It covers 72 % of the labelled inner surface and is right on 95 % of
what it covers, so 68 % (616 cm²) is correctly numbered. Our automatic numbering covers 74 % and is right on 66 %:
49 % (437 cm²). On vertices both methods cover, Kelly is right 96 % of the time and we are right 73 % of the time. Our
contribution is real but narrower:
- **Our chains lift the fitter.** Without them, the fitter is right on 20 % of covered vertices under the same strict
  per-band rule; with them, 66 %.
- **We beat Kelly at the scroll bottom.** In z 400-1500, Kelly's index is weak (79 %) and ours is better on both
  correct area and same-vertex accuracy.
- **Our fit adds area in Kelly's holes.** It numbers surface wherever the scroll is, including where Stevens has no
  patch, and adds 62 cm² of correctly numbered area there. That fill is only ~40 % accurate, so it would need a
  human or Kelly-side check before use.
- **Kelly's index is consistent over the whole scroll; our numbers are per band.**

In short: don't claim to rival Kelly on winding accuracy. The defensible claims are the chain lift over the fitter and
the coverage of surface that patches don't reach.
