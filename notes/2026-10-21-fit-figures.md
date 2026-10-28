# 2026-10-21 Before/after figures: auto annotations in the villa spiral fit (PHerc0139, z 3000-4500)

Claim being illustrated (notes/2026-10-21-annotation-round2.md, Round 3): arm19 (+ automatic winding annotations)
vs arm0 (baseline) on band z 3000-4500: matched pair acc +12.3 pp, wrong-winding area -4.7 pp. arm2 = human-oracle
annotations (upper bound). All coordinates are 9.362 um L0 voxels of volume 20250728140407 (zyx). Figures live in
release/figures/ (PNG, 256-colour quantized, 0.24-0.98 MB). Each has a release/ script; shared helpers are in
release/fit_traces.py.

## Figures

| file | what it shows | script |
|---|---|---|
| fit_slice_z4200.png | **lead.** CT slice with human / arm0 / arm19 traces, coloured by "right winding" | fig_slices.py |
| fit_slice_z3300.png | the same at a second site (inner windings) | fig_slices.py |
| fit_slice_z3300_outer.png | counter-example: an outer site where arm19 is worse | fig_slices.py |
| fit_wrong_winding_map.png | top-down angle x z wrong-winding share for arm0 / arm19 / arm2, plus a fixes-vs-breaks panel | fig_wrong_map.py (+ fit_vertex_dump.py) |
| fit_render_strips.png | flattened VC3D renders of human w029, w030 and the fit windings on them (arm0 / arm19 / arm2) | fig_renders.py |
| fit_render_zoom.png | zoom of the render strips: arm0's extra turn j48 vs the real sheet, and arm19's banding | fig_renders.py |
| fit_render_strips_w031_w039.png | supplementary: renders for human w031 and w039 (arm0 duplicate turn j57/j58) | fig_renders.py |

### Slice figures (fig_slices.py)
- Panels:
  - human meshes: oracle copies, tab10 colours by winding. Blacklisted w041, w042, w045 and w046 are dashed, marked "x" and not used as reference.
  - arm0 and arm19 traces: z-iso contours of meshes_step4. Each fit is labelled in the human numbering using spiral_eval's per-bin seam offsets.
  - Fit colours: green = on the human sheet it is numbered as; red = on another human sheet; grey = no human sheet within 4 vox.
  - The % in each title = right / wrong share of traced length within 4 vox of a human trace.
- Sites, with right/wrong % of traced length:

  | figure | z | centre (y, x) | crop | angle | arm0 right/wrong | arm19 right/wrong | windings |
  |---|---|---|---|---|---|---|---|
  | fit_slice_z4200 | 4200 | 3312, 4205 | 360 | 0 deg | 28 / 72 | 94 / 6 | about w026-w040 |
  | fit_slice_z3300 | 3300 | 2510, 3950 | 420 | 330 deg | 55 / 45 | 76 / 24 | about w028-w040 |
  | fit_slice_z3300_outer | 3300 | 2420, 4130 | 300 | about 330 deg, outer | 96 / 4 | 76 / 24 | about w043-w050 |

- Commands:
  - `uv run python release/fig_slices.py --z 4200 --center 3312 4205 --size 360 --out release/figures/fit_slice_z4200.png`
  - `uv run python release/fig_slices.py --z 3300 --center 2510 3950 --size 420 --out release/figures/fit_slice_z3300.png`
  - `uv run python release/fig_slices.py --z 3300 --center 2420 4130 --size 300 --out release/figures/fit_slice_z3300_outer.png`
  - Each command streams one CT L0 slice crop from the open-data bucket. Nothing else is downloaded.
- All sites are outside the w033-w037 / villa #2167 gap sector (about 145-260 deg).

### Wrong-winding map (fig_wrong_map.py)
- How it is built:
  - fit_vertex_dump.py re-runs the spiral_eval vertex metric per vertex, one surface at a time. It uses the reference cache ref_z3000-4500_s2_m0_v2.npz, tol 4, and 10 deg x 250 z seam-offset bins, and writes data/scratch/fit_figures/vertex/<arm>.npz/.json.
  - Its numbers match the scorer: wrong_area seam / global = arm0 48.84 / 56.48, arm19 44.15 / 46.11, arm2 35.30 / 39.87.
  - Runtime is about 13 min per arm, about 1 GB RAM.
- Cells are 3 deg x 25 z, pooled over windings.
- The 4th panel shows where arm19 fixes arm0 (18.4 % of matched vertices) vs breaks it (14.0 %): net +4.4 pp.
- The #2167 sector (145-260 deg) is marked.

### Render figures (fig_renders.py + pod)
- Pipeline:
  - On the pod, spiral/render_compare.sh ran s05_snap (KEEP_MIN_SHEET=0, m7 L0 rows 15-24), then s06_flatten (lasagna flatten_fast_nofilter), then s06b_montage --max-mnodes 0.01 (one winding per montage), then s07_render (VC3D, 66 layers, CT streamed over https), then render_mid.py (mean of the 7 middle layers).
  - Inputs were the meshes/ w###_spliced surfaces (20-vox grid) for arm0, arm19 and arm2.
  - Human meshes: oracle w029, w030, w031 and w039, masked to z 3000-4500 with meta scale corrected to 0.195. They went straight to montage + render, with no snap and no flatten.
- Local step: each render is resampled onto an angle x z canvas (0.25 deg x 4 z; the zoom uses 0.06 deg x 1 z). Human meshes are cut to their first 360 deg.
- Outlines: red = on a human sheet other than the winding's dominant one; yellow = on no human sheet within 4 vox.
- Blue shading marks where sheet identity is ambiguous, and these zones are left out of the percentages: the angle ranges where a human mesh covers an angle twice, plus the fit seam at 171-178 deg.
- Stats (from data/scratch/fit_figures/renders/render_stats_<set>.json):

  | fit winding | main sheet | on main | other | none | numbered as | numbered right |
  |---|---|---|---|---|---|---|
  | arm0 j47 | w029 | 90 % | 3 % | 7 % | w029 | 67 % |
  | arm19 j42 | w029 | 86 % | 3 % | 11 % | w029 | 84 % |
  | arm2 j46 | w029 | 91 % | 2 % | 7 % | w029 | 85 % |
  | **arm0 j48 (extra turn)** | w029 | 68 % | 14 % | 18 % | w030 | 28 % |
  | arm0 j49 | w030 | 90 % | 3 % | 7 % | **w031** | 22 % |
  | arm19 j43 | w030 | 88 % | 2 % | 10 % | w030 | 87 % |
  | arm2 j47 | w030 | 92 % | 2 % | 6 % | w030 | 87 % |
  | arm0 j50 | w031 | 85 % | 4 % | 11 % | w032 | 27 % |
  | arm19 j44 | w031 | 81 % | 4 % | 15 % | w031 | 78 % |
  | arm0 j57 | w039 | 56 % | 23 % | 21 % | w039 | 36 % |
  | arm0 j58 (duplicate of j57) | w039 | 60 % | 25 % | 15 % | w040 | 37 % |
  | arm19 j52 | w039 | 65 % | 19 % | 16 % | w039 | 64 % |

- Zoom windows:
  - 0-45 deg x z 3300-3900: arm0 j48 shows dark air gaps; w030, arm0 j49, arm19 j43 and arm2 j47 show papyrus fibre.
  - 285-330 deg x z 3500-4100: arm19 vertical banding.
- Commands:
  - `uv run python release/fig_renders.py --overview release/figures/fit_render_strips.png --zoom release/figures/fit_render_zoom.png --zoom-win 0 45 3300 3900 --zoom-win 285 330 3500 4100`
  - `uv run python release/fig_renders.py --set w031_039 --overview release/figures/fit_render_strips_w031_w039.png --zoom /dev/null`

## What the figures say
- arm19's gain is in the inner windings. Wrong-area by winding group (point estimates, no bootstrap):

  | windings | arm0 | arm19 | arm2 |
  |---|---|---|---|
  | w023-w040 | 59.3 | 40.0 | 32.7 |
  | w028-w040 | 49.9 | 27.4 | 32.2 |
  | w043-w059 | 29.9 | 39.3 | 29.9 |

  - On w028-w040, arm19 even beats arm2.
  - On w043-w059, arm19 is 9.5 pp worse than arm0 (arm2 ties arm0). The outer-site slice figure shows this.
- arm0's inner failure is mostly a counting failure, not bad surfaces. arm0 inserts an extra turn j48 between w029 and w030: only 36 % of it is within 4 vox of any human sheet before snap. Everything outward is then numbered one too high (j49 sits cleanly on w030 but is numbered w031). arm0 also lays two windings (j57, j58) on w039.
- arm19's surfaces are slightly noisier: 10-15 % of the area is on no sheet, vs 7-11 % for arm0. There is vertical banding at 270-360 deg, consistent with Round 3's higher switch rate (+17.6 per 1000).

## Caveats
- **Selected sites.** The slice sites were picked to show the effect, plus one counter-example. Headline numbers must come from spiral_eval (pair acc, wrong area), not from the slice percentages.
- **The outer human labels are inconsistent.**
  - w041/w042 are coincident (median 2.5 vox apart), and so are w045/w046 (0.1 vox); all four are blacklisted.
  - w040 to w043 is 30.6 vox and w046 to w047 is about 35 vox, i.e. two turns each, so a sheet is missing.
  - All three arms put 2 fit windings between w040 and w043.
  - A naive relabel (k >= 43 becomes k - 1) flips arm19 - arm0 wrong-area from -4.7 to +13.5 pp, because the arms also disagree on w044-w047.
  - So the band-wide wrong-area number is fragile to the outer labelling. Site-local pair accuracy and the inner-winding comparison are the robust parts. Flag this in the write-up.
- **Seam-bin artefact.** The fits' seam (173-176 deg) falls inside the 170-180 deg offset bin, which gives a dark stripe in every arm on the wrong map. It is annotated in the figure and is small.
- **Human meshes overlap themselves** (each spans 365-400 deg, with seams at about 120-140 and 225-260 deg). They also cross each other in compressed zones: z 4200 at w028-w031, z 3300 at w038-w040, and w048/w049 at the z 3300 outer site. Single-pixel "red" there can be reference noise.
- **Snap and flatten.**
  - Snap moves fit surfaces by up to about 15 vox.
  - Human meshes were neither snapped nor flattened, so their renders are slightly sharper and are not a like-for-like texture comparison.
  - s06b ran with --max-mnodes 0.01 to force one winding per montage (not the stage default).
- **Pod.** The task asked for runpod. runpod had no L40S (HTTP 503; re-listed, no pod created), so a massedcompute L40S was used.

## Cost
- **Pod:**
  - vss-p0139-render (<pod-id>), 1x L40S on massedcompute, $0.82/h.
  - Created 2026-10-21T16:59:13Z, terminated 17:30:41Z: 0.52 h, about **$0.43** (cap $25).
  - Logged in data/scratch/p0139_spiral/pods_arms.txt. `prime pods list` was empty afterwards.
  - Pod time: setup about 8 min. render_compare took 8.5 min in total: m7 fetch 10 s; snap 2-3 s and flatten 53-113 s per arm; renders 6-44 s each.
- **Local:** 3 vertex dumps (about 13 min each, about 1 GB RAM), CT slice crops, and figure rendering. No GPU used.

## Data (local only)
- data/scratch/fit_figures/vertex/: per-vertex dumps.
- data/scratch/fit_figures/renders/<arm|human>/: mid-layer renders, montage tifxyz, snap reports, logs.
- data/scratch/fit_figures/pod/: the bundle sent to the pod.
- data/scratch/fit_figures/purity.json: per fit winding, the share of area within 4 vox of each human sheet.

## Label audit, band 1 (2026-10-21, 18:10Z; human meshes, CT, m7 and grad_mag only)
Outputs are in data/scratch/label_audit/. Scripts: eval/label_subsets.py, eval/ct_gap_audit.py, eval/subset_wrong_area.py
and eval/label_audit_figs.py. Figures: fig_ct_gap_ray.png, fig_ct_turn_counts.png and fig_ct_sector2167.png.

**Subsets (frozen in second-band-plan 7b).**
- 16 consistent: w023, w029-w033, w037, w044, w047, w048, w052-w054, w057-w059.
- 21 inconsistent: w024-w028, w034-w036, w038-w043, w045, w046, w049-w051, w055, w056.
- Coincident pairs: only w041/w042 and w045/w046. Everything else fails on spacing; pass rates are 43-77 % of angle bins.

**CT method.**
- Rays: 24 angles x 6 z (3136-4386), with the human seams (115-145 and 220-265 deg) avoided. A dense set covers
  140-270 deg in 5 deg steps for #2167.
- Per ray, the crossing radius of each mesh is taken, then three counts are made between two meshes:
  - CT peaks;
  - m7 runs strictly between the two meshes' own runs (a run within 4 vox of a mesh is that mesh's sheet);
  - integrated lasagna grad_mag, i.e. turns, scaled on dk=1 controls.
- Calibration on controls whose windings are consistent:
  - turns from m7 (runs + 1): 1.05 / 1.93 / 2.95 / 3.88 for dk = 1 / 2 / 3 / 4;
  - turns from grad_mag: 1.00 / 2.08 / 3.21 / 4.25.
- CT peak counting overcounts (0.36 false peaks per dk=1 gap) and is not used for verdicts.
- "Local" estimates divide each ray by that ray's own control spans (w023-w033 + w052-w059). CIs are 95 % ray
  bootstrap, 2000 resamples.

| span | labels | m7 turns, share of rays | turns, local gm / m7 [CI] | verdict |
|---|---|---|---|---|
| w041 vs w042 | 0 (dup) | m7 between 0.03; median gap -0.1 vox | gm 0.33 | same sheet (certain) |
| w045 vs w046 | 0 (dup) | 0.00; gap 0.1 vox | 0.08 | same sheet (certain) |
| w040-w041, w042-w043 | 1 | 1 turn on 87 % / 90 % | 1.07 / 1.03 (raw) | one turn each; w041=w042 is a real sheet midway |
| **w040-w043** | 3 | 2 turns 66 %, 3 turns 18 %, 1 turn 14 % | 1.93 [1.78, 2.04] / 2.12 [2.00, 2.12] | **2 turns: labels overcount by 1** (~95 %) |
| w037-w043 | 6 | 5 turns 60 % | 5.15 (raw) | 5 turns (same overcount) |
| **w046-w047** | 1 | 2 turns 59 %, 3 turns 30 % (clustered at z >= 3886, 160-350 deg), 1 turn 10 % | 2.11 [1.96, 2.32] / 2.27 [2.12, 2.43] | **one untraced sheet** (~90 %); a second one on ~30 % of rays |
| w044-w047 | 3 | 3 turns 53 %, 4 turns 34 % | 3.22 (raw) | 3 turns on most rays: the w045=w046 duplicate and the untraced sheet cancel |
| w043-w047 | 4 | 4 turns 50 %, 5 turns 32 % | 4.07 [3.94, 4.26] / 4.50 [4.25, 4.86] | about 4 |
| w047-w052 | 5 | 4 turns 47 %, 5 turns 34 % | 4.11 [3.97, 4.31] / 4.25 [4.25, 4.47] | about 4.2; w049-w051 is 1 turn on 57 % of rays where w050 is missing |
| w033-w040, outside 145-260 deg | 7 | | 6.61 [6.31, 6.92] / 7.00 [6.38, 7.00] | labels right |
| **w033-w040, #2167 sector** | 7 | | 4.64 [4.09, 5.07] / 4.98 [4.12, 5.49] | **labels overcount by ~2** |
| w034-w036, #2167 sector (dense rays) | 2 | median gap -0.5 vox, gm 0.58 | | **w036 lies on w034** (certain) |
| w033-w052, outside the sector | 19 | | 17.10 [16.39, 17.64] / 17.94 [17.00, 18.06] | outer group 1-2 too high vs inner |
| w023-w029 consecutive (spacing-inconsistent) | 1 each | m7 between 0.01-0.13 | 0.98-1.18 | labels right (spacing irregular, not mislabelled) |

**Corrections** (data/scratch/label_audit/relabel_band1.json):
- `supported`: w041, w042 -> 41, and every k >= 43 -> k - 1. This is the CT-majority global map outside the #2167 sector.
  It is identical to the earlier "naive relabel".
- `outer_minus2`: `supported`, plus w051+ one lower again. This is the grad_mag end of the w033-w052 interval.
- `alt_w047_unchanged`: w043-w046 -1 and w047+ unchanged. CT disfavours it (it would need w046-w047 = 3 turns, seen on
  ~30 % of rays).
- `supported_plus_2167`: also k >= 36 -1. This is only true inside 145-260 deg, so it is a bound. Inside the sector no
  global map works; use excl_flag there.

**Seam-aware wrong area, arm19 - arm0** (B - A, pp, spiral_eval bootstrap; `eval/subset_wrong_area.py --split 41`).
- Validation: the original labels reproduce spiral_eval exactly: -4.69 [-7.16, -2.14].

| labels | all | consistent | inconsistent | inner (w023-w040) | outer (w043-w059) |
|---|---|---|---|---|---|
| original | -4.69 [-7.16, -2.14] | -3.08 [-5.87, -0.20] | -6.40 [-8.78, -3.79] | -25.74 [-30.05, -21.11] | +12.97 [+10.49, +15.71] |
| original, excl_flag | -6.06 [-9.22, -2.87] | -3.19 [-6.95, +0.60] | -9.22 [-12.15, -6.26] | -29.80 | +14.33 |
| **supported** | **+13.51 [+10.93, +16.01]** | +20.12 [+17.07, +22.93] | +6.20 [+3.80, +8.67] | -3.89 [-9.35, +2.00] | +28.09 [+23.00, +33.22] |
| **supported, excl_flag** | **+14.11 [+11.05, +17.16]** | +21.16 | +6.20 | -7.22 | +32.42 |
| outer_minus2 | +0.55 [-0.62, +1.87] | +4.68 [+3.53, +5.78] | -3.99 [-5.97, -1.93] | -20.73 | +18.41 |
| outer_minus2, excl_flag | -0.16 [-1.59, +1.34] | +3.88 | -4.67 | -26.13 | +22.14 |
| alt_w047_unchanged (disfavoured) | -9.10 [-11.98, -6.03] | -7.47 | -10.83 | -24.21 | +3.56 |
| supported_plus_2167 (bound) | +7.99 [+6.43, +9.64] | +7.73 | +8.38 | +3.27 | +11.78 |

Offsets refitted inside each group, so the group does not depend on how the other group is numbered (`--own-offsets`):

| group | original or supported | excl_flag |
|---|---|---|
| inner w023-w040 | **-19.30 [-21.35, -17.07]** | -23.43 [-25.25, -21.54] |
| inner consistent (w023, w029-w033, w037) | -15.59 [-17.74, -13.40] | -18.97 |
| outer w043-w059 | **+9.46 [+7.50, +11.40]** | +9.65 [+7.13, +12.19] |
| outer consistent | +9.81 [+7.76, +11.84] | +9.73 |

`supported` shifts the outer group as a block, so it does not change these. Under `alt_w047_unchanged`, outer becomes
-0.57. Under `supported_plus_2167`, inner becomes -0.57.

**Conclusions (band 1).**
- **The band-wide -4.7 pp does not survive a CT-supported relabel.**
  - Under the CT-majority map it is +13.5 [+10.9, +16.0] (+14.1 with the #2167 sector excluded).
  - Under the other CT-compatible bound (outer_minus2) it is +0.6 [-0.6, +1.9].
  - Only the CT-disfavoured alt map keeps it negative.
- The pooled number is decided by how the human labels number the outer group relative to the inner group, and those
  labels are wrong:
  - outer is 1-2 turns too high outside #2167;
  - outer is ~2-3 turns too high inside #2167.
  The per-bin modal offset ties the two groups together.
- **Robust, label-order-independent parts:**
  - arm19 is better on the inner windings: -19.3 pp [-21.4, -17.1] with own offsets.
  - arm19 is worse on the outer windings: +9.5 pp [+7.5, +11.4].
  - Both hold on the consistent subsets.
- Caveats:
  - The inner result assumes the original inner labels outside #2167, which CT supports (w033-w040 = 6.6-7.0 turns).
  - All three fit arms put two windings between w040 and w043, i.e. they follow the human count, not the CT. The
    fitter can stack two windings on one sheet, as arm0's j57/j58 do on w039, so this is not independent evidence.
  - The m7 rule that a run within 4 vox of a mesh belongs to that mesh can merge an adjacent sheet (fig_ct_gap_ray:
    w046-w047). The controls are counted the same way, and grad_mag does not use this rule.

## Pair metrics under CT label fixes (2026-10-21; human labels, CT audit and existing fits only; blind to arm20-25)
Tool: `eval/pairs_exclude.py` (flags / dump / spiral / bench; usage in its docstring). Outputs are in
data/scratch/label_audit/: flagged_pairs_{spiral_band1,spiral_band2,bench}.csv, spiral_ct_fix_armN.json,
bench_ct_fix.json and pairs_dump/armN.csv.

**CT-flagged set.** It is built from scoring_pairs.csv, sites.json, the umbilicus and the CT audit; no fit is read.
lo/hi are the two winding labels of a pair. A pair is flagged if any of these holds:
- it crosses the w041/w042 duplicate (lo <= 41 < hi). This includes every w040|w043 straddle, which the duplicate
  explains;
- it crosses the w045/w046 duplicate (lo <= 45 < hi);
- it crosses w046|w047 (lo <= 46 < hi);
- [lo, hi] meets w034-w036 and either end lies at 145-260 deg (#2167 sector, any z);
- it is one of the 172 skip pairs.

The flag CSVs list every pair in the set with one column per flag and with exp_<key> = |map(wa) - map(wb)| for each
relabel map. Band 1 and bench use relabel_band1.json; band 2 uses relabel_band2.json.

| set | mix pairs (sites) | dup/gap w040-w047 | #2167 sector | skip | flagged | relabel changes |
|---|---|---|---|---|---|---|
| spiral band 1 (z 3000-4500) | 7,548 (33) | 0 | 0 | 77 (already dropped) | 77 | supported 0, outer_minus2 77 (= the skip pairs), supported_plus_2167 340 |
| spiral band 2 (z 1750-3000) | 5,872 (21) | 0 | 545 (6 sites) | 0 | 545 | minus1 0, minus2 57 |
| benchmark (87 sites) | 24,972 (87) | 0 | 675 (7 sites; 545 at z 1750-3000, 130 at z > 6000) | 172 | 847 | supported 0, outer_minus2 172 (= the skip pairs), supported_plus_2167 1,105 |

Why so few pairs are flagged:
- No scoring pair has a winding in w040-w047. The pair windings are w024-w039 and w048-w058, and a pair spans
  at most 4 windings inside one 192^3 site. So none of the w040-w047 errors (the duplicates and the hidden sheet) can
  reach the pair truth.
- `supported` shifts all of w043+ as a block, so it changes no pair delta.
- All 33 band-1 sites lie at 270-60 deg, outside the #2167 sector. spiral_eval's own w035-gap flag also finds 0 band-1
  pairs.
- The sector pairs in band 2 and the benchmark never cross w034|w035|w036 in a way a sector relabel would change:
  w035 +-1 and w035 +-2 relabels change 0 of the 675.

Validation:
- `pairs_exclude.py dump` (the pair half of spiral_eval) gives jA/jB identical to spiral_eval --dump-pairs for arm0
  on all 7,548 pairs. Peak memory is 14-16.5 GB per arm, because the fit quads, not the vertex sample, take the memory.
- The `original, all` rows reproduce arm0_vs_armN.json exactly (estimate and CI) for acc_matched, acc_all and contact.
- The bench `original, all` row reproduces SUBMISSION_DRAFT section 1 exactly.

**Band 1 spiral pairs, B - arm0** (pp, pairs_mix, skip dropped, spiral_eval site bootstrap, 1000 resamples, seed 1):
- excl_ct removes no further pairs, so it equals `all`.
- `supported` changes no pair, so it equals `all`.
- `supported_plus_2167` (k >= 36 -1 everywhere) is only a stress bound. The CT says the w033-w040 labels are right
  outside the sector, and every band-1 pair lies outside it.

| arm | (i) all = (ii) excl_ct = (iii) supported: acc_matched | acc_all | contact acc_matched | supported_plus_2167 bound: acc_matched | acc_all | contact |
|---|---|---|---|---|---|---|
| arm2 | +14.10 [+8.03, +21.22] | +11.58 [+5.81, +16.82] | +9.88 [+2.02, +17.14] | +10.74 [+3.88, +18.51] | +9.42 [+4.17, +14.79] | +1.22 [-8.74, +10.40] |
| arm8 | +12.06 [+5.29, +19.19] | +9.88 [+2.96, +16.50] | +3.02 [-7.75, +15.01] | +12.12 [+5.09, +19.41] | +9.96 [+3.08, +16.48] | +2.77 [-8.49, +14.42] |
| arm13 | +14.57 [+8.74, +21.30] | +11.46 [+7.51, +15.83] | +9.64 [-1.75, +20.62] | +14.48 [+8.66, +21.22] | +11.44 [+7.46, +15.81] | +9.12 [-1.32, +19.21] |
| arm17 | +12.38 [+7.73, +17.18] | +3.94 [-0.20, +7.81] | +10.39 [+4.95, +15.16] | +8.20 [+2.73, +13.22] | +1.41 [-2.30, +4.98] | +4.61 [-4.30, +11.79] |
| arm18 | +4.06 [+0.35, +8.59] | -3.05 [-7.78, +1.60] | +4.60 [-1.47, +10.09] | +4.21 [+0.34, +8.80] | -3.09 [-8.12, +1.54] | +2.28 [-4.95, +8.58] |
| **arm19** | **+12.25 [+7.86, +17.17]** | +4.93 [+0.54, +9.27] | +12.05 [+2.08, +20.92] | +7.99 [+2.81, +13.16] | +2.32 [-1.38, +6.06] | +4.51 [-5.48, +15.00] |

The band-1 spiral set has n = 7,471 non-skip pairs, of which 733 are contact pairs.

**Benchmark (B): m7 1.18x vs native** (frozen graph b12 r40 k30 and direct count; p0139_paired site bootstrap, 87
sites, 1000 resamples, seed 1). GT = graph on the GT masks.
- outer_minus2 is identical to the original truth except on the 172 skip pairs, where it changes all of them.
- supported_plus_2167 applied everywhere is a bound. Outside the sector it lowers GT agreement (85.4 -> 83.8 on
  non-skip pairs).

| truth / subset | n (sites) | m7 1.18x graph [CI] | graph diff vs native [CI] | direct diff [CI] | contact | GT |
|---|---|---|---|---|---|---|
| (i) original, all | 24,972 (87) | 82.07 [78.9, 84.9] | **+2.08 [1.26, 2.93]** | +1.85 [1.10, 2.59] | 58.3 | 84.87 |
| original, non-skip | 24,800 (87) | 82.53 | +2.09 [1.26, 2.95] | +1.85 [1.10, 2.59] | 58.4 | 85.42 |
| **(ii) original, excl_ct** | 24,125 (87) | 83.03 [79.9, 85.8] | **+2.00 [1.14, 2.85]** | +1.74 [1.01, 2.47] | 58.9 | 85.53 |
| (iii) supported, all | 24,972 (87) | 82.07 | +2.08 [1.26, 2.93] (no pair changes) | +1.85 | 58.3 | 84.87 |
| outer_minus2, all | 24,972 (87) | 82.45 | +2.07 [1.25, 2.93] | +1.85 [1.12, 2.59] | 58.2 | 85.44 |
| supported_plus_2167, all (bound) | 24,972 (87) | 79.18 | +1.25 [0.43, 2.15] | +1.18 [0.53, 1.86] | 55.7 | 83.21 |
| touches w033-w037, non-skip | 10,310 (62) | 76.80 | +4.52 [2.83, 6.20] | +4.10 [2.58, 5.60] | 56.5 | 78.58 |
| **touches w033-w037, excl_ct** | 9,635 (62) | 77.66 | **+4.47 [2.71, 6.19]** | +3.98 [2.52, 5.50] | 57.3 | 78.38 |
| touches w033-w037, non-skip, supported_plus_2167 (bound) | 10,310 (62) | 69.80 | +2.51 [0.70, 4.22] | +2.47 [1.19, 3.80] | 51.9 | 74.57 |
| rest, non-skip (= rest excl_ct) | 14,490 (87) | 86.60 | +0.36 [-0.28, 1.02] | +0.26 [-0.26, 0.75] | 60.7 | 90.28 |

- The w033-w037 split CIs differ from notes/2026-10-20-p0139-label-audit.md by <= 0.1 pp: here all 87 sites are
  resampled with a mask, and the estimates are identical.
- **The skip pairs are a labelling artefact the CT explains.** All 172 cross w050, which is absent at their sites:
  170 are w049-w051/52/53 and 2 are w048-w051.
  - Under outer_minus2 (w049->w051 = 1 turn), GT agreement on them rises from 5.8 % to 87.8 %, m7x from 15.7 to 70.9
    and m7n from 15.1 to 71.5.
  - Dropping them or relabelling them leaves the m7 gain unchanged.

**Verdict.**
- **arm19 +12.3 pp pair gain: survives unchanged.**
  - The fixes do not move it: excl_ct removes no pair beyond the 77 skip pairs spiral_eval already drops, and
    `supported` changes no pair delta.
  - Reason: the band-1 pairs never touch w040-w047, and no band-1 site lies in the #2167 sector. The pair metric,
    unlike the seam-aware wrong area, does not depend on how the outer group is numbered relative to the inner one.
  - Even the stress bound supported_plus_2167 (k >= 36 -1 everywhere, contradicted by the CT outside the sector)
    leaves +8.0 [+2.8, +13.2].
  - The same holds for arms 2, 8, 13 and 17. arm18 stays at +4.1 [+0.4, +8.6].
- **m7 1.18x +2.08 pp: survives.**
  - Excluding the 847 CT-flagged pairs gives +2.00 [1.14, 2.85] (direct +1.74 [1.01, 2.47]), and `supported`
    changes nothing.
  - The w033-w037 concentration also survives: +4.47 [2.71, 6.19] on 9,635 non-flagged pairs, against +0.36
    elsewhere. Outside the sector the CT finds the w033-w040 labels right.
  - Only the CT-contradicted global k >= 36 shift shrinks it, to +1.25 [0.43, 2.15], still above 0.
  - Caveat: the CT audit covers z 1750-4500. Benchmark sites at z 4500-8300 are checked by labels only.
- **Band 2** (for the arm20-25 read-out, 7c):
  - The CT-flagged set is the 545 #2167-sector pairs; minus1 changes no pair and minus2 changes 57.
  - Apply after scoring:
    `uv run python eval/pairs_exclude.py spiral --dump-a <arm20 pairs csv> --dump-b <arm21 pairs csv> --flags data/scratch/label_audit/flagged_pairs_spiral_band2.csv --relabel-keys minus1 minus2 --out <json>`.
    The dumps come from `spiral_eval.py --dump-pairs X.csv`, which writes X.csv and X_b.csv, or from
    `pairs_exclude.py dump --z0 1750 --z1 3000`.
