# 2026-10-19: Dataset111_Contact0172, contact-aware surface labels from PHerc0172 traces. PRIVATE (CC BY-NC data)

Goal: fine-tune m7 on labels that put an explicit background separator between DISTINCT touching windings.
Binary masks fuse touching sheets, so contact-pair winding accuracy sits at ~56-58 % (GT masks ~50 %).

Script: `scripts/contact_set.py`. Output: `data/nnunet/nnUNet_raw/Dataset111_Contact0172/`
- `imagesTr`, `labelsTr` (uint8 tif, 192^3, 7.91 um native)
- `dataset.json`: CT channel; labels bg 0 / surface 1 / ignore 2; `.tif`, SimpleTiffIO
- `manifest.json`: per crop stem, origin, windings, contact/separator voxels, per-winding QC
- `qc.json`: per-winding aggregates and totals

Previews: `data/scratch/contact_set_preview_{0,1,2}.png`. Build log: `data/scratch/contact_set/build.log`.

## Data
- Meshes: 44 traced windings w052-w095 (`*-on-20241024131839-7.91um.tifxyz`; byte-identical to the ...838 copies).
  - 20-vox quad grid, one full wrap each. Cached under `data/meshes/PHerc0172/`.
  - The 9 auto_grown_* traces are skipped.
- CT: `20241024131839-7.910um-53keV-masked.zarr` L0, streamed per crop.
  - Same voxels at the same indices as `...131838` (NCC 0.97-1.0 at dz=0); different intensity window.
  - The segments' surface volumes were rendered from 131838.
- Orientation: all 44 meshes have consistent normals; outward = towards w+1.

## Main finding: the published meshes are only coarsely on the sheets
- Surface volumes (33 layers, ±16 vox) of w070 at L0:
  - The sheet wanders ±5-10 vox from the mesh layer over ~100-vox scales.
  - The sheet peak is within ±2 vox of the mesh in only 15-22 % of pixels.
  - Mesh-layer CT is ~equal to off-layer CT (c6 > 0 in ~50 % of pixels).
- In contact crops, mean CT at mesh points ≈ crop mean (90.2 vs 89.7, 76.6 vs 76.1).
  - No global shift fixes this (best ±6-vox shift gains < 3 grey levels). It is local trace error, not registration.
- Fine for ink detection (±16 layers); useless for a 3-vox band.
- Hence each winding is snapped to the CT per crop (below). Raw-mesh bands would be wrong on ~half the area.
- Also: consecutive windings are 15-19 vox apart (median) while visible CT sheets are ~8-10 vox apart.
  - Most slabs between traced windings hold an extra, untraced bright layer: delaminated fibre layer or skipped sheet.
  - Background is therefore scarce.
- Trace pathologies (global vertex search):
  - Consecutive meshes coincide (|gap| < 2 vox) on 1-20 % of vertices per pair and cross (gap < -2) on 0.1-8 %.
  - w075/w076 overlap massively (median gap -0.5): the same sheet traced twice over large areas.
  - A stray piece of w093 runs through the w060-w078 region.

## Recipe (per 192^3 crop)
1. Snap: build fine UV lattices of each winding (0.7 vox).
   - Move them along the outward normal onto the CT ridge: offset h, |h| ≤ 8, coarse-to-fine smoothing 8/4/2 vox.
   - h is bounded by half the gap to traced neighbours, so windings cannot swap.
   - Flag a lattice sample (handled like a mesh hole) when any of these hold:
     - weak ridge (peak − darkest point 2-6 vox away < 0.15·(p95−p5));
     - raw ridge > 1 vox from the smooth h;
     - h at its bound;
     - CT structure-tensor normal disagrees (|cos| < 0.7);
     - another winding snapped within 2.5 vox (same ridge).
   - Masks are cleaned: fill holes < 6², open r = 2, drop islands < 12² vox².
2. Per winding: signed point-to-plane distance sd (EDT nearest sample + its normal).
   - Band |sd| ≤ 1.5 → 1; centre |sd| ≤ 0.5 is always 1.
3. Separator 0: two nearest windings differ, |d1−d2| < 1 and d2 ≤ 2.5 (not on a centre voxel).
4. Background 0 only in the slab between consecutive windings k, k+1, where both nearest samples are unflagged.
5. Ignore 2: everything else, plus the following:
   - within 4 vox of mesh holes, edges or flagged samples;
   - bright untraced material in slabs (> 4.5 vox from traced surfaces, above the surface/background midpoint,
     components ≥ 200 vox, dilated 2);
   - crossings (dilated 3);
   - coincident surfaces (d2 ≤ 1, i.e. ≤ 2 vox apart; dilated 1);
   - windings with < 20 % confident samples in the crop, or snapped c6 ≤ 0 (whole neighbourhood |sd| ≤ 3).
6. Deviation from the brief: "hard contact" (< 2 vox apart, keep both centre lines) is NOT kept.
   - Distinct sheets here are 4-6 vox thick, so centre lines < 2 vox apart are trace conflicts, not contacts.
   - They are ignored and counted as `coincident_vox`. `hard_contact_vox` counts separators between surfaces
     < 3 vox apart.
7. Crop selection:
   - Contact: 64-vox cells of the global contact points (2 ≤ gap ≤ 6), scored as contacts − (crossings +
     coincidences) within the crop. Greedy, non-overlapping.
   - Open: random winding vertices with no contact/crossing within 110 vox.
   - Reject crops with > 10 % masked-zero CT.

## Numbers (build 2026-10-19, 5 workers, ~3.5 h on a loaded machine)
- Crops: 320 (240 contact + 80 open), 192^3. 352 jobs; 7 rejected for masked CT, extras dropped.
  - Origins z 964-13137.
- Separators:
  - Every contact crop has separator voxels: median 3286 per crop (p10 1564, p90 5151). Open crops: median 378.
  - Totals: separator 843,771; contact-midline voxels (surfaces ≤ 6 vox apart, not ignored) 2,001,852.
  - hard_contact (separator between surfaces < 3 vox apart): 1,004.
  - coincident (ignored): 2,453,412. crossing (ignored): 258,322.
- Label fractions, median per crop:
  - contact crops: bg 0.165, surface 0.084, ignore 0.752; 14 windings per crop.
  - open crops: bg 0.223, surface 0.068, ignore 0.721.
  - Untraced-material ignore: 142 M voxels, 6 % of all voxels.
- Label on material: mean CT under label 1 = 119.2 vs label 0 = 58.1 (median over crops).
  - Per-crop difference p5/p50/p95 = 49.5 / 61.4 / 74.7.
  - Surface is darker than background in no crop.
- Alignment QC, median over windings:

  | | c3 (I0 − I±3) | c6 (I0 − I±6) |
  |---|---|---|
  | raw mesh | 0.2 | 2.6 |
  | snapped | 39.4 | 56.9 |

  - c6_mesh > 0 for only 59 % of (crop, winding) pairs. Windings with median c6_mesh ≤ 0: w053, w055, w056, w057,
    w065, w082.
  - The snapped numbers are biased upwards by construction: the snap maximises the ridge. The visual check is the
    real test.
  - Confident (unflagged) sample fraction per winding: 0.53-0.77, median 0.70. Worst: w076 0.53, w075 0.62,
    w068/w082 0.64.
  - Mean |h|: 2.7 vox.
  - 140 of 4318 (crop, winding) instances dropped.
- Previews (each with a 64^2 zoom on the densest separator window):
  - 0: two consecutive pairs on one fused bright band, separator along the interface.
  - 1: dense region, sparse separators.
  - 2: a poor region where traces cross a void; 91 % ignore, as intended.

## Caveats
- Labels are ignore-heavy by design. Untraced layers make background confident only in the minority of slabs.
- Snapping can still lock onto an untraced neighbouring layer where the trace is ≥ 4 vox off.
  - The bounds, normal and same-ridge tests limit this, but do not remove it.
  - Spot-check more previews before trusting the separators at scale.
- The separator sits on the midline between snapped centre lines. At true contact there is often no visible dark
  gap at 7.91 um, so it marks the interface inside a fused bright band (the case we want to teach).
- Crops come from the traced z-range only (z ≈ 1.1k-13.7k).
- The repo's ordered snap (stages/snap_ordered.py) snaps to model predictions and could cross-check this.

## Checks before training (2026-10-19, main session)
- Six random crops overlaid (data/scratch/contact_spot.png) plus one zoom (contact_zoom.png): surface labels sit on
  bright lines; background mostly in dark gaps; untraced sheets mostly ignore.
- Background purity: share of background voxels brighter than the surface-label 25th percentile (CT smoothed s=1):
  Dataset111 median 0.12 (p90 0.17) vs Dataset101 DS059 0.48 and Dataset107 m7 pseudo-labels 0.67. Cleaner than the
  sets we already train on.
- Dataset112 = 320 crops x2 + 700 Dataset107 = 1340 cases. Fine-tune at lr 1e-3 (nnU-Net's default 1e-2 drove m7's
  pseudo-dice from 0.31 to 0.07 in one epoch; 1e-3 reached 0.50 in 60 iterations).

## Control result: m7ctrl150 (m7 fine-tuned 150 ep on the 700 Dataset107 crops only, lr 1e-3, H200, ~$16)
| model | PHerc0139 graph | paired vs m7 1.18x | contact | direct | hf2um official | sep | VOI_merge |
|---|---|---|---|---|---|---|---|
| m7 public 1.18x | 82.07 | ref | 58.3 | 79.45 | 0.494 | 49.8 | 1.040 |
| m7ctrl150 1.18x (hf2um native) | 81.14 | -0.93 [-2.27, 0.25] | 54.6 | 79.42 | 0.488 | 58.7 | 0.872 |
- Re-tuning m7 on its own bucket pseudo-labels moves it to the m7d300 profile (fewer merges, lower contact). The
  contact arm (m7ct150) must beat this control at contacts for the contact labels to count.

## Contact arm result: m7ct150 (Dataset112: 320 contact crops x2 + 700 Dataset107, 150 ep, lr 1e-3, A100) — NULL
| model | PHerc0139 graph | contact | contact undercount -1 | compressed | direct | hf2um official | sep | VOI_merge |
|---|---|---|---|---|---|---|---|---|
| m7 public 1.18x | 82.07 | 58.3 | 30.5 | 75.7 | 79.45 | 0.494 (native 0.495) | 49.8 | 1.040 |
| m7ctrl150 | 81.14 | 54.6 | 33.5 | 73.2 | 79.42 | 0.488 | 58.7 | 0.872 |
| m7ct150 | 81.08 | 55.3 | 33.9 | 72.7 | 79.46 | 0.492 | 58.5 | 0.901 |
- m7ct150 vs control, paired: -0.06 [-0.49, +0.36]; contact +0.7. vs m7: -0.99 [-2.43, +0.24].
- In-distribution check (12 Dataset111 training crops, native 7.91 grid): predicted fg on near-surface background
  m7 22.5 %, control 30.7 %, contact arm 10.4 % (it learned thinner surfaces); on separator-like background (bg within
  2.5 vox of two different surface components) 17.5 / 22.9 / 22.1 %. It did not learn the separators even on its own
  training crops. Separators are ~0.1 % of labelled voxels (75 % of each crop is ignore); plain CE+Dice gives them
  no weight.
- Pod cost this round ~$35 (broken 4xH100 ~$7, H200 ~$19, A100 ~$8.5).
- Next if pursued: a separator class with a heavily weighted loss (or a boundary head), or exact-contact sim data.

## Separator round (2026-10-20)
- Labels: scripts/make_ds114.py marks label 3 = background voxels in a 1-vox gap between surface voxels on opposite
  sides (13 directions); median 6.4k per crop (the builder logged 3.3k of its own; gap 2 gave 17.5k and hugged band
  edges). Dataset114 (label 3 trained as bg + weighted CE, train/run.py --sep-weight) and Dataset116 (3-class
  bg/surface/separator/ignore; seg head is re-initialised anyway since load_pretrained_weights skips seg_layers).
- Smoke (20 cases, batch 1): the CE term collapses surfaces early (pseudo-dice 0.68 -> 0.19 at w 0.3, 0.02 at w 1
  after 60 it, gap-2 labels); with gap-1 labels over 300 it: w 0 0.70, 0.05 0.58, 0.1 0.65, 0.3 0.51.
- eval/sep_check.py baseline on 12 PHerc0172 training crops (surface on separators / on near-surface bg / recall):
  m7 26.5 / 23.3 / 28.5; m7ctrl150 37.1 / 32.0 / 42.9; m7ct150 78.8 / 10.4 / 90.6. The contact arm fuses the 1-vox
  separators 79 % of the time on its own training crops.
- Runs: m7sep01 (A100, w 0.1), m7sep03 (A40, w 0.3), m7c3 (GH200 aarch64, 3-class). Control = m7ct150.
- eval/p0139_crossings.py now masks seg == 1 (was > 0; identical for 2-class models); infer_nnunet.py --argmax.

## Separator round results (2026-10-20)
150 epochs each from m7, lr 1e-3. sep01/sep03: Dataset114 (gap-1 separator labels as bg + w * CE at separators);
c3: Dataset116 3-class head. ~$28 compute plus ~$5-8 for an orphaned RTX6000Ada pod (vss-sep03, idle 5 h).

In distribution (eval/sep_check.py, 12 PHerc0172 contact crops, native grid):

| model | separators called surface | near-surface bg called surface | surface recall |
|---|---|---|---|
| m7 | 26.5 | 23.3 | 28.5 |
| m7ct150 | 78.8 | 10.4 | 90.6 |
| m7ctrl150 | 37.1 | 32.0 | 42.9 |
| m7sep01 | 9.0 | 5.8 | 78.9 |
| m7sep03 | 3.2 | 3.7 | 69.8 |
| m7c3 | 61.1 | 10.8 | 90.5 |

Held out (PHerc0139 at s1.1836, graph b12 r40 k30, paired vs m7pub 1.18x; hf2um native, --argmax):

| model | graph | paired | contact | direct | hf2um official | compressed sep | VOI_merge |
|---|---|---|---|---|---|---|---|
| m7pub | 82.07 | ref | 58.3 | 79.45 | 0.495 | 54.3 | 1.059 |
| m7ctrl150 | 81.14 | -0.93 | 54.6 | 79.42 | 0.488 | 58.7 | 0.872 |
| m7sep01 | 81.02 | -1.05 [-2.26, 0.13] | 55.4 | 79.28 | 0.494 | 59.4 | 0.912 |
| m7sep03 | 81.44 | -0.62 [-1.76, 0.48] | 55.9 | 79.63 | 0.491 | 60.3 [+0.1, +12.6] | 0.903 |
| m7c3 | 81.57 | -0.50 [-1.53, 0.55] | 55.9 | 79.89 | 0.491 | 60.0 | 0.902 |

- The separator loss works in distribution (separators called surface 79 % -> 3-9 %, specific to the gaps), the
  3-class head mostly doesn't (61 %).
- None of it transfers to PHerc0139 winding: all arms sit 0.5-1 pp below m7 1.18x and lower at contacts.
- On hf2um, fewer merges and better compressed separation, but the no-contact control gets most of that, so it's
  the fine-tune, not the separators.
- Verdict: null for October. Separators learned on PHerc0172 mesh contacts don't generalise to other scrolls at 9 um.
  Open: whether separated masks help the spiral fit's track extraction (a mask arm), and more varied contact labels.
