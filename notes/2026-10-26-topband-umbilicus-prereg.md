# 2026-10-26 PHerc1667 top band (z 8000-9000): umbilicus repair, pre-registration

Status: pre-registered before any fit with the repaired umbilicus. Protocol as in test 8 and the whole-scroll note
(notes/2026-10-26-p1667-wholescroll-prereg.md): choices are frozen here with UTC timestamps before the outputs they
govern exist; later changes go into dated entries below the Log line.

**What kind of test this is.** This is a **repair motivated by a seen result, on the band that showed it**. In the
whole-scroll seed-1 read-out every fresh band's M2 arm beat base by 43-60 pp inner wrong area except z 8000-9000:
base 78.50 % -> M2 75.10 % (-3.39 [-4.58, -2.49]), switches +3.8 (data/scratch/wholescroll/eval/pool_s1_fresh.json).
The whole-scroll prereg (section 1a) had flagged the cause tested here **before any fit**: the umbilicus has control
points only for z 672-8800, and the fitter extrapolates it. z 8000-9000 is **no longer fresh**. The claim on offer is
"the pre-flagged cause, once fixed, removes the failure", not a fresh confirmation of the method.

## 1. Diagnosis (label-free; done 2026-10-26T21:00-21:45Z; scripts and figures in data/scratch/topband/)

### 1a. Where the umbilicus comes from and how each component reads it
- `data/eval/p1667/umbilicus_p1667_L2.json` (sha e6a862b8...) comes from eval/p1667_labels.py: the centre of the
  innermost labelled wrap (k = 11) of the merged human mesh, Kasa circle fit per 64 z, median of 5. It has 128 control
  points at z 672-8800 (L2, 9.596 um). prep_inputs.py copies it into each band's fit inputs as `umbilicus.json`.
- **Chains** (tools/autowind/core.py `umbilicus_fn`) and the **scorer** (eval/spiral_eval.py `Umbilicus.angle`, used
  for the 10 deg offset bins) use `np.interp`, which **clamps** above z 8800 at the last point (1991.17, 1950.42).
- **Fitter** (villa 0fb38c45 `umbilicus.json_umbilicus_z_to_yx`): `scipy.interp1d(fill_value='extrapolate')` evaluated
  at z_begin..z_end-1 (8000-8999), then `gaussian_filter1d(sigma=75, mode='nearest')` (transforms.UmbilicusTransform).
  The umbilicus loss (`loss_weight_umbilicus` 1.25) pulls the spiral origin onto this curve. Above 8800 the last
  segment (8736 -> 8800: dx -57.4, dy +22.6 per 64 z) is extended linearly: raw centre at z 9000 = (1812, 2021),
  **193 vox** from the clamped one. Smoothed (what the fit uses), the fitter's centre departs from the clamped one by
  3 vox at z 8700, 29 at 8800, 58 at 8850, 96 at 8900, 134 at 8950 and 163 at 8999.

### 1b. Is the extrapolated centre plausible? (m7 mask only; core_check.py, look_top.png, zoom_top.png)
- Above z ~8400 the cross-section is a crescent: the inner turns on one side are missing (scroll top damaged).
- A scroll centre must lie in the sheet-free core. Distance from each candidate to the nearest m7 sheet (L1 slices)
  and number of sheets crossed going from it to the clamped centre:

| z | extrapolated: d_sheet / sheets crossed / offset | clamped (held) point: d_sheet |
|---|---|---|
| 8848 | 54 / 0 / 46 | 38 |
| 8896 | 44 / 0 / 92 | 8 |
| 8944 | 2 / **6** / 139 | 17 |
| 8992 | 0 / **4** / 185 | 30 |
| 9040 | 6 / **4** / 231 | 30 |
| 9088 | 0 / **2** / 277 | 35 |
| 9152 | 31 / 0 / 339 | 271 |

  From z ~8940 the fitter's centre lies **on the sheets, 2-6 turns outward of the core edge**, so it cannot be the
  scroll centre; the held point stays in sheet-free space at every z 8800-9216.
- A direct label-free centre estimate (least-squares intersection of mask-gradient normal lines, centre_est.py) is
  **not precise enough** to define a trend: inside z 672-8800 it is 39-155 vox (median per z range, either radius setting) from
  the human umbilicus, and above 8800 its two radius settings disagree in sign (inner, r 200 L1 px: x keeps falling by
  ~60 vox to z 8880 and then fails, isotropy < 0.35; wide, r 450: x rises by ~70 vox, then is flat; both break down
  above z ~9024 as the crescent thins). The held point lies between the two.
- **Bottom (z 344-672), checked the same way:** the extrapolated centre there lies in sheet-free space (d_sheet
  155-640 vox, no sheets crossed to the clamped centre), and the label-free estimate is unusable (isotropy 0.2-0.5,
  jumps of 300-1300 vox). It passes the core check, and this test does not touch z < 8000, so it is **left unchanged**.

### 1c. Effect visible in the existing top-band fits (label-free; step_check.py, fit_top.png)
Rays (5 deg) from the clamped umbilicus through the m7 mask. Each sheet crossing takes the winding index of the
nearest fit point (<= 4 vox). Clean step = consecutive crossings differ by exactly 1. Innermost 15 crossings per ray:

| arm | z 8000-8800 | z 8850-8950 | reference: z6000 band (good) |
|---|---|---|---|
| base s1 / s2 | 0.44-0.69 | 0.29-0.54 | base 0.81 |
| M2 s1 / s2 | 0.73-0.87 | 0.63-0.72 | M2 0.91 |

Both arms lose clean steps at z >= 8850, where the fitter's centre moves into the sheets. The fitted spiral's origin
follows the extrapolated point (fit_top.png, z 8990). **Caveat, written before any fit:** over z 8000-8800, where the
umbilicus is the human one, M2's inner step consistency is near the good-band level and far above base. The
band-wide label shortfall (M2 75 % wrong vs 20-33 % elsewhere) is therefore mostly **not** visible to this check.
The umbilicus is a plausible cause of damage in the top ~150-250 slices, but probably not of the whole band-wide failure.

## 2. The repair (frozen)
- **New umbilicus** `data/scratch/topband/umbilicus_p1667_L2_top.json`, sha256
  **e0936d53db8f0ff1270de272bc3e96903f45dcd229519973fbd104246c152451** (make_umbilicus.py 4abd603b...): the 128
  original points (z 672-8800) **byte-identical** (checked: the serialised point block of the original file is
  reproduced exactly), plus 7 points at z 8864, 8928, ..., 9248 equal to the z 8800 point (x 1991.17, y 1950.42).
  The umbilicus is **held constant above z 8800**. Why the hold and not an estimated trend: (i) it passes the core check
  at every z (1b), and the extrapolation fails it; (ii) the label-free estimators do not agree on a trend; (iii) it is
  exactly what the chains and the scorer already use, so the fitter now agrees with every other component.
- Consequences, verified numerically: `np.interp` (chains, scorer) is **identical at every z 0-9269**; the fitter's
  interp1d is identical on z 672-8800; the fitter's smoothed centre is unchanged below z 8600 and moves by the
  amounts in 1a above it.
- **Fit inputs:** `data/p1667_top/z8000/inputs/` = `cp -a data/p1667_ws/z8000/inputs` with only `umbilicus.json`
  replaced (sha 236651f8...; the new file plus the metadata keys prep_inputs.py adds). Masks: data/p1667_ws/z8000/masks
  (unchanged).
- **Chains: the frozen whole-scroll chain set is reused unchanged** (p1667z8000_m7_ray_c, sha 82dbf46f...; copied to
  data/scratch/topband/windings/). The chain command with the new umbilicus would give an identical centre at every z
  (np.interp identical), so rerunning it could only change file metadata. The M2 arm therefore differs from
  wsf8000_m2s* **only** in the fitter's umbilicus.json.

## 3. Pre-registration
### 3a. Arms (recipe identical to whole-scroll wsf8000_* except the umbilicus)
`spiral/arms/top8000_{base,bases2,m2s1,m2s2}.env` = wsf8000_*.env with only ARM changed (diff checked).
Pinned villa 0fb38c45 via run_arm.sh (0310894f...) + export_fine.sh 4 (779d504b...), recipe.toml (9ba8494e...), 30k
steps. M2 FIT_OVERRIDES `{"pcl_unattached_pcl_min_point_spacing": 0.0, "loss_weight_unattached_pcl_radius": 10.0}`;
seed 2 adds `"optimizer_random_seed": 2`. Four fits: base-fixed s1, M2-fixed s1, base-fixed s2, M2-fixed s2.

### 3b. Scoring (frozen)
Vertex dumps `release/fit_vertex_dump.py --scroll p1667 --z0 8000 --z1 9000 --tol 4` (collect_top.sh; reference sample
ref_p1667_z8000-9000_s2_m0_v2.npz, sha 703a50a0..., the one the whole-scroll run used). Scorer: the frozen whole-scroll
`eval/wholescroll_pool.py` (dfdb1f57...) `band_parts()`, called through `data/scratch/topband/score_top.py`
(db5d1431...), which only swaps the arm-name lookup (validated before this freeze: with the original pair it gives
-3.393 [-4.580, -2.485], switches +3.804, identical to pool_s1_fresh.json). Inner (k < 34) own-offset seam wrong area,
paired 30 deg x 250 z block bootstrap, 1000 resamples, 95 % percentile CI. Note: the scorer's angle bins use the
original umbilicus file, unchanged (its np.interp equals the new file's everywhere).

### 3c. Primary claim and pass rule
Pairs P1 = top8000_base -> top8000_m2s1 (seed 1), P2 = top8000_bases2 -> top8000_m2s2 (seed 2).
Gain threshold: half the median seed-1 gain of the four other fresh bands (z400 -49.73, z1500 -43.15, z3000 -59.62,
z6000 -52.81; median -51.27): **-25.6 pp**.
**PASS ("the pre-flagged cause, once fixed, removes the failure")** if, for **both** P1 and P2:
1. inner diff (M2-fixed - base-fixed) 95 % CI upper < 0;
2. inner diff point estimate <= -25.6 pp;
3. seam switches per 1000, M2 - base, point estimate <= +5.

**PARTIAL** if 1 and 3 hold for both seeds, both diffs are more negative than the original pairs' (-3.39 s1; s2 from
pool_s2 when that is computed, else wsf8000_bases2 -> wsf8000_m2s2 via score_top O2), but 2 fails: the repair helps but
does not remove the failure. Otherwise **FAIL**. A missing arm after 2 resumes leaves the verdict "incomplete".

### 3d. Secondary read-outs (reported, not part of the claim)
1. **Umbilicus effect on its own:** U1 = wsf8000_base -> top8000_base, U2 = bases2 pair (base-fixed vs base-original);
   M1/M2 = wsf8000_m2s1/m2s2 -> top8000_m2s1/m2s2.
2. **Where the change lands:** inner wrong area per z slab 8000-8750 and 8750-9000 (slabs aligned with the 250-z offset
   bins) for every pair (score_top.py slabs). The original arms' slab split is computed **only after** this freeze.
3. Label-free inner step consistency per z (step_check.py, NIN=15) for the four new fits next to the originals.
4. Areas (labelled, correctly numbered, fit cm^2) as in wholescroll_pool.

### 3e. Predictions (honest, written before any fit)
- U1/U2 (base): small, -3 to +3 pp band-wide; slab 8750-9000 maybe a few pp better.
- M1/M2 (M2-fixed vs M2-original): slab 8750-9000 better by 5-25 pp; slab 8000-8750 within +-3 pp; band-wide -1 to -6 pp.
- P1/P2: -5 to -12 pp. **P(PASS) ~ 0.1, P(PARTIAL) ~ 0.5.** Reason (1c): the fitter's centre changes only above z
  ~8650, and the M2 fit is already locally consistent over z 8000-8800, so most of the band-wide shortfall probably has
  another cause (candidates, not tested here: the crescent geometry with the inner turns missing on one side, debris
  near the core, a different inner-winding mix because k 28-34 drop out above z 7750).
- A FAIL here means the pre-flagged cause is real at the very top but is not why the band fails.

### 3f. Running rules
- Pods `vss-top-<arm>`: single GPU >= 48 GB VRAM, x86_64, driver >= 580 (boot3 host check), <= $2.50/h, non-spot, no
  GH200/GB200, no lambdalabs; no premium cards above $2.50/h. At most 4 live pods. **Budget cap $25**, gate: no new pod
  if spend + (live burn + new price) x 5 h > $25. s1 pair first.
- Priority: while the whole-scroll grabber still has arms to place, grab_top.py leaves it the cheapest N offers (N = its
  open arms), so the primary whole-scroll verdict is not delayed by this repair.
- Scripts (copies, own dir data/scratch/topband/pods/; the whole-scroll scripts and pods are never touched):
  grab_top.py (c65813a1...), boot3.sh (e485ae7d...), boot_and_launch.sh (0571ef2e...), collect_top.sh (b244c482...),
  watchdog_top.sh (016fd4f2...), prime.py (6599588a...). collect_top.sh terminates each pod once its arm is pulled; a
  final pod list confirms that none is left. STOP file in data/scratch/topband/pods/ stops new pods.
- No reruns except resuming a failed pod with the same env (same arm name, new pod allowed, at most 3 pods per arm).
- All shas: data/scratch/topband/prereg_shas.txt.

## Log (append-only)

### 2026-10-26T21:45:56Z: sections 1-3 frozen
- sha256 of this note up to and including the "## Log (append-only)" line: 63881b516686ee675bcdae1bb173ff02b0f667d30eb01e8fe8ad4a89997b6a5d.
- data/scratch/topband/prereg_shas.txt sha256 a8ae89e58b32d6d2afa163ed0939dd19504c0105ca8b442fb9c0d560517b0c17.
- At this time: no fit with the repaired umbilicus exists; no vss-top-* pod has been created. Label metrics computed so
  far on z 8000-9000: only the already-published original pair (score_top.py O1 validation, --no-slabs).

### 2026-10-26T21:59:30Z: post-freeze observation (secondary 3d.2 on the original arms) and operational additions; no recipe or rule change
- Original pairs, inner wrong area by slab (score_top.py O1/O2, computed after the freeze as 3d.2 allows):
  s1 base -> M2: z 8000-8750 78.19 -> 75.43 (-2.76), z 8750-9000 79.53 -> 73.95 (-5.58); band -3.39 [-4.58, -2.49].
  s2 base -> M2: z 8000-8750 78.27 -> 72.06 (-6.21), z 8750-9000 79.27 -> 72.01 (-7.26); band -6.45 [-8.06, -5.14],
  switches -6.5.
  **The failure is band-wide, not concentrated at the top.** The repair leaves the fitter's centre unchanged below
  z ~8650, so it can reach the 8000-8750 slab only through the fit's coupling along z. This makes PASS less likely than
  the 0.1 written in 3e. The prediction is left as written; nothing in sections 2-3 changes. The fits still run as
  pre-registered (they measure the umbilicus effect at the top: U1/U2, M1/M2 slab 8750-9000).
- O2 (the original s2 pair, needed for PARTIAL) is now known: -6.45.
- Operational (new files; frozen ones untouched): pods/guard_top.py (sha256 88a659cb42eb1c8e012e967535e25a5e73f7589212d718bfc07114dbbd2ed2d1)
  deletes a vss-top-* pod whose arm FAILED, whose boot ended BOOT_RC != 0, that is older than 9 h, or when spend
  passes $25. finish_top.py (efa40ef5da599fa39bc278191fccaeed256b0da2aafb74bb69b16e74c8c94bcb) waits for the four
  dumps, runs score_top.py and step_check.py, applies rule 3c mechanically and appends section 4 to this note.
- 2026-10-26T21:59:33Z: coordinator stopped the fits (pods/STOP) before any pod existed: post-freeze split shows the failure is band-wide (P(PASS) ~0.1) and the band's value vs Kelly is small. No fits run, $0 spent. The diagnosis (fitter's linear umbilicus extension lands on sheets above z~8940) stands as a label-free finding.
