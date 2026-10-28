# 2026-10-16: simulator v1 (sim/, forward/)

Implements sim/DESIGN.md. Geometry is an implicit numpy/torch model, not Blender Geometry Nodes, because
exact identity at any resolution needs labels evaluated per point. Blender is kept as a viewer and as a possible future
crumple stage.

## Geometry (sim/scroll.py)
- Tables over (wrap k, theta, z): inner-face radius R, thickness t, gap g, spiral arc length U.
- Wrap placement: target radius T_k = Archimedean nominal + A * D(x), then R_{k+1} = max(R_k + t_k, T_{k+1}).
  - D is a smooth 4D-RFF displacement field: radial correlation 500 um, tangential 2.5 mm.
  - Where D compresses wraps radially, sheets touch with gap exactly 0, and the contact pushes outer wraps outward.
  - A is bisected so that the contact area fraction equals `contact_frac` (0.15).
- Rejected designs, recorded so they are not retried:
  1. Gap noise along arc length, accumulated wrap by wrap. R random-walks outward: the k=60 radius spans 7.6-16 mm
     around one wrap, with 100-400 um jumps between theta samples.
  2. A per-wrap contact ramp. Too sharp, giving a sawtooth.
- Smoothness check (seed 1): max |dR| between theta samples is 2/4/25 um at wraps 5/20/60, for arc steps of 10/20/46 um.
- World map: squash (1, 0.62), twist, then a smooth warp (Lipschitz < 0.5).
  - Material-space depths are converted to world normal distance with |grad r| (`Scroll.metric_scale`).
  - So material `t_mean` = 85 um means 53-85 um of world thickness, depending on direction.
- Instance label: the winding number relative to a per-patch reference, with the cut on the opposite side of the scroll.
  It is continuous along each sheet across the theta = 0 seam.
- Fiber texture:
  - Recto strands run along u (arc length), verso strands along z. Coordinates are relative to the patch reference,
    and U is stored in float64.
  - TF32 is disabled (RFF phases reach about 1e4 rad).
  - In fused stacks, the only boundary cue is the switch between recto streaks and verso dots.
- Speed: tables 7-12 s per scroll; density sampling about 10M points/s on the RTX 3090 (a 320^3 block at 2.4 um takes 3 s).
  A full cell (256^3 at 7.91 um plus 256^3 at 2.4 um, labels at both) takes ~30 s and ~40 MB of zarr.

## Texture revision after the eye test (2026-10-17, data/scratch/real_vs_sim{,_v2}.png)
Real vs sim at 7.91 um and 2.4 um, side by side. v1 problems:
1. Isotropic 50 um density blobs (porosity field). Clearly visible at 7.91 um; absent in real data.
2. Strands far too thin. A units bug: RFF scales are Gaussian correlation lengths, but `fiber_across_um=11` was divided
   by 2pi, giving l = 1.75 um. At 2.4 um this showed as salt-and-pepper speckle, not 7-15 um strands.
3. Sheets too clean and regular. Real sheets split into sub-layers (thin dark cracks parallel to the sheet, visible at
   7.91 um), have frayed edges, irregular thickness and sparse bright inclusions.

Changes (all texture length scales are now correlation lengths, with no 2pi):
- Strands l = 5 um across, 80 um along; void density between strands 0.25 of strand density.
- Porosity is laminar, in sheet coordinates (u, z, depth): 150 um in-plane, 20 um across the sheet, sd 0.15.
- Delamination cracks: a thresholded field (150 um in-plane, 4 um across, threshold 1.5 sd) cuts density by 90%.
  These are intra-sheet gaps that are not sheet boundaries: the distractor a merge-aware model has to learn to ignore.
- Inclusions: at most one soft sphere (r 4-12 um, +1.2 density) per 200 um world cell, p = 0.06 per cell, from an integer hash.
- Edge roughness 4 um, correlation 8 um (still clamped to +-gap/2, so contacts stay exact).
- Thickness field: correlation 600 um (was 1500), relative sd 0.25 (was 0.18).

v2 at 7.91 um: fibrous streaks and sub-layer cracks, no blobs, visually close to real. Remaining differences: real sheets
fray and branch into the gaps more, and real thickness is more irregular. At 2.4 um the strands are still more
plate-like than real fibrils. This is accepted for now, because resolution is an in-sim question.

### v3: face loosening (2026-10-17, after the first sim-set preview)
The v2 preview showed dark sheet interiors with only the faces bright: at 7.91 um, sheets read as 2-3 vox lines inside
8 vox labels. Real 2.4 um edge profiles do the opposite: grey rises from ~140 at +10 um to ~180 at +38 um inside the
material. The v2 fit only matched the edge profile to +18 um and measured material above Otsu, so it never saw this.
Changes:
- Density deficit near faces: `face_loose` 0.35, decaying inward over 15 um.
- At touching faces the same deficit is scaled by `contact_loose` (0 = fused).
- Fewer voids: void density 0.35, crack threshold 1.8 sd.
- The fit now matches edge profiles to +40 um at both resolutions.

## Domain randomization (`sim/generate.py --randomize`)
Per scroll seed, ScrollParams are drawn from `RANDOM_RANGES`:
- interface: interface_dip 0-0.7, contact_loose 0-1;
- geometry: contact_frac 0.08-0.30, t_mean 65-105, g_mean 40-100, displacement correlations, squash;
- texture: face_loose 0.2-0.5, crack threshold 1.5-2.4, void density 0.25-0.5, porosity 0.08-0.2, strand width,
  roughness, inclusion rate.

Contact-interface contrast is swept rather than fixed, because no real label resolves identity at contacts.
contact_loose gives a ~15 um dip, which survives at 7.91 um. Half the scrolls have a weak cue (< 0.5), so a model
cannot rely on it alone. Sim test sets should report results binned by contact_loose.

## Real-data targets (forward/measure.py -> forward/calib_real.json "stats_v2")
Measured on 8 PHerc1667 patches: the central 320^3 of the 2.4 um crops and the central 160^3 of the 7.91 um jeong200 patches.
- 2.4 um:
  - Air 46.5, material 141 (sd 28).
  - Air residual noise sd 3.6; autocorrelation at lags 1-4 = 0.79 / 0.48 / 0.29 / 0.18 (heavier tail than one Gaussian).
  - Dark phase-contrast fringe on the air side of each boundary: about 7 grey levels below background, 5-20 um out.
- 7.91 um:
  - Air 78, material 131 (sd 19).
  - Air residual sd 10.9 with a long autocorrelation tail (0.55 at lag 4). Part of this is structure in the "air", not noise.
- 7.91 vs blurred 2.4 (sigma 2.5 vox), least squares over 8 registered pairs: a = 0.719, b = 41.9, residual sd 11.0.

## GT surface band position (jeong200, measured)
Along the band normal, oriented so material is +:
- The band covers t = -2 .. +2 vox around its centre.
- The CT half-max edge sits at t ~ -2.5.

So the Dataset059 label is the first ~4.5 vox (~35 um) of the sheet from its recto face, not a band centred on the face.
Sim: `surface` = in-sheet & world depth < 34 um, plus 2 um in front of the face (sim/render.py).

## CT forward model (forward/ct.py, fitted by forward/fit.py -> forward/ct_params.json)
- I2 = air + gain * G_psf * rho - alpha * Laplacian(G_fringe * rho) + n2.
- I7 = J - alpha7 * Laplacian(G_f7 * J) + n7, with J = a * resample(G_2.5vox * I2_clean) + b.
- Noise: a mixture of two Gaussian-correlated fields.
- Both noise models are fitted on pure synthetic noise against the real air-residual std and autocorrelation.
  Imaging parameters are fitted on sim renders of the v2 texture: edge profile plus air and material statistics.

Fit results (2026-10-17, `forward/ct_params.json`, log in data/scratch/ct_fit.log):
- 2.4 um: air 38.7, gain 130, PSF sigma 2.3 um, fringe alpha 332 with sigma 5.1 um, noise sd 3.2 (corr 2.2 um).
- 7.91 um: a 0.55 (at its lower bound; registration regression gives 0.72), b 57.5, halo alpha 256 with sigma 13 um,
  noise sd 12.1 (s1 4.7 um, s2 27 um, w2 0.75).

| stat | 2.4 sim / real | 7.91 sim / real |
|---|---|---|
| Otsu | 88.0 / 86.2 | 103.8 / 101.3 |
| material mean | 141.8 / 141.1 | 133.2 / 130.9 |
| material sd | 28.8 / 28.4 | 17.9 / 19.0 |
| material p95 | 188.5 / 192.5 | 166.5 / 164.0 |
| air mean | 38.7 / 46.5 | 75.2 / 78.3 |
| air residual sd | 3.8 / 3.6 | 17.7 / 10.9 |

**Update (v3 texture, warm refit, then noise retarget):**
- 2.4 um: alpha 517, sigma 6.7 um, gain 146. Material mean, sd and p95 are 142.1 / 29.5 / 193.0 sim vs
  141.1 / 28.4 / 192.5 real.
- 7.91 um: a 0.55 (at bound), b 53.2, halo alpha 234 with sigma 11.4 um. Material mean 136.6 sim vs 130.9 real.
- The v3 eye test still showed blobby 7.91 air. Measuring real 7.91 air as a function of distance from material
  (16 Scroll 4 patches) gives:

  | air >= | noise sd | acf lags 1-4 |
  |---|---|---|
  | 30 um | 10.8 | 0.81 / 0.65 / 0.53 / 0.45 |
  | 60 um | 5.8 | 0.73 / 0.45 / 0.27 / 0.19 |
  | 90 um | 5.0 | 0.69 / 0.38 / 0.17 / 0.08 |

  The 30 um target was halo-dominated in real data as well. The 7.91 noise is now fitted to the >= 90 um statistics:
  sd 5.0, s1 4.7 um, s2 11.3 um, w2 0.59 (`calib_real.json` "noise7_far").

The 7.91 air mismatch in the table above is therefore measurement geometry, not noise. Noiseless sim scans already have an air residual sd of
11-22 at >= 30 um from material (5-10 at >= 60 um): halos overlap across the narrow sim gaps, while real air statistics
come mostly from wide gaps. The pure-noise model reproduces the real air autocorrelation (0.81 / 0.64 / 0.55 / 0.45).
Accepted. Training also uses nnU-Net intensity augmentation, and the README realism check is the formal gate.
The 2.4 air mean is 8 grey levels low; not chased.
