# 2026-10-16: contact check (does 2.4 um CT separate sheets that are fused at 7.91 um?)

Design written before running. Results appended below, whatever they are.

## Question
At sites where GT says two sheets sit < 4 vox apart at 7.91 um (Jeong's "compressed" bin, where models fuse
sheets), does the 2.4 um scan of the same physical site show a dark valley between the two sheet surfaces?

## Data
- 36 Scroll 4 (PHerc1667) patches of the Jeong 200-patch pool, 7.91 um (DLS `20231117161658`), GT = recto surface bands ~4-5 vox thick.
- 2.4 um scan: `s3://vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr`.

## Registration (7.91 -> 2.4 um)
- All 20 PHerc1667 segments carry tifxyz meshes on both scans; same UV parameterization, grid step 20 vox in each scan's units.
  Per segment, fit the UV scale between grids, then use grid points as 3D correspondences.
- Global affine (one segment): z flipped, ~99 deg rotation in xy, scale 3.27-3.29 (nominal 3.297), median residual 6.9 vox at 2.4 um.
- Per patch: local affine from correspondences near the patch, then refine by maximizing NCC between the real 7.91 patch and
  the 2.4 data resampled (antialiased) onto the patch grid. **Patches with NCC < 0.6 after refinement are excluded** (reported).

## Sites
- GT normals from the structure tensor of the smoothed GT; centre-line of each band from GT ridge (max inside-EDT).
- Compressed site: from a band centre point, along the normal, next band centre at distance d in [1.5, 4) vox. Window = [0-1, d+1].
- Positive control: wide sites, d in [8, 15) vox. Window = [0-1, d+1].
- Negative control (texture confound): isolated band points (no other band within 15 vox along the normal); window of
  length matched to the compressed d distribution, starting at the band centre, pointing into the sheet side chosen at random (seeded).
- Up to 400 sites per class per patch, seeded sampling (seed 0).

## Arms (same physical line per site)
- A: real 7.91 um, sampled every 1 vox (7.91 um).
- B: native 2.4 um, sampled every 2.4 um.
- C: the same 2.4 um data blurred to 7.91-like resolution (Gaussian, sigma fitted once by best NCC to real 7.91 patches),
  sampled every 7.91 um. B vs C isolates resolution with everything else fixed (same scan, same registration).

## Valley score
Inside the window, for each interior sample t: depth(t) = min(max I left of t, max I right of t) - I(t), normalized by
(that flanking minimum peak - background), background = 5th percentile of the patch. A site is "resolved" if max depth >= 0.15.

## Decision rule (fixed now)
Resolution angle **alive** if, at compressed sites, resolved(B) - resolved(C) >= 10 points AND resolved(B) at compressed
sites exceeds resolved(B) at negative-control windows by >= 10 points. Otherwise the multi-resolution angle is dropped and the
project keeps only identity-through-contact. Positive control must show high resolved rates in all arms, or the harness is broken.

## Amendments (made before any 2.4 um valley score was computed)
1. **Site bins.** GT band centres are never < ~9.5 vox apart (centre-to-centre along the normal: Scroll 4 p1 = 9.75,
   Scroll 1 p1 = 9.5, nothing < 8 in either; bands are ~4.5 vox thick and never merge). Jeong's "< 4 vox" must therefore be
   an edge-to-edge gap, i.e. centre spacing roughly < 9-10. New bins: **compressed d < 10.5**, wide d in [16, 24).
2. **Negative control redefined.** Window of length drawn from the compressed d distribution, starting at band centre - 1 of a
   *wide* site and pointing toward its neighbour (so it lies in the sheet body, not across an inter-sheet gap that has a sheet
   behind it). This measures how often within-sheet texture or noise alone makes a "valley".
3. **Primary endpoint = difference in differences**: [B - C] at compressed sites minus [B - C] at negative-control windows.
   This cancels effects that resolution has everywhere (fiber texture, per-voxel noise at 2.4 um). Alive if DiD >= 10 points
   AND B - C at compressed sites >= 10 points.
4. Secondary: B repeated on 2.4 um data smoothed with sigma 0.7 vox (noise suppression), reported alongside.
5. Registration result before scoring: 2/2 test patches NCC 0.92 after translation refinement. Eye test (data/scratch/xscan_*.png):
   native 2.4 um splits compressed bundles into many thin strands (7-15 um, likely fiber bundles); the real 7.91 um scan is blurrier
   than 2.4 um blurred with sigma 1.5, so the C-arm sigma is calibrated by NCC as planned.

## Results

### Registration (all patches)
Translation-refined local affines give NCC 0.87-0.94 between real 7.91 um patches and 2.4 um data resampled onto them.
Best blur for the 2.4 -> 7.91 forward degradation is sigma = 2.5 vox at 2.4 um (~6 um Gaussian) on every patch checked:
**the real 7.91 um scan is effectively a ~6 um-blurred version of the 2.4 um scan.** This is a measured forward-model
parameter for the simulator, and the registered pairs are calibration data.

### Pre-registered valley test
Quick run (3 patches, 2 with sites): compressed A 91.0%, C 95.3%, B 100%; control A 85.8%, C 85.6%, B 100%;
DiD(B) = -9.6 points, B - C at compressed = +4.8. **By the letter of the rule: not alive.**
**Full run** (all 36 Scroll 4 patches; 34 have sites; none excluded for NCC; calibrated sigma = 2.5, mean NCC 0.916):

| class | n sites | A real 7.91 | C 2.4->7.91 | B native 2.4 | Bs 2.4 smoothed |
|---|---|---|---|---|---|
| compressed (d < 10.5) | 2778 | 91.0% | 92.5% | 99.8% | 99.8% |
| wide (d 16-24) | 12756 | 99.9% | 99.9% | 100% | 100% |
| control (sheet-body windows) | 12756 | 86.4% | 86.1% | 97.4% | 96.9% |

- B - C at compressed sites: +7.3 points (rule needs >= 10).
- DiD(B) = -4.1 points, 95% CI from the patch bootstrap [-7.5, -0.3] (rule needs >= +10). DiD(Bs) = -3.5.
- **By the letter of the rule: not alive**, consistent with the quick run.
- The ceiling problem from the quick run holds. A already resolves 91% of GT-compressed sites, and positive controls sit
  at 99.9-100% in every arm.

Interpretation: the test did not probe the question. The real 7.91 um arm already shows a valley at 91% of
"compressed" sites, so these are not fused at 7.91 um. The cause is structural: GT only labels sheets annotators could separate
at 7.91 um (centre spacing never < ~9.5 vox), so GT-defined sites cannot reach the fused regime. Resolution reveals structure
everywhere (control B 100% vs C 86%), consistent with fiber-scale texture (7-15 um strands) becoming visible at 2.4 um.

### Exploratory (not pre-registered): mesh-defined contacts
Idea: the tifxyz traces of consecutive wraps (w028-w041) give sheet identity independent of GT; where wrap k and k+1 traces
come within 15-40 um there should be a contact. Consecutive-wrap spacing on the 2.4 um meshes: median 155 um, 10.9% < 32 um.
Eye test of filtered sites (both meshes locally planar, site inside material; data/scratch/mesh_contacts2.png): the traces
are offset by roughly 5-15 um from material boundaries; at several "close" sites both traces hug the same edge with one in
air. **The traces are not precise enough to define contact identity at 2.4 um.** No quantitative test run on them.

### Verdict on the resolution gate
**Not testable on current real labels**, rather than refuted. No real label source has sheet identity at contacts accurate
to < ~10 um: GT only covers sheets separable at 7.91 um, mesh traces are ~5-15 um off. Decision:
- Angle (a), identity through contacts, becomes the primary goal. The simulator is the only source of exact identity at contacts.
- Angle (b), resolution, is demoted to an in-sim question (render identical geometry at 7.91 / 2.4 um with the calibrated
  forward model and measure separability), plus a later end-to-end real-data check: does a surface model run on 2.4 um data
  produce fewer merges at registered eval patches than the same model on the 7.91 um data?
- Kill criteria unchanged for the main endpoint.
