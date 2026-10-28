# Simulator design (draft 2026-10-16)

Goal: synthetic PHerc-like volumes with **exact sheet identity through contacts**, rendered at 7.91 and 2.4 um from
the same geometry. Everything below is grounded in measurements from notes/2026-10-16-*.md; numbers marked (measured).

## Targets from real data
| quantity | value | source |
|---|---|---|
| wrap pitch, consecutive traced wraps, PHerc1667 | median 155 um, p10 28 um, 10.9% < 32 um (traces noisy at low end) | (measured) mesh spacing |
| GT band centre spacing at 7.91 um | median ~16 vox (126 um), never < ~9.5 vox | (measured) jeong200 GT |
| GT label | recto-surface band ~4.5 vox thick at 7.91 um, binary | (measured) |
| internal texture | strands 7-15 um wide, visible at 2.4 um, mushed at 7.91 um | (observed) registered pairs |
| 2.4 -> real 7.91 um degradation | Gaussian sigma 2.5 vox @2.4 (~6 um) then resample; NCC 0.87-0.94 | (measured) 36 registrations |

## Geometry (Blender, headless, seeded)
- Procedural first: Geometry Nodes spiral (Archimedean, variable pitch) -> layered displacement fields (low-freq
  squash/ellipse, mid-freq waviness, local buckles) -> compression that drives pitch into the 15-60 um band in
  designated "contact zones". Full cloth simulation across hundreds of wraps is too slow/unstable; use Blender cloth
  only for local crumple patches if procedural buckles look wrong.
- Sheet = two fiber layers (recto horizontal, verso vertical fibers), thickness drawn per zone; fibers as instanced
  curves so the 7-15 um strand texture exists at 2.4 um.
- Contacts: allow sheet-sheet distance -> 0 with no gap; optionally a thin low-density interface (tunable) so in-sim
  separability can be swept.

## Labels (per voxel, both resolutions)
- `instance`: wrap index (exact through contacts).
- `surface`: recto band in Dataset059 format (binary, ~4.5 vox at 7.91 um) so existing models/metrics apply unchanged.
- `uv`: unrolled coordinates.
- `contact`: distance to nearest *other* sheet (um), so sites can be binned exactly (the thing real data cannot give).

## CT forward model
- Render density at 1.2 um, then per target resolution: Gaussian PSF + resample + noise.
  7.91 um: sigma calibrated so that "sim 2.4 -> 7.91" matches the measured real relation (sigma ~6 um).
- Phase-contrast edge enhancement and noise power spectrum: fit from registered real pairs and air regions.
- Validation: for registered real pairs, the forward model applied to real 2.4 um must reproduce real 7.91 um
  (already NCC ~0.92 with a pure Gaussian; improve with noise/edge terms).

## Output
- Patches as 256^3 cells with the patch = cell trimmed by 10 vox (matches the Jeong/Dataset059 layout), uint8 CT.
- Zarr + JSON sidecar with seed and full parameters (README rule).

## Real-data evaluation for identity-through-contact
- Merge-sensitive endpoints on jeong200 (instance VOI_merge, component counts) alongside the official blend.
- Site-level separation: Jeong's frozen weld sites / PHerc1218 repaired fusion sites
  (kaggle `jhjeong0815/pherc1218-topological-repair`) - check licence and format before use.
- End-to-end resolution check later: same model on registered 2.4 um vs 7.91 um patches, compare merge rates.
