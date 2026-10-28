# PHerc0139 winding labels: CT-verified audit over the full z range (2026-10-22)

This extends the band-1 and band-2 label audits (notes/2026-10-21-fit-figures.md, notes/2026-10-21-second-band-plan.md
7b) to every z where the public human meshes exist (z 1750-8500). It uses the same calibrated detectors: m7 mask runs
and lasagna grad_mag turns. The work was done locally with no pods, so the cost was $0.

## Outputs

| file | content |
|---|---|
| eval/winding_audit.py | reusable CLI (`extract`, `analyse`, `report`, `audit`); sha256 e35495994702b4c665fb9dc5539573ea9b4bdd9a0d3568feb43d75b0a1218b2c |
| data/scratch/label_audit/p0139_label_corrections.json | per (slab, sector) and per slab: old label -> CT-consistent relative winding, conf, trusted; duplicate / untraced / overcount lists; per-mesh summary; validation |
| data/scratch/label_audit/p0139_label_corrections.csv | the same mapping, one row per (z slab, sector or `all`, mesh): 5,740 rows |
| data/scratch/label_audit/full/audit_pairs.csv | every (slab, sector, label pair) verdict with m7 / gm CIs |
| data/scratch/label_audit/full/{rays.npz, audit.json, pairs_rays.npz} | raw rays, full analysis |
| release/fig_label_audit.py | figure script; outputs are release/figures/label_audit_*.png |

All of these are local (derived labels are not published).

## Method

- **Rays.** There is one z level every 48 slices (z = 24 + 48k, so each level stays inside one m7 chunk row), from
  1752 to 8472. That gives 141 levels × 72 angles (5° apart) around the umbilicus.
  - Mesh crossings come from each mesh's marching-squares contour at the level, intersected with the ray.
  - If a mesh crosses one ray at radii more than 5 vox apart (a fold or overlap), it is dropped on that ray.
- **m7.**
  - The mask is the bucket m7 L0 th0.2 prediction, streamed and never stored. Each ray samples it along 9 lines
    (3 z × 3 tangential, ±2 vox) at a 0.5-vox step and takes the maximum.
  - Each mesh crossing is assigned to the mask run that contains it, or to a run within 4 vox. A crossing with no
    run nearby gets its own virtual sheet. If one run holds crossings more than 8 vox apart, it is split.
  - Turns between two meshes = the difference of their sheet indices.
- **gm.** Lasagna grad_mag (u8/1000) is integrated along the centre line. Rows 34-69 were resampled to fill
  z 4352-8832 (data/p0139/inputs_audit, 2.5 GB). Coverage was checked first: every level has m7 and gm, with no NaN.
- **Calibration.**
  - The gm scale is set per 500-slice slab. It is 1 / the median gm of label-adjacent (dk = 1) pairs with
    |gap| ≥ 4 vox, measured on the even-angle half only. The scale ranges from 0.68 to 0.85.
  - The odd-angle half is held out for validation.
- **Verdict per (slab [, 30° sector], label pair).**
  - T* = the modal m7 turn count. It counts only if it holds on ≥ 50 % of rays and the gm median is within 0.5 of it;
    otherwise the pair is `ambiguous`.
  - Classes:
    - `ok`: turns = expected;
    - `duplicate`: 0 turns;
    - `untraced`: more turns than labels, so a sheet has no mesh;
    - `overcount`: fewer turns than labels;
    - `order`: negative turns.
  - Confidence = the share of 300 angle-cluster bootstrap replicates that reproduce T*. The report uses
    conf ≥ 0.8.
- **Mapping.**
  - Each cell gets a robust weighted least-squares solve of x_k' − x_k = edge turns. The edge value is the mode
    for dk ≤ 2 and the rounded median for longer edges. The anchor is x_029 = 29. Edges with residual > 0.5 are
    dropped while the anchor's component stays connected.
  - New label = round(x_k). Confidence comes from a 200-replicate angle bootstrap.
  - A shift is **trusted** if conf ≥ 0.8 and the same solve on per-edge rounded gm turns (x_gm_int) gives the same
    label.

### Seam convention (new, and essential)

- Each human mesh is one turn cut at its own seam, and the seams fall into groups:
  - A, w023-w034: ~140°, drifting to ~190° by z 7000;
  - B, w036-w049: ~250-265°;
  - C, w051-w059: ~60-100°.
- **w035 and w050 are partial bridge meshes.** They fill the arcs where the seam rotates from one group to the next:
  - w035 is absent at ~150-260°;
  - w050 covers ~90-265° and is absent elsewhere.
- So the label difference equals the physical turn count only on one side of both meshes' seams.
- Expected turns are E = dk + c(k2) − c(k), where c = [the ray lies before the mesh's seam, walking outward (cw in
  arctan2 angle) from a reference angle].
  - The reference angle is one value for the whole scroll: 115°, the best ray agreement of raw m7 turns with E.
  - Agreement is 0.7845 at 115°, 0.7832 at 110° and 0.7838 at 120°; 250° gives only 0.735.
  - Per-slab fits are near-ties (Δ < 0.015) and made the low slabs unstable, so they are not used.
- Rays within 10° of either seam are dropped. Verdicts are reported in label convention (T = T_raw − (E − dk)).
- Consequence: **w049 → w051 is 1 physical turn where w050 is absent, and that is exactly what the labels imply.** The
  172 "skip" pairs are a seam-convention effect, not missing sheets. This replaces the earlier reading that w050 is
  missing or untraced.

## Validation

- **Clean gaps.**
  - Controls are chosen from labels only, with no CT: dk = 1 pairs whose local spacing ratio is 0.6-1.6 on ≥ 80 % of
    angles and whose |gap| is ≥ 4 vox on ≥ 95 % of rays. There are 245 control (slab, pair) cells, with 75,674
    held-out rays.
  - m7 turns per ray: mean 1.040, P(1) = 0.866, P(0) = 0.049, P(2) = 0.077.
  - gm turns: median 0.981, p10-p90 0.61-1.56 (validation figure).
- **False flags on the held-out set.** 4 of 245 cells are flagged (1.6 %):
  - 2 are tiny cells (6 and 8 rays) at w038 and w040, z 2250;
  - 2 are w052-w053 and w053-w054 at z 2250-2750, where gm agrees with m7 (0.44 and 1.88). That is probably a real
    local error (w053 on w052's sheet) that the label-only control selection missed.
  - Excluding the tiny cells, the false-flag rate is ≤ 0.8 %.
  - Note: the earlier run with per-slab reference angles had 0 flags.
- **Split-half reproducibility.**
  - Verdicts are made on the calibration half and re-measured on the held-out half, over 503 (slab, pair) cells.
  - 450 / 459 `ok` cells are reproduced (98.0 %), and 40 / 44 flagged cells are reproduced (90.9 %).
  - On held-out rays of `ok` gaps: m7 mean 1.044, gm median 0.988.
- **Spot checks.** There are 4 CT crops, each with mesh traces, the ray, and CT / m7 / gm profiles:
  - release/figures/label_audit_dup_w041_w042.png (z 3624, 300°): w041 and w042 lie on one CT sheet, at radius
    1017 vox.
  - release/figures/label_audit_dup_untraced_w045_w047.png (z 3816, 0°):
    - w045 = w046, both on one sheet;
    - between w046 and w047 there is one clear CT sheet with no mesh (m7 2, gm 1.7).
  - release/figures/label_audit_sector2167.png (z 3960, 190°): w034 and w036 lie < 1 vox apart on one sheet, where
    1 turn is expected (#2167 sector).
  - release/figures/label_audit_bridge_w050.png (z 3816, 15°): w050 is absent, and w049 → w051 is one sheet step
    (m7 1), as the seam convention predicts.
  - The overview and validation figures are release/figures/label_audit_overview.png and
    release/figures/label_audit_validation.png.

## Findings

All values below are from all-angle cells (seams excluded). m7 and gm are mean / median turns minus the label
difference; conf is from the bootstrap.

| winding range | z 1750-2750 | z 2750-6750 | z 6750-7750 | z 7750-8500 |
|---|---|---|---|---|
| w023-w034 | ok | ok (every dk = 1 pair, conf 1.0) | ok | ok (few rays) |
| w034-w035-w036 (#2167 / w035 bridge) | no meshes | w034-w035 ok outside 145-260°. In 145-260° w034 → w036 is **0-1 turns where 1 is expected**: overcount at z 4250-4750 (conf 1.0, m7 −0.82, gm −0.89) and in sector 180-210° at 3750-4250. Ambiguous at 3250, 5750-6750. At z 3750, sectors 60-150° also show w035 on w034's sheet. | ambiguous | ambiguous |
| w036-w041 | z 2250: w039 on w038's sheet, then an untraced sheet before w040 (net 0); 12-15 rays | ok | ok | ok |
| **w041 = w042** | — | **duplicate z 3250-7250** (m7 −1.20 to −0.62, gm −1.05 to −0.85, conf 0.83-1.0). Ambiguous at 2750. Organisers blacklist both ("coincides 32 %"). | separate at 7250-7750 (m7 −0.34, gm −0.39, partial) | ambiguous |
| w042-w045 | — | ok | ok | ok |
| **w045 = w046** | — | **duplicate z 2750-8250**, except ambiguous at 5750. gm is −1.00 except −0.83 at 5250 and −0.57 at 5750. m7 is −1.0 except −0.54 to −0.45 at 4750-5750, where conf is 0.6-0.84. Organisers blacklist both ("coincides 82 %"). | duplicate | duplicate |
| **w046 → w047** | — | **an untraced sheet: 2 turns** (m7 +0.67 to +1.39, gm +0.72 to +1.40) in 10 of 11 slabs (conf 0.96-1.0, except 0.61 at 3250). Ambiguous at 3750 (m7 +1.50, gm +1.60). At z 3250-4250, 150-330°, it is 3 turns. | 2 turns | 2 turns |
| w047-w050, w050-w051 | — | ok | ok | ok |
| w049 → w051 (w050 absent) | ambiguous | **ok: 1 physical turn** (label convention 1.91-2.02, conf 1.0) at 3750-6250 | ok at 6750-7250 (1.68-1.70, conf 0.92-0.99) | ambiguous |
| w051-w059 | z 2250-2750: w053 on w052's sheet, then an untraced sheet before w054 (conf 0.71 / 1.0) | ok | ok | ok |
| long edges across missing meshes | w034 → w051 (dk 17): m7 14.4-15.8, gm 15.2-16.5 → ambiguous | — | — | w034 → w051: m7 13.0-13.6, gm 12.3-14.7 → ambiguous |

Pooled CT-consistent numbering (anchor w029; per-slab `all` cells, trusted unless marked):

| z slab | w023-w036 | w037-w041 | w042 | w043-w045 | w046 | w047-w059 |
|---|---|---|---|---|---|---|
| 3250, 4250, 4750, 6250 | 0 | 0 | −1 | −1 | −2 | −1 |
| 3750 | 0 | 0 | −1 | −1 | −2 | 0 (untrusted: w046 → w047 = 3 turns on half the angles) |
| 5250 | 0 | 0 | −1 | −1 | −2 (untrusted) | −1 (untrusted) |
| 5750 | 0 | 0 | −1 | −1 | −1 (untrusted) | 0 (untrusted) |
| 6750 | 0 (w035-w041 untrusted) | 0 (untrusted) | −1 (untrusted) | −1 (untrusted) | −2 (untrusted) | −1 (untrusted) |
| 7250 | 0 | 0 | 0 | 0 | −1 | 0 |
| 2750 | 0 (w035-w036 untrusted) | −1 (untrusted) | −2 (untrusted) | −2 (untrusted) | −3 (untrusted) | −2 (untrusted) |
| 7750 | 0 | −1 (w037-w039 trusted) | −1 (untrusted) | −1 (untrusted) | −2 (untrusted) | −1 (untrusted) |
| 1750-2250, 8250 | 0 | mixed (untrusted) | — | — | — | w051+ −2 / −3 (untrusted; long edges only) |

Mid-z summary:
- The CT-consistent numbering is: w042 → 41, w043-w045 → k − 1, w046 → 44, w047+ → k − 1.
- This is the band-1 `supported` map (data/scratch/label_audit/relabel_band1.json) with one refinement: w046 goes to
  44, the same as w045, not 45.
- It rejects `outer_minus2`, which would shift w051+ down by another 1. w049 → w051 is the expected 1 physical turn.

### How much of the label set is affected

- **Local, pair level.**
  - 88.8 % of the ray-weighted pair evidence in (slab, sector) cells has a confident verdict.
  - Of that evidence: 91.8 % ok, 5.0 % duplicate, 2.9 % untraced sheet, 0.3 % overcount, < 0.1 % order.
- **Meshes with real errors.**
  - 4 meshes are in duplicate pairs: w041/w042 and w045/w046. All 4 are already blacklisted by the organisers.
  - There is one untraced sheet between w046 and w047, which puts all 13 meshes w047-w059 one turn off relative to
    the inner set.
  - The #2167 sector (w034 → w036 at 145-260°, z 3750-4750) is regional.
  - Two single-slab local slips occur at z 2250-2750 (w039, w053).
  - w035 and w050 are not errors. They are bridge meshes.
- **Mesh area (radius-weighted crossings in (slab, sector) cells).**
  - 12.7 % of the area gets a trusted non-zero shift, 50.0 % is untrusted, and 1.6 % has no cell.
  - The untrusted share is high because the per-sector solve chains outward from w029. One uncertain edge in a sector
    (w045/w046 at z 4750-5750, w046 → w047 = 2 vs 3, the #2167 gap) makes every outer label in that cell untrusted.
  - The pooled per-slab map is the robust proposal. The sector cells are detail, and their shifts that disagree with
    the pooled map cluster in 60-210°, between the seam groups.

## Impact on benchmarks

- **Band-1 wrong area** (arm19 − arm0, notes/2026-10-21-second-band-plan.md):
  - −4.7 pp under the original labels becomes +13.5 pp under the CT correction (`supported`).
  - The full audit confirms that `supported` is the CT-consistent numbering for z 3250-4750, and it rules out the
    `outer_minus2` bound (+0.5 pp).
  - The w046 → 44 refinement only touches a blacklisted mesh.
  - The pooled sign therefore hinges on the w041/w042 duplicate and the w046 → w047 untraced sheet, which are the two
    most confident findings. Reporting the label-order-independent own-offset inner / outer numbers next to it stays
    the safe practice.
- **Pair metrics are unaffected.**
- **Skip pairs:** GT agreement goes from 5.8 % to 87.8 % under the w050 fix.
  - The audit explains why: where w050 is absent, w049 → w051 is one physical turn, and the label difference of 2 is
    the bridge-mesh seam convention.
  - So the fix (count that gap as 1) is CT-supported: `ok` in all 8 slabs from z 3750 to 7750, with conf ≥ 0.92.
    It is ambiguous at 2750-3750 and 7750.
- **Band 2 (z 1750-3000)** stays bounded, not corrected.
  - w034 → w051 there is ambiguous (m7 14.4-15.8, gm 15.2-16.5 against the 16 the mid-z map implies).
  - The `minus1` / `minus2` maps stay bounds.

## Tool usage

```
# PHerc0139 (preset: sites.json meshes + published affine, bucket m7 streamed, local lasagna)
uv run python eval/winding_audit.py extract --scroll PHerc0139 --z0 1750 --z1 8500 --out data/scratch/label_audit/full
uv run python eval/winding_audit.py analyse --out data/scratch/label_audit/full --anchor 29
uv run python eval/winding_audit.py report  --out data/scratch/label_audit/full \
    --corrections data/scratch/label_audit/p0139_label_corrections.json
uv run python release/fig_label_audit.py            # figures (streams 4 small CT crops)

# any scroll with human meshes (tifxyz dirs named ...wNNN...), a surface mask zarr and a lasagna grad_mag zarr
uv run python eval/winding_audit.py audit --meshes DIR --umbilicus umb.json [--umbilicus-frame volume|mesh] \
    [--mesh-affine affine.json] --mask s3://.../surface.zarr/0 --gm path/grad_mag.ome.zarr/2 --gm-scale 4 \
    --z0 Z0 --z1 Z1 --out OUTDIR --anchor K
```

- Key options:
  - `--slab 500` and `--sector 30` set the cell size.
  - `--z-step 48 --z-offset 24`: keep the levels inside the mask's chunk rows.
  - `--seam-margin 10` and `--aref` (default: one fitted angle).
  - `--outward-ccw` is for scrolls that wind the other way, and `--no-seams` restores label difference = expected.
  - `--min-rays 12` and `--min-conf 0.8`.
  - `--boot` / `--boot-map` set the bootstrap sizes.
- Runtime and memory:
  - `extract` is the slow step (it streams the mask chunks).
  - `analyse` takes about 2 min (about 1.2 M pair rows) and `report` takes seconds.
- The `--meshes DIR` path is written for generic tifxyz directories but has only been run on the PHerc0139 preset.

## Limitations

- **m7 merges compressed sheets.** P(0) = 0.049 on clean controls, and duplicates show m7 −0.5 at 4750-5750 while gm
  stays −1.0. Where sheets are under ~6 vox apart (w031-w032 at z 6250, the #2167 sector), verdicts become ambiguous
  rather than wrong. The gm agreement requirement protects against false calls but costs coverage.
- **Seam bookkeeping.**
  - The seam angle per mesh and level is estimated from the contour gap. It jumps for a few meshes and levels
    (for example w036 40° at z 3000, w048 25° at 3768), which adds noise in the low slabs.
  - The reference angle is flat within ±5° (agreement 0.783-0.785).
  - Sector-level shifts in 60-210° should not be applied without the pooled map.
- **Numbering is relative.** It is anchored at w029. Long edges across mesh-free z (w034 → w051 at z < 2750 and
  > 7750) cannot be resolved to ±1.
- **Mapping ignores physical continuity across slabs.** It is per slab and sector. A mesh that changes sheet inside a
  cell shows up only as a residual or a dropped edge.
- **CT is used only visually** (the 4 spot checks). The detectors are m7 and lasagna, so a shared failure of both on
  the same sheet would not be caught. In the spot checks, the CT agrees with every flag shown.
- **Costs.** No pods. About 2.5 GB of lasagna was resampled into data/p0139/inputs_audit, about 2 GB of m7 chunks were
  streamed (not stored), and 4 small CT crops were streamed.
