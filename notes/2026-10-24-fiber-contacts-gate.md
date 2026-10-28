# Can automated fiber paths separate merged sheets at chain contacts? Pre-set gate

**Gate frozen: 2026-10-24T20:49:45Z (UTC).** Written before any fiber geometry was read near a chain step. Up to this
point only collection metadata had been read (extents, counts, provenance; see "Data found"). Nothing in the "Gate"
section may change after this timestamp. Later sections are append-only and timestamped.

## Data found (before the gate)
- **Automated Fiber Volumes (qualzz, villa #1930):** `https://dl.ash2txt.org/community-uploads/qual/AutomatedFiberVolumes/`
  - `Paris4_v1.afv` (412 MB, SQLite, VCFC v1). Frame `PHercParis4/20260411134726`, native L0 XYZ at 2.4 µm (the volume
    our Paris4 chains use, at OME level 2). 512,698 traces, 5.27 M points, z 1549-75752 native (whole scroll height).
  - Provenance (verbatim fields): "bidirectional midpoint CUDA integration of unoriented dyadic fiber-axis tensors"
    on the lasagna fiber prediction at **level 4 (38.4 µm)**; integration step 16 native; simplification max error 8
    native; "Estimated overlapping streamlines, not unique physical fibers, H/V labels, or validated merges". Family is
    'unknown' for all traces. In our L2 frame (9.6 µm) that is a 4-vox field, a 4-vox step and up to 2 vox of
    simplification error. Sheets at merged contacts are ~3-6 vox apart, so separation is marginal a priori.
  - Others: PHerc0175A (2 files), PHerc1447, PHerc0125 (9.362 µm). **None for PHerc0139 or PHerc1667.**
- **Dense lasagna fiber predictions (S3 `<scroll>/representations/predictions/fibers/`):**
  - Paris4 on 20260411134726. PHerc0139 only on the 2.399 µm scan 20260102150214, not our 9.362 µm frame. None for
    PHerc1667.
  - Published at level 3 or coarser (19.2 µm, `presence`/`nx`/`ny`). That is too coarse to resolve touching sheets, so
    it is not used here.
- **qualzz fiber model and generator:**
  - Model: HF `Qualzz20/afv_fiber_9um` (nnUNet, `checkpoint_final.pth` 566 MB).
  - Generator: villa PR #1934 (open, branch `Qualzz:vc3d-afv-generator`, `vesuvius/src/vesuvius/afv_spline_generator`).
  - Training code: `github.com/Qualzz/afv-fiber-9um-training`.
  - PR example: one 512³ zone at 9.362 µm in 81 s on an RTX 3090.

## Question
At chain steps where the m7 ray count undercounts (a merged contact: the step is counted 1 but the label says 2), do
the automated fiber traces show an extra sheet between the two mask crossings, often enough and cleanly enough to beat
the label-free detectors? The baseline is held-out raise precision of 31.6-41.5 % (notes/2026-10-24-contact-detector-gate.md),
and on Paris4 z9000: gm 0.263 35.6 %, F 35.9 %.

## GATE

### Test set (the contact-detector gate's step tables)
- Steps are chain single steps (d0 = 1) of the no-raise chains in `data/scratch/fourth_test/dev_cache/<region>.pkl`.
  Labels use the region's primary variant, both ends labelled, k < kmax, exactly as in `dev_steps.evaluate` and the
  contact-detector gate.
- Positive ("merged contact"): true winding difference == 2. Negative: true difference == 1 (a correct non-contact
  step). Steps with other differences (0, >= 3) stay in the denominator of precision as non-positives.
  - Secondary metric: precision counting true difference >= 2.
- **Arm A (primary):** PHercParis4 z 9000-10500, all labelled single steps (16,804 td1, 3,258 td2), fibers from
  `Paris4_v1.afv`.
- **Arm B (conditional):** qualzz's 9 µm generator (PR #1934 code, `Qualzz20/afv_fiber_9um`, default settings) run
  locally on sampled 512³ zones. Arm B runs only if the generator runs locally within about 2 h of setup effort and
  $0 spend. Otherwise it is reported "not tested".
  - B1 is PHerc0139 band 1 (9.362 µm volume 20250728140407, level 0 = chain frame). B2 is Paris4 z9000 (level 2 of
    20260411134726, 9.6 µm, chain frame).
  - Zone selection is label-blind: seed 0, random draws from the midpoints of all labelled single steps (any true
    difference) of the region, keeping a draw only if its 512³ box does not overlap an earlier one. Up to 12 zones per
    region.
  - Scored on the labelled single steps whose whole tube (below) lies inside a zone, at least 24 vox from the zone
    faces.
- Matched control set (for AUC and figures only): each positive is paired with the td1 step of the same region and z
  level whose start radius is nearest (without replacement).

### Coordinates
- Chain points are zyx in the chain frame. Paris4 is OME level 2: chain = (native − 1.5) / 4. PHerc0139 is 9.362 µm
  level 0: chain = native.
- Fiber polylines (XYZ native) are converted to the chain frame and resampled to 0.5 vox spacing.

### Fiber rule (no fitted parameters in the primary rule)
For step j from crossing a to crossing b (length L, unit ray direction u):
1. Tube: the fiber samples whose perpendicular distance to the line a-b is <= rho = 4 vox and whose axial position
   t = (p − a)·u lies in [−3, L + 3].
2. Each fiber with samples in the tube contributes one value: the median t of its in-tube samples. A fiber is skipped
   if the spread (max − min) of its in-tube t exceeds 3 vox, because it runs along the ray instead of across it and
   is not tied to one sheet.
3. Interior fibers: the fibers with median t in [0.3 L, 0.7 L].
4. **Fiber raise:** the step is called ">= 2 windings" iff at least **m = 2** distinct interior fibers exist.
5. Coverage: a step is "covered" iff at least one fiber contributes in its tube. Uncovered steps are never raised.

Continuous score for AUC and PR curves: number of interior fibers, with ties broken by the interior fraction
(interior / all contributing).

Secondary, cross-fitted rule (honest held-out, reported in full):
- Grid: m in {1, 2, 3, 4, 6} × interior window in {[0.3, 0.7], [0.2, 0.8], [0.25 L+2 vox clipped]} → the last
  becomes [max(0.2 L, 2.5), min(0.8 L, L − 2.5)].
- Arm A is split into two z folds: z < 9760 and z >= 9760 (chain z levels). The grid point with the highest
  precision at recall >= 30 % is chosen on one fold and scored on the other, then swapped. The held-out numbers are
  pooled. If both arm-B regions exist, the cross-fit is leave-one-region-out over B1 and B2 instead (in addition).
- Exploratory only (not gated): fiber raise AND gm-0.263 raise, and fiber features added to the F logistic.

### Pass bar (per tested region; arm A decides the verdict, arm B regions are reported with the same bar)
- **Precision (true diff == 2) >= 60 %**, with the cluster-bootstrap 95 % CI lower bound **> 41.5 %**.
  - Bootstrap over chains (rays), 2,000 resamples, seed 0.
- **Recall >= 30 %** of all labelled td2 single steps in the region, uncovered ones included.
- Report coverage: the fraction of labelled single steps covered, and the fraction of td2 steps covered.
- Verdict:
  - **PASS** if the primary rule or the cross-fitted rule meets the bar on arm A. If only the cross-fitted rule
    passes, say so. Two shots are taken; this is disclosed.
  - **FAIL** if neither passes.
  - **NOT TESTABLE** if coverage of td2 steps is < 30 %, because then recall >= 30 % is impossible and the data
    cannot answer the question. That case is reported as "fails recall by coverage" with precision on covered steps.
- PHerc1667 is not part of the gate (no fiber data).
- Figures: 3-4 example contact sites (CT + m7 mask + fibers + chain crossings + label crossings), plus 2 matched td1
  sites. Saved in `data/scratch/fiber_contacts/`.

### Process rules
- Fibers are read only through local SQLite (python -I, read-only immutable URI) from `data/untrusted/afv_fibers/`.
- No pickles or torch files from untrusted sources except with `weights_only=True`.
- Spend cap $10. Local GPU first, with lease `~/.gpu-leases/vss-fib.json`.

---

## Log (append-only)
### L1 (2026-10-24T21:10Z): arm A result (Paris4_v1.afv, Paris4 z9000)
- Extraction: `data/scratch/fiber_contacts/fiber_steps.py`, 17 s. 48,193 traces touch the region. 4.0 M samples
  (0.5 vox) lie within ±4 vox of the chain z levels. Scoring: `fiber_eval.py`, output `result_paris4_z9000.json`.
- **Coverage:** 9.0 % of the 20,906 labelled single steps, 10.2 % of td2 steps and 8.6 % of td1 steps have any fiber in
  the tube. That is about 0.1 contributing fibers per step.
- **Primary rule (m = 2, [0.3 L, 0.7 L]):** 10 steps raised (0.05 %), precision 0.0 % (CI 0-0), recall 0.0 %.
  - The gm 0.263 raise on the same steps: 35.6 % precision, 38.3 % recall.
- Grid, in-sample, for information:
  - m = 1: 486-742 raises, precision 18-19 %, recall 2.7-4.3 %.
  - m >= 2: 0-19 raises.
  - Cross-fit over z folds: no grid point reaches recall >= 30 % on either training fold, so nothing was chosen.
- Matched pairs (3,258 td2 vs same-z nearest-radius td1): AUC of the interior-fiber score 0.503, AUC of any-fiber
  count 0.510. The fibers carry no information at merged contacts.
- **Why (alignment check, `align_check.py`, z 9760, 20k samples):**
  - Fiber samples fall on the m7 mask 25.4 % of the time. Shifted copies give 21-26 % (best shift +4/+4 vox: 26.1 %).
    Copies jittered ±20 vox give 23.7 %. Swapping xy gives 18.4 %, so the frame is right but the sheet localisation is
    nearly absent.
  - This fits the provenance: streamlines integrated on a 38.4 µm (4-vox) field with a 4-vox step and 2-vox
    simplification. They follow fiber direction, not sheet position at voxel scale.
- **Arm A verdict per the gate:** td2 coverage 10.2 % < 30 % → "NOT TESTABLE, fails recall by coverage". Precision on
  covered steps is 0 % (10 raises). Even uncapped, the signal is at chance (AUC 0.50). Arm B (native 9 µm spline
  generator) follows, because it is the actual Sept-prize method and resolves 9 µm.
### L2 (2026-10-24T21:35Z): arm B running; mask-CT misregistration found. Deviation fixed before any arm-B scoring
- **Arm B runs.** qualzz's generator (PR #1934 head 2d8577e, `Qualzz20/afv_fiber_9um`, defaults: no mirror,
  threshold 60, min length 100, black distance 16, no extend) is driven by `data/scratch/fiber_contacts/afv_run.py`.
  - The driver forces `torch.load(weights_only=True)`; the checkpoint loads fine that way. It stubs the heavy
    `vesuvius` package imports. The PR's own loader uses `weights_only=False`.
  - Venv: `data/untrusted/afv_gen/venv` (torch 2.14.1+cu130, nnunetv2).
  - Zones come from `zones.py` (label-blind, seed 0, 12 per region), are run by `run_zones.sh`, and are written to
    `afv_b/`. About 2 min per 512³ zone on the shared 3090, under GPU lease vss-fib with 5 GB.
- **The 9 µm fibers are on papyrus; the m7 mask is locally offset from the CT.**
  - PHerc0139 b1 zone 0, z 3488: CT at fiber samples is 135.8, against 92.6 / 86.0 for copies jittered by ±3 / ±6
    vox. The fibers track the bright sheets precisely.
  - By contrast, the Paris4_v1.afv traces give 99.7 against 97.5 / 94.9, i.e. not localised (arm A conclusion
    unchanged).
  - But the m7 mask (the frame of the chain points) is locally misregistered to the CT
    (`mask_ct_offset.py`, 96³ crops, best shift of mask(q + d) vs CT(q)):
    - PHerc0139 b1: (−4,−4,−1), (2,−1,−3), (1,−1,3), (1,1,−1). CT contrast in vs out of the mask goes from
      13.2 to 27.1 and from 9.9 to 24.4 at the best shift.
    - Paris4 z9000: (−3,−1,0), (0,0,−1), (−1,−1,2), (2,−4,−4).
  - The shifts vary from place to place, so this is not a global offset. Up to 4 vox is comparable to a whole step's
    interior window, so fiber positions relative to the chain crossings carry this error.
- **Deviation (label-free, fixed now, before scoring arm B).** `register.py` estimates a local mask→CT shift per
  128³ sub-block of each zone:
  - argmax of CT contrast in vs out of mask(q + d), |d| <= 5, FFT correlation, then a 3×3×3 median over blocks.
  - Fibers are then mapped into the mask frame as p + d before the frozen rule is applied.
  - Arm B is reported both **as frozen (no registration)** and **registered**. The verdict notes which one passes, if
    either. Arm A is not re-run with registration, because its traces are not on papyrus in the first place.
- Side finding, for the backlog: this misregistration also limits every CT-profile feature computed at chain
  crossings. The contact-detector CT pilot had AUC <= 0.67.
### L3 (2026-10-24T22:40Z): arm B, as frozen (no registration)
24 zones generated, all rc 0, 88-133 s each, 531-1,398 fibers per 512³ zone (after the default min length 100; zone
0 had 5,992 before cleanup). Scored with `fiber_steps_b.py` + `fiber_eval.py` on the labelled single steps inside the
zones (`subset_<region>_B.json`). Outputs: `result_p0139_b1_B.json`, `result_paris4_z9000_B.json`.

| region (zones) | labelled single / td2 | coverage all / td2 / td1 | fibers per tube td1 / td2 | primary raised, prec, recall | m = 1 [0.3-0.7] (in-sample) | matched AUC | gm 0.263, same steps |
|---|---|---|---|---|---|---|---|
| PHerc0139 b1 | 1,058 / 72 | 22.6 / 34.7 / 21.8 % | 0.24 / 0.38 | 0, –, 0 % | 55 raised, 10.9 %, 8.3 % | 0.520 | 41.3 % prec, 36.1 % rec |
| Paris4 z9000 | 2,647 / 385 | 22.3 / 21.6 / 22.2 % | 0.25 / 0.24 | 1, 0 %, 0 % | 156 raised, 13.5 %, 5.5 % | 0.493 | 41.7 % prec, 35.3 % rec |

- The Paris4 z-fold cross-fit chooses nothing: no grid point reaches recall >= 30 % on either training fold.
- Fiber position along the step (`tdiag.py`, fibers per step by t/L bin) is flat-ish for td1 and td2 alike. For
  PHerc0139 td1, density in [0.1, 0.5) L is 0.044 / 0.033, the same as at the crossings (0.046 / 0.044). There is no
  "fibers at the two ends, empty middle" pattern for correct steps, which fits the mask-CT offset.
- Coverage is the binding limit. About 0.25 fibers per tube means a missing interior sheet carries a fiber in the
  tube in well under 30 % of cases. That holds even with perfect localisation, so the recall bar cannot be met by
  construction at the generator's default density.
### L4 (2026-10-24T23:35Z): arm B registered, exploratory denser fibers, side finding
**Registered** (`reg_<region>.json`, fibers p → p + d):
- The share of fiber samples on the m7 mask rises only from 0.29 to 0.33 (PHerc0139, 12 zones, `regcheck.py`) and
  from 0.27 to 0.29 (Paris4). Copies jittered ±8 vox give 0.21 / 0.23.
- PHerc0139 b1: coverage 22.9 / 26.4 / 22.7 % (all / td2 / td1). The primary rule raises 0. Matched AUC 0.507.
  - m = 1 (in-sample) gives precision 6.2-8.2 % at 4-8 % recall; the base rate is 6.8 %.
- Paris4 z9000: coverage 23.6 / 21.8 / 23.8 %. The primary rule raises 1 step, which is correct: precision 100 %,
  CI [0, 100], recall 0.3 %. Matched AUC 0.496.
  - m = 1 gives 14.3-14.6 % precision at 5.7-8.1 % recall; the base rate is 14.5 %.
  - The cross-fit chooses nothing.
- Outputs: `result_*_Breg.json`.

**Exploratory, not gated: denser fibers.** The PHerc0139 zones were re-run with `--min-length 0` (`afv_bd/`). That
gives 3,647-6,133 fibers per zone, about 5x the default.
- Unregistered: coverage 33.0 / 45.8 / 32.3 %, 0.38 / 0.54 fibers per tube. The primary rule raises 2 steps, both
  wrong. Matched AUC 0.541; any-fiber count AUC 0.609 (td2 steps are longer, so their tubes catch more fibers).
- Registered: coverage 35.0 / 37.5 %, primary 2 raises, both wrong, AUC 0.507.
- m = 1 precision is 6.9-11.1 %, i.e. the base rate.
- The extra fibers are short pieces, so coverage grows far less than the fiber count.

**Side finding (label-free, `radial_d.py`): the m7 mask sits inward of the CT sheets.** Over 768 blocks of 128³ per
region, the radial component of the best mask→CT shift is:
- PHerc0139 b1: −1.91 vox (sd 1.90; 87 % of blocks negative).
- Paris4 z9000: −1.64 vox (sd 2.21; 79 % negative).

The tangential mean is −0.04 / −0.05. So the CT sheet lies about 2 vox outward of the m7 surface voxels, a systematic
one-face bias. This matters for anything that samples CT at chain crossings (the contact detector's CT profiles) and
is worth checking in the m7/m7-contact training targets.

Figures (`data/scratch/fiber_contacts/`):
- `sites_p0139_b1_reg.png`: 4 random td2 sites and 2 td1 controls; CT, m7, 9 µm fibers, chain and label crossings.
- `sites_paris4_z9000_reg.png`: the same for Paris4.
- `armA_paris4_v1_z9760.png`: Paris4_v1.afv traces on CT.

At step scale the fibers are a few dots (vertical fibers) and the odd horizontal segment per 96² window. At the
merged sites the tube holds 0-1 fibers and never an interior one. In one PHerc0139 td2 case (z 3808) the label's extra
winding lies in an empty CT gap, where no fiber can exist.

## VERDICT (2026-10-24T23:40Z): FAIL. Automated fiber paths do not separate merged sheets at our miscount sites.

| arm / region | td2 coverage | primary: raised, precision [95 % CI], recall | matched AUC | gate |
|---|---|---|---|---|
| A, Paris4_v1.afv, Paris4 z9000 (all 20,906 steps) | 10.2 % | 10, 0.0 % [0, 0], 0.0 % | 0.503 | fails by coverage; 0 % on covered |
| B, 9 µm generator, PHerc0139 b1 (1,058 steps) | 34.7 % | 0, –, 0.0 % | 0.520 | **fail** |
| B, same, registered | 26.4 % | 0, –, 0.0 % | 0.507 | fails by coverage |
| B, 9 µm generator, Paris4 z9000 (2,647 steps) | 21.6 % | 1, 0 %, 0.0 % | 0.493 | fails by coverage |
| B, same, registered | 21.8 % | 1, 100 % [0, 100], 0.3 % | 0.496 | fails by coverage |
| baseline gm 0.263, same steps (B: PHerc0139 / Paris4) | – | 41.3 % / 41.7 % precision, 36.1 % / 35.3 % recall | – | – |

- **Neither the primary nor the cross-fitted rule meets the bar anywhere.** The cross-fit never found a setting with
  training recall >= 30 %.
- With the threshold relaxed to a single interior fiber (in-sample), precision equals the base rate (6-15 %).
  Every AUC is 0.49-0.54.
- Reasons, in order of weight:
  1. **Density.** The traces are sparse at step scale: 0.1 fibers per tube for Paris4_v1 and 0.25-0.4 for the 9 µm
     generator, even without its length filter. A sheet hidden between two mask crossings has a fiber in a 4-vox tube
     in well under a third of cases, so recall >= 30 % is out of reach by construction.
  2. **Localisation.**
     - The published Paris4 collection is streamlines of a 38.4 µm field and is not on papyrus at voxel scale.
     - The 9 µm fibers are on papyrus, but the chain frame (m7) is offset by about 2 vox inward plus a local ±2-4 vox.
       Registration recovers little.
  3. **Some "merged" steps have no second sheet in the CT** at that z. The extra label winding falls in a gap or a
     fused band, so a fiber detector cannot see it.
- The question needed "fibers crossing the ray segment imply 1 vs >= 2 sheets". The fibers that exist pass near the
  segment too rarely, and with too little position information relative to the chain crossings, to answer it.
- No wiring sketch, because the gate did not pass. A fiber-based contact signal would need a dense, voxel-accurate
  product (presence at 9 µm, not paths) registered to the chain mask. That is a different test from this one.

**Spend:** $0. No pods were created; the local 3090 ran 36 generator runs of about 1.5-2 min each, under lease vss-fib
(5 GB; removed after).

**Disk:**
- `data/untrusted/afv_fibers/` 0.39 GB (Paris4_v1.afv plus manifests).
- `data/untrusted/afv_gen/` 6.7 GB (venv 6.2 GB, model 0.54 GB, sparse PR clone).
- `data/scratch/fiber_contacts/` 0.15 GB.
- New CT chunk cache in `data/scratch/contact_det/ctcache/` about 20.7 GB (raw 128³ npy from the registration;
  shared directory, left in place, re-streamable). m7cache +0.26 GB.

z 7000-8500 was not touched.
