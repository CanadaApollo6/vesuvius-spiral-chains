# Scout: their surface + our numbering (Stevens patches, Kelly index), 2026-10-26

This was a read-only scout. Nothing was posted, no fit was run, and no code from the community repos was run.

Downloads, each into its own untrusted dir:
- shallow clones of the three repos;
- Kelly's `solve_edges.json` (29 MB, sha256 matches their table);
- Stevens' `scroll4_patches.zip` sample (1.3 MB, 13 patches);
- the zip central directories of `s4_good_patches.zip` and `s4_10_components_tifxyz.zip`, read over HTTP range (25 MB).

## 1. Stevens' patches: public, tifxyz, no winding numbers

All files sit at **https://dl.ash2txt.org/community-uploads/will/**. Listing and sizes are as of today.

| File | Size | Contents (from the zip index) |
|---|---|---|
| `s4_good_patches.zip` (2026-07-31) | 1.58 GiB (3.23 GB unzipped) | 47,797 patches `s4_good_patches/patch_N/{x,y,z}.tif + meta.json` |
| `s4_bad_patches.zip` (2026-07-31) | 328 MiB | 9,171 patches. These are the July $2.5k "removed sheet-switching patches" set; Kelly's README confirms the good/bad split |
| `s4_10_components_tifxyz.zip` (2026-09-01) | 1.21 GiB (stored, not compressed) | 10 large tifxyz `patch_0..9`, uuid `WSP20260901_patch_N`, bboxes spanning z 500-10,720. Most likely the flattened components behind the Aug $20k prize (365 cm²). The date and uuid suggest this, but the file is **not confirmed** as that prize's output |
| `scroll4_patches.zip` | 1.3 MiB | 13-patch sample |
| `s4_demerged_v1.7z`, `pvfs_2025_10_19.zarr.7z` (64 GiB) | | Older material. Not examined |

- **Format:** standard tifxyz. It has float32 `x.tif/y.tif/z.tif` with -1 for empty cells. `meta.json` holds
  `{"format":"tifxyz","type":"seg","uuid","bbox","scale":[0.25,0.25]}`.
- **Grid:** patches are about 50-120 cells square. Vertex spacing is 4.0 vox, so a patch is about 3 mm across.
- **Frame:** Scroll 4 volume **20231117161658, level 0, 7.91 µm**. Zarr L0 shape is (11174, 3340, 3440) and the patch
  coordinates fit inside it. Stevens' repo is pipeline4 to pipeline9, written in C++. Its internal format is `.bin`
  records `{x,y,px,py,pz}`, and `bin2tifxyz` converts them.
- **No winding numbers:** neither the meta.json files nor the repo (github.com/WillStevens/scrollreading, GPL-3.0)
  store a winding, umbilicus index or sheet number. The components carry 2D layout only (`patchPositions_<comp>.txt`,
  `rel.csv`).
- **License:** the code is GPL-3.0. The data upload states no license; treat it as dl.ash2txt (CC BY-NC) and keep
  derived outputs local.

**Frame check:** our PHerc1667 frame is 20251217075048 L2 (9.596 µm). The public `metadata.json` (local copy,
data/untrusted/dinovol/metadata) has the affine 20231117161658 → 20251217075048 L0. It is a rotation with axis flips,
scale 3.27, derivation `20231117161658-20251217075048⁻¹`, and I saved it to
`data/scratch/patch_scout/affine_s4_7p91_to_2p399.json`. I mapped 10 sample patches through it with /4 for L2 and
looked them up on our local m7 masks:
- **84-99.8 % of patch vertices land on the (2-voxel-dilated) mask**;
- with a 5-vox shift the figure drops to 39-57 %;
- the undilated mask base rate is 8-14 %.

So the frames register, and Stevens' patches can be compared directly with our chains and our human labels.

Mapping Kelly's patch centroids the same way, **about 12,000 patches have a centroid in each of our two PHerc1667
bands**: L2 z 4500-6000 (11,960) and z 6500-8000 (12,252). Each band covers about 36 index windings.

## 2. Kelly's tool (github.com/tf-kelly/scrolls, MIT; Sept $1k)

- **What it numbers:** exactly Stevens' release. 56,934 of the 56,968 patches (good + bad) get an integer winding.
  `community/wrap_index.csv` is in the repo, with columns `patch,k_q3c,component,thN`. There are 4 components, and
  56,926 patches are in component 0.
- **How:**
  1. Patch pairs that overlap become "joins", 174,641 in total.
  2. Each join gets integer difference terms (a, b, d, w) from three measurements: radial separation divided by an
     assumed 134 µm pitch; a CT sheet-crossing count from their checker; and a same-sheet rule (max normal separation
     below 0.59 of the pitch gives d = 0).
  3. One HiGHS LP minimises the total absolute disagreement, and the result is rounded.
  4. The axis is rebuilt from patch centroids, with no umbilicus input. Handedness is chosen by the LP objective.
  - Code: `phase/tools/vc_sheet_check/vc_sheet_check/solve.py`, `_vendor/our_solver.py`.
  - Release `sept-submission` adds `same_windings.json` (334 MB; 1.28 M point pairs), `relative_windings.json`
    (39 MB), `abs_winding.json` (32 MB; 6 points per patch, wind + 13) and `solve_edges.json`. All are villa
    point-collection v1 with `between_patches__A__B` names.
- **Accuracy:** the repo reports **no patch-level accuracy against human windings**.
  - 15,154 of the 174,641 joins stay unsatisfied (23,857 of 514,297 terms).
  - The index-steered fit matches the reference on turn-changing joins 0.963 (slab 2, 672 joins) and 0.848 (held-out
    HO-A, 402 joins). Unconstrained fits score 0.616 and 0.595; the oracle scores 0.958 and 0.948. This is from
    docs/CLAIMS.md B1/B2, and the anchors came from the reference.
  - On six of Stevens' pages (components), at least 98.9 % of the area falls on one index turn (B14).
- **Limitations they state:**
  - Numbers are relative within a component.
  - The outermost windings (published w038-w040) get 1 turn where the published segmentation has 3 (F14).
  - The contact regime (sheets under 25-50 µm apart) is blind (F1).
  - The pitch is fixed by hand.
  - Patches that cross sheets still get one number.
  - At index turn changes, a blind reader judged 12/20 as different sheets and 7/20 as the same.
  - Whole-scroll fitting is "still work in progress".
- **How they would accept outside evidence:**
  - Append (a, b, d, w) terms in `build_ab_terms` (`solve.py:176-189`). The points need patch ids.
  - Or write `between_patches__A__B` relative collections for the fitter.
  - Our chains slot in directly as extra terms.

## 3. Can our chains assign global winding numbers to these patches?

**Yes, mechanically.** A chain point within tolerance of a patch surface ties that patch to the chain's winding at
that point. If one chain touches patches a and b at winds i and j, then k_b − k_a = j − i. That is the cross-sheet,
radial evidence the patch graph lacks: Kelly's terms come from overlaps along a sheet plus a pitch/CT-crossing
estimate. Chains also flag sheet-switching patches directly, because one chain touching one patch at two different
winds means the patch spans two windings.

Per band:
- ~576 rays × ~32 points = ~18.5k chain points;
- against ~12k patches of about 7k vertices each;
- we expect many patches touched by at least one chain, but the hit rate is **not measured** yet. Step 1 measures it.

**Scoring.** No new labels are needed. We have the PHerc1667 human merged mesh (`data/eval/p1667/labels_merged_v4`,
k 11-41) in the same frame. Per patch, the human k is the mode of the label at the nearest merged-mesh vertex over the
patch's vertices within 2.5 vox. Patches with no match, or with more than 10 % minority k, are reported separately as
"human-ambiguous/switching".
- **M1 (pairwise, offset-free):** for patch pairs tied by a chain, the fraction with Δk(chain) = Δk(human). It is
  directly comparable to our step precision.
- **M2 (global):** after one offset per component, the % of patch area whose k matches human, for:
  - (a) Kelly's index alone;
  - (b) chains only (an integer solve on chain terms plus Kelly's same-sheet overlap terms);
  - (c) Kelly + chains (their LP with our terms appended, or our own solve on their `solve_edges.json`).
- **M3:** the precision and recall of "chain says this patch spans two windings" against the bad set (Stevens' removed
  patches) and against human-ambiguous patches.
- **M4 (no labels, whole scroll):** the Kelly-vs-chain disagreement rate. It also yields a ranked list of disputed
  joins, the human-check target that our HITL result says is worth a human's time.
- Both bands have been seen: z 6500-8000 is the confirmation band, and z 4500-6000 is the ink-lowband. The umbilicus
  came from the human mesh, so that must be disclosed. **Preregister M1-M3 and the tolerances before scoring.**
- Do not use Paris4. Their PCU fits and the z 8500-9000 / z 10800-11800 restrictions don't apply here.

**Data and cost.**
- `s4_good_patches.zip` is 1.58 GiB. That is under the 2 GB line but large, so I am **reporting before downloading**.
- `s4_bad_patches.zip` (0.33 GiB) is needed for M3. Together they come to **~1.9 GiB (2.03 GB), just over the line,
  so this needs approval**.
- `s4_10_components_tifxyz.zip` (1.21 GiB) is optional and a later step: number Stevens' prize surfaces.
- Kelly's `same_windings.json` (334 MB) is not needed. `solve_edges.json` (already local) has the joins and centroids.
- We only extract patches in band, about 24k patches, ~0.8 GB unzipped.
- Compute is CPU only:
  - a KD-tree per ray neighbourhood over patch vertices;
  - an integer solve over 12k nodes per band takes seconds (Kelly's whole-scroll LP: 87 s on 4 CPUs).
  - Estimate: under 2 h CPU, **$0**, no GPU, no pod.
- Our chains for both bands already exist (`data/scratch/auto_windings/p1667/band_m7_p1667_ray_c`,
  `data/scratch/ink_lowband/autowind_z4500`), so no new chain run is needed.
- About one working day of agent time.

**Risks.**
- Chain-to-patch hit rate could be low, since rays are 15° apart and patches about 3 mm wide.
- Chain step errors (step precision 95.8 % on PHerc1667 per release/TECHNICAL_REPORT.md, so ≈ 4 % of steps wrong) feed directly into M1. Mitigate with `--split-perr` and votes across
  chains/z-offsets.
- The human mesh is one merged read (81 % single-wrap agreement in its own build).
- The outermost windings, where Kelly fails, may be where our chains are also weakest (dense packing).

## Recommended next step

Approve **downloading `s4_good_patches.zip` + `s4_bad_patches.zip` (~2.03 GB total)** into
`data/untrusted/stevens_s4_patches/`. Then:
1. Write the scoring prereg.
2. Run a **hit-rate pilot** on band z 6500-8000. Go only if at least 30 % of in-band patches are touched by at least
   one chain point.
3. If go, run M1-M4 on both bands, with Kelly's index as baseline.

Cost: $0 compute, about 1 day. If M2(c) beats M2(a), the submission angle is "machine winding numbers for the community
patch set, plus a ranked dispute list for human checking". It plugs into both Kelly's solver and the fitter in the
format already confirmed in notes/2026-10-26-chain-export.md.

## Outcome (added 2026-10-26)

The download was approved and done, and scoring ran under notes/2026-10-26-patch-numbering-prereg.md:
- the gate passed (73 % coverage);
- "chains improve Kelly" came out **mixed**: B2 +0.83 pp significant, B1 +0.18 pp not;
- chains are worse than Kelly pairwise;
- the chain sheet-switch flag is useless.

See that note for the table.
