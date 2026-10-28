# 2026-10-27: Does HercuViT (sheet-instance segmentation) separate touching sheets better than m7? Pre-set gate

**Gate frozen: 2026-10-26T23:05Z (UTC).** Sections 0-4 were written after scouting HercuViT (code, checkpoint,
licence, training data), installing it in its own venv, and running it on the author's demo region (PHerc1447,
not one of ours). They were also written after a driver check on that demo region and after counting the windows
the dev rays need (label ray geometry only). Before this time no HercuViT output was computed on any dev region,
and no mask was sampled along a ray. Nothing in sections 0-4 changes after this time. The log and results are
append-only. Run directory: `data/scratch/hercuvit_gate/`. Untrusted code and weights: `data/untrusted/hercuvit/`.

## 0. Question
Our chains undercount at tight packing because m7 fuses touching sheets or misses one of them. At Paris4 contacts
m7 resolves 17.6 % of touching pairs, fuses 30.8 % and misses a sheet in 51.7 %
(notes/2026-10-26-surface-model-gate.md). No other surface mask passed that gate. HercuViT is a new community
network that predicts sheet **instances**: which voxels belong to which sheet. That lets a model keep two sheets
apart even when their voxels touch. Question: does HercuViT resolve touching sheet pairs more often than m7, on
the same rays and with the same metric? A second, instance-aware reading uses its instance labels.

## 1. What HercuViT is (scouted before the gate)
- **Code:** github.com/jimmylomro/hercUNet, branch `hercuvit`, commit `de4acb7a5f1e4282adbca21d9a34c0a523041451`
  (2026-10-26). Author JaimeLomRo (HF `jimmylomro`). MIT licence.
- **Checkpoint:** HF `jimmylomro/hercuvit-v0`, revision `b490c58514d577129ddadd4948abfe49fd737ef5`.
  - `checkpoint_latest.pth`: sha256 `6f7593b4…034b`, which matches the Hub LFS oid. It is `train_v5` at epoch 190
    (global step 241,806), 31.1 M params, MIT.
  - The author updates this repo in place as better checkpoints come out. This gate is about this revision only.
- **Architecture:**
  - Encoder: the Dinovol ViT `scrollprize/dinovol_ps8_9.6um_101626` (width 288, 10 blocks, patch 8), fine-tuned.
  - CNN pixel decoder with UNet stem skips, giving 32-d per-voxel mask features.
  - Decoupled 2-class surface head.
  - Mask2Former-style decoder: 64 queries, 6 layers, per-voxel ownership = argmax over the queries.
  - Post-processing: merge mutually confused masks (> 1/3 both ways), prune small low-confidence masks, voxel
    floor, then one quad-mesh fit per sheet into an `.asv` SQLite store.
  - Disjoint 128³ windows. No stitching across windows.
- **Training data** (HF dataset `jimmylomro/hercunet-corpus`, revision 8314b273; `submission/corpus/*.txt`):
  - 4,031 HercuLabels pseudo-label windows from 22 scrolls: Scroll 5 (id "5") plus PHerc0125, 0175A, 0175B, 0191,
    0211, 0257, 0268, 0306B, 0343, 0358, 0483A, 0483B, 0490A, 0490B, 0800, 0813, 0826, 0846B, 1218, 1447, 1545.
  - 100 mined "m7 rehearsal" windows (coherent m7 predictions) from 20 of the same scrolls.
  - No human meshes. **None of PHercParis4, PHerc0139 or PHerc1667 is in the decoder's training windows.**
- **In-sample status per dev region:**
  - Decoder and heads: out-of-sample on all three dev scrolls (no windows, no labels from them).
  - Encoder: Dinovol was self-supervised on 40 volumes. Those include the Paris4 and PHerc1667 2.4 µm scans at
    pyramid level 2 (the exact volumes and level we run here) and the PHerc0139 2.399 µm scan at level 2 (a sister
    scan of our 9.362 µm volume). Dinovol's pretraining also used a pseudo-object loss on surface-m7 predictions
    (notes/2026-10-24-dinovol-probe-gate.md).
  - So all three regions are CT-in-sample for the encoder (unlabelled), and label-out-of-sample for the whole
    model. This exposure is the same for all three regions and for the control, so it does not favour one of them.
    The Paris4 human meshes may have been traced on m7-family predictions. If that matters, it favours m7.
- **Input scale:** 128³ windows at about 9.6 µm. Training resampled to 9.6 µm, and the engine warns outside
  7.5-10 µm. Our grids (Paris4 / PHerc1667 CT L2 at 9.6 µm, PHerc0139 L0 at 9.362 µm) are all in that band, so
  no resampling is done. Normalisation: per-window robust normalisation (1-99 % clip, median / MAD), as Dinovol.
- **Target:** the author states it predicts the sheet **medial crest**, a few voxels inboard of the faces that
  manual labels mark. Our labels are traced meshes, so some face offset is expected. Matching tolerance stays at
  3 voxels, as in the earlier gate, and the offset is reported.
- **VRAM:** 5.7 GiB peak with the instance head at batch 4 (measured on the demo region).
- **Safety:**
  - The package's loader calls `torch.load(..., weights_only=False)` (`hercuvit/infer/loader.py`).
  - The checkpoint loads with `weights_only=True`: it holds only tensors, dicts and primitives
    (`inspect_ckpt.py`). Our driver loads it that way.
  - Every HercuViT process runs inside `unshare -rn` (no network), under `env -i` (no tokens, no AWS or HF
    variables, sandbox HOME), with `HF_HUB_OFFLINE=1` (CLI) or `python -I` (our driver).
  - CT is staged first by our own script (`stage_ct.py`, project env): chunk objects of the needed windows only.
    HercuViT never fetches anything.

### Demo region (done before the gate; not a dev region)
- Command: the author's demo command on PHerc1447 `11618:11810,3197:3389,3552:3744`, with `--mesh local --surf off
  --overlap 0`. The 27 covering CT chunks (128³) were staged as a local `file://` zarr, and the checkpoint came
  from the local copy instead of `--model-hf`.
- Result: 27 windows (3x3x3 grid), **274 sheet meshes**, 9 s.
- `demo_check.png` (CT slice, instance overlay, meshes): sheets follow the CT layers, and the per-window colours
  break at window seams, as the author's README figure shows. This counts as reproduced roughly. PHerc1447 is in
  HercuViT's training set.
- **Driver check:** our `hvit_infer.py` uses the package's own normalise, mask-logit, merge and prune functions,
  with a `weights_only=True` load. On the same 27 windows its surface was **bit-identical** to the CLI's
  `--surf local` output (max |Δ| 0, `check_demo.py`). It kept 274 instances, the same count as the CLI's 274 meshes.

## 2. Measurement (dev rays only; same rays as the surface-model gate)
- **Regions, k sets, ray subset:** identical to notes/2026-10-26-surface-model-gate.md section 2.
  - Paris4 z 7000-8500 (z levels 7072-8352), PHerc0139 band 1, PHerc0139 band 2 (reported only), and PHerc1667
    (control).
  - Every 4th z level starting at the 2nd, all 24 angles.
  - m7's per-ray results are the earlier gate's `data/scratch/surf_gate/out/<region>_m7.json`, used unchanged.
- **Windows** (`stage_ct.py`):
  - Per gate level z0: disjoint 128³ windows with z range [z0-64, z0+64) on two xy grids. Grid A has origins at
    multiples of 128; grid B is A shifted by +64 in y and x.
  - A grid's windows are the tiles a dev ray crosses (ray ± 40 vox along r, ± 6 vox sideways).
  - Counts: A / B windows are Paris4 3,162 / 3,184, b1 1,317 / 1,303, b2 971 / 977, PHerc1667 1,385 / 1,358.
  - Outputs are kept for z0 ± 8. The highest Paris4 window ends at z 8416.
- **Inference per window** (`hvit_infer.py`, the model's shipped settings):
  - No TTA, fp16 autocast.
  - Surface p = softmax(sem)[1], stored as uint8 round(255 p).
  - Instances: owner = argmax over the queries; merge at 1/3; gate p > 0.4; prune at mass 0.10 / conf 0.70; drop
    instances with fewer than 2,097 gated voxels (the mesh fit's floor).
  - Windows with CT max < 25 are skipped, as the engine does.
- **Stitching the two grids (geometry only, never by result):**
  - A voxel takes its value from the grid in which it lies farther from its window's xy border (Chebyshev
    distance; ties go to A). If that grid has no window there, the voxel comes from the other grid.
  - This keeps every scored voxel at least about 16-32 voxels from a window side. z is centred: rays sit at the
    window's middle slice.
- **Standard reading (S), the same code as the earlier gate:**
  - `measure.py` is imported unchanged; a wrapper `measure_hvit.py` registers the new masks and writes into this
    run directory.
  - Mask `hvit` = stitched surface, uint8 >= 102 (p >= ~0.4, the model's own gate).
  - Mask `hvit_inst` (secondary, reported only) = stitched kept-instance voxels.
  - Contacts (Δk 1, spacing < 8 vox) are resolved, fused or missed. Also: sheet recall, extra runs per label,
    face offset, and resolved over all Δk 1 pairs.
- **Instance-aware reading (I), new code `measure_inst.py`:**
  - Each ray is sampled every 0.5 vox, as in measure.py. At each sample, the instance key is taken from the first
    voxel with a kept instance among the centre and its 6 neighbours (centre first). The key is
    (grid, window, owner id).
  - A label crossing's instance in grid g is the key of the nearest sample within 3 vox of the crossing that has a
    key in grid g. No such sample means no instance.
  - **Pairs:** each Δk 1 pair uses the grid the stitch rule picks at the pair's midpoint, and both crossings are
    read in that grid.
    - **resolved**: both have instances, in the same window, with different ids.
    - **fused**: same window, same id.
    - **missed**: either crossing has no instance.
    - **boundary**: the two instances are in different windows of that grid. Window ids are not comparable, so
      the pair cannot be scored.
  - Boundary pairs are dropped from **both** HercuViT and m7 before any rate is computed, so the comparison stays
    on the same pairs. Their count is reported, plus a sensitivity reading that scores them as missed.
  - **Instance recall:** share of label crossings (k in the set) that have an instance, read in the stitch-rule
    grid at the crossing.
  - **Instance extras:** runs of consecutive samples with the same key. A new run starts at a key change or after a
    gap of more than 5 samples. Runs touching the segment ends are dropped. A run is extra if its centre lies
    between the first and last labelled crossing and no labelled crossing is within 3 vox of its interval. The
    count is per labelled crossing.
  - The m7 baseline for reading I is m7's standard categories on the same pairs. A mask's "instances" along a ray
    are its runs.
- **CIs:** cluster bootstrap over rays, 2,000 resamples, seed 0. Δ = HercuViT − m7, paired on the same rays and
  pairs.

## 3. GATE (fixed now)
**Reading I (primary; the question asked).** HercuViT passes iff, in **both** Paris4 z7000 and PHerc0139 b1:
1. instance-resolved at contacts is at least m7 resolved + **10 pp**, with paired 95 % CI lower bound > 0;
2. instance-fused at contacts is lower than m7 fused (point estimate);
3. instance recall is no lower than m7 recall − 2 pp;
4. instance extras are no more than m7 extra runs + 0.05 per labelled crossing;
5. boundary pairs are at most 10 % of that region's contacts. Above that, the region cannot be assessed, and that
   counts as a fail.

And on **PHerc1667 (control)**, for instance recall and instance-resolved over all Δk 1 pairs (any spacing): the
point Δ is at least −2 pp **and** the paired 95 % CI upper bound is > 0 (no significant loss).

**Reading S (drop-in mask swap).** The earlier gate's items 1-4, applied to mask `hvit` in both Paris4 and b1,
with the same PHerc1667 control rule as above (recall and all-pair resolved).

PHerc0139 b2 is reported, not gated. `hvit_inst` under S, the CI on the m7-fused subset, and the 8-12 / >= 12
spacing rates are reported only.

**Reading the result:**
- **PASS-I:** HercuViT's instances keep touching sheets apart better than m7, across scrolls. Using them in
  autowind would need chains built from instances (section 5 describes what that takes). No fits here.
- **PASS-S:** its surface mask could replace m7 in place (mask store, tracks and surf-SDT all from the same
  source; test 11's lesson). No fits here.
- **Both fail:** HercuViT v0 at this revision does not beat m7 at contacts by a margin that matters. Stop, and
  recheck only if a newer checkpoint lands, under a new note.
- A Paris4-only gain is reported as such and does not pass.

## 4. Rules and budget
- **Reserved regions:** Paris4 z 8500-9000 (blinded for the human session) and Paris4 z 10800-11800 (reserved for
  the confirmation test) are not used in any way. `stage_ct.py` asserts every Paris4 window ends below z 8500.
- **Compute:** $0, local 3090 only, lease `~/.gpu-leases/vss-hercuvit.json` (8 GB), no pods. Jobs over about
  6 GB run under `flock /tmp/vss_eval.lock` unless the GPU has more than 20 GB free, which is logged.
- **Data:** crops only. CT chunks are staged under `data/scratch/hercuvit_gate/ct/`, about 44 GB in total, and
  deleted after scoring. CC BY-NC data: nothing is published, posted or committed.
- **Frozen files untouched:** measure.py is imported read-only. tools/auto_windings.py, eval/p0139_graph.py,
  tools/autowind/postproc.py and tools/check_viewer* are not touched.
- **Our readers:** scripts that read HercuViT outputs run with `python -I` in the project env.

---

## Log (append-only)
### L1 (2026-10-26T23:06-23:43Z): inference on the dev windows
- `run_hvit.sh`, offline under `unshare -rn`, `env -i`, `python -I`, eval lock. GPU free 22.3 GB at each start
  (`mem_log.txt`). Peak 5.7 GiB, about 6.2 windows/s, 13,657 windows in 37 min. 2 windows were skipped (air or
  off-volume).
- About 33.5 k kept instances on Paris4 per grid, roughly 10.5 per window.
- Stitch: the per-level A / B shares are balanced (about 0.2 each of the bounding box; the rest of the box has no
  window).

### L2 (2026-10-26T23:30Z, post hoc, not gated): a false-split check, added after the Paris4 reading-I extras
After the Paris4 reading I showed 0.57 extra instance runs per label (m7: 0.09), `diag_extra.py` looked at those runs:
- median length 4 vox, median 5.8 vox from the nearest label;
- 20 % share an instance key with another run within 8 vox.

They look like instance fragments inside the thick surface band, not shattered noise far from sheets. To ask
whether reading I's contact gain is real separation or this fragmentation, `diag_null.py` places two pseudo
crossings at r ± d/2 around **isolated** labelled sheets:
- isolated = both labelled neighbours have Δk 1 and are >= 12 vox away;
- d = that region's median contact spacing.

It reads them exactly as real crossings and counts **split** = both read, but different instances or runs. Only one
sheet is there, so any split is false. This check was designed after seeing Paris4 and is reported only.

## Results (2026-10-26T23:50Z): GATE FAILED on both readings. m7 stays.
Gate subset, 2,000 cluster-bootstrap resamples over rays. Δ = HercuViT − m7 on the same rays with paired 95 % CI.
Contacts = Δk 1 label pairs < 8 vox apart. "res all" = resolved over every Δk 1 pair. Out files:
`data/scratch/hercuvit_gate/out/summary.json` (S) and `summary_inst.json` (I). m7 rows are the earlier gate's.

**Reading S (standard metric; mask `hvit` = surface p >= 0.4, stitched)**

| region | mask | contacts | resolved % (Δ [CI]) | fused % | missed % | recall % (Δ [CI]) | extra / label (Δ) | offset | res all % (Δ [CI]) |
|---|---|---|---|---|---|---|---|---|---|
| Paris4 z7000 | m7 | 1088 | 17.6 | 30.8 | 51.7 | 78.9 | 0.092 | -0.26 | 58.5 |
| | hvit | | 25.3 (**+7.7** [+4.9, +10.7]) | **52.0** | 22.7 | 85.8 (+6.9 [+5.9, +7.9]) | 0.167 (**+0.074**) | +1.55 | 56.9 (-1.6 [-3.0, -0.1]) |
| PHerc0139 b1 | m7 | 261 | 24.1 | 51.0 | 24.9 | 88.3 | 0.193 | +0.09 | 74.0 |
| | hvit | | 23.8 (**-0.4** [-5.9, +5.2]) | **63.2** | 13.0 | 91.5 (+3.2 [+1.8, +4.7]) | 0.328 (**+0.134**) | +2.06 | 70.7 (-3.3 [-5.8, -0.7]) |
| PHerc0139 b2 (not gated) | m7 | 64 | 14.1 | 50.0 | 35.9 | 82.9 | 0.180 | +0.12 | 67.5 |
| | hvit | | 14.1 (+0.0 [-10.6, +11.1]) | 65.6 | 20.3 | 88.2 (+5.3) | 0.272 (+0.092) | +1.97 | 69.5 (+1.9) |
| PHerc1667 (control) | m7 | 42 | 54.8 | 33.3 | 11.9 | 91.9 | 0.091 | -0.06 | 86.8 |
| | hvit | | 19.0 (-35.7 [-52.5, -20.8]) | 69.0 | 11.9 | 84.0 (**-7.9** [-9.5, -6.4]) | 0.442 (+0.351) | +2.16 | 66.2 (**-20.7** [-23.3, -18.1]) |

`hvit_inst` (kept-instance voxels as the mask) is within 0.5 pp of `hvit` everywhere (out/summary.json).

**Reading I (instance-aware, primary)**

| region | contacts (boundary) | I-resolved % vs m7 (Δ [CI]) | I-fused vs m7 | I-missed vs m7 | I-recall vs m7 (Δ [CI]) | I-extra vs m7 runs | res all vs m7 (Δ [CI]) |
|---|---|---|---|---|---|---|---|
| Paris4 z7000 | 1088 (5) | 46.4 vs 17.5 (+28.8 [+24.9, +32.6]) | 29.5 vs 30.8 | 24.2 vs 51.6 | 85.2 vs 78.9 (+6.3 [+5.3, +7.3]) | **0.566 vs 0.092** | 64.1 vs 58.2 (+5.9 [+4.2, +7.6]) |
| PHerc0139 b1 | 261 (1) | 48.5 vs 24.2 (+24.2 [+17.3, +30.7]) | 39.2 vs 50.8 | 12.3 vs 25.0 | 91.1 vs 88.3 (+2.8 [+1.3, +4.3]) | **0.854 vs 0.193** | 75.8 vs 73.4 (+2.4 [-0.1, +5.0]) |
| PHerc0139 b2 (not gated) | 64 (0) | 40.6 vs 14.1 (+26.6 [+11.1, +42.4]) | 37.5 vs 50.0 | 21.9 vs 35.9 | 88.1 vs 82.9 (+5.2) | 0.743 vs 0.180 | 74.4 vs 67.8 (+6.6) |
| PHerc1667 (control) | 42 (1) | 51.2 vs 53.7 (-2.4 [-16.7, +12.2]) | 36.6 vs 34.1 | 12.2 vs 12.2 | 83.6 vs 91.9 (**-8.2** [-9.8, -6.6]) | 1.252 vs 0.091 | 69.6 vs 86.7 (**-17.0** [-19.8, -14.2]) |

- Boundary-as-missed sensitivity at contacts: Paris4 +28.6 [+24.7, +32.5], b1 +24.1 [+17.2, +30.6]. This is
  unchanged; boundary pairs are at most 0.5 % of contacts.
- Over all pairs, boundary pairs are 2-5 %. Those pairs are wide, not contacts.

**Gate items failed** (bold above):
- **Reading I:**
  - Item 4 (extras) fails in both tight regions: Paris4 0.566 > 0.142; b1 0.854 > 0.243.
  - The PHerc1667 control fails on recall (-8.2 pp, CI excludes 0) and on all-pair resolved (-17.0 pp, CI excludes 0).
  - Items 1, 2, 3 and 5 pass in both tight regions.
- **Reading S:**
  - Paris4: item 1 (+7.7 < +10), item 2 (fused 52.0 vs 30.8) and item 4 (+0.074 > +0.05) fail.
  - b1: item 1 (-0.4, CI includes 0), item 2 (63.2 vs 51.0) and item 4 (+0.134) fail.
  - The control fails: recall -7.9, all-pair resolved -20.7.

**Post-hoc false-split check (L2; not gated)**

Split / same / missing % at isolated single sheets, pseudo pair d apart:

| region (d vox, n) | instance reading | m7 runs | hvit runs |
|---|---|---|---|
| Paris4 (6.78, 7,307) | **28.2** / 34.9 / 36.9 | 3.2 / 59.7 / 37.1 | 13.9 / 50.1 / 36.0 |
| PHerc0139 b1 (5.43, 1,326) | **23.2** / 45.9 / 31.0 | 0.6 / 79.5 / 19.9 | 8.7 / 61.2 / 30.0 |
| PHerc0139 b2 (6.02, 654) | **25.2** / 43.0 / 31.8 | 2.1 / 71.9 / 26.0 | 8.7 / 59.6 / 31.7 |
| PHerc1667 (7.07, 2,624) | **31.7** / 26.5 / 41.8 | 0.7 / 59.9 / 39.4 | 12.2 / 47.0 / 40.8 |

### What the numbers say (post hoc, no verdict attached)
- **The instance reading's contact gain is mostly fragmentation, not separation.**
  - A single isolated sheet read at two points a contact-width apart comes out as two different instances 23-32 %
    of the time. m7 does that 0.6-3 % of the time.
  - Look only at pairs where both points were read. Real contacts come out "resolved" 61 % (Paris4) and 55 % (b1)
    of the time. Single sheets come out "split" 45 % and 34 % of the time.
  - The excess over this false-split baseline is about +16 pp (Paris4) and +21 pp (b1), conditional on both read.
    That is some real identity signal, but well below the headline +24-29 pp. It comes with 0.6-0.9 extra instance
    runs per label, which a chain would read as extra windings.
- **As a surface mask, HercuViT finds more sheets but fuses more.**
  - At contacts it cuts misses (Paris4 51.7 -> 22.7 %, b1 24.9 -> 13.0 %) but fuses far more (30.8 -> 52.0 %,
    51.0 -> 63.2 %). This is the same trade P4-2um L2 and F0 made in the earlier gate.
  - The band is thick: about 29 % of the voxels on the demo region pass p >= 0.4, and the figure shows blobby, broken
    bands.
  - The face offset is +1.5 to +2.2 vox (outward) against the labels. m7's is within ±0.3.
- **The wide-spacing control gets worse:** PHerc1667 recall falls by 8 pp and all-pair resolved falls by 17-21 pp on
  both readings. The wide-spacing regime is where the chain already works, so HercuViT v0 would cost windings there.
- **Leakage does not explain any of this.** The decoder never saw these scrolls. The encoder saw all three (CT
  only), equally.
- This is one checkpoint (train_v5 epoch 190, revision b490c585), and the author says a more converged one is
  coming. The instance head's fragmentation (own-sheet accuracy 0.65 per the write-up) is the thing to recheck on a
  new revision. The cheap check: rerun `diag_null.py` and reading I on the same windows (37 min of GPU), under a new
  note.

### What integration would have needed (not proposed; the gate failed)
- **Reading I path:** autowind builds chains from mask crossings along rays. Using instances would mean:
  - Replace "consecutive runs" with "consecutive distinct instances" in the step builder.
  - Resolve instance identity across window seams. HercuViT has no cross-window stitching; the two-grid
    overlap used here is one way, with IoU matching of instances in the overlap.
  - Derive the kimimaro tracks and the surf-SDT from the same instance-gated mask, so every positional input moves
    together (test 11's lesson).
  - Store the per-window instances (or the `.asv` meshes) as the mask source.
  - It would need a fix for the fragmentation first: merge instances that share a sheet across the band.
    Otherwise the extras turn into overcounted windings.
- **Reading S path:** `autowind chains --mask <stitched surf_S store>`, with tracks and surf-SDT from the same store.
  The pipeline grid already matches (9.6 / 9.362 µm), so no resampling is needed. It failed its own items, so this
  is moot.

### Figure
`release/figures/hercuvit_vs_m7_p4z7_z7584.png` (`figure.py`). The crop is the 256² window at Paris4 z7584 with the
most labelled contacts (13), chosen from label geometry only. It shows:
- m7's thin, regular bands;
- HercuViT's thicker, broken surface;
- HercuViT's instance colours, which change along a single sheet inside a window. This is the fragmentation the
  false-split check measures.

Not for publication (CC BY-NC data).

### Spend and housekeeping
- **$0.** No pods. Local 3090 under lease `~/.gpu-leases/vss-hercuvit.json` (8 GB), released at 23:44Z. About 40 min
  of GPU time: 37 min on dev windows, under 1 min on the demo.
- **Downloads:**
  - Code: git clone (commit de4acb7).
  - Checkpoint: 333 MB, sha256 6f7593b4…034b = Hub LFS oid, revision b490c585.
  - venv: 6.2 GB under `data/untrusted/hercuvit/`.
  - CT: 22,668 chunk objects (about 44 GB), only the chunks the windows touched. Deleted after scoring.
  - Kept: the sparse prediction zarrs `data/scratch/hercuvit_gate/pred/` (8 GB) and `out/`.
- **Safety:**
  - No HercuViT process had network access or secrets.
  - Our driver used `weights_only=True`. The CLI demo runs used the package loader (`weights_only=False`) on a
    checkpoint already verified to load with `weights_only=True`.
- **Frozen files untouched:** `data/scratch/surf_gate/measure.py` was imported read-only (sha256 a6598fb9…085f).
  `tools/` and `eval/` were not touched.
- **Reserved regions:** none of Paris4 z 8500-9000 or z 10800-11800 was read. The highest Paris4 window ends at z 8416.
- **Scripts:** `data/scratch/hercuvit_gate/` holds:
  - `stage_ct.py`, `hvit_infer.py` and `run_hvit.sh`: staging and inference;
  - `stitch.py`, `measure_hvit.py` and `measure_inst.py`: stitching and the two readings;
  - `diag_extra.py` and `diag_null.py`: the post-hoc checks;
  - `figure.py`;
  - `inspect_ckpt.py`, `check_demo.py` and `render_demo.py`: the checkpoint and demo checks.

  Logs are in `logs/`.
