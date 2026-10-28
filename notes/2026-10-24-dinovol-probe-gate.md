# dinovol frozen-feature probe for merged sheet contacts: pre-set gate

**Gate frozen: 2026-10-24T20:54Z (UTC)**, written before any dinovol feature was extracted at a contact site and
before any probe was fitted. Done before this point: code checkout, checkpoint download and config inspection, a model
load / speed self-test on random input, and the export of the step subset (endpoints, labels, old features; no new
features). Nothing in the Gate section changes after this timestamp; later sections are append-only and timestamped.

Question: do frozen, pretrained dinovol 3D features separate merged sheet contacts better than our current label-free
signals? This decides whether fine-tuning dinovol for contact detection becomes the next model bet.

Background: notes/2026-10-24-contact-detector-gate.md (old detector F; LORO held-out raise precision 41.5 / 31.6 / 35.9 %
at recall 25.5 / 30.4 / 62.4 % on b1 / b2 / Paris4; gate failed).

## Checkpoint (chosen before any contact-site feature)

- Code: ScrollPrize/villa `dinovol/` at commit 322cb0899ac9efe2588d5a3395dce6dc2dea3c89 (sparse checkout,
  data/untrusted/dinovol/villa/src). Own venv data/untrusted/dinovol/venv (torch 2.14.1+cu130, timm 1.0.20), run with
  `python -I`. Weights loaded with `torch.load(weights_only=True)` (worked; no full unpickling needed).
- Candidates inspected (both downloaded, sha256 = Hub LFS oid, embedded configs read):
  - `scrollprize/dinovol_ps8_9.6um_101626/teacher_305k.pt` (sha256 8753064…dd77): width 288, 10 blocks, 6 heads,
    16-ch CNN stem, patch 8, global/local crops 128³/64³, step 305k.
  - `scrollprize/dinovol_8um_ps192_102126/teacher_step_640000.pt` (sha256
    2354c493f050070e17ea0cda5cd3c1a6ebdde19a70a9c28bc9d368b80faabb86): width 480, 10 blocks, 10 heads, 32-ch CNN stem
    (the README's current `pretrain_8_9um.json` recipe), patch 8, global/local crops 192³/96³, step 640k.
- Resolution: **both embed the same 40-volume training list at the same voxel sizes** (7.91-9.604 µm: ESRF 8.64 /
  9.362 µm volumes at scale 0; 2.4 µm volumes incl. PHercParis4 and PHerc1667 at pyramid level 2 = 9.6 µm; PHerc0139
  from its 2.399 µm scan at level 2 = 9.596 µm) and the same zoom range (global crop scale 0.18-0.45 of the source
  window, i.e. 0.84-1.15x resampling). The "8um" / "9.6um" names do not reflect a resolution difference. Our CT:
  PHerc0139 9.362 µm (113 keV, level 0; a sister scan of the same scroll; other 9.362 µm 113 keV scrolls are in the
  training list), PHercParis4 and PHerc1667 at level 2 = 9.6 µm (the exact training volumes and level).
- **Chosen: `dinovol_8um_ps192_102126`, step 640000, sha256 2354c493…b86.** Resolution is a tie, so the tie is broken
  by: current recipe, larger model, 2x the training steps (the likely starting point of any fine-tune), and a wider
  native context (192³ = ~1.8 mm, ~9 sheet pitches).
- Training-data note: both models are self-supervised on these same scrolls (no winding labels), and the pretraining
  includes a pseudo-object loss on the surface-m7 prediction, the same family of mask our chains come from. No label
  leakage, but the features may partly re-encode the mask.

## Steps (dev regions only; z 7000-8500 of Paris4 stays untouched)

Pre-registered subset (data/scratch/dino_probe/export_steps.py -> steps_<region>.npz): labelled single steps
(d0 = 1, both ends labelled, k < kmax; primary label variant), same chains / labels as loro.py, at these z levels
(chosen in 4-level groups that share CT chunk rows, with >= ~200 positives per tight region; the counts of labelled
steps per z level were looked at, nothing else):

| region | z levels | steps | true diff == 2 |
|---|---|---|---|
| PHerc0139 b1 | 3296-3488, 4064-4256 (8) | 3,319 | 259 |
| PHerc0139 b2 | 2208-2912 (12) | 2,778 | 222 |
| Paris4 z9000 | 9504-9696 (4) | 3,393 | 563 |
| PHerc1667 (control) | 7264-7456 (4) | 1,422 | 12 |

Event y = true winding difference == 2 (as in the old gate, L2). CT streamed only as 128³ chunks touched by the
crops (~8,100 chunks, ~17 GB, cached in data/scratch/contact_det/ctcache/).

## Features

- **DINO (primary arm).** Per step a -> b: one 192³ crop centred at round((a+b)/2), zero outside the volume,
  robust normalisation as in pretraining (per crop, 1-99 % clip, median / MAD), EMA teacher backbone, bf16 autocast.
  Points p_t = a + t (b - a), t in {0, .25, .5, .75, 1}.
  - tokens: final-layer `x_norm_patchtokens` (480-d), trilinear at p_t (dinovol's `sample_grid_points`):
    f_mid = mean over t in {.25, .5, .75}, f_end = mean over t in {0, 1}.
  - dense: the checkpoint's dense head object embedding ("dense", 64-d, L2-normalised; trained to separate sheets as
    objects) at p_t: e_mid = mean over interior t, plus the 10 pairwise cosines among the 5 points.
  - feature vector = [f_mid, f_end, e_mid, 10 cosines] = 1,034 dims.
- **OLD (baseline arm).** The old final set F (10 features: logI, logIr, L_pitch, logperr, rlmax, split, m_gapmin,
  g_max, nb_d2, nb_cnt), same steps.
- Secondary arms (reported, not gated): DINO+OLD (concatenation); TOK only ([f_mid, f_end]); DENSE only
  ([e_mid, cosines]).

## Probe (identical code for every arm)

StandardScaler + L2 logistic regression (sklearn, lbfgs), training samples weighted so each training region has equal
total weight. C chosen from the fixed grid {1e-4, 1e-3, 1e-2, 1e-1, 1} by inner CV **on the training regions only**:
fit on one training region, score average precision on the other, both directions, mean; refit on both training
regions with the best C. LORO over the three tight regions (fit on two, score the third). Same code for OLD.

## Metrics

- **Primary: held-out precision at matched recall.** Recall targets = the old detector's held-out recall per region:
  b1 25.5 %, b2 30.4 %, Paris4 62.4 %. Threshold = the score of the ceil(target x n_pos)-th highest-scoring positive
  in the held-out region; precision = share of steps with score >= threshold that are true-2. Both arms at the same
  recall.
- CIs: cluster bootstrap over chains of the held-out region (2,000 resamples, seed 0), threshold recomputed per resample
  for each arm, 95 % percentile CI for each arm's precision and for the paired difference DINO - OLD.
- Secondary: held-out AUC and average precision; old-detector reference numbers on the full regions.
- **Control (PHerc1667).** Each fold's DINO model with a threshold fixed on its training regions only: the score at
  which recall over the two training regions (region-equal weights) is 30 %. Raised fraction and spurious raises
  (raised steps with true diff != 2) as a share of PHerc1667 labelled single steps. Same for OLD (reported).

## GATE (primary arm DINO)

Pass iff both:
1. In **at least 2 of the 3** held-out tight regions: DINO precision at matched recall **>= 55 %**, AND the paired
   bootstrap 95 % CI of (DINO - OLD) precision has its **lower bound > 0**.
2. Control: under **each** of the 3 fold models, PHerc1667 spurious raises **<= 2.0 %** of its labelled single steps.

If a secondary arm passes while DINO fails, the verdict is still FAIL (noted as a signal needing a fresh
pre-registration). No drift / step-precision gate here (features exist only for labelled single steps of the subset);
this probe answers the ranking question only.

If pass: outline what a fine-tune would need (labels, compute on a pod); do not run it. If fail: write up, stop.

---

## Log (append-only)
### A1 (2026-10-24T21:58Z): addendum after organiser info; before any probe fit or score (primary arm unchanged)
Written after new information from the organisers' Discord thread (via the coordinator) and before any probe was fitted
or any dinovol feature was scored against labels (feature extraction for the frozen arm was running; only an unlabelled
sanity check of p1667 features had been looked at: finite, unit-norm dense embeddings, mean cos(A, B) = -0.31).
The frozen primary arm DINO, its gate and its checkpoint stay exactly as above.

**Organiser info.** `dinovol_ps8_9.6um_101626` (~350k steps, 128³) and `dinovol_8um_ps192_102126` (continued from it
for ~300k more steps, 192³). Training: 3D DINOv3-style crop-level DINO (cls token), a cross-view iBOT, plus
contrastive supervision (weight 0.1; DINO + iBOT = 99.52 % of the loss) from meshes, fibers, ink and surface
predictions: DetCon-style object loss, point-to-point same-object loss, ink prototype loss.
Checked against the embedded configs: both share `output_dir .../pretrain_8_9um_objects_meshes_ink_160k` (consistent
with "continued"); the only config differences are crop / source-window size, width, CNN channels, batch and
object-point sampling rates. **No voxel-size difference exists in the configs** (same 40 volumes, same `pixel_size_um`
and pyramid levels, same 0.84-1.15x zoom range), so the native-spacing rule cannot separate them; per region our CT is
9.362 µm (PHerc0139), 9.6 µm (Paris4), 9.596 µm (PHerc1667), all inside the shared training range. The frozen choice
(ps192) stands; the 9.6 µm checkpoint is added below as a secondary arm.

**Per-voxel output.** The checkpoint has a per-voxel head: the dense head (`PointDenseHead`, a 3D FPN over the 24³
tokens and the 48³/96³/192³ CNN skips, evaluated at arbitrary points) whose 64-d "dense" readout is the object-instance
embedding that the contrastive object losses train (dinovol_2/loss/object_losses.py acts on dense object embeddings).
The frozen primary arm already uses it. So the 8-voxel token grid is not the resolution limit for the dense features;
**no finer-input-scale arm is added.** (Token features stay 8-voxel-coarse: a tight-region gap of ~18-19 vox is ~2
tokens; this is disclosed, not tested further.)

**Leakage (features in-sample, probe not).** Our dev labels are the public human meshes of these scrolls (band oracles
"cut from the human meshes": PHerc0139 w0xx meshes, Paris4 and PHerc1667 public meshes). dinovol's object labels come
from `metadata.json` segments (dinovol_2/dataset/object_labels.py: every segment of the sample with a tifxyz on, or an
affine to, the training volume; ink labels attached). Checked against the public metadata.json (Last-Modified
2026-10-20): every listed segment of the three scrolls is loadable by that code for the training volume: PHerc0139 39
segments (native to the 2.4 µm training scan 20260102150214), PHerc1667 20 (tifxyz-transformed to 20251217075048),
PHercParis4 82 (71 native + 11 transformed). The exact mesh cache used (`/ephemeral/sean/.../mesh_cache_8_9um`) is not
public, so membership of our specific label meshes cannot be confirmed, but overlap is very likely total. Also: the
Paris4 surface-m7 prediction and PHerc0139 / Paris4 fiber predictions were object labels. **Therefore every result
here, pass or fail, is in-sample for the features on all four regions (b1, b2, Paris4, PHerc1667); only the probe is
held out (LORO).** A pass would justify a fine-tune bet but is not a held-out claim; a fail is the stronger statement
(even with the label sheets possibly seen in pretraining, frozen features do not separate contacts).

**New secondary arms (not gated; the gate remains the primary arm DINO).** Defined now, before looking:
1. **Instance-similarity scores, zero fitted parameters** (sign fixed a priori: higher = more likely a hidden / merged
   contact), on the ps192 dense embeddings e (L2-normalised), A = t 0, B = t 1:
   - I1 "third object in the gap" = -max(cos(e_.5, e_A), cos(e_.5, e_B)).
   - I2 = -mean over t in {.25, .5, .75} of max(cos(e_t, e_A), cos(e_t, e_B)).
   - I3 "end self-consistency" = -min(c_A, c_B), with c_X = mean cos(e_X, e_{X+o}) over 8 offsets o along the sheet
     at X: +-8 and +-16 vox in z and along the in-plane tangent v = (0, -u_x, u_y) (u = in-plane ray direction A -> B).
     At radii ~1000-3000 vox, 16 vox is ~0.3-0.9 degrees of the sheet's own continuation.
   - I4 "own continuation vs neighbour" = -min(c_A - cos(e_A, e_B), c_B - cos(e_A, e_B)).
   Same metrics as the primary (held-out precision at the old recall per region, chain bootstrap CIs, paired difference
   vs OLD, AUC, AP). No LORO needed (nothing fitted); control threshold per fold = the 30 %-recall rule on that fold's
   training regions.
   The "two clusters inside a merged mask blob" variant needs mask-blob geometry not in the step tables; not run.
2. **DINO96**: checkpoint `dinovol_ps8_9.6um_101626/teacher_305k.pt` (sha256
   875306424046f69ac2767b1447ea99f9dbc6a9c7d28147d8a03ce0346e00dd77) at its native 128³ crop, identical feature
   recipe (288-d tokens: [f_mid, f_end] + dense e_mid + 10 cosines = 650 dims) through the identical probe, plus its
   I1-I4.
Extraction: a second pass (data/scratch/dino_probe/extract2.py) with both checkpoints saves the 5 along-ray points
plus the 16 offset points; the frozen arm keeps the first-pass features (dino_<region>.npz).

## Result, primary arm (2026-10-24T22:08Z): GATE FAILED

data/scratch/dino_probe/probe.py -> probe_result.json. Features: dino_<region>.npz (pass 1, ps192 step 640000).
Held-out precision (true diff == 2) at the old detector's recall; chain-bootstrap 95 % CIs (2,000 resamples).

| held-out (n / pos / chains) | recall | **DINO** prec [CI] | OLD prec [CI] | DINO - OLD [CI] | AUC DINO / OLD | AP DINO / OLD | C DINO / OLD |
|---|---|---|---|---|---|---|---|
| PHerc0139 b1 (3,319 / 259 / 205) | 25.5 % | **49.6** [42.5, 58.7] | 38.5 [30.5, 50.0] | +11.1 [-1.3, 21.3] | 0.788 / 0.779 | 0.343 / 0.270 | 1e-3 / 1e-2 |
| PHerc0139 b2 (2,778 / 222 / 306) | 30.4 % | **43.6** [34.4, 53.1] | 36.6 [24.9, 52.1] | +7.0 [-6.7, 19.1] | 0.839 / 0.796 | 0.343 / 0.293 | 1e-3 / 1 |
| Paris4 z9000 (3,393 / 563 / 117) | 62.4 % | **39.2** [35.8, 43.9] | 37.5 [33.5, 41.4] | +1.7 [-2.1, 6.3] | 0.792 / 0.766 | 0.374 / 0.361 | 1e-3 / 1 |

Gate item 1 (>= 55 % and CI(DINO - OLD) lower bound > 0 in >= 2 of 3): **0 of 3** (no region reaches 55 %; no
difference CI excludes 0). Item 2 (PHerc1667 spurious raises <= 2.0 % under each fold model, training-calibrated 30 %
recall threshold): **holds**, 0.0 / 0.0 / 0.49 % (OLD: 0.98 / 0.84 / 2.67 %). **Verdict: FAIL.**

Secondary arms of the frozen gate (same probe, same CIs):

| held-out | DINO+OLD | TOK only | DENSE only |
|---|---|---|---|
| b1 | 49.6 [42.9, 58.2], diff +11.1 [-0.4, 21.1] | 33.3, diff -5.2 [-16.9, 3.1] | 44.7, diff +6.2 [-8.2, 15.3] |
| b2 | 43.0 [34.7, 53.0], diff +6.4 [-6.2, 17.8] | 28.8, diff -7.8 [-21.5, 5.0] | 48.6 [33.3, 59.8], diff +12.0 [-6.4, 23.8] |
| Paris4 | 41.7 [37.2, 45.5], diff +4.2 [0.0, 7.8] | 31.6, diff -5.9 [-9.7, -2.1] | 31.7, diff -5.8 [-10.1, -1.9] |

None reaches 55 % anywhere. Reading: frozen dinovol adds a modest, consistent ranking gain over the old label-free
features (AUC +0.01 to +0.04, AP +0.01 to +0.07, precision +2 to +11 pp at matched recall, all CIs touching 0), and it
is much quieter on the wide-packing control (p1667 raises 0-0.7 % vs OLD 1-2.9 %). Most of the signal is in the
dense-head object embeddings plus their cosines; the final-layer patch tokens alone are worse than OLD. Held-out
precision stays near 40-50 %, the same ceiling the old work found (~50 %), far from the >= 70 % the count rule needs.
All of this is in-sample for the features (addendum A1: our label meshes were very likely dinovol object labels).
Secondary arms of A1 (instance scores I1-I4, DINO96) follow when pass 2 finishes.

### Result, A1 secondary arms (2026-10-24T23:13Z)
data/scratch/dino_probe/probe2.py -> probe2_result.json (pass-2 features dino2_{ps192,c96}_<region>.npz; the ps192
along-ray features reproduce pass 1 to bf16 rounding, max |diff| 0.002). Held-out precision at the old recall [CI];
diff vs OLD CI; AUC; PHerc1667 spurious raises at the training-calibrated 30 %-recall threshold (max over folds).

| arm | b1 (R 25.5) | b2 (R 30.4) | Paris4 (R 62.4) | p1667 spurious (max) |
|---|---|---|---|---|
| OLD | 38.5 [30.5, 50.0], AUC .779 | 36.6 [24.9, 52.1], .796 | 37.5 [33.5, 41.4], .766 | 2.67 % |
| I1 ps192 (0 params) | 30.7, d [-22.0, 3.9], .672 | 34.2, d [-16.1, 11.0], .742 | 37.5, d [-4.3, 3.3], .761 | 0.49 % |
| I2 ps192 (0 params) | 27.9, d [-22.7, -2.0], .671 | 33.7, d [-16.6, 8.7], .768 | 39.2, d [-3.3, 6.6], .786 | 0.91 % |
| I3 ps192 | 11.1, .543 | 12.1, .591 | 16.5, .508 | 2.81 % |
| I4 ps192 | 10.7, .532 | 12.2, .592 | 15.7, .480 | 11.46 % |
| DINO96 probe | 36.6, d [-13.1, 7.7], .780 | 38.9, d [-9.5, 15.2], .808 | 41.5, d [-1.0, 7.3], .786 | 2.46 % |
| I1 / I2 c96 | 19.5 / 19.0 | 29.6 / 27.9 | 34.0 / 34.3 | 24.9 % |
| I3 / I4 c96 | 10.4 / 9.2 | 11.6 / 10.7 | 16.0 / 16.6 | 12.9 % |

Reading: the dense object embeddings clearly encode sheet identity (unlabelled check on p1667: mean cos to the same
crossing's own sheet 0.97 at 8 vox and 0.88 at 16 vox, vs -0.31 between the two crossings of a step), but that identity
does not mark merged contacts. The zero-parameter "third object in the gap" scores (I1, I2) roughly match OLD on Paris4
and b2 and are worse on b1; the end self-consistency scores (I3, I4) are at chance (AUC 0.48-0.59), i.e. a crossing
that hides two sheets looks as self-consistent along the sheet as a clean one. The 9.6 µm checkpoint is no better than
OLD. No secondary arm reaches 55 % anywhere; no "secondary passes while primary fails" case arises.

## Verdict (2026-10-24T23:13Z): FAIL. Frozen dinovol features are not the next model bet for contact detection.

- Primary DINO (ps192, 640k): 49.6 / 43.6 / 39.2 % held-out precision at the old recall vs OLD 38.5 / 36.6 / 37.5 %;
  gain +11.1 / +7.0 / +1.7 pp with every CI including 0; 0 of 3 regions >= 55 %. Control fine (spurious <= 0.49 %).
- This is despite features that are very likely in-sample for our label meshes (A1), which favours dinovol; the
  held-out-scroll result would probably be weaker still.
- Useful by-products: dinovol features make the detector far quieter on wide packing (p1667 raises 0-0.7 % vs 1-2.9 %),
  and DINO+OLD gave the only difference CI touching 0 from above (Paris4 +4.2 [0.0, 7.8]). A fine-tune could still move
  the ceiling, but this probe gives no evidence that the pretrained representation already carries the needed signal;
  per the gate, no fine-tune outline is written and nothing further is run.

Spend: $0 (no pods). Local RTX 3090 under lease ~/.gpu-leases/vss-dino.json (declared 10 GB; measured 8.4 GB for
ps192 192³ bf16; lease removed). GPU time ~2.2 h (pass 1 ps192 ~66 min at 0.30-0.41 s/step; pass 2 c96 ~13 min,
ps192 ~51 min). CT: ~9,000 new 128³ chunks streamed (ctcache 1.9 -> 22 GB: p0139 12 GB, paris4 8.3 GB, p1667
1.6 GB). Other disk: data/untrusted/dinovol 6.7 GB (venv, two checkpoints 1.1 GB, code, metadata.json),
data/scratch/dino_probe 0.45 GB. Paris4 z7000-8500 untouched.
