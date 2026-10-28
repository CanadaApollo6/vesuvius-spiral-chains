# 2026-10-19: hf2um, a second held-out ~9 um surface benchmark from the organisers' surface-labels-2026-07

## Data (`scripts/hf2um_set.py`, `data/eval/hf2um/`, local only, CC BY-NC)
- Source: HF bucket `scrollprize/datasets`, `surfaces/2um_032726/<vol>.zarr` (rasterized per-winding meshes, 0 = labelled
  background, 1 = surface, 2 = ignore). Also `surfaces/kaggle/` (320^3 `sample_*.tif`, 0/1/2): the Kaggle set, not used
  (real350 is trained on Kaggle data).
- Label levels 0/1 are only partly uploaded (HTTP 404 inside labelled regions). Level 2 is complete and is ~9 um:
  2.399 um x 4 = 9.60 um (MANBp, PHerc1667), 2.215 um x 4 = 8.86 um (PHerc0343P). CT: matching open-data volume, level 2.
- Left out: PHerc0139 (w0xx volumes, our test scroll), 0500P2 (in Dataset107), s1 (Kaggle).
  None of the three used scrolls was trained on as a 2 um scan. PHerc0343 (not 0343P) is in Dataset109 (round 2 only).
- 104 crops of 160^3: manbp 36, p1667 40, p0343p 28. Non-overlapping, ignore < 30 %, surface > 2 %, seed 0.
- Bands are 1-2 vox thick (rasterized), GT foreground 3 % vs ~13 % predicted.
  - `labelsSep/` dilates the band by 1 vox for `eval/separation.py` only (its sites need band EDT >= 2).
  - Compressed sites: p1667 8804, p0343p 8597, manbp 349, so separation mostly reflects 1667 and 0343P.
- HF rate-limits anonymous reads (429); the script retries and reads with the local HF token (read only).
- CT intensities differ from training (mean 62-96 vs Dataset107 102); every model sees the same shift.

## Results (`data/scratch/eval_hf2um.sh`, `eval/hf2um_summary.py`, 2026-10-19)
Spacing declared as um / 7.91 (um / 9.362 for r9ft300), fold 0, no TTA. Paired patch bootstrap vs real350
(1000 resamples). Official = Kaggle leaderboard blend on raw labels with ignore = 2.

| model | official | diff [CI] | VOI_merge | cc | compressed sep | diff [CI] | wide sep |
|---|---|---|---|---|---|---|---|
| real350 | 0.472 | | 0.905 | 56.8 | 41.6 | | 66.6 |
| **m7d300** | 0.476 | +0.004 [-0.001, +0.010] | 0.862 | 42.3 | **57.7** | **+16.0 [+8.3, +24.8]** | 78.1 |
| m7dsq | 0.477 | +0.005 [-0.001, +0.010] | 0.859 | 46.3 | 57.9 | [+8.8, +24.2] | 78.0 |
| m7d | 0.476 | +0.004 [-0.001, +0.010] | 0.863 | 41.9 | 53.7 | [+3.8, +20.6] | 77.2 |
| m7mix | 0.475 | +0.003 [-0.002, +0.009] | 0.871 | 46.8 | 54.0 | [+5.0, +20.2] | 76.7 |
| r9ft300 | 0.444 | -0.028 [-0.034, -0.024] | 0.930 | 78.4 | 29.3 | [-19.0, -5.7] | 60.0 |

- m7 distillation transfers to scrolls and a scanner setup (2.4 um, 78 keV, downsampled) it never saw.
  - Compressed separation +16 pts, mostly through hit rate (62.5 vs 47.0 %: more labelled sheets are found).
  - Fewer merges (VOI_merge -5 %) and fewer fragments (cc 42 vs 57).
  - The official blend ties (+0.004), as it did on jeong200: it barely responds to fused sheets.
- Per volume: m7d300's official score is +0.032 on 0343P [+0.023, +0.043] and -0.013 on MANBp [-0.020, -0.006].
  Its separation gain is +11-35 on 0343P and +0-21 on 1667.
- The ranking matches PHerc0139: m7d300 ≈ m7dsq > m7d ≈ m7mix >> real350 > r9ft300.
  - r9ft300 (our 9 um real labels) generalises worst, consistent with its PHerc0139 deficit.
- Use: a second model-selection set next to the PHerc0139 tune half, for the round-2 checkpoints.

### The teacher: m7 on the 64 crops it covers (MANBp, 0343P)
The bucket has m7 run at CT level 2 for MANBp and 0343P (`scripts/hf2um_m7.py`), on the same grid as the crops.
There is none for 1667.

| model (64 crops) | official | diff vs real350 [CI] | VOI_merge | cc | compressed sep | diff [CI] |
|---|---|---|---|---|---|---|
| real350 | 0.454 | | 0.864 | 77.2 | 31.0 | |
| m7 (teacher) | **0.488** | +0.035 [+0.027, +0.042] | 0.846 | 30.9 | 37.6 | [-6.7, +19.9] |
| m7d300 | 0.460 | +0.007 [-0.001, +0.015] | 0.807 | 52.8 | **52.8** | [+9.4, +34.5] |
| m7dsq | 0.461 | +0.007 [-0.001, +0.015] | 0.800 | 60.1 | 54.5 | [+12.0, +35.1] |

- m7 wins the official blend: higher surface Dice (0.773 vs 0.727) and fewer fragments.
- The student separates compressed sheets far better than its teacher (52.8 vs 37.6), as on PHerc0139 (81.1 vs 80.5).
  A likely reason is that our nnU-Net plus sigmoid threshold produces thinner bands than m7's th0.2 binary.
- m7 may have been trained on these organiser labels: its training data is undocumented, and the MANBp and 0343P
  labels predate the m7 run of 2026-04-13. That would favour m7 on this set.
- Round-2 target: close the surface Dice and fragment gap to m7 without giving back separation.

### Caveat: the labels are sparse (figure `data/scratch/hf2um_xsec.png`)
- Only the traced windings are labelled. Visible sheets between them are labelled 0 (background), not ignore.
  - In the 0343P crop shown, ~7 labelled lines lie among 12+ visible sheets.
- So the official blend and precision-type terms penalise correct predictions of untraced sheets. Treat them as
  secondary on this set.
- Recall-type measures are sound: hit rate and separation of adjacent labelled sheets.
- The figure shows the two crops with the largest separation gain. They are cherry-picked for illustration;
  the numbers are in the tables above.
