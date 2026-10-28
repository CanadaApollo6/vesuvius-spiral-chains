# 2026-10-22: PHerc1667 gm-threshold calibration session (human calibrates, machine applies)

This session calibrates the gm raise threshold of the auto winding chains on the PHerc1667 confirmation region
(z 6500-8000, chain set `data/scratch/auto_windings/p1667/band_m7_p1667_ray_gmxc`, which raises at gm_p >= 0.18).
The method is the one in notes/2026-10-22-hitl-budget.md section 3 and data/scratch/hitl/calib.py:
1. The user checks random single steps (count before gm = 1) by eye in tools/check_viewer.py.
2. tools/calib_apply.py fits P(true step >= 2 | logit gm_p) on the answered checks.
3. The threshold is set where the number of raised single steps equals the predicted number of true doubles among
   all 17,291 single steps of the chain set.

Planned for about Oct 25. Local CPU only, $0.

## Launch command (for the user)
```
uv run python tools/check_viewer.py --queue data/scratch/hitl/check_queue_p1667_calib.json \
    --out data/scratch/hitl/answers_p1667_calib.jsonl
```
- The browser opens at http://127.0.0.1:8765/. The server picks PHerc1667 from the queue's `"scroll": "p1667"` field.
- **Resumable.** Stop with Ctrl-C or close the tab, then run the same command again. It continues at the first
  unanswered item.
- The tutorial (5 examples, about 2 min) runs once for this log file; press `t` to replay it. Its examples are
  PHerc0139 (z 4500-6000), as before. PHerc1667 sheets are spaced wider (15-28 vox against 10-16), but they are read
  the same way.
- All 200 items and the tutorial are pre-rendered in data/scratch/hitl/cache_p1667/, so no item waits on S3.

**Afterwards (the agent runs this, not the user):**
```
uv run python tools/calib_apply.py --queue data/scratch/hitl/check_queue_p1667_calib.json \
    --answers data/scratch/hitl/answers_p1667_calib.jsonl --json data/scratch/hitl/p1667_calib_result.json
```
It prints the threshold, a 2000-resample bootstrap 95 % CI, the predicted doubles, and the raise counts at the new
threshold and at 0.18. `--mid-only` restricts the fit to the calib.py pool (gm_p 0.10-0.70).

## Expected session time
| | items | at 10 s | at 15 s |
|---|---|---|---|
| whole queue | 200 | 33 min | 50 min |
| 30 min session | about 120-180 | | |

Add the tutorial (about 2 min).
- **Stopping early is fine.** The order is a fixed random shuffle of all three strata, so any prefix is still a
  random sample. Each prefix holds about 75 % mid-range steps.
- **Target.** Section 3 needs 100-150 answered checks to pin the threshold to about +-0.03. 'u', 's' and open-air
  answers do not count towards this. So a 30 min session should still reach the target unless many answers drop.

## Queue (data/scratch/hitl/check_queue_p1667_calib.json)
- **Built by** data/scratch/hitl/export_queue_p1667.py, seed 1667 (numpy `default_rng`), recorded in the file's
  `sampling` block.
- **Draws.** Uniform without replacement over single steps:
  - 150 with gm_p in [0.10, 0.70], from a pool of 4,937;
  - 30 with gm_p < 0.10 (pool 12,081);
  - 20 with gm_p > 0.70 (pool 273).

  The tails give the logistic support at both ends. The 200 items are shuffled together, and `rank` is the shuffled
  position.
- **Spread.**
  - 164 of 577 chains, with at most 4 items per chain;
  - all 23 ray z levels and all 24 ray angles.
- **Steps.** 89 of the 200 are gm-raised at 0.18. 18 are bridged (no graph solve), so an endpoint can be in open air.
- **Item schema.** As check_queue_band2.json, without `oracle_count`, `ct_count_*`, `oracle_bracket` or
  `label_error_zones`. Two fields are added:
  - `stratum` (mid / low / high);
  - `p_err`, from the label-free z4500 uncertainty model (informational).
- **Frame.** Points are in PHerc1667 level 2 of 20251217075048-2.399um (9.596 um; `p_L2 = p_2399 / 4`), given both as
  zyx and as xyz (xyz = relative_windings `p`).
- **No controls in the queue** (`controls: []`). Their answers are sealed (below), so every item the user sees is a
  calibration item.

## Answer conventions
| key | meaning | log | in the calibration |
|---|---|---|---|
| `0`-`9` | sheets strictly between A and B | `n_between` = digit, `winding_step` = digit + 1 | used: y = (winding_step >= 2) |
| `s` | A and B on the same sheet | `winding_step` = 0 | unanswered (dropped) |
| `u` | unsure | `winding_step` = null | unanswered (dropped) |
| `a` (new) | A or B sits in open air, not on a sheet | `winding_step` = null | unanswered (dropped) |

- The last answer per item wins: Backspace goes back, and answering again overwrites.
- Blind controls (`is_control`) never enter the fit. This queue has none.
- `a` is a new key for this session (button "open air"). Before it, open air had to be answered `u`; both are dropped
  the same way.
- Log lines carry `"scroll": "p1667"`.

## Blinding statement
- PHerc1667 labels in z 6500-8000 are confirmation-test labels. No label, oracle or CT file was opened to choose,
  filter or order the queue. The export script reads only the two auto chain sets and the PHerc0139-fitted uncertainty
  model.
- **The queue file holds no label-derived field or count.** No label-based metric was computed on the queue.
- **Sealed controls.**
  - The label counts of the queue items are in data/scratch/hitl/sealed/p1667_calib_answers.json, written by
    data/scratch/hitl/seal_p1667_controls.py (band_oracle_p1667 rays, same endpoint matcher as hitl_lib).
  - They cover 96 of the 200 items. Only that count was looked at; no value was printed or summarised.
  - The viewer, the export and calib_apply.py never read the sealed file. Open it only after the session, to measure
    human accuracy.
- **Nothing visible to the user.** The browser gets only images, profiles and the item id. gm_p, chain counts, p_err
  and strata stay on the server.
- No arm result in data/scratch/confirm_region/ was read for this work.

## Viewer changes (tools/check_viewer.py, check_viewer_ui.html; backward compatible)
- **Per-scroll sources (`SCROLLS`).**
  - p0139: unchanged (9.362 um volume 20250728140407 + the 2.399 um volume via the published affine; cache
    data/scratch/hitl/cache/).
  - p1667: base = level 2 of the PHerc1667 2.399 um zarr (9.596 um, 9269 x 3808 x 3808). The 2.4 um view uses L0 and
    L1 of the same zarr, mapped by the pyramid scale (`L0 = 4 p + 1.5`, voxel centres) plus the usual +-3 vox NCC
    search. The cache is data/scratch/hitl/cache_p1667/.
  - The umbilicus is only implied: the queue steps already lie on the umbilicus rays of the chain set (umbilicus
    data/eval/p1667/umbilicus_p1667_L2.json). The viewer needs only A and B.
- **Choosing the scroll.** The queue's `"scroll"` field picks it; files without one are p0139. `--scroll` overrides it
  for serve, precache and shots. Tutorial items always render from p0139.
- **Image URLs.** These are now `/img/<scroll>/<key>/<name>.png`; the old `/img/<key>/<name>` still serves p0139.
- **UI.**
  - The resolution label and scale bar follow the item's voxel size (9.6 um / 96 um for PHerc1667).
  - New `a` (open air) key and button; the end card counts it.
- p0139 renders are byte-identical, so the existing band-1/2 caches stay valid (RENDER_VERSION unchanged).

## Tests
- `uv run python tools/check_viewer.py shots --queue data/scratch/hitl/check_queue_band2.json` passes (band-2
  backward compatibility, served from the existing cache).
- `uv run python tools/check_viewer.py shots --queue data/scratch/hitl/check_queue_p1667_calib.json --screens
  data/scratch/hitl/screens_p1667` passes (PHerc1667 end to end; screenshots in data/scratch/hitl/screens_p1667/).
- `uv run python tools/calib_apply_test.py` passes. It uses synthetic answers on the band-2 queue:
  - exact drop counts for u / s / a / controls / revisits;
  - the CLI end to end;
  - 95 % CI coverage of the true threshold: 39 / 40 sessions;
  - the `--mid-only` filter.
- **Precache.** 200 items rendered in 3.3 min with 8 workers, with 0 2.4 um errors; the tutorial was already in the
  p0139 cache. data/scratch/hitl/cache_p1667/ is 52 MB.
- **Registration.** L2 against L0: the NCC median is 0.996 (p10 0.994), and the best shift is 0 for all 200 items.
  Screenshots are in data/scratch/hitl/screens_p1667/.
