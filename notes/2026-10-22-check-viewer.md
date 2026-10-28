# 2026-10-22: check viewer for human sheet counts on uncertain chain steps (tools/check_viewer.py)

A small local web app for checking the most uncertain auto-chain steps on PHerc0139 by eye, blind, in about
20-30 min. Each item is one chain step: points A (inner) and B (outer) on a radial ray at one z, in 9.362 um L0
voxels. The answers replace the automatic count on those steps in the human-in-the-loop fit experiment. Local CPU
only, no pods, $0. CT is streamed as byte ranges of the uncompressed zarr chunks, and only the z planes needed are read.

## How to run
```
# band 1 (z 3000-4500); the browser opens at http://127.0.0.1:8765/
uv run python tools/check_viewer.py --queue data/scratch/hitl/check_queue_band1.json \
    --out data/scratch/hitl/answers_2026-10-25_band1.jsonl
# band 2 (z 1750-3000)
uv run python tools/check_viewer.py --queue data/scratch/hitl/check_queue_band2.json \
    --out data/scratch/hitl/answers_2026-10-25_band2.jsonl
```
- **Resumable.** Stop with Ctrl-C (or just close the tab) and run the same command again: it continues at the
  first unanswered item. The tutorial runs once per log file; press `t` to replay it.
- **Order.** The queue's 300 steps plus its 40 blind controls (answerable steps, flagged `is_control` in the log) are
  shuffled with a fixed seed (`--seed 0`). `--top N` keeps only the N highest-ranked steps (the export's p_err
  order); `--no-controls` drops the controls. The user will not finish all 340 in 30 min (about 150-250 at 5-10 s
  each). Every queue step is uncertain by construction, so any shuffled prefix is a random sample of them.
- **Cache.** The first 300 items in session order for both bands, the stand-in queue and the tutorial are already
  rendered in data/scratch/hitl/cache/ (**see the cache section for sizes**). Items past 300 are rendered on the fly,
  5 ahead of the user. After the queue file changes, re-run:
  `uv run python tools/check_viewer.py precache --queue data/scratch/hitl/check_queue_band1.json --n 300`
- Other flags: `--port`, `--no-browser`, `--no-hires`, `--prefetch`. The server listens on 127.0.0.1 only.

## What the user sees
- **Header:** a progress bar ("k answered, item i / n") and the current resolution (9.4 um, or 2.4 um in purple).
- **Left: "slice seen from above".** An axial 9.362 um slice at the step's z, rotated so that the ray runs left to
  right (A on the left). The crop is about 3x the step length (112-320 vox wide, 0.7x as tall), so sheets can be
  followed in from where they are separate. A is a cyan ring and B an orange ring. Two dashed yellow lines bound a
  corridor around A->B and leave the middle of it uncovered.
- **Right: "side view".** The vertical plane through the A-B line (horizontal = the same u axis, up = +z, +-0.35 W
  in z). Sheets that touch in the axial slice often separate here.
- **Below: "brightness along the line".** The 9.362 intensity along A->B, averaged over +-1.5 vox across the line
  and over z-1..z+1, with A, B and the A-B span shaded. It shares the x axis with the slice above it.
- **`h`: 2.4 um toggle.** Both images switch to the 2.399 um scan: the axial view from L0 at 0.25-vox sampling, the
  side view from L1 (4.8 um), and the 2.4 um profile is added in purple. Each is registered to the 9.362 frame with
  the published affine plus a per-item NCC shift search (+-3 vox). Median NCC is 0.91 axial and 0.93 side; p10 is
  0.76 / 0.83. The 2.4 view is off by default and its use is logged (see the tips for why).
- **Right panel:** the question, answer buttons, key help, and a warning about the 2.4 um view.
- **Blind.** The browser gets only images, profiles and the item id. Chain counts, gm_p, p_err and oracle or CT
  counts never leave the server.

## What to do
- **The question:** "How many sheets lie between A and B? Not counting the sheets that A and B sit on."
- **Keys:**
  - `0`-`9`: number of sheets between;
  - `s`: A and B are on the same sheet (rare);
  - `u`: unsure;
  - `<-` / Backspace: back (answering again overwrites the earlier answer; the last answer per item wins);
  - `->`: forward over answered items;
  - `h`: 2.4 um;
  - `o`: hide the markers;
  - `c`: cycle contrast / darker (helps in bright, dense regions);
  - `t`: tutorial.
- **Tutorial (about 2 min, 5 examples, with feedback):**

  | # | case | answer |
  |---|---|---|
  | 1 | wide gap | 0 |
  | 2 | one clear sheet | 1 |
  | 3 | close but separate neighbours | 0 |
  | 4 | touching sheets: the middle sheet merges with A's sheet below the line | 1 |
  | 5 | one thick sheet that splits into fibre stripes at 2.4 um (said explicitly) | 1 |

  - After each answer, red triangles mark the sheets between A and B, and a short explanation is shown.
  - The examples come from z 4500-6000, outside both check bands. A candidate was kept when:
    - the human-mesh winding difference is complete on the ray (every intermediate winding crosses it);
    - all windings involved are label-consistent and untouched by the CT relabel (w029-w032, w052-w054,
      w057-w059; this avoids w041=w042, w045=w046, missing w050 and the #2167 sector);
    - the m7 auto chain gives the same count;
    - it looked right by eye.
  - z 4500-6000 has no CT relabel, so this filter stands in for CT-corrected counts.
  - A "two between" example (n2_touch_0) was dropped: its profile shows a third peak that could be fibre texture.
  - Picks and texts: `TUTORIAL_PICKS` in check_viewer.py; `tutorial-pick` then `tutorial-build` regenerate
    data/scratch/hitl/tutorial.json.

## Tips for counting sheets (also in the UI)
- Bright = papyrus, dark = air. Do not count the sheets A and B sit on. The rings may sit 1-2 vox off the sheet's
  centre line.
- **Follow sheets along their length.** Start at the edge of the slice, where sheets are separate, and trace each one
  into the corridor. Two sheets that merge into one thick band still count as two. This continuity is what the pixel
  profile cannot see.
- Check the side view. Contacts are local, and the same two sheets often show a dark line a few voxels up or down.
- A profile peak counts only if it continues as a band in the slice. Small bumps from texture are not sheets.
- **2.4 um:** each sheet shows its fibre layers as several bright stripes with dark lines between them. Those
  intra-sheet lines are as deep as a real contact line, which is why automatic 2.4 um counting failed
  (notes/2026-10-22-hires-step-count.md). Use 2.4 um only to confirm a contact you already see at 9.4 um.
- Answer `u` instead of guessing. 5-15 s per item is about right.

## Log format (jsonl, one line per answer; last line per item_id wins)
`{"item_id": "b1_c322_k0", "answer": "1", "n_between": 1, "winding_step": 2, "is_control": false, "seconds": 6.4,
"timestamp": "2026-10-25T14:03:11+00:00", "hires_opened": false, "pos": 17, "revisit": false, "A": [z,y,x],
"B": [z,y,x], "queue": "...", "session": "..."}`
- **Conventions.**
  - `answer` is the raw key: a digit, `s` or `u`.
  - `n_between` = sheets strictly between A and B.
  - **`winding_step` = n_between + 1, or 0 for `s`.** This matches the export's question ("turns between p_inner and
    p_outer; 0 if same sheet") and the queue's `chain_count`. Use winding_step to replace chain counts.
- **Timing.**
  - `seconds` runs from the moment the item's images finished loading to the key press.
  - `hires_opened` is true if the 2.4 um view was on at any time during that visit. The toggle is sticky across
    items, so it stays true while the user leaves it on.
- **Tutorial answers** go to `<out>.tutorial.jsonl` (answer, truth, correct, seconds, plus a `tutorial_done` event).

## Test (headless, end to end)
- **Command:** `uv run python tools/check_viewer.py shots --queue data/scratch/hitl/check_queue_band1.json`
  (tools/check_viewer_test.py).
- **What it does:**
  - starts the server on a temp log;
  - drives headless chromium over the DevTools protocol with keys only: tutorial Enter / 1 / Enter x5, then 9 main
    answers including `h`, `u` and one Backspace re-answer;
  - checks the jsonl (9 lines, 8 items, hires flag, revisit flag) and that /api/state resumes at item 9.
- **Screenshots** in data/scratch/hitl/screens/:
  - 1_tutorial_touching_reveal.png: tutorial 4/5 after answering, with red markers and the explanation;
  - 2_item_9p4um.png: a queue item, default view;
  - 3_item_2p4um.png: a queue item with the 2.4 um toggle on.
- **Load time per item:**
  - Browser, from item switch to images loaded: 0-9 ms (cached and prefetched; the 4 next items' JSON and images
    are preloaded).
  - Server, cached item: about 6 ms JSON, under 1 ms per PNG.
  - Cold renders (items past the cache): 0.6-1.5 s single-threaded; 3-6 s when the user jumps around and requests
    pile up behind the prefetch workers. In normal sequential use the 5-ahead prefetch hides this.

## Cache
- data/scratch/hitl/cache/<sha1(A,B)[:14]>/ holds ax9.png, rz9.png, ax24.png, rz24.png and meta.json (frame,
  profiles, registration shifts, timing). About 0.27 MB per item; keys are queue-independent.
- **Sizes:**
  - band 1, first 300 + tutorial: 84 MB;
  - band 2, first 300 + tutorial: 91 MB;
  - whole cache directory, including the stand-in queue and tutorial candidates: 253 MB (about 950 items).
- Precache wall time: about 5.5 min per 300 items with 8 workers. Under that load a render takes 7 s median
  (fetch9 1.8 s, 2.4 um 5 s).

## Files
- tools/check_viewer.py: server, renderer and the modes `precache`, `standin`, `tutorial-pick`, `tutorial-build`
  and `shots`.
- tools/check_viewer_ui.html: single-page UI.
- tools/check_viewer_test.py: headless end-to-end test.
- data/scratch/hitl/check_queue_standin_band1.json: stand-in queue, 400 random gm-raised steps of
  band_m7_ray_gmxc (built before the real queues existed). data/scratch/hitl/tutorial.json and
  tutorial_candidates.{json,png}.

## Limitations
- **The 2.4 um side view comes from L1 (4.8 um)** to keep each item to about 35 MB of reads. The axial 2.4 view is
  full L0.
- **Registration.** The 2.4 registration is an in-plane shift per view. The z offset (measured in the side view,
  typically <= 1 vox) is not applied to the axial 2.4 slice.
- **No 2.4 um render** when the 2.4 crop is mostly empty: 1 of 300 in band 1, 0 in band 2, 1 in the stand-in. The toggle then shows
  9.4 um and says "no 2.4 um data".
- **Long bridged steps.** Some steps are 100-270 vox long (p99). The crop is capped at 320 vox, so context there is
  under 3x, and such steps may hold many sheets (keys up to 9).
- **Truth for the tutorial** is human meshes plus m7 agreement on label-consistent windings, checked by eye, not a CT
  relabel (none exists for z 4500-6000). There is no "two between" example.
- **Brightness.** The window is per item (1-99.7 percentile of the 9.362 crop). Dense, bright regions can look washed
  out; `c` cycles darker settings.
- Tested in headless chromium at 1920x1080. The layout rescales on resize. Not tried on the user's Hyprland desktop
  browser.
