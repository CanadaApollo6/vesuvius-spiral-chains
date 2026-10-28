# HITL session: how to do it (about 30 min, one sitting)

You count papyrus sheets between two points. Your answers decide which automatic winding steps get corrected. We then
measure how much correctly numbered surface one minute of your checking buys. The design is in
`notes/2026-10-24-hitl-session-prereg.md`. You don't need to read it.

## Start
From the repo root:

```
uv run python tools/check_viewer.py --queue data/scratch/hitl_session/queue_p4z9000_t60.json --out data/scratch/hitl_session/answers_hitl_session.jsonl
```

A browser tab opens at http://127.0.0.1:8765/. If it doesn't, open that address yourself. Keep the terminal open.

## Order and timing
1. **Practice: 4 examples, about 3 min.** These come from a different part of the scroll. Each one tells you the right
   answer after you press a key. Practice is never scored, and its time doesn't count.
2. **The check queue.** Items run in a fixed order: ray by ray, inner to outer along each ray. The header shows the
   item time used out of **25:00**. It turns orange in the last 2 minutes. At 25:00 a "time is up" card appears, and
   then you are done. Expect roughly 5-10 s per item.
3. **Stop by closing the tab and pressing Ctrl-C in the terminal.**

## What each item shows
- **Main view:** a slice through the scroll at 9.6 um. Bright bands are papyrus sheets and dark is air. Ring **A**
  (inner) and ring **B** (outer) each sit on a sheet.
- **Side view:** the same corridor seen from another angle. Sheets often separate there when they touch in the main
  view.
- **Profile:** brightness along the line from A to B. Each sheet is a peak.
- **Optional:** `h` switches to the 2.4 um view; `o` hides the markers; `c` changes contrast.

## How to answer (one key per item)
- **`0`-`9`:** the number of sheets strictly **between** A and B. Do not count the sheets A and B sit on.
- **`s`:** A and B are on the same sheet (rare).
- **`a`:** A or B sits in open air, not on a sheet.
- **`u`:** you are unsure. Use it whenever you can't decide. A `u` is safe: the step is simply cut. A wrong number is
  not safe.
- **`←` / Backspace** goes back to change an answer. The last answer counts. Going back uses item time too.

Two conventions, both shown in the practice:
- **Ring on a double band.** If a ring sits on a thick band made of two touching sheets, treat it as being on the
  band's **left** sheet (the inner one, nearer A).
- **Touching sheets still count.** A sheet that touches A's sheet, or another sheet, still counts as between if it is
  between. Follow the bands away from the corridor to see where they split.

## Breaks
You can stop at any time with Ctrl-C and run the same command later. The viewer resumes at the first unanswered
item. The clock only runs while an item is on screen, and the welcome-back card shows the time used so far.

## Please don't
- Don't look at labels, meshes, chains, fit outputs, the oracle simulation (`data/scratch/hitl_session/sim/`), or
  any score or result file before or during the session. Use only the viewer.
- Don't edit the answer file. Don't let anyone (or any agent) answer for you.

## When you are done
Tell the agent "HITL session done". It runs the pre-registered pipeline: answers -> chains -> fits on pods -> score.
