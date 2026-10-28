"""autowind post-processing of a finished `autowind chains --raise auto` run (f6 screen, notes/2026-10-24-f6-bias-screen.md;
f7 `--min-pts`, notes/2026-10-24-f7-switch-aware-cut.md).

New, optional and separate: no existing code path changes, so every published chain set still reproduces byte for byte.

  uv run python -m autowind.postproc --run <autowind --raise auto output dir> --mode cut --out <dir>

Input: the run directory written by `autowind chains --raise auto`:
  packing.json         the S_pack decision ('gm' = tight packing, 'off' = wide)
  noraise/chains.json  the no-raise chains (only present when the decision was 'gm')
  chains.json          the raised chains with the frozen grad_mag logistic's per-step P ("gm_p")
Branch (S_pack rule, unchanged):
  decision 'off' -> relative_windings.json and chains.json are copied byte for byte (= the confirmed no-raise recipe);
  decision 'gm'  -> the chosen mode is applied to the NO-RAISE chains, using gm_p only to rank / select steps.
Modes:
  cut   hard-cut every single step (no-raise step 1) with gm_p >= --thr; no shared point; pieces with < 2 points are
        dropped; wind re-zeroed per piece. No step value is changed (no +1 is ever guessed).
        --min-pts N (f7, default 2 = the f6 recipe, byte for byte): pieces with < N points are dropped instead of < 2.
        f7 switch-aware variants: --thr 0.10 --min-pts 4 (c10m4) and --thr 0.15 --min-pts 4 (c15m4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import core  # noqa: F401  (puts tools/ and eval/ on sys.path)
import auto_windings as aw  # noqa: E402  (frozen; imported, never modified)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cut_chains(raw: list[dict], gmc: list[dict], thr: float, min_pts: int = 2) -> tuple[list[dict], dict]:
    """raw: no-raise chains; gmc: the same chains after gmfix (same points, with gm_p). Returns the cut pieces."""
    if len(raw) != len(gmc):
        raise RuntimeError("no-raise and raised chain files differ in chain count")
    out, n1, ncut, ndrop_pts = [], 0, 0, 0
    for a, b in zip(raw, gmc):
        if a["pts"] != b["pts"]:
            raise RuntimeError("no-raise and raised chains differ in points")
        n = len(a["pts"]) - 1
        if n < 1:
            continue
        d = np.diff(a["wind"]).astype(int)
        p = np.asarray(b.get("gm_p") or [0.0] * n, float)
        cut = (d == 1) & (p >= thr)
        n1 += int((d == 1).sum())
        ncut += int(cut.sum())
        s = 0
        for e in [int(x) for x in np.flatnonzero(cut)] + [n]:  # piece = points s..e (steps s..e-1 uncut)
            if e - s + 1 >= max(2, min_pts):
                w0 = a["wind"][s]
                piece = dict(z=a["z"], theta=a["theta"], pts=a["pts"][s:e + 1], wind=[w - w0 for w in a["wind"][s:e + 1]],
                             r=a["r"][s:e + 1], perr=a.get("perr", [])[s:e], gm_p=[float(x) for x in p[s:e]])
                if "rl" in a:
                    piece["rl"] = a["rl"][s:e + 1]
                out.append(piece)
            else:
                ndrop_pts += e - s + 1
            s = e + 1
    st = dict(single_steps=n1, steps_cut=ncut, cut_frac=round(ncut / max(n1, 1), 4), pieces=len(out),
              points=sum(len(c["pts"]) for c in out), dropped_singleton_points=ndrop_pts,
              median_piece_points=float(np.median([len(c["pts"]) for c in out])) if out else 0.0)
    return out, st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autowind.postproc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="output dir of `autowind chains --raise auto`")
    ap.add_argument("--mode", required=True, choices=["cut"])
    ap.add_argument("--thr", type=float, default=0.263, help="cut: gm_p threshold (default = the auto raise threshold)")
    ap.add_argument("--min-pts", type=int, default=2, help="cut: drop pieces with fewer points (default 2 = f6 recipe)")
    ap.add_argument("--out", required=True, type=Path)
    o = ap.parse_args(argv)
    run, out = o.run, o.out
    out.mkdir(parents=True, exist_ok=True)
    pack = json.loads((run / "packing.json").read_text())
    info = dict(postproc="f6", mode=o.mode, thr=o.thr, run=str(run), decision=pack["decision"], S_pack=pack["S_pack"],
                inputs={"relative_windings.json": _sha(run / "relative_windings.json")})
    if o.min_pts != 2:
        info["min_pts"] = o.min_pts
    if pack["decision"] != "gm":  # wide packing: the confirmed no-raise recipe, byte for byte
        for f in ("relative_windings.json", "chains.json"):
            shutil.copyfile(run / f, out / f)
        info["applied"] = "none (S_pack >= threshold): copied the run's no-raise set unchanged"
    else:
        raw = json.loads((run / "noraise" / "chains.json").read_text())
        gmc = json.loads((run / "chains.json").read_text())
        info["inputs"]["noraise/chains.json"] = _sha(run / "noraise" / "chains.json")
        info["inputs"]["chains.json"] = _sha(run / "chains.json")
        ch, st = cut_chains(raw, gmc, o.thr, o.min_pts)
        aw.write_json(ch, out / "relative_windings.json", prefix="autocut")
        (out / "chains.json").write_text(json.dumps(ch))
        info["applied"] = st
    info["sha256"] = {"relative_windings.json": _sha(out / "relative_windings.json"), "chains.json": _sha(out / "chains.json")}
    (out / "postproc.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
