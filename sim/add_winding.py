"""Add the winding-coordinate label `wind_7p91` (float32) to existing sim cells, re-rendered on identical geometry.

  uv run python sim/add_winding.py data/sim/v1 [--limit N]

s = wrap index (relative to the cell reference) + radial fraction to the next wrap's recto face (see Scroll.sample
"wind"). The scroll is rebuilt from the cell's stored ScrollParams (tables are deterministic in the seed) and anchored
at the stored centre, so s lines up with the stored CT and instance labels. Check per cell: on in-sheet voxels,
floor(s) maps one-to-one onto the stored instance labels.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "forward"))
from ct import VOX_LO  # noqa: E402
from render import render_grid  # noqa: E402
from scroll import Scroll, ScrollParams  # noqa: E402


def check(wind: np.ndarray, inst: np.ndarray) -> float:
    """Fraction of in-sheet voxels whose floor(s) disagrees with the majority instance label for that floor."""
    m = inst > 0
    f = np.floor(wind[m]).astype(np.int64)
    i = inst[m].astype(np.int64)
    bad = 0
    for v in np.unique(f):
        sel = i[f == v]
        bad += sel.size - np.bincount(sel).max()
    return bad / max(1, m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cells = sorted(args.root.glob("*.zarr"))[: args.limit or None]
    S, seed = None, None
    for c in cells:
        g = zarr.open_group(str(c), mode="r+")
        if "wind_7p91" in g and not args.force:
            continue
        a = dict(g.attrs)
        if a["scroll_seed"] != seed:
            t0 = time.time()
            S, seed = Scroll(ScrollParams(**a["scroll_params"])), a["scroll_seed"]
            print(f"scroll {seed}: tables {time.time() - t0:.1f}s", flush=True)
        meta = a["cell"]
        S.set_reference(torch.tensor(meta["centre_world_um"], dtype=torch.float32, device=S.dev))
        shape = g["instance_7p91"].shape
        wind = render_grid(S, np.asarray(meta["lo_origin_world_um"]), shape, VOX_LO, want_density=False,
                           keys=("wind",))["wind"].astype(np.float32)
        bad = check(wind, g["instance_7p91"][:])
        g.create_array("wind_7p91", data=wind, chunks=(128, 128, 128), overwrite=True)
        nan = float(np.isnan(wind).mean())
        print(f"{c.name}: s {np.nanmin(wind):.2f}..{np.nanmax(wind):.2f}, nan {nan:.3f}, floor/instance mismatch {bad:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
