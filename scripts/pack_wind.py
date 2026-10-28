"""Pack the 7.91 um arrays the winding trainer needs into per-cell .npy directories (compact, mmap-able on a pod).

  uv run python scripts/pack_wind.py data/sim/v1 data/winding/pack_v1

Per cell: ct_7p91 (uint8), wind_7p91 (float16, s minus floor(min s), so values stay < ~100 and float16 resolves
< 0.06 wrap), instance_7p91 (uint8), contact_7p91 (float16 um). The constant shift is harmless: the loss is
offset-invariant.
"""

import sys
from pathlib import Path

import numpy as np
import zarr

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cells = sorted(p for p in src.glob("*.zarr") if (p / "wind_7p91").exists())
for c in cells:
    out = dst / c.stem
    if (out / "contact_7p91.npy").exists():
        continue
    out.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(c), mode="r")
    s = g["wind_7p91"][:]
    s = (s - np.floor(np.nanmin(s))).astype(np.float16)
    inst = g["instance_7p91"][:]
    assert inst.max() < 256, c
    np.save(out / "ct_7p91.npy", g["ct_7p91"][:])
    np.save(out / "wind_7p91.npy", s)
    np.save(out / "instance_7p91.npy", inst.astype(np.uint8))
    np.save(out / "contact_7p91.npy", g["contact_7p91"][:].astype(np.float16))
print(f"packed {len(cells)} cells into {dst}")
