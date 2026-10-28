"""render_mid.py - per montage of <ds>/render/<m>.zarr: mean of the 7 middle layers (d = -3..3 around layer 33) and the
d = 0 layer, as uint8 tifs in <ds>/mid/, plus the montage tifxyz + layout.json copied next to them (small, for the
laptop figure). Runs on the pod after s07 (render_compare.sh).

    $PY -I render_mid.py <ds>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

ds = Path(sys.argv[1])
out = ds / "mid"
out.mkdir(exist_ok=True)
for z in sorted((ds / "render").glob("*.zarr")):
    a = zarr.open(str(z), mode="r")
    a = a["0"] if hasattr(a, "keys") and "0" in a else a
    n = a.shape[0]
    c = n // 2  # 66 layers: layer 33 = d 0
    mid = np.asarray(a[c - 3:c + 4]).astype(np.float32)
    tifffile.imwrite(out / f"{z.stem}_mean7.tif", np.clip(mid.mean(0), 0, 255).astype(np.uint8), compression="zlib")
    tifffile.imwrite(out / f"{z.stem}_d0.tif", np.asarray(a[c]).astype(np.uint8), compression="zlib")
    print(z.stem, a.shape, a.dtype)
    m = ds / "montage" / f"{z.stem}.tifxyz"
    if m.exists() and not (out / m.name).exists():
        shutil.copytree(m, out / m.name)
shutil.copy(ds / "montage" / "layout.json", out / "layout.json")
json.dump({"layers": int(n), "centre": int(c)}, open(out / "mid.json", "w"))
