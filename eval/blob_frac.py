"""Blob fraction: share of predicted surface voxels lying inside solid masses (Euclidean distance to the nearest
background voxel > r), the failure mode where touching sheets fuse into a slab (cf. Kyles' anti-blob gate).

  uv run --with pandas python eval/blob_frac.py data/preds/xing_m7d300_s11836_seg data/preds/xing_m7pub_s11836_seg ...

Per mask: foreground fraction and blob fraction at r = 2, 3, 4 vox. PHerc0139 masks (p0139_*) are grouped by the site
class from the pairs CSV (compressed / wide); hf2um masks by volume. Prints pooled means per directory and group.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
cls = pd.read_csv(REPO / "data/preds/graph_m7d300_s11836.csv", usecols=["stem", "site_cls"]).drop_duplicates()
cls = dict(zip(cls.stem, cls.site_cls))
rows = []
for d in map(Path, sys.argv[1:]):
    for f in sorted(d.glob("*.tif")):
        m = tifffile.imread(f) > 0
        if not m.any():
            continue
        e = ndi.distance_transform_edt(m)
        stem = f.stem
        grp = cls.get(stem, stem.split("_")[0])
        rows.append({"dir": d.name, "stem": stem, "group": grp, "fg": m.mean(),
                     **{f"blob_r{r}": float((e[m] > r).mean()) for r in (2, 3, 4)}})
df = pd.DataFrame(rows)
cols = ["fg", "blob_r2", "blob_r3", "blob_r4"]
out = pd.concat([df.groupby("dir")[cols].mean().assign(group="all").set_index("group", append=True),
                 df.groupby(["dir", "group"])[cols].mean()]).sort_index()
print((100 * out).round(2).to_string())
