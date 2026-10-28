"""Stream a 3D crop from a remote OME-Zarr volume into data/crops/.

Reads only the chunks that overlap the requested box. Writes:
  data/crops/<name>.zarr       local zarr array (same dtype as source)
  data/crops/<name>.json       provenance: source, level, origin, shape, time
  data/crops/<name>.png        orthogonal mid-slices for a quick look

Example:
  uv run python scripts/stream_crop.py \
      --url s3://vesuvius-challenge-open-data/PHerc1667/<volume>.zarr \
      --level 0 --origin 4000 3000 3000 --size 512 512 512 --name p1667_dense_a
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
CROPS = REPO / "data" / "crops"


def open_level(url: str, level: str):
    storage = {"anon": True} if url.startswith("s3://") else None
    root = zarr.open(url, mode="r", storage_options=storage)
    if isinstance(root, zarr.Array):
        return root, {}
    return root[level], dict(root.attrs)


def save_preview(vol: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, y, x = (s // 2 for s in vol.shape)
    views = [("z (axial)", vol[z]), ("y", vol[:, y]), ("x", vol[:, :, x])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lo, hi = np.percentile(vol, [1, 99])
    for ax, (title, img) in zip(axes, views):
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True, help="OME-Zarr group (or array) URL")
    p.add_argument("--level", default="0", help="multiscale level path inside the group")
    p.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--name", required=True)
    p.add_argument("--note", default="", help="free-text reason for this crop")
    args = p.parse_args()

    arr, attrs = open_level(args.url, args.level)
    print(f"source shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")

    lo = np.array(args.origin)
    hi = lo + np.array(args.size)
    if (lo < 0).any() or (hi > np.array(arr.shape)).any():
        raise SystemExit(f"box {lo.tolist()}..{hi.tolist()} outside volume {arr.shape}")

    t0 = time.time()
    vol = arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    dt = time.time() - t0
    print(f"read {vol.nbytes / 1e6:.1f} MB in {dt:.1f}s")

    CROPS.mkdir(parents=True, exist_ok=True)
    out = CROPS / f"{args.name}.zarr"
    dst = zarr.create_array(out, shape=vol.shape, dtype=vol.dtype, chunks=(128, 128, 128), overwrite=True)
    dst[:] = vol

    meta = {
        "name": args.name,
        "source_url": args.url,
        "level": args.level,
        "origin_zyx": lo.tolist(),
        "size_zyx": list(args.size),
        "source_shape": list(arr.shape),
        "dtype": str(vol.dtype),
        "multiscales": attrs.get("multiscales"),
        "stats": {"min": float(vol.min()), "max": float(vol.max()), "mean": float(vol.mean())},
        "note": args.note,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "CC BY-NC 4.0 (Vesuvius Challenge open data)",
    }
    (CROPS / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    save_preview(vol, CROPS / f"{args.name}.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
