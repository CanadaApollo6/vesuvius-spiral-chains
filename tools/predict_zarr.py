"""Surface masks for a box of any OME-Zarr scroll volume with an nnU-Net v2 checkpoint (e.g. m7d300), tiled.

  uv run --with nnunetv2 --with "zarr==2.18.4" --with "numcodecs<0.16" --with s3fs --with tifffile \
      python tools/predict_zarr.py --model <trainer dir> --zarr s3://vesuvius-challenge-open-data/<scroll>/volumes/<vol>.zarr \
      --level 0 --um 9.362 --box 5000:5512,3000:3512,3000:3512 --out mask.zarr

Reads the box (plus a margin) in tiles, predicts each tile and writes uint8 {0, 255} into a zarr of the box's shape.
--um is the input voxel size: spacing = um / --ref-um (7.91, the grid m7d300 was evaluated at on PHerc0139 and hf2um).
Public buckets are read anonymously; nothing is downloaded beyond the requested box.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from infer_nnunet import load_predictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--level", default="0")
    ap.add_argument("--um", type=float, required=True, help="voxel size of the volume at --level (um)")
    ap.add_argument("--ref-um", type=float, default=7.91)
    ap.add_argument("--box", required=True, help="z0:z1,y0:y1,x0:x1 at --level")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--margin", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    so = {"anon": True} if args.zarr.startswith("s3://") else {}
    src = zarr.open_group(args.zarr, mode="r", storage_options=so)[args.level] if so else \
        zarr.open_group(args.zarr, mode="r")[args.level]
    box = [tuple(int(v) for v in s.split(":")) for s in args.box.split(",")]
    shape = tuple(b - a for a, b in box)
    out = zarr.open(str(args.out), mode="w", shape=shape, chunks=(128, 128, 128), dtype="u1")
    pred = load_predictor(args.model, fold=args.fold, tta=args.tta)
    spacing = [s * args.um / args.ref_um for s in pred.configuration_manager.spacing]
    T, M = args.tile, args.margin
    starts = [range(0, n, T) for n in shape]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                lo = [max(a + o - M, 0) for (a, _), o in zip(box, (z, y, x))]
                hi = [min(a + o + T + M, src.shape[d]) for d, ((a, _), o) in enumerate(zip(box, (z, y, x)))]
                ct = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
                _, prob = pred.predict_single_npy_array(ct[None], {"spacing": spacing}, None, None, True)
                surf = prob[1] > prob[0]
                # keep the tile core (without margins) and clip to the box
                o0 = [a + o for (a, _), o in zip(box, (z, y, x))]
                o1 = [min(o + T, b) for o, (_, b) in zip(o0, box)]
                core = surf[tuple(slice(p - l, q - l) for p, q, l in zip(o0, o1, lo))]
                out[tuple(slice(p - a, q - a) for p, q, (a, _) in zip(o0, o1, box))] = core.astype(np.uint8) * 255
                print(f"tile z{z} y{y} x{x}: surface {core.mean():.3f}", flush=True)
    out.attrs.update({"source": args.zarr, "level": args.level, "box": args.box, "model": str(args.model),
                      "spacing_declared": spacing})


if __name__ == "__main__":
    main()
