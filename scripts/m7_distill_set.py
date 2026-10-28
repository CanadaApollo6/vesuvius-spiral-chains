"""Dataset107_M7Distill: 192^3 CT crops from ~9 um scrolls, labelled with the bucket's m7 surface prediction.

  uv run --with zarr --with s3fs --with tifffile python scripts/m7_distill_set.py [--per-scroll 300]

Why: m7 + the frozen patch graph scores 80.5 % on PHerc0139 against real350's 76.6 %, but m7's weights are private
and its training data is undocumented. Distilling m7 from other scrolls into our own nnU-Net, with PHerc0139 never
included, gives a held-out number and an open checkpoint.

Crops sit on m7's 192^3 chunk grid (one m7 chunk each). The chunks are chosen at m7 level 4 (1/16) where the surface
fraction is 0.08-0.40 (papyrus, not empty or blank). Random with seed 0, then crops with > 10 % masked-zero CT are
dropped. Labels: 1 = m7 > 0, else 0 (no ignore). Native resolution with spacing 1, as Dataset103. Images are .tif
(Tiff3DIO).
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import zarr

REPO = Path(__file__).resolve().parents[1]
B = "s3://vesuvius-challenge-open-data/"
SCROLLS = {  # scroll: (CT volume, m7 L0 prediction), none of them PHerc0139
    "PHerc0814": ("20250804134230-9.362um-1.2m-113keV-masked.zarr", "20250804134230-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc0500P2": ("20250820143440-9.362um-1.2m-113keV-masked.zarr", "20250820143440-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc1447": ("20250521151220-8.640um-1.2m-116keV-masked.zarr", "20250521151220-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc0826": ("20250821151701-9.362um-1.2m-113keV-masked.zarr", "20250821151701-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc0211": ("20250821151803-9.362um-1.2m-113keV-masked.zarr", "20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc0125": ("20250821151825-9.362um-1.2m-113keV-masked.zarr", "20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
}
MORE = ["PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0257", "PHerc0268", "PHerc0306B", "PHerc0358", "PHerc0800",
        "PHerc0813", "PHerc1203", "PHerc1218", "PHerc1545", "PHerc0343"]  # --more: keys discovered in the bucket
C = 192
SO = {"anon": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scroll", type=int, default=300)
    ap.add_argument("--out", type=Path, default=REPO / "data/nnunet/nnUNet_raw/Dataset107_M7Distill")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--more", action="store_true", help="add the MORE scrolls (--per-scroll-more crops each)")
    ap.add_argument("--per-scroll-more", type=int, default=200)
    args = ap.parse_args()
    scrolls = {sc: (v, m, args.per_scroll) for sc, (v, m) in SCROLLS.items()}
    if args.more:
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        for sc in MORE:
            m7k = [q.split("/")[-1] for q in fs.ls(f"vesuvius-challenge-open-data/{sc}/representations/predictions/surfaces")
                   if "-m7-L0-" in q and q.endswith(".zarr")][0]
            vol = [q.split("/")[-1] for q in fs.ls(f"vesuvius-challenge-open-data/{sc}/volumes")
                   if q.split("/")[-1].startswith(m7k.split("-")[0])][0]
            scrolls[sc] = (vol, m7k, args.per_scroll_more)
    (args.out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (args.out / "labelsTr").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sc, (ctk, m7k, per) in scrolls.items():
        assert "0139" not in sc
        m7g = zarr.open_group(f"{B}{sc}/representations/predictions/surfaces/{m7k}", mode="r", storage_options=SO)
        ct = zarr.open_group(f"{B}{sc}/volumes/{ctk}", mode="r", storage_options=SO)["0"]
        m7 = m7g["0"]
        l4 = np.asarray(m7g["4"][:]) > 0  # 12^3 per L0 chunk
        n = np.array(l4.shape) // 12
        frac = l4[: n[0] * 12, : n[1] * 12, : n[2] * 12].reshape(n[0], 12, n[1], 12, n[2], 12).mean((1, 3, 5))
        cand = np.argwhere((frac >= 0.08) & (frac <= 0.40))
        rng = np.random.default_rng(0)
        pick = cand[rng.permutation(len(cand))[: int(per * 1.3)]]
        print(sc, "candidate chunks", len(cand), "picked", len(pick), flush=True)

        def one(c):
            z, y, x = (c * C).tolist()
            stem = f"{sc.lower()}_z{z}_y{y}_x{x}"
            fi, fl = args.out / "imagesTr" / f"{stem}_0000.tif", args.out / "labelsTr" / f"{stem}.tif"
            if fi.exists() and fl.exists():
                return stem
            a = np.asarray(ct[z:z + C, y:y + C, x:x + C])
            if a.shape != (C, C, C) or (a == 0).mean() > 0.10:
                return None
            lab = (np.asarray(m7[z:z + C, y:y + C, x:x + C]) > 0).astype(np.uint8)
            tifffile.imwrite(fi, a, compression="zlib")
            tifffile.imwrite(fl, lab, compression="zlib")
            return stem

        with ThreadPoolExecutor(args.workers) as ex:
            got = [s for s in ex.map(one, pick) if s][:per]
        manifest += [{"scroll": sc, "stem": s} for s in got]
        print(sc, "kept", len(got), flush=True)
    keep = {m["stem"] for m in manifest}
    for d in ("imagesTr", "labelsTr"):  # drop extras beyond per-scroll
        for f in (args.out / d).glob("*.tif"):
            if f.name.replace("_0000.tif", "").replace(".tif", "") not in keep:
                f.unlink()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "surface": 1}, "numTraining": len(manifest),
        "file_ending": ".tif", "overwrite_image_reader_writer": "Tiff3DIO",
        "name": "m7 pseudo-labels on ~9 um crops of " + "/".join(scrolls) + "; not PHerc0139"}, indent=1))
    print("total", len(manifest))


if __name__ == "__main__":
    main()
