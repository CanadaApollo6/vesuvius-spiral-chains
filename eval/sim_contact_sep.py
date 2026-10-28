"""Are fused contacts visible in sim CT? Merge rate of a surface model on held-out sim cells, for adjacent sheets that
touch vs adjacent sheets that don't.

  uv run --with nnunetv2 python eval/sim_contact_sep.py --model <nnU-Net trainer dir> --out data/preds/simsep_<tag>

Cells: Dataset102 fold-0 validation cells (never seen by simpre). Per cell, GT recto band B_i = surface & instance == i.
Sample points on sheet i next to sheet i+1, touching (contact < 2 um) or apart (>= 16 um); in a 25^3 window around
each, merged if B_i and B_{i+1} share their dominant predicted component (cc3d, 26-connected, window only); missed if
either band has < 30% of its window voxels predicted. If a sim-trained model keeps touching sheets apart on sim, the information is in
the rendered CT and the winding regression (train/winding.py) is what fails to use it.
"""

import argparse
import json
import sys
from pathlib import Path

import cc3d
import numpy as np
import zarr
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))


def pairs_in_cell(seg: np.ndarray, inst: np.ndarray, surf: np.ndarray, contact: np.ndarray, rng: np.random.Generator,
                  r: int = 12, per_pair: int = 6) -> list[dict]:
    """Local test at sample points: window of radius r around a point of sheet i near sheet i+1; components computed in
    the window only (as eval/separation.py). Touching points: contact < 2 um; apart points: contact >= 16 um."""
    touch = np.nan_to_num(contact.astype(np.float32), nan=1e6)
    band = surf > 0
    ids = [i for i in np.unique(inst) if i > 0]
    rows = []
    for i in ids:
        j = i + 1
        bi, bj = band & (inst == i), band & (inst == j)
        if bi.sum() < 200 or bj.sum() < 200:
            continue
        near_j = ndi.binary_dilation(inst == j, iterations=2)
        for kind, cand in (("touching", (inst == i) & near_j & (touch < 2.0)),
                           ("apart", (inst == i) & ndi.binary_dilation(inst == j, iterations=int(24 / 7.91) + 2)
                            & (touch >= 16.0))):
            pts = np.argwhere(cand)
            if len(pts) == 0:
                continue
            for c in pts[rng.choice(len(pts), min(per_pair, len(pts)), replace=False)]:
                lo, hi = np.maximum(c - r, 0), np.minimum(c + r + 1, seg.shape)
                w = tuple(slice(x, y) for x, y in zip(lo, hi))
                L = cc3d.connected_components(np.ascontiguousarray(seg[w] > 0), connectivity=26)
                Li, Lj = L[bi[w]], L[bj[w]]
                if (bi[w].sum() < 10) or (bj[w].sum() < 10):
                    continue
                hi_i, hi_j = (Li > 0).mean(), (Lj > 0).mean()
                missed = min(hi_i, hi_j) < 0.3
                di = np.bincount(Li[Li > 0]).argmax() if (Li > 0).any() else -1
                dj = np.bincount(Lj[Lj > 0]).argmax() if (Lj > 0).any() else -2
                rows.append({"i": int(i), "kind": kind, "missed": bool(missed), "merged": bool((not missed) and di == dj)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", type=Path, default=REPO / "data" / "sim" / "v1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from infer_nnunet import load_predictor

    pred = load_predictor(args.model, fold="0")
    spacing = list(pred.configuration_manager.spacing)
    split = json.loads((REPO / "data/nnunet/nnUNet_preprocessed/Dataset102_SimV1/splits_final.json").read_text())[0]
    rows, rng = [], np.random.default_rng(0)
    for name in split["val"][: args.limit or None]:
        g = zarr.open_group(str(args.cells / f"{name}.zarr"), mode="r")
        ct = g["ct_7p91"][:].astype(np.float32)
        seg = pred.predict_single_npy_array(ct[None], {"spacing": spacing}, None, None, False)
        r = pairs_in_cell(seg, g["instance_7p91"][:], g["surface_7p91"][:], g["contact_7p91"][:], rng)
        rows += [x | {"cell": name} for x in r]
        c = [x for x in r if x["kind"] == "touching"]
        print(f"{name}: {len(r)} points, {len(c)} touching, merged {sum(x['merged'] for x in c)}"
              f"/{sum(not x['missed'] for x in c)} hit", flush=True)
    out = {}
    for cls in ("touching", "apart"):
        sel = [x for x in rows if x["kind"] == cls]
        seen = [x for x in sel if not x["missed"]]
        out[cls] = {"points": len(sel), "missed": sum(x["missed"] for x in sel),
                    "merged_given_hit": 100 * sum(x["merged"] for x in seen) / max(len(seen), 1),
                    "separated_all": 100 * sum((not x["merged"]) and not x["missed"] for x in sel) / max(len(sel), 1)}
    out["model"] = str(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1))
    args.out.with_suffix(".rows.json").write_text(json.dumps(rows))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
