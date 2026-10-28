"""Run a downloaded nnU-Net v2 surface model on 7.91 um patches (jeong200 tifs or sim cell zarrs).

  uv run --with nnunetv2 python eval/infer_nnunet.py --model data/models/bshepp_nnunet --out data/preds/bshepp \
      [--images data/eval/jeong200/imagesTr] [--sim-cells data/sim/v1] [--limit N] [--tta] [--fold 0]

Checkpoints are untrusted pickles: loaded with torch weights_only=True (numpy scalars allowlisted) and handed to
nnUNetPredictor.manual_initialization, so nnU-Net's own unrestricted torch.load is never called.
Writes {stem}.tif, uint8, 1 = predicted surface (p(class 1) > p(class 0); class 2 = Kaggle 'unlabeled' ignored).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

REPO = Path(__file__).resolve().parents[1]
SAFE = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]


def load_predictor(model_dir: Path, fold: str = "all", tta: bool = False):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    with torch.serialization.safe_globals(SAFE):
        f = model_dir / f"fold_{fold}" / "checkpoint_final.pth"
        if not f.exists():  # e.g. scrollprize/surface_m7_nnunet ships only checkpoint_best.pth
            f = f.with_name("checkpoint_best.pth")
        ck = torch.load(f, map_location="cpu", weights_only=True)
    dataset_json = json.loads((model_dir / "dataset.json").read_text())
    plans = PlansManager(json.loads((model_dir / "plans.json").read_text()))
    conf = plans.get_configuration(ck["init_args"]["configuration"])
    n_in = determine_num_input_channels(plans, conf, dataset_json)
    lm = plans.get_label_manager(dataset_json)
    net = get_network_from_plans(conf.network_arch_class_name, conf.network_arch_init_kwargs,
                                 conf.network_arch_init_kwargs_req_import, n_in, lm.num_segmentation_heads,
                                 allow_init=True, deep_supervision=False)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta, perform_everything_on_device=True,
                           device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pred.manual_initialization(net, plans, conf, [ck["network_weights"]], dataset_json, ck["trainer_name"],
                               ck["inference_allowed_mirroring_axes"])
    return pred


def inputs(args):
    if args.images:
        for f in sorted(Path(args.images).glob("*_0000.tif")):
            yield f.name.replace("_0000.tif", ""), tifffile.imread(f)
    if args.sim_cells:
        import zarr

        for c in sorted(Path(args.sim_cells).glob("*.zarr")):
            yield c.stem, zarr.open_group(str(c), mode="r")["ct_7p91"][:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--sim-cells", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--fold", default="all", help="fold_<fold>/checkpoint_final.pth (our train/run.py models: 0)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest.json with per-stem 'um': declare spacing um / --ref-um x plans spacing (data/eval/hf2um)")
    ap.add_argument("--ref-um", type=float, default=7.91, help="voxel size of the model's training grid")
    ap.add_argument("--argmax", action="store_true", help="surface = argmax class 1 (separator-class models)")
    ap.add_argument("--save-prob", type=Path, default=None, help="also write the surface softmax as uint8 (x255) tifs here")
    args = ap.parse_args()
    if not args.images and not args.sim_cells:
        args.images = str(REPO / "data" / "eval" / "jeong200" / "imagesTr")
    args.out.mkdir(parents=True, exist_ok=True)
    pred = load_predictor(args.model, fold=args.fold, tta=args.tta)
    # all inputs are 7.91 um patches at the model's own grid: declare its plans spacing so nnU-Net never resamples
    # (bshepp plans: 7.91; our train/run.py models: nnU-Net's default 1.0)
    spacing = list(pred.configuration_manager.spacing)
    um = {m["stem"]: m["um"] for m in json.loads(args.manifest.read_text())} if args.manifest else {}
    for i, (stem, vol) in enumerate(inputs(args)):
        if args.limit and i >= args.limit:
            break
        dst = args.out / f"{stem}.tif"
        if dst.exists():
            continue
        seg, prob = pred.predict_single_npy_array(vol[None].astype(np.float32), {"spacing": [sp * um[stem] / args.ref_um for sp in spacing] if um else spacing},
                                                  None, None, True)
        # Kaggle label 2 is "unlabeled" (ignore); models trained on it predict it too. Decide surface vs background only.
        surf = seg == 1 if args.argmax or prob.shape[0] == 2 else prob[1] > prob[0]
        tifffile.imwrite(dst, surf.astype(np.uint8), compression="zlib")
        if args.save_prob:
            args.save_prob.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(args.save_prob / f"{stem}.tif", np.rint(255 * prob[1]).astype(np.uint8), compression="zlib")
        print(f"{stem}: surface frac {surf.mean():.3f} (argmax class-1 frac {(seg == 1).mean():.3f})", flush=True)


if __name__ == "__main__":
    main()
