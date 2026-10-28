"""Ablation launcher on nnU-Net v2 (3d_fullres), trainers built in-process so no files go into the nnunetv2 package.

  export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed \
         nnUNet_results=$PWD/data/nnunet/nnUNet_results
  uv run --with nnunetv2 python train/run.py --dataset Dataset101_DS059real --arm real    --epochs 250
  uv run --with nnunetv2 python train/run.py --dataset Dataset101_DS059real --arm squeeze --epochs 250
  uv run --with nnunetv2 python train/run.py --dataset Dataset102_SimV1     --arm real    --epochs 100 --tag simpre
  uv run --with nnunetv2 python train/run.py --dataset Dataset101_DS059real --arm real    --epochs 250 \
      --pretrained data/nnunet/nnUNet_results/Dataset102_SimV1/nnUNetTrainer_simpre__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth \
      --tag simft

Arms:
  real     nnU-Net default augmentation.
  squeeze  + SqueezeWarpTransform: local squeeze / stretch of the patch along a random direction (sheets pushed
           together or pulled apart) plus a smooth low-frequency warp; our stand-in for Camillo's decohesion / warp /
           squeeze augmentations (not found in the public vesuvius library, 2026-10-16).
Validation split: fixed 40 random cases (seed 0) held out of training, written to splits_final.json as fold 0.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

N_VAL = 40


class SqueezeWarpTransform(BasicTransform):
    """Output -> input map: along a random direction n through a random centre c,
    a' = a + k * s_a * tanh(a / s_a) * w_perp, with k = 1/f - 1 (f < 1 squeezes sheets together, f > 1 pulls apart),
    then a smooth random displacement (4^3 control grid, amplitude warp_vox). Image bilinear, segmentation nearest."""

    COARSE_STEP = 3.0

    def __init__(self, p=0.5, factor=(0.5, 1.6), slab=(6.0, 24.0), perp=(24.0, 96.0), warp_vox=3.0):
        super().__init__()
        self.p, self.factor, self.slab, self.perp, self.warp_vox = p, factor, slab, perp, warp_vox

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict["image"].shape[1:]
        if np.random.rand() > self.p:
            return {"grid": None}
        dev = data_dict["image"].device
        # The displacement is smooth (slab >= 6 vox), so it is built on a coarse grid (~COARSE_STEP vox) and upsampled
        # trilinearly: mean deviation from the full-resolution field 0.002 vox, max 0.6 vox (at slab = 6). At full
        # resolution it took 2.4 s per 205^3 sample on a busy cloud pod and starved the GPU (2026-10-17).
        m = [max(2, math.ceil((s - 1) / self.COARSE_STEP) + 1) for s in shape]
        axes = [torch.linspace(0, s - 1, mi, device=dev) for s, mi in zip(shape, m)]
        X = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)  # (3, mz, my, mx) voxel coords of coarse nodes
        n = torch.randn(3, device=dev)
        n = n / n.norm()
        c = torch.tensor([np.random.uniform(0.25, 0.75) * s for s in shape], device=dev)
        r = X - c[:, None, None, None]
        a = (r * n[:, None, None, None]).sum(0)
        rp = r - a[None] * n[:, None, None, None]
        f = math.exp(np.random.uniform(math.log(self.factor[0]), math.log(self.factor[1])))
        k = 1.0 / f - 1.0
        s_a = np.random.uniform(*self.slab)
        w = torch.exp(-(rp**2).sum(0) / (2 * np.random.uniform(*self.perp) ** 2))
        disp = (k * s_a * torch.tanh(a / s_a) * w)[None] * n[:, None, None, None]
        if self.warp_vox > 0:
            ctrl = torch.randn(1, 3, 4, 4, 4, device=dev) * self.warp_vox
            disp = disp + F.interpolate(ctrl, size=tuple(m), mode="trilinear", align_corners=True)[0]
        # grid_sample wants (x, y, z) in [-1, 1]. The identity part is linear, so upsampling it with the
        # displacement is exact
        scale = torch.tensor([2.0 / (s - 1) for s in shape], device=dev)[:, None, None, None]
        g = F.interpolate(((X + disp) * scale - 1)[None], size=tuple(shape), mode="trilinear", align_corners=True)[0]
        return {"grid": g.flip(0).permute(1, 2, 3, 0)[None]}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params["grid"] is None:
            return img
        return F.grid_sample(img[None].float(), params["grid"], mode="bilinear", padding_mode="border",
                             align_corners=True)[0].to(img.dtype)

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        if params["grid"] is None:
            return seg
        return F.grid_sample(seg[None].float(), params["grid"], mode="nearest", padding_mode="border",
                             align_corners=True)[0].round().to(seg.dtype)


class SepLoss(torch.nn.Module):
    """nnU-Net's loss with label 3 (contact separator, scripts/make_ds114.py) trained as background, plus
    w * mean CE over separator voxels at full resolution: separators are ~0.1-1 % of labelled voxels, so plain
    CE+Dice ignores them."""

    def __init__(self, base: torch.nn.Module, w: float):
        super().__init__()
        self.base, self.w = base, w

    def forward(self, output, target):
        ds = isinstance(target, (list, tuple))
        tg = list(target) if ds else [target]
        t0 = [torch.where(t == 3, torch.zeros_like(t), t) for t in tg]
        loss = self.base(output, t0 if ds else t0[0])
        o, sep = (output[0] if ds else output), tg[0][:, 0] == 3
        if sep.any():
            loss = loss - self.w * torch.log_softmax(o.float(), 1)[:, 0][sep].mean()
        return loss


def make_trainer(tag: str, epochs: int, squeeze: bool, lr: float | None = None, iters: int | None = None,
                 sep_weight: float | None = None):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class T(nnUNetTrainer):
        # explicit signature: nnUNetTrainer.__init__ records its args via inspect.signature(self.__init__) + locals()
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = epochs
            if lr is not None:
                self.initial_lr = lr  # fine-tuning: nnU-Net's 1e-2 default disturbs pretrained weights
            if iters is not None:
                self.num_iterations_per_epoch = iters
                self.num_val_iterations_per_epoch = max(10, iters // 5)

        def _build_loss(self):
            loss = super()._build_loss()
            return SepLoss(loss, sep_weight) if sep_weight is not None else loss

        def validation_step(self, batch: dict) -> dict:
            if sep_weight is not None:  # pseudo-dice one-hots the target: separators count as background there
                t = batch["target"]
                fix = lambda x: torch.where(x == 3, torch.zeros_like(x), x)
                batch["target"] = [fix(x) for x in t] if isinstance(t, list) else fix(t)
            return super().validation_step(batch)

        @staticmethod
        def get_training_transforms(*a, **k):
            t = nnUNetTrainer.get_training_transforms(*a, **k)
            if squeeze:
                # before nnU-Net's SpatialTransform: acts on the larger pre-crop region, so the far-field shift
                # (up to ~k * slab voxels) and border replication fall outside the final patch
                t.transforms.insert(0, SqueezeWarpTransform())
            return t

    T.__name__ = T.__qualname__ = f"nnUNetTrainer_{tag}"
    return T


def write_split(pre: Path, data_id: str = "nnUNetPlans_3d_fullres") -> None:
    f = pre / "splits_final.json"
    if f.exists():
        return
    cases = sorted(json.loads((pre / "dataset.json").read_text()).get("_cases", [])) or sorted(
        p.name.replace(".b2nd", "").replace(".npz", "") for p in (pre / data_id).glob("*")
        if p.suffix in (".b2nd", ".npz") and not p.name.endswith("_seg.b2nd"))
    rng = np.random.default_rng(0)
    val = sorted(rng.choice(cases, min(N_VAL, len(cases) // 5), replace=False).tolist())
    f.write_text(json.dumps([{"train": [c for c in cases if c not in val], "val": val}], indent=1))
    print(f"wrote {f}: {len(cases) - len(val)} train / {len(val)} val")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", choices=["real", "squeeze"], default="real")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plans", default="nnUNetPlans")
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--continue-training", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--sep-weight", type=float, default=None, help="label-3 separator CE weight (Dataset114)")
    args = ap.parse_args()
    tag = args.tag or f"{args.arm}{args.epochs}"
    pre = Path(os.environ["nnUNet_preprocessed"]) / args.dataset
    plans = json.loads((pre / f"{args.plans}.json").read_text())
    write_split(pre, plans["configurations"]["3d_fullres"]["data_identifier"])
    dataset_json = json.loads((pre / "dataset.json").read_text())
    plans["continue_training"] = args.continue_training  # this nnU-Net version pops it in nnUNetTrainer.__init__
    T = make_trainer(tag, args.epochs, args.arm == "squeeze", args.lr, args.iters, args.sep_weight)
    t = T(plans, "3d_fullres", 0, dataset_json, device=torch.device("cuda"))
    if args.continue_training:
        t.load_checkpoint(str(Path(t.output_folder) / "checkpoint_latest.pth"))
    elif args.pretrained:
        from nnunetv2.run.load_pretrained_weights import load_pretrained_weights

        t.initialize()
        load_pretrained_weights(t.network, args.pretrained, verbose=True)
    t.run_training()


if __name__ == "__main__":
    main()
