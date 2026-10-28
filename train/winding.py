"""Winding-field model (campaign W2): per voxel, regress the winding coordinate s from 7.91 um CT.

  export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...    (only the plans file is read)
  uv run --with nnunetv2 python train/winding.py --cells data/sim/v1 --out data/winding/w1 --epochs 60 \
      [--init data/nnunet/nnUNet_results/Dataset101_DS059real/nnUNetTrainer_real350__nnUNetPlans__3d_fullres]

s (sim/add_winding.py) = wrap index + radial fraction to the next wrap: continuous through contacts, +1 per wrap, so
recto faces are its integer level sets and a fused contact is a missing unit step. A 128^3 patch (~1 mm) cannot know
the absolute wrap number or which way the umbilicus lies, so the loss is invariant to a constant offset and to sign
(both are fixed later, globally, from the umbilicus). Network: nnU-Net's PlainConvUNet from the Dataset101 plans, one
input and one output channel, optionally initialised from a real-data surface model (all weights whose shapes match),
so it starts from real-CT features.
Validation (sim, held-out scroll seeds): after the best sign and shift per crop, the wrap error rate = share of
in-sheet voxels with |s_hat - s| > 0.5, overall and at contacts (gap_min < 2 um).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
# Dataset101 CTNormalization (nnU-Net plans): clip to [p0.5, p99.5], then z-score
CT_MEAN, CT_SD, CT_LO, CT_HI = 124.7, 40.35, 45.0, 229.0
VAL_SEEDS = (119, 120)
SAFE = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]


RES = "7p91"  # array suffix of the cells (--res; sim/generate.py --stage 9p362 writes *_9p362)


def keys() -> tuple:
    return tuple(f"{k}_{RES}" for k in ("ct", "wind", "instance", "contact"))


def open_cell(p: Path):
    """A sim cell: zarr group (sim/generate.py output) or a directory of .npy files (scripts/pack_wind.py, cloud)."""
    if p.suffix == ".zarr":
        return zarr.open_group(str(p), mode="r")
    return {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in keys()}


def has_wind(p: Path) -> bool:
    return (p / f"wind_{RES}").exists() or (p / f"wind_{RES}.npy").exists()


def norm_ct(x: np.ndarray) -> np.ndarray:
    return ((np.clip(x.astype(np.float32), CT_LO, CT_HI) - CT_MEAN) / CT_SD).astype(np.float32)


class SimCrops(Dataset):
    """Random patch^3 crops (ct, s) from sim cells, with random flips and axis permutations."""

    def __init__(self, cells: list[Path], patch: int, n: int):
        self.cells, self.patch, self.n = cells, patch, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([torch.initial_seed() % 2**32, i])
        P = self.patch
        for _ in range(20):
            g = open_cell(self.cells[rng.integers(len(self.cells))])
            o = [int(rng.integers(0, n - P + 1)) for n in g[f"ct_{RES}"].shape]
            sl = tuple(slice(a, a + P) for a in o)
            s = np.asarray(g[f"wind_{RES}"][sl], dtype=np.float32)
            fin = np.isfinite(s)
            if fin.mean() > 0.5 and np.nanmax(s) - np.nanmin(s) > 2:
                break
        ct = norm_ct(g[f"ct_{RES}"][sl])
        perm = rng.permutation(3)
        ct, s = ct.transpose(perm), s.transpose(perm)
        for ax in range(3):
            if rng.random() < 0.5:
                ct, s = np.flip(ct, ax), np.flip(s, ax)
        return torch.from_numpy(ct.copy())[None], torch.from_numpy(s.copy())[None]


def intensity_aug(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """nnU-Net-like intensity augmentation on normalized CT, per sample (GPU)."""
    B = x.shape[0]
    dev = x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
    out = []
    for b in range(B):
        v = x[b]
        if r(1) < 0.2:  # blur
            sig = 0.5 + r(1).item()
            k = torch.arange(-3, 4, device=dev, dtype=v.dtype)
            w = torch.exp(-(k**2) / (2 * sig**2))
            w = w / w.sum()
            for ax in range(3):  # separable; ax 0/1/2 = z/y/x of the (1, 1, Z, Y, X) tensor
                shape = [1, 1, 1, 1, 1]
                shape[2 + ax] = 7
                pad = [0] * 6  # F.pad order: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
                pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = 3
                v = F.conv3d(F.pad(v[None], pad, mode="replicate"), w.view(shape))[0]
        if r(1) < 0.15:  # noise
            v = v + torch.randn(v.shape, device=dev, generator=gen) * 0.1 * r(1)
        if r(1) < 0.15:  # brightness
            v = v * (0.75 + 0.5 * r(1))
        if r(1) < 0.15:  # contrast
            m = v.mean()
            v = (v - m) * (0.75 + 0.5 * r(1)) + m
        if r(1) < 0.3:  # gamma on min-max rescaled values
            lo, hi = v.min(), v.max()
            g = 0.7 + 0.8 * r(1)
            v = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1) ** g * (hi - lo) + lo
        out.append(v)
    return torch.stack(out)


def wind_loss(pred: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Offset- and sign-invariant Huber regression over voxels where s is defined."""
    losses = []
    for b in range(pred.shape[0]):
        m = torch.isfinite(s[b])
        p, t = pred[b][m].float(), s[b][m]
        t = t - t.mean()
        best = None
        for sgn in (1.0, -1.0):
            d = p - sgn * t
            d = d - d.mean()
            l = F.huber_loss(d, torch.zeros_like(d), delta=0.5)
            best = l if best is None else torch.minimum(best, l)
        losses.append(best)
    return torch.stack(losses).mean()


def phase_loss(pred: torch.Tensor, s: torch.Tensor, lam: float) -> torch.Tensor:
    """wind_loss on channel 0 plus lam * squared error of channels 1-2 against (cos 2 pi s, sin 2 pi s). Integer level
    sets of s are the recto faces, so the phase is absolute (only the sign is ambiguous); the sign is shared by both
    terms. The phase is a sharp local target: it marks every face, including inside fused contacts."""
    losses = []
    for b in range(pred.shape[0]):
        m = torch.isfinite(s[b, 0])
        t = s[b, 0][m]
        ps, pc, pn = (pred[b, c][m].float() for c in range(3))
        ct, st = torch.cos(2 * math.pi * t), torch.sin(2 * math.pi * t)
        best = None
        for sgn in (1.0, -1.0):
            d = ps - sgn * (t - t.mean())
            d = d - d.mean()
            l = F.huber_loss(d, torch.zeros_like(d), delta=0.5) + lam * ((pc - ct) ** 2 + (pn - sgn * st) ** 2).mean()
            best = l if best is None else torch.minimum(best, l)
        losses.append(best)
    return torch.stack(losses).mean()


def decode_phase(p: np.ndarray) -> np.ndarray:
    """Phase unwrapping guided by the coarse channel: s = s_c + wrap(phi - s_c), phi = atan2(sin, cos) / 2 pi."""
    phi = np.arctan2(p[2], p[1]) / (2 * np.pi)
    wrap = lambda x: x - np.rint(x)  # noqa: E731
    # channel 0 is offset-free: first shift it onto the phase anchor (circular mean of phi - s_c)
    s_c = p[0] + np.angle(np.mean(np.exp(2j * np.pi * (phi - p[0])))) / (2 * np.pi)
    return s_c + wrap(phi - s_c)


def grad_loss(pred: torch.Tensor, s: torch.Tensor, scales=(1, 3, 9)) -> torch.Tensor:
    """Offset-free by construction: Huber on finite differences of s along each axis at several scales (error in wraps
    per k voxels, divided by k), best sign per sample. Penalizes local slope errors (a fused contact is a missing
    step over ~one sheet thickness) without charging a single miscount to every voxel of the crop."""
    losses = []
    for b in range(pred.shape[0]):
        best = None
        for sgn in (1.0, -1.0):
            tot = 0.0
            for k in scales:
                for ax in range(3):
                    n = s.shape[2 + ax] - k
                    dt = s[b, 0].narrow(ax, k, n) - s[b, 0].narrow(ax, 0, n)
                    dp = pred[b, 0].float().narrow(ax, k, n) - pred[b, 0].float().narrow(ax, 0, n)
                    m = torch.isfinite(dt)
                    tot = tot + F.huber_loss((dp[m] - sgn * dt[m]) / k, torch.zeros_like(dt[m]), delta=0.05)
            tot = tot / (3 * len(scales))
            best = tot if best is None else torch.minimum(best, tot)
        losses.append(best)
    return torch.stack(losses).mean()


def pair_scores(pred: np.ndarray, s: np.ndarray, contact: np.ndarray, rng: np.random.Generator, n: int = 3000,
                tol: float = 0.25) -> dict:
    """Exact, pitch-proof check: voxel pairs a, b = a +- L*u along the true winding direction u (L 2-30 vox). Correct if
    |d_hat - d| < tol wrap, d = s(b) - s(a). Also for pairs whose segment crosses a contact (gap < 2 um), and for a
    crop-oracle-pitch baseline d_base = (b - a).u * median |grad s| (a perfect pitch estimator that resolves nothing)."""
    s = s.astype(np.float32)
    gs = np.stack(np.gradient(s), 0)
    mag = np.sqrt((gs**2).sum(0))
    ok = np.isfinite(mag) & (mag > 1e-3)
    idx = np.argwhere(ok)
    if len(idx) < 100:
        return {}
    a = idx[rng.integers(0, len(idx), n)]
    u = gs[(slice(None), *a.T)].T / mag[tuple(a.T)][:, None]
    step = (rng.uniform(2, 30, n) * rng.choice([-1.0, 1.0], n))[:, None] * u
    b = np.rint(a + step).astype(int)
    keep = np.all((b >= 0) & (b < np.array(s.shape)), 1)
    a, b, u = a[keep], b[keep], u[keep]
    keep = np.isfinite(s[tuple(b.T)])
    a, b, u = a[keep], b[keep], u[keep]
    fin = np.isfinite(s)
    sg = 1.0 if np.mean((pred[fin] - pred[fin].mean()) * (s[fin] - s[fin].mean())) >= 0 else -1.0
    d = s[tuple(b.T)] - s[tuple(a.T)]
    dh = sg * (pred[tuple(b.T)] - pred[tuple(a.T)])
    base = ((b - a) * u).sum(1) * float(np.median(mag[ok]))
    t = np.linspace(0, 1, 16)[:, None, None]
    seg = np.rint(a[None] + t * (b - a)[None]).astype(int)  # [16, n, 3]
    cv = np.nan_to_num(contact[seg[..., 0], seg[..., 1], seg[..., 2]].astype(np.float32), nan=1e6)
    cross = (cv < 2.0).any(0)
    good, bgood = np.abs(dh - d) < tol, np.abs(base - d) < tol
    return {"n": int(len(d)), "ok": int(good.sum()), "base_ok": int(bgood.sum()), "n_c": int(cross.sum()),
            "ok_c": int(good[cross].sum()), "base_ok_c": int(bgood[cross].sum())}


def wrap_errors(pred: np.ndarray, s: np.ndarray, inst: np.ndarray, contact: np.ndarray) -> dict:
    """After the best sign and median shift: share of in-sheet voxels with |error| > 0.5 (all / at contacts)."""
    m = inst > 0
    best = None
    for sgn in (1.0, -1.0):
        e = pred[m] - sgn * s[m]
        e = e - np.median(e)
        bad = np.abs(e) > 0.5
        if best is None or bad.mean() < best[0].mean():
            best = (bad, e)
    bad = best[0]
    c = np.nan_to_num(contact[m].astype(np.float32), nan=1e6) < 2.0
    return {"err": float(bad.mean()), "err_contact": float(bad[c].mean()) if c.any() else float("nan"),
            "n_contact": int(c.sum()), "n": int(m.sum())}


def build_net(init: Path | None, out: int = 1):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    src = init if init else None
    if src:
        plans = PlansManager(json.loads((src / "plans.json").read_text()))
    else:
        import os

        pre = Path(os.environ["nnUNet_preprocessed"]) / "Dataset101_DS059real"
        plans = PlansManager(json.loads((pre / "nnUNetPlans.json").read_text()))
    conf = plans.get_configuration("3d_fullres")
    net = get_network_from_plans(conf.network_arch_class_name, conf.network_arch_init_kwargs,
                                 conf.network_arch_init_kwargs_req_import, 1, out, allow_init=True,
                                 deep_supervision=False)
    if src:
        with torch.serialization.safe_globals(SAFE):
            ck = torch.load(src / "fold_0" / "checkpoint_final.pth", map_location="cpu", weights_only=True)
        own = net.state_dict()
        keep = {k: v for k, v in ck["network_weights"].items() if k in own and own[k].shape == v.shape}
        net.load_state_dict(keep, strict=False)
        print(f"init from {src.name}: {len(keep)}/{len(own)} tensors", flush=True)
    return net


@torch.no_grad()
def validate(net, cells: list[Path], patch: int, dev) -> dict:
    """Fixed crops (seed 0) from held-out cells: 2 per cell."""
    net.eval()
    rng = np.random.default_rng(0)
    rows, prs, prs_raw = [], [], []
    prng = np.random.default_rng(1)
    for c in cells:
        g = open_cell(c)
        for _ in range(2):
            o = [int(rng.integers(0, n - patch + 1)) for n in g[f"ct_{RES}"].shape]
            sl = tuple(slice(a, a + patch) for a in o)
            x = torch.from_numpy(norm_ct(g[f"ct_{RES}"][sl]))[None, None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)[0].float().cpu().numpy()
            p = decode_phase(out) if out.shape[0] == 3 else out[0]
            sw, ct = np.asarray(g[f"wind_{RES}"][sl], dtype=np.float32), np.asarray(g[f"contact_{RES}"][sl])
            rows.append(wrap_errors(p, sw, g[f"instance_{RES}"][sl], ct))
            prs.append(pair_scores(p, sw, ct, prng))
            if out.shape[0] == 3:
                prs_raw.append(pair_scores(out[0], sw, ct, np.random.default_rng(len(prs))))
    net.train()
    n = sum(r["n"] for r in rows)
    nc = sum(r["n_contact"] for r in rows)
    return {"err": sum(r["err"] * r["n"] for r in rows) / n,
            "err_contact": sum(r["err_contact"] * r["n_contact"] for r in rows if r["n_contact"]) / max(nc, 1),
            "crops": len(rows), **pair_summary(prs),
            **({"pair_coarse": pair_summary(prs_raw)["pair"]} if prs_raw else {})}


def pair_summary(prs: list[dict]) -> dict:
    t = {k: sum(r.get(k, 0) for r in prs) for k in ("n", "ok", "base_ok", "n_c", "ok_c", "base_ok_c")}
    return {"pair": t["ok"] / max(t["n"], 1), "pair_base": t["base_ok"] / max(t["n"], 1),
            "pair_contact": t["ok_c"] / max(t["n_c"], 1), "pair_contact_base": t["base_ok_c"] / max(t["n_c"], 1),
            "pair_n_contact": t["n_c"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=REPO / "data" / "sim" / "v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="nnU-Net trainer folder (plans.json, fold_0/checkpoint_final.pth)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--res", default="7p91", help="cell array suffix: 7p91, 9p362, 8p64")
    ap.add_argument("--grad-w", type=float, default=0.0, help="weight of the multi-scale gradient loss (grad_loss)")
    ap.add_argument("--phase-w", type=float, default=0.0, help="> 0: 3 outputs (s, cos, sin) and phase_loss")
    args = ap.parse_args()
    global RES
    RES = args.res
    args.out.mkdir(parents=True, exist_ok=True)
    cells = sorted(p for p in args.cells.iterdir() if p.name.startswith("s") and has_wind(p))
    seed_of = lambda p: int(p.name[1:4])  # noqa: E731
    train = [p for p in cells if seed_of(p) not in VAL_SEEDS]
    val = [p for p in cells if seed_of(p) in VAL_SEEDS]
    print(f"{len(train)} train / {len(val)} val cells", flush=True)
    dev = torch.device("cuda")
    net = build_net(args.init, out=3 if args.phase_w else 1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.iters
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))
    dl = DataLoader(SimCrops(train, args.patch, args.iters * args.batch), batch_size=args.batch,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    log = (args.out / "log.jsonl").open("a")
    for ep in range(args.epochs):
        t0, tot = time.time(), 0.0
        for x, s in dl:
            x, s = intensity_aug(x.to(dev, non_blocking=True), gen), s.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(x)
            loss = phase_loss(p, s, args.phase_w) if args.phase_w else wind_loss(p, s)
            if args.grad_w:
                loss = loss + args.grad_w * grad_loss(p[:, :1], s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / args.iters, "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1)}
        if (ep + 1) % args.val_every == 0 or ep == args.epochs - 1:
            if val:
                rec |= {"val_" + k: v for k, v in validate(net, val, args.patch, dev).items()}
            torch.save({"net": net.state_dict(), "epoch": ep, "args": vars(args) | {"cells": str(args.cells),
                        "out": str(args.out), "init": str(args.init)}}, args.out / "checkpoint.pth")
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
