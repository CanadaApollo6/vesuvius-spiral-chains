"""Generate synthetic scroll cells: the same geometry rendered as 7.91 um and 2.4 um CT, with exact labels.

  uv run python sim/generate.py --out data/sim/v1 --scroll-seed 1 --cells 4 [--cell-seed 0] [--size 256] [--hi-size 256]

Per cell writes <out>/<name>.zarr with arrays
  ct_7p91 uint8, surface_7p91 uint8 (Dataset059-style recto band), instance_7p91 uint16 (sheet id, exact through
  contacts), contact_7p91 float16 (world um to the nearest other sheet, NaN outside sheets)
  ct_2p4, surface_2p4, instance_2p4, contact_2p4: central hi_size^3 block at 2.399 um, same physical region centre
and attrs {scroll params, CT params, seeds, world origins/voxel sizes}; plus <name>.png preview.
The 7.91 scan is derived from the clean 2.4 um image exactly as measured on real data (forward/ct.py).

  --stage 9p362 (or 8p64): ESRF Grand Prize-style cells instead. ct_<key> straight from density with the fitted
  ScanStage (forward/fit9.py), plus instance/surface/contact/wind_<key> on the same grid; no 2.4 block.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "sim"), str(REPO / "forward")]
from ct import (VOX_HI, VOX_LO, CTParams, ScanStage, gauss3, hi_clean, hi_clean_halo, hi_scan, lo_scan,  # noqa: E402
                slab_map, stage_halo, stage_scan)
from render import render_grid  # noqa: E402
from scroll import Scroll, ScrollParams  # noqa: E402

LABEL_KEYS = ("instance", "surface", "contact_um")


def pick_centre(S: Scroll, rng: np.random.Generator, size_um: float, min_sheet=0.3, max_sheet=0.8, tries=200):
    """Random world point well inside the scroll whose neighbourhood is part sheet, part gap."""
    p = S.p
    R = S.R.cpu()
    r_out = float(R[-1].min())
    zmax = p.z_extent_um / 2 - size_um / 2 - 100
    for _ in range(tries):
        rad = rng.uniform(p.r0 + size_um, r_out - size_um)
        ang = rng.uniform(0, 2 * math.pi)
        c = np.array([rad * math.cos(ang) * p.squash[0], rad * math.sin(ang) * p.squash[1], rng.uniform(-zmax, zmax)])
        ca, sa = math.cos(p.squash_angle), math.sin(p.squash_angle)
        c[:2] = [ca * c[0] - sa * c[1], sa * c[0] + ca * c[1]]
        probe = torch.tensor(c, dtype=torch.float32, device=S.dev) + (torch.rand(2000, 3, device=S.dev) - 0.5) * size_um
        o = S.sample(probe, want_density=False)
        frac = float((o["instance"] > 0).float().mean())
        if min_sheet <= frac <= max_sheet and bool((o["k"] >= 0).all() & (o["k"] < p.n_wraps).all()):
            return c, frac
    raise RuntimeError("no valid cell centre found")


def make_cell(S: Scroll, ctp: CTParams, centre, size: int, hi_size: int, gen: torch.Generator) -> dict:
    S.set_reference(torch.tensor(centre, dtype=torch.float32, device=S.dev))
    step = VOX_LO / VOX_HI
    margin = int(math.ceil(3.5 * math.hypot(ctp.blur_lo_vox_hi, ctp.psf_sigma_um / VOX_HI, ctp.fringe_sigma_um / VOX_HI))) + 2
    n_hi = int(math.ceil((size - 1) * step)) + 1 + 2 * margin
    lo_origin = np.asarray(centre) - (size - 1) / 2 * VOX_LO  # world xyz of 7.91 voxel (0,0,0)
    hi_origin = lo_origin - margin * VOX_HI
    t0 = time.time()
    rho = render_grid(S, hi_origin, (n_hi,) * 3, VOX_HI, keys=("density",))["density"]
    t_render = time.time() - t0
    # forward model in z-slabs: full 2.4 um volumes (~850^3) stay on the CPU, only the pre-blurred one goes to the GPU
    rho = torch.from_numpy(rho)
    clean = slab_map(lambda x: hi_clean(x, ctp), rho, hi_clean_halo(ctp), "cpu", dev=S.dev)
    del rho
    # 2.4 um block centred on the cell centre
    c_hi = margin + (size - 1) * step / 2  # hi-voxel coordinate of the centre
    a = int(round(c_hi - (hi_size - 1) / 2))
    hi = hi_scan(clean[a : a + hi_size, a : a + hi_size, a : a + hi_size].to(S.dev), ctp, gen)
    hi_block_origin = hi_origin + a * VOX_HI
    blur_halo = int(math.ceil(3.5 * ctp.blur_lo_vox_hi)) + 1
    b = slab_map(lambda x: gauss3(x, ctp.blur_lo_vox_hi), clean, blur_halo, S.dev, dev=S.dev)
    del clean
    lo = lo_scan(b, ctp, gen, out_shape=(size,) * 3, offset_hi=(margin,) * 3, preblurred=True)
    del b
    torch.cuda.empty_cache()
    lab_lo = render_grid(S, lo_origin, (size,) * 3, VOX_LO, want_density=False, keys=LABEL_KEYS)
    lab_hi = render_grid(S, hi_block_origin, (hi_size,) * 3, VOX_HI, want_density=False, keys=LABEL_KEYS)
    return {
        "ct_7p91": lo.to(torch.uint8).cpu().numpy(), "ct_2p4": hi.to(torch.uint8).cpu().numpy(),
        **{f"{k.replace('_um', '')}_7p91": v for k, v in lab_lo.items()},
        **{f"{k.replace('_um', '')}_2p4": v for k, v in lab_hi.items()},
        "_meta": {"centre_world_um": list(map(float, centre)), "lo_origin_world_um": lo_origin.tolist(),
                  "hi_origin_world_um": hi_block_origin.tolist(), "vox_lo_um": VOX_LO, "vox_hi_um": VOX_HI,
                  "axis_order": "zyx (array) = world (z, y, x); world z = scroll axis", "render_s": t_render,
                  "n_hi_render": n_hi},
    }


def make_cell_stage(S: Scroll, st: ScanStage, key: str, centre, size: int, gen: torch.Generator) -> dict:
    """ESRF cell: ct_<key> from density via stage_scan, labels (incl. the winding coordinate) on the same grid."""
    S.set_reference(torch.tensor(centre, dtype=torch.float32, device=S.dev))
    step = st.vox_um / VOX_HI
    pad = int(math.ceil(3.5 * abs(st.fringe_sigma_um) / st.vox_um)) + 1  # output voxels lost to the fringe filter edge
    margin = stage_halo(st) + int(math.ceil(pad * step))
    n_hi = int(math.ceil((size - 1) * step)) + 1 + 2 * margin
    origin = np.asarray(centre) - (size - 1) / 2 * st.vox_um  # world xyz of output voxel (0,0,0)
    hi_origin = origin - margin * VOX_HI
    t0 = time.time()
    rho = render_grid(S, hi_origin, (n_hi,) * 3, VOX_HI, keys=("density",))["density"]
    t_render = time.time() - t0
    b = slab_map(lambda x: gauss3(x, st.psf_sigma_um / VOX_HI), torch.from_numpy(rho),
                 int(math.ceil(3.5 * st.psf_sigma_um / VOX_HI)) + 1, S.dev, dev=S.dev)
    del rho
    ct = stage_scan(b, st, gen, out_shape=(size + 2 * pad,) * 3, offset_hi=(margin - pad * step,) * 3, preblurred=True)
    ct = ct[pad:-pad, pad:-pad, pad:-pad]
    del b
    torch.cuda.empty_cache()
    lab = render_grid(S, origin, (size,) * 3, st.vox_um, want_density=False, keys=LABEL_KEYS + ("wind",))
    return {
        f"ct_{key}": ct.to(torch.uint8).cpu().numpy(),
        **{f"{k.replace('_um', '')}_{key}": v for k, v in lab.items()},
        "_meta": {"centre_world_um": list(map(float, centre)), "origin_world_um": origin.tolist(), "vox_um": st.vox_um,
                  "stage": key, "axis_order": "zyx (array) = world (z, y, x); world z = scroll axis",
                  "render_s": t_render, "n_hi_render": n_hi},
    }


def save_cell(out: Path, name: str, cell: dict, attrs: dict) -> None:
    g = zarr.open_group(str(out / f"{name}.zarr"), mode="w")
    for k, v in cell.items():
        if k.startswith("_"):
            continue
        if k.startswith("contact"):
            v = v.astype(np.float16)
        if k.startswith("wind"):
            v = v.astype(np.float32)
        g.create_array(k, data=v, chunks=(min(128, v.shape[0]),) * 3)
    g.attrs.update(attrs | {"cell": cell["_meta"]})
    if "ct_7p91" in cell:
        preview(out / f"{name}.png", cell)


def preview(path: Path, cell: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = cell["ct_7p91"], cell["ct_2p4"]
    zl, zh = lo.shape[0] // 2, hi.shape[0] // 2
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax[0, 0].imshow(lo[zl], cmap="gray", vmin=40, vmax=200)
    ax[0, 1].imshow(np.ma.masked_equal(cell["surface_7p91"][zl], 0), cmap="autumn", alpha=1.0)
    ax[0, 1].imshow(lo[zl], cmap="gray", vmin=40, vmax=200, alpha=0.6)
    inst = cell["instance_7p91"][zl]
    ax[0, 2].imshow(np.where(inst > 0, inst % 10 + 1, 0), cmap="tab20", interpolation="nearest")
    ax[0, 3].imshow(np.minimum(np.nan_to_num(cell["contact_7p91"][zl], nan=200), 200), cmap="magma")
    ax[1, 0].imshow(hi[zh], cmap="gray", vmin=20, vmax=220)
    ax[1, 1].imshow(lo[:, lo.shape[1] // 2], cmap="gray", vmin=40, vmax=200)
    insth = cell["instance_2p4"][zh]
    ax[1, 2].imshow(np.where(insth > 0, insth % 10 + 1, 0), cmap="tab20", interpolation="nearest")
    ax[1, 3].imshow(np.minimum(np.nan_to_num(cell["contact_2p4"][zh], nan=200), 200), cmap="magma")
    titles = ["CT 7.91 (z mid)", "surface label 7.91", "instance 7.91", "contact um 7.91 (dark = touching)",
              "CT 2.4 (central block)", "CT 7.91 (y mid, scroll axis vertical)", "instance 2.4", "contact um 2.4"]
    for a, t in zip(ax.ravel(), titles):
        a.set_title(t)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


# per-scroll domain randomization (--randomize). Contact-interface contrast (interface_dip, contact_loose) cannot be
# measured on real data (no label has identity at contacts to < 10 um), so it is swept rather than fixed; the other
# ranges bracket the defaults.
RANDOM_RANGES = {
    "interface_dip": (0.0, 0.7),
    "contact_frac": (0.08, 0.30),
    "t_mean": (65.0, 105.0),
    "g_mean": (40.0, 100.0),
    "t_rel_sd": (0.15, 0.35),
    "disp_corr_radial_um": (300.0, 800.0),
    "disp_corr_tangential_um": (1200.0, 3500.0),
    "contact_loose": (0.0, 1.0),
    "face_loose": (0.2, 0.5),
    "delam_thresh": (1.5, 2.4),
    "void_density": (0.25, 0.50),
    "porosity_sd": (0.08, 0.20),
    "fiber_across_um": (4.0, 7.0),
    "surface_rough_um": (2.0, 6.0),
    "incl_prob": (0.0, 0.12),
    "squash_angle": (0.0, float(np.pi)),
}


def jitter_stage(st: ScanStage, rng: np.random.Generator) -> ScanStage:
    """Per-cell scanner variation around a fitted ESRF stage (real scans differ: PHerc1447 at 8.64 um is ~10 grey darker
    than PHerc0826, fringe depth varies 8-18 grey between crops of one scan)."""
    return ScanStage(**{**asdict(st), "air": st.air + rng.uniform(-4, 4), "gain": st.gain * rng.uniform(0.85, 1.12),
                        "psf_sigma_um": st.psf_sigma_um * rng.uniform(0.8, 1.25),
                        "fringe_alpha": st.fringe_alpha * rng.uniform(0.5, 1.5),
                        "fringe_sigma_um": st.fringe_sigma_um * rng.uniform(0.85, 1.2),
                        "n_std": st.n_std * rng.uniform(0.8, 1.4)})


def randomize(scroll_seed: int) -> dict:
    rng = np.random.default_rng([scroll_seed, 7919])
    out = {k: float(rng.uniform(*r)) for k, r in RANDOM_RANGES.items()}
    out["squash"] = (1.0, float(rng.uniform(0.45, 0.85)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "sim" / "v1")
    ap.add_argument("--scroll-seed", type=int, default=1)
    ap.add_argument("--cell-seed", type=int, default=0)
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--hi-size", type=int, default=256)
    ap.add_argument("--params", type=Path, default=None, help="JSON overrides for ScrollParams")
    ap.add_argument("--randomize", action="store_true", help="draw ScrollParams from RANDOM_RANGES (seeded by scroll seed)")
    ap.add_argument("--stage", default=None, help="ESRF stage key in forward/ct_params.json (9p362, 8p64)")
    ap.add_argument("--stage-jitter", action="store_true", help="per-cell scanner variation (jitter_stage)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    over = randomize(args.scroll_seed) if args.randomize else {}
    over |= json.loads(args.params.read_text()) if args.params else {}
    sp = ScrollParams(**{**over, "seed": args.scroll_seed})
    ctp = CTParams.load()
    st = ScanStage.load(args.stage) if args.stage else None
    t0 = time.time()
    S = Scroll(sp)
    print(f"scroll seed {sp.seed}: tables {time.time() - t0:.1f}s, contact area {S.contact_area_frac:.3f}", flush=True)
    rng = np.random.default_rng([args.scroll_seed, args.cell_seed])
    jrng = np.random.default_rng([args.scroll_seed, args.cell_seed, 31])  # separate stream: centres don't move
    size_um = args.size * (st.vox_um if st else VOX_LO)
    for i in range(args.cells):
        name = f"s{args.scroll_seed:03d}_c{args.cell_seed:03d}_{i:03d}"
        centre, frac = pick_centre(S, rng, size_um)
        gen = torch.Generator(device=S.dev).manual_seed(hash((args.scroll_seed, args.cell_seed, i)) % (2**31))
        t1 = time.time()
        if st:
            st_i = jitter_stage(st, jrng) if args.stage_jitter else st
            cell, sfx = make_cell_stage(S, st_i, args.stage, centre, args.size, gen), args.stage
            ct_attrs = {"stage_key": args.stage, "stage_params": asdict(st_i)}
        else:
            cell, sfx = make_cell(S, ctp, centre, args.size, args.hi_size, gen), "7p91"
            ct_attrs = {"ct_params": asdict(ctp)}
        attrs = {"scroll_params": sp.to_json(), **ct_attrs, "scroll_seed": args.scroll_seed,
                 "cell_seed": args.cell_seed, "cell_index": i, "generator": "sim/generate.py"}
        save_cell(args.out, name, cell, attrs)
        n_inst = int(cell[f"instance_{sfx}"].max())
        print(f"{name}: centre {np.round(centre).tolist()} sheet frac {frac:.2f} sheets {n_inst} "
              f"surface frac {cell[f'surface_{sfx}'].mean():.3f} {time.time() - t1:.1f}s "
              f"(render {cell['_meta']['render_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
