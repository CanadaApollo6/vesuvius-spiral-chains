"""Render a Scroll onto a regular voxel grid (slab by slab, on the GPU)."""

import torch

from scroll import Scroll


# GT surface band (Dataset059 style), measured on real jeong200 patches: starts at the recto face and runs ~4.5 vox
# (7.91 um) into the sheet. In world um:
SURFACE_DEPTH_UM = 34.0
SURFACE_FRONT_UM = 2.0  # small margin in front of the face (gap side)


def derived_labels(S: Scroll, pts: torch.Tensor, o: dict) -> dict:
    """World-metric labels: recto surface band, distance to the nearest other sheet (contact), depth in sheet."""
    sc = S.metric_scale(pts)
    in_sheet = o["instance"] > 0
    depth_w = o["depth"] / sc
    front_w = o["to_next"] / sc
    gap_after = ~in_sheet & (o["depth"] >= o["thick"]) & (o["k"] >= 0) & (o["k"] < S.p.n_wraps)
    surface = (in_sheet & (depth_w < SURFACE_DEPTH_UM)) | (gap_after & (front_w < SURFACE_FRONT_UM))
    return {"surface": surface.to(torch.uint8), "contact_um": torch.where(in_sheet, o["gap_min"] / sc, torch.nan),
            "depth_um": torch.where(in_sheet, depth_w, torch.nan)}


@torch.no_grad()
def render_grid(S: Scroll, origin_um, shape, vox_um: float, want_density: bool = True,
                keys=("density", "instance", "gap_min"), max_pts: int = 1 << 20) -> dict:
    """Sample S on a zyx grid. origin_um = world (x, y, z) of voxel (0, 0, 0); world z = scroll axis = array axis 0.

    Returns numpy arrays (z, y, x) for the requested keys. Call S.set_reference() near the grid centre first.
    """
    nz, ny, nx = shape
    dev = S.dev
    ox, oy, oz = (float(v) for v in origin_um)
    yy, xx = torch.meshgrid(torch.arange(ny, device=dev, dtype=torch.float32) * vox_um + oy,
                            torch.arange(nx, device=dev, dtype=torch.float32) * vox_um + ox, indexing="ij")
    xy = torch.stack([xx.flatten(), yy.flatten()], -1)
    per = max(1, max_pts // (ny * nx))
    bufs = {k: [] for k in keys if want_density or k != "density"}
    for z0 in range(0, nz, per):
        zs = torch.arange(z0, min(nz, z0 + per), device=dev, dtype=torch.float32) * vox_um + oz
        pts = torch.cat([xy.repeat(len(zs), 1), zs.repeat_interleave(ny * nx)[:, None]], -1)
        o = S.sample(pts, want_density=want_density)
        if any(k in bufs for k in ("surface", "contact_um", "depth_um")):
            o |= derived_labels(S, pts, o)
        for k in bufs:
            bufs[k].append(o[k].reshape(len(zs), ny, nx).cpu())
    out = {}
    for k, v in bufs.items():
        a = torch.cat(v)
        out[k] = a.numpy().astype("uint16") if k == "instance" else a.numpy()
    if "instance" in out:
        out["instance"] = relabel(out["instance"])
    return out


def relabel(inst):
    """Map wrap ids in a grid to 1..n (0 stays background); keeps order (inner wraps first)."""
    import numpy as np

    u = np.unique(inst)
    u = u[u > 0]
    lut = np.zeros(int(inst.max()) + 1, dtype=np.uint16)
    lut[u] = np.arange(1, len(u) + 1, dtype=np.uint16)
    return lut[inst]
