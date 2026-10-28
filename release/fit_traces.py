"""Shared helpers for the PHerc0139 (and, via use_scroll("p1667"), PHerc1667) fit figures: z-slice traces of tifxyz surfaces, human-frame winding mapping, CT crops.

All coordinates are 9.362 um L0 voxels of 20250728140407 (zyx). Not a figure script on its own.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import find_contours

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402

ARMS = REPO / "data" / "scratch" / "p0139_spiral"
ORACLE = REPO / "data" / "scratch" / "spiral_eval" / "oracle"        # human 2.399 meshes in the 9.362 frame (build-oracle)
VDUMP = REPO / "data" / "scratch" / "fit_figures" / "vertex"
CT_URL = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
CT_BASE = 0          # zarr level of the fit frame (PHerc1667: level 2 of the 2.399 um volume = 9.596 um)
SCROLL = "p0139"
VDUMPS = [VDUMP]     # HumanFrame looks for <arm>.json in these, in order


def use_scroll(name: str) -> None:
    """Switch the module (and spiral_eval) to a scroll: p0139 (default) or p1667 (z 6500-8000 confirmation band,
    fit frame = L2 of 20251217075048, human meshes = the labelled merged mesh cut per winding, build-oracle)."""
    global ORACLE, VDUMP, VDUMPS, CT_URL, CT_BASE, SCROLL
    se.set_scroll(name)
    SCROLL = name
    if name == "p1667":
        C = REPO / "data" / "scratch" / "confirm_region" / "p1667"
        ORACLE = C / "oracle_z6500"
        VDUMP = C / "vertex"
        VDUMPS = [C / "vertex_fig", C / "vertex"]
        CT_URL = "s3://vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr"
        CT_BASE = 2


SCROLL_INFO = {"p0139": ("PHerc0139", 9.362), "p1667": ("PHerc1667", 9.596)}


def blacklisted_windings() -> set[int]:
    """Human windings the scorer leaves out of the vertex sample (sites.json 'blacklisted'; none on PHerc1667)."""
    if SCROLL != "p0139":
        return set()
    meta = json.loads((se.ROOT / "sites.json").read_text())
    return {s["winding"] for s in meta["segments"] if "blacklisted" in s and s["winding"] is not None}


def surface_dirs(root: Path, spliced: bool = False) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(root.iterdir()):
        if not (p / "z.tif").exists() or (p.name.endswith("_spliced") != spliced):
            continue
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        w = meta.get("winding", meta.get("winding_index"))
        out.append((int(w) if w is not None else int(re.findall(r"w(-?\d+)", p.name)[-1]), p))
    return out


def slice_traces(root: Path, z0: float, box: tuple[float, float, float, float], spliced: bool = False) -> list[dict]:
    """Iso-z contours of every surface under root at z = z0 inside box (y0, y1, x0, x1). [{w, yx (N, 2)}]."""
    y0, y1, x0, x1 = box
    out = []
    for w, d in surface_dirs(root, spliced):
        z = tifffile.imread(d / "z.tif").astype(np.float32)
        ok = np.isfinite(z) & (z > 0)
        if not ((z[ok] <= z0).any() and (z[ok] >= z0).any()):
            continue
        y = tifffile.imread(d / "y.tif").astype(np.float32)
        x = tifffile.imread(d / "x.tif").astype(np.float32)
        ok &= (y > 0) & (x > 0)
        if (d / "mask.tif").exists():
            m = tifffile.imread(d / "mask.tif")
            ok &= (m[..., 0] if m.ndim == 3 else m) != 0
        near = ok & (y >= y0 - 50) & (y <= y1 + 50) & (x >= x0 - 50) & (x <= x1 + 50) & (np.abs(z - z0) < 60)
        if not near.any():
            continue
        r, c = np.nonzero(near)
        sl = (slice(max(r.min() - 2, 0), r.max() + 3), slice(max(c.min() - 2, 0), c.max() + 3))
        zs, ys, xs, oks = z[sl], y[sl], x[sl], ok[sl]
        zf = np.where(oks, zs, np.nan)
        for cont in find_contours(zf, z0):
            rr, cc = cont[:, 0], cont[:, 1]
            r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
            r1, c1 = np.minimum(r0 + 1, zs.shape[0] - 1), np.minimum(c0 + 1, zs.shape[1] - 1)
            fr, fc = rr - r0, cc - c0

            def interp(a):
                return ((1 - fr) * (1 - fc) * a[r0, c0] + (1 - fr) * fc * a[r0, c1] + fr * (1 - fc) * a[r1, c0]
                        + fr * fc * a[r1, c1])
            valid = oks[r0, c0] & oks[r0, c1] & oks[r1, c0] & oks[r1, c1]
            yx = np.stack([interp(ys), interp(xs)], 1)
            yx[~valid] = np.nan
            inb = (yx[:, 0] >= y0) & (yx[:, 0] < y1) & (yx[:, 1] >= x0) & (yx[:, 1] < x1)
            yx[~inb] = np.nan
            if np.isfinite(yx).all(1).sum() >= 2:
                out.append({"w": w, "yx": yx})
    return out


class HumanFrame:
    """Maps a fit's winding j to the human label frame with spiral_eval's seam offsets (per 250 z x 10 deg bin)."""

    def __init__(self, arm: str):
        info = json.loads(next(d / f"{arm}.json" for d in VDUMPS if (d / f"{arm}.json").exists()).read_text())
        self.s, self.g = info["sign"], info["global_offset"]
        self.zbin, self.abin = info["zbin"], info["abin"]
        self.bins = {int(k): v for k, v in info["bin_off"].items()}
        self.umb = se.Umbilicus()

    def k(self, j: np.ndarray, zyx: np.ndarray) -> np.ndarray:
        ang = self.umb.angle(zyx)
        b = np.floor(zyx[:, 0] / self.zbin).astype(np.int64) * 1000 + (ang // self.abin).astype(np.int64)
        off = np.array([self.bins.get(int(x), self.g) for x in b])
        return self.s * (np.asarray(j) - off)


def ct_slice(z: int, box: tuple[int, int, int, int], level: int = 0) -> np.ndarray:
    import zarr
    f = 2 ** level
    a = zarr.open(CT_URL, mode="r", storage_options={"anon": True})[str(CT_BASE + level)]
    y0, y1, x0, x1 = (v // f for v in box)
    return np.asarray(a[z // f, y0:y1, x0:x1])


def save_png(fig, path: Path, dpi: int) -> None:
    """Save, then quantize to 256 colours (CT greys + a few line colours survive; ~half the bytes)."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    im = Image.open(path).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
