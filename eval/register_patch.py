"""Register a 7.91 um Scroll 4 eval patch to the 2.399 um scan.

Local affine from mesh correspondences (eval/xscan.py), refined by a translation
search that maximizes NCC between the real 7.91 patch and the 2.4 um data
blurred and resampled onto the patch grid. The 2.4 um bounding-box crop is cached
under data/xscan_crops/.

  uv run python eval/register_patch.py s4_z1536_y512_x2304 [more stems...]
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xscan import VOL_2P399, Correspondences, apply  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "data" / "eval" / "jeong200"
CACHE = REPO / "data" / "xscan_crops"
PATCH_OFFSET = 10  # patch voxel (0,0,0) = filename origin + 10 (verified 2026-10-16)
CORE = 160  # register / analyse the central CORE^3 of each patch
SCALE = 7.91 / 2.399


@dataclass
class Registered:
    stem: str
    ct: np.ndarray  # 7.91 patch, zyx uint8
    gt: np.ndarray  # 7.91 GT, zyx bool
    hi: np.ndarray  # 2.4 um crop, zyx uint8
    hi_origin: np.ndarray  # zyx origin of the crop in the 2.4 volume
    m: np.ndarray  # 4x3 affine: lo xyz (scroll voxels @7.91) -> hi xyz (@2.399), translation refined
    ncc: float
    sigma: float  # blur (in 2.4 voxels) used for 7.91-like rendering

    def patch_to_lo_xyz(self, p_zyx: np.ndarray) -> np.ndarray:
        z0, y0, x0 = parse_origin(self.stem)
        p = np.asarray(p_zyx, float)
        return np.c_[p[:, 2] + x0 + PATCH_OFFSET, p[:, 1] + y0 + PATCH_OFFSET, p[:, 0] + z0 + PATCH_OFFSET]

    def hi_local_zyx(self, lo_xyz: np.ndarray) -> np.ndarray:
        h = apply(self.m, lo_xyz)
        return np.c_[h[:, 2], h[:, 1], h[:, 0]] - self.hi_origin

    def sample_hi(self, p_zyx: np.ndarray, vol: np.ndarray | None = None, order: int = 1) -> np.ndarray:
        """Sample a 2.4 um volume (default: native crop) at patch-space points (zyx, may be fractional)."""
        q = self.hi_local_zyx(self.patch_to_lo_xyz(p_zyx))
        return map_coordinates(self.hi if vol is None else vol, q.T, order=order, mode="nearest")


def parse_origin(stem: str):
    return tuple(map(int, re.search(r"_z(\d+)_y(\d+)_x(\d+)", stem).groups()))


def core_grid(n: int, step: int = 1) -> np.ndarray:
    lo = (n - CORE) // 2
    r = np.arange(lo, lo + CORE, step)
    return np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)


def fetch_hi(stem: str, m: np.ndarray, n: int, margin: int = 40):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{stem}.npz"
    if path.exists():
        z = np.load(path)
        return z["hi"], z["origin"]
    lo = (n - CORE) // 2
    corners = np.array([[a, b, c] for a in (lo, lo + CORE) for b in (lo, lo + CORE) for c in (lo, lo + CORE)], float)
    z0, y0, x0 = parse_origin(stem)
    lo_xyz = np.c_[corners[:, 2] + x0 + PATCH_OFFSET, corners[:, 1] + y0 + PATCH_OFFSET, corners[:, 0] + z0 + PATCH_OFFSET]
    h = apply(m, lo_xyz)
    h_zyx = np.c_[h[:, 2], h[:, 1], h[:, 0]]
    vol = zarr.open(VOL_2P399, mode="r", storage_options={"anon": True})["0"]
    a = np.maximum(np.floor(h_zyx.min(0)).astype(int) - margin, 0)
    b = np.minimum(np.ceil(h_zyx.max(0)).astype(int) + margin, vol.shape)
    hi = vol[a[0] : b[0], a[1] : b[1], a[2] : b[2]]
    np.savez(path, hi=hi, origin=a)
    return hi, a


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


def register(stem: str, corr: Correspondences | None = None, sigma: float = 1.5, search: int = 12) -> Registered:
    ct = tifffile.imread(POOL / "imagesTr" / f"{stem}_0000.tif")
    gt = tifffile.imread(POOL / "labelsTr" / f"{stem}.tif") == 1
    n = ct.shape[0]
    corr = corr or Correspondences()
    z0, y0, x0 = parse_origin(stem)
    c = np.array([x0 + PATCH_OFFSET + n / 2, y0 + PATCH_OFFSET + n / 2, z0 + PATCH_OFFSET + n / 2])
    fit = corr.local_affine(c)
    if fit is None:
        raise RuntimeError(f"{stem}: too few mesh correspondences")
    m = fit[0].copy()
    hi, origin = fetch_hi(stem, m, n)
    blurred = ndi.gaussian_filter(hi.astype(np.float32), sigma)

    reg = Registered(stem, ct, gt, hi, origin, m, 0.0, sigma)
    pts = core_grid(n, step=4).astype(float)
    target = ct[tuple(pts.astype(int).T)].astype(np.float32)
    base = reg.hi_local_zyx(reg.patch_to_lo_xyz(pts))

    def score(t_zyx):
        return ncc(target, map_coordinates(blurred, (base + t_zyx).T, order=1, mode="nearest"))

    best = (score(np.zeros(3)), np.zeros(3))
    for step in (4, 2, 1, 0.5):  # coarse-to-fine translation search in 2.4 um voxels
        rng = np.arange(-search, search + 1e-9, step) if step == 4 else np.arange(-2 * step, 2 * step + 1e-9, step)
        centre = best[1]
        for dz in rng:
            for dy in rng:
                for dx in rng:
                    t = centre + np.array([dz, dy, dx])
                    s = score(t)
                    if s > best[0]:
                        best = (s, t)
    t_xyz = best[1][::-1]
    reg.m[3] += t_xyz
    reg.ncc = best[0]
    return reg


def main() -> None:
    corr = Correspondences()
    out = {}
    for stem in sys.argv[1:]:
        r = register(stem, corr)
        out[stem] = {"ncc": r.ncc, "hi_shape": list(r.hi.shape), "m": r.m.tolist()}
        print(f"{stem}: NCC={r.ncc:.3f} hi crop {r.hi.shape}")
    print(json.dumps({k: v["ncc"] for k, v in out.items()}))


if __name__ == "__main__":
    main()
