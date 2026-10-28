"""Cross-scan registration for PHerc1667: 7.91 um DLS scan <-> 2.399 um scan.

Every PHerc1667 segment ships tifxyz meshes on both scans with the same UV
parameterization (grid step 20 vox in each scan's own units). Grid points are
therefore 3D correspondences. This module builds a correspondence table from
all segments and fits local affines around any point.

  uv run python eval/xscan.py build      # download meshes, write data/meshes/corr_7p91_2p399.npz

Coordinates here are xyz (tifxyz convention); zarr volumes are indexed zyx.
"""

import sys
from pathlib import Path

import numpy as np
import s3fs
import tifffile
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
MESHES = REPO / "data" / "meshes"
CORR = MESHES / "corr_7p91_2p399.npz"
BUCKET = "vesuvius-challenge-open-data/PHerc1667/segments"
LO, HI = "20231117161658-7.91um", "20251217075048-2.399um"
NOMINAL = 7.91 / 2.399

VOL_7P91 = "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr"
VOL_2P399 = "s3://vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr"


def load_grid(d: Path) -> np.ndarray:
    return np.stack([tifffile.imread(d / f"{c}.tif") for c in "xyz"], -1).astype(np.float64)


def valid(g: np.ndarray) -> np.ndarray:
    return (g > 0).all(-1) & np.isfinite(g).all(-1)


def affine_fit(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least-squares M (4x3) with [a, 1] @ M ~= b."""
    x = np.c_[a, np.ones(len(a))]
    if w is not None:
        x, b = x * w[:, None], b * w[:, None]
    return np.linalg.lstsq(x, b, rcond=None)[0]


def apply(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.c_[p, np.ones(len(p))] @ m


def match_grids(a: np.ndarray, b: np.ndarray, n: int = 30000, seed: int = 0):
    """Fit the UV scale/offset mapping grid a -> grid b; return paired xyz points (all valid a points)."""
    ii, jj = np.nonzero(valid(a))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(ii), min(n, len(ii)), replace=False)
    bval = valid(b).astype(float)

    def sample(p, i, j):
        ci, cj = i * p[0] + p[2], j * p[1] + p[3]
        ok = (ci >= 0) & (ci <= b.shape[0] - 1) & (cj >= 0) & (cj <= b.shape[1] - 1)
        ok[ok] = map_coordinates(bval, [ci[ok], cj[ok]], order=1) > 0.999
        pb = np.stack([map_coordinates(b[..., k], [ci[ok], cj[ok]], order=1) for k in range(3)], -1)
        return a[i[ok], j[ok]], pb

    def cost(p):
        pa, pb = sample(p, ii[sub], jj[sub])
        if len(pa) < 100:
            return 1e9
        return float(np.median(np.linalg.norm(apply(affine_fit(pa, pb), pa) - pb, axis=1)))

    p0 = [(b.shape[0] - 1) / (a.shape[0] - 1), (b.shape[1] - 1) / (a.shape[1] - 1), 0.0, 0.0]
    res = minimize(cost, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 600})
    pa, pb = sample(res.x, ii, jj)
    return pa, pb, res.x, res.fun


def build() -> None:
    fs = s3fs.S3FileSystem(anon=True)
    all_a, all_b, seg_id = [], [], []
    segs = [p for p in fs.ls(BUCKET) if "merged" not in p]
    for k, seg in enumerate(segs):
        name = seg.rsplit("/", 1)[1]
        sid = name.split("-", 1)[0]
        grids = {}
        for tag in (LO, HI):
            d = MESHES / name / tag
            d.mkdir(parents=True, exist_ok=True)
            for c in "xyz":
                if not (d / f"{c}.tif").exists():
                    fs.get(f"{seg}/mesh/{sid}-on-{tag}.tifxyz/{c}.tif", str(d / f"{c}.tif"))
            grids[tag] = load_grid(d)
        pa, pb, p, med = match_grids(grids[LO], grids[HI])
        print(f"[{k + 1}/{len(segs)}] {name}: uv scale {p[0]:.4f},{p[1]:.4f} off {p[2]:.2f},{p[3]:.2f} "
              f"pairs {len(pa)} global-affine median residual {med:.2f} vox@2.4")
        all_a.append(pa)
        all_b.append(pb)
        seg_id.append(np.full(len(pa), k, np.int16))
    np.savez_compressed(CORR, lo=np.concatenate(all_a).astype(np.float32), hi=np.concatenate(all_b).astype(np.float32),
                        seg=np.concatenate(seg_id), segments=np.array([s.rsplit("/", 1)[1] for s in segs]))
    print(f"wrote {CORR}")


class Correspondences:
    def __init__(self, path: Path = CORR):
        z = np.load(path)
        self.lo, self.hi = z["lo"].astype(np.float64), z["hi"].astype(np.float64)
        from scipy.spatial import cKDTree

        self.tree = cKDTree(self.lo)

    def local_affine(self, center_xyz: np.ndarray, radius: float = 250.0, min_pts: int = 200):
        """Gaussian-weighted local affine lo->hi around center (lo xyz). Returns (M, n, median residual) or None."""
        idx = self.tree.query_ball_point(center_xyz, radius)
        if len(idx) < min_pts:
            return None
        a, b = self.lo[idx], self.hi[idx]
        w = np.exp(-0.5 * (np.linalg.norm(a - center_xyz, axis=1) / (radius / 2)) ** 2)
        m = affine_fit(a, b, np.sqrt(w))
        r = np.linalg.norm(apply(m, a) - b, axis=1)
        return m, len(idx), float(np.median(r))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        raise SystemExit(__doc__)
