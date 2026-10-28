"""Reusable 2.399 um / 9.362 um PHerc0139 access (notes/2026-10-22-hires-step-count.md).

- to24 / to9: 9.362 L0 zyx <-> 2.399 L0 zyx with the published affine (p9_xyz = A p2399_xyz + t,
  data/eval/surface_labels_2026_07_sample/transform_0139_2p399_to_9p362.json). No local shift: the affine is locally
  off by ~1-2 vox at 9.362 (median 1.4); register (NCC) if you need better.
- umb(z) -> (x, y) of the umbilicus in the 9.362 frame; ray_xyz9(z, theta, r) -> zyx points on a probe ray.
- SlabReader: anonymous byte-range reads of the uncompressed zarr v2 volumes (128^3 chunks, C order); only the z slices
  a request needs are fetched. R24 = 2.399 L0 (volume 20260102150214, shape 76953 x 26511 x 26511), R9 = 9.362 L0.
  R24.sample(P) gives trilinear samples at points P (..., 3) zyx in that volume's voxels (rows along axis 0 are read
  in boxes of `seg`). ~0.3 s latency per request, ~65 requests/s per process with s3fs cat_ranges.
"""
import numpy as np
from scipy.ndimage import map_coordinates
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402

BUCKET = "vesuvius-challenge-open-data/PHerc0139/volumes/"
V24 = BUCKET + "20260102150214-2.399um-0.2m-78keV-masked.zarr"
V9 = BUCKET + "20250728140407-9.362um-1.2m-113keV-masked.zarr"
SHAPE24 = (76953, 26511, 26511)
SHAPE9 = (20974, 6621, 6621)
CH = 128
DR = 0.125          # radial sample step (9.362 vox) used by tools/hires_step_count.py
TUBE24 = 3          # +-3 lines (2.399 vox) in z and tangent
TUBE9 = 1           # +-1 line (9.362 vox)
PAD = 24.0          # extra radius before the first and after the last chain point (9.362 vox)

_A, _T = se.affine()
_AINV = np.linalg.inv(_A)
_UMB = se.Umbilicus()
_FS = None


def fs():
    global _FS
    if _FS is None:
        import fsspec
        _FS = fsspec.filesystem("s3", anon=True)
    return _FS


def to24(zyx9):
    """9.362 L0 zyx -> 2.399 L0 zyx through the inverse published affine."""
    zyx9 = np.asarray(zyx9, float)
    return ((zyx9[..., ::-1] - _T) @ _AINV.T)[..., ::-1]


def to9(zyx24):
    zyx24 = np.asarray(zyx24, float)
    return (zyx24[..., ::-1] @ _A.T + _T)[..., ::-1]


def umb(z):
    return np.interp(z, _UMB.z, _UMB.x), np.interp(z, _UMB.z, _UMB.y)


def ray_xyz9(z, theta, r):
    cx, cy = umb(z)
    th = np.radians(theta)
    r = np.asarray(r, float)
    return np.stack([np.full_like(r, z), cy + r * np.sin(th), cx + r * np.cos(th)], -1)


class SlabReader:
    """Sparse reads from an uncompressed zarr v2 array (128^3 chunks, C order): per chunk only a z-slice range."""

    def __init__(self, root, level, shape):
        self.root, self.level, self.shape = root, level, np.array(shape)
        self.nbytes = 0
        self.nreq = 0

    def fetch(self, need):
        """need: {(cz, cy, cx): [z0, z1)} -> {(cz, cy, cx): (z0, array (z1 - z0, 128, 128))}."""
        keys = sorted(need)
        paths = [f"{self.root}/{self.level}/{cz}/{cy}/{cx}" for cz, cy, cx in keys]
        st = [need[k][0] * CH * CH for k in keys]
        en = [need[k][1] * CH * CH for k in keys]
        out = {}
        for i0 in range(0, len(keys), 400):
            res = fs().cat_ranges(paths[i0:i0 + 400], st[i0:i0 + 400], en[i0:i0 + 400], on_error="return")
            for k, b in zip(keys[i0:i0 + 400], res):
                z0, z1 = need[k]
                self.nreq += 1
                if isinstance(b, bytes) and len(b) == (z1 - z0) * CH * CH:
                    self.nbytes += len(b)
                    out[k] = (z0, np.frombuffer(b, np.uint8).reshape(z1 - z0, CH, CH))
                else:  # missing chunk (masked / outside) -> zeros
                    out[k] = (z0, None)
        return out

    def sample(self, P, seg=48):
        """Trilinear samples at points P (..., 3) zyx in this array's voxels; P[i] along axis 0 is a ray coordinate.
        Points are grouped in segments of `seg` rows along axis 0, each read as one dense box."""
        shp = P.shape[:-1]
        P2 = P.reshape(shp[0], -1, 3)
        boxes = []
        need = {}
        for i0 in range(0, shp[0], seg):
            Q = P2[i0:i0 + seg]
            lo = np.floor(Q.reshape(-1, 3).min(0)).astype(int) - 1
            hi = np.floor(Q.reshape(-1, 3).max(0)).astype(int) + 3
            lo = np.clip(lo, 0, self.shape - 1)
            hi = np.clip(hi, lo + 1, self.shape)
            boxes.append((i0, lo, hi))
            for cz in range(lo[0] // CH, (hi[0] - 1) // CH + 1):
                z0 = max(lo[0], cz * CH) - cz * CH
                z1 = min(hi[0], (cz + 1) * CH) - cz * CH
                for cy in range(lo[1] // CH, (hi[1] - 1) // CH + 1):
                    for cx in range(lo[2] // CH, (hi[2] - 1) // CH + 1):
                        k = (cz, cy, cx)
                        a = need.get(k)
                        need[k] = (z0, z1) if a is None else (min(a[0], z0), max(a[1], z1))
        data = self.fetch(need)
        out = np.zeros(P2.shape[:-1], np.float32)
        for i0, lo, hi in boxes:
            box = np.zeros(hi - lo, np.uint8)
            for cz in range(lo[0] // CH, (hi[0] - 1) // CH + 1):
                for cy in range(lo[1] // CH, (hi[1] - 1) // CH + 1):
                    for cx in range(lo[2] // CH, (hi[2] - 1) // CH + 1):
                        z0, arr = data[(cz, cy, cx)]
                        if arr is None:
                            continue
                        c0 = np.array([cz, cy, cx]) * CH
                        a = np.maximum(lo, c0)
                        b = np.minimum(hi, c0 + CH)
                        box[a[0] - lo[0]:b[0] - lo[0], a[1] - lo[1]:b[1] - lo[1], a[2] - lo[2]:b[2] - lo[2]] = \
                            arr[a[0] - c0[0] - z0:b[0] - c0[0] - z0, a[1] - c0[1]:b[1] - c0[1], a[2] - c0[2]:b[2] - c0[2]]
            Q = P2[i0:i0 + seg].reshape(-1, 3) - lo
            v = map_coordinates(box.astype(np.float32), Q.T, order=1, mode="nearest")
            out[i0:i0 + seg] = v.reshape(-1, P2.shape[1])
        return out.reshape(shp)


R24 = SlabReader(V24, 0, SHAPE24)
R9 = SlabReader(V9, 0, SHAPE9)


