"""CT forward model: density at 2.4 um -> synthetic 2.4 um and 7.91 um scans (uint8), GPU.

2.4 um:  I2 = air + gain * (G_psf * rho) - alpha * Laplacian(G_fringe * rho) + n2
         (the Laplacian term gives the propagation phase-contrast fringe: dark on the air side of every boundary)
7.91 um: I7 = J - alpha7 * Laplacian(G_fringe7 * J) + n7,  J = a * resample_7.91(G_2.5vox * I2_clean) + b
         (measured: real 7.91 ~= real 2.4 blurred with sigma 2.5 vox @2.4 then resampled; notes/2026-10-16-contact-check.md)
ESRF stages (Grand Prize scans, 1.2 m propagation, 113-116 keV), directly from density on the 2.4 grid:
         I = air + gain * resample_vox(G_psf * rho) - alpha * Laplacian(G_fringe * .) + n   (ScanStage, stage_scan)
         keyed by voxel size ("9p362", "8p64"); fitted by forward/fit9.py to real crops (data/real9).
Noise: mixture of two Gaussian-correlated fields, fitted to air statistics of the real scans (forward/fit.py).
Defaults come from forward/fit.py (see forward/ct_params.json).
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
VOX_HI, VOX_LO = 2.399, 7.91


@dataclass
class CTParams:
    # 2.4 um
    air: float = 46.5
    gain: float = 150.0  # grey levels per unit density
    psf_sigma_um: float = 1.5
    fringe_alpha: float = 0.0  # grey levels * um^2 per unit density
    fringe_sigma_um: float = 4.0
    n2_std: float = 3.6
    n2_s1_um: float = 2.4
    n2_s2_um: float = 7.0
    n2_w2: float = 0.2
    # 7.91 um
    blur_lo_vox_hi: float = 2.5  # Gaussian sigma in 2.4 um voxels before resampling (measured)
    lo_a: float = 0.67
    lo_b: float = 48.0
    lo_fringe_alpha: float = 0.0  # 7.91 scan's own phase-contrast halo (different beamline), grey * um^2
    lo_fringe_sigma_um: float = 20.0
    n7_std: float = 6.8
    n7_s1_um: float = 8.0
    n7_s2_um: float = 30.0
    n7_w2: float = 0.3

    @classmethod
    def load(cls, path: Path = HERE / "ct_params.json") -> "CTParams":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})

    def save(self, path: Path = HERE / "ct_params.json", extra: dict | None = None) -> None:
        keep = {k: v for k, v in json.loads(path.read_text()).items() if k == "stages"} if path.exists() else {}
        path.write_text(json.dumps(asdict(self) | (extra or {}) | keep, indent=1))


def _kernel(sigma: float, dev) -> torch.Tensor:
    r = max(1, int(math.ceil(3.5 * sigma)))
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gauss3(v: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """Separable Gaussian blur of a [Z,Y,X] tensor (reflect padding)."""
    if sigma_vox <= 0:
        return v
    k = _kernel(sigma_vox, v.device)
    r = (len(k) - 1) // 2
    x = v[None, None]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="reflect" if min(v.shape) > r else "replicate"), k.view(shape))
    return x[0, 0]


def laplacian(v: torch.Tensor) -> torch.Tensor:
    p = F.pad(v[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    c = p[1:-1, 1:-1, 1:-1]
    return (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] - 6 * c)


def corr_noise(shape, std: float, s1_vox: float, s2_vox: float, w2: float, gen: torch.Generator, dev) -> torch.Tensor:
    def unit(s):
        w = torch.randn(shape, generator=gen, device=dev)
        f = gauss3(w, s)
        return f / f.std()

    n = math.sqrt(1 - w2) * unit(s1_vox) + (math.sqrt(w2) * unit(s2_vox) if w2 > 0 else 0)
    return std * n


def slab_map(fn, vol: torch.Tensor, halo: int, out_device, slab: int = 64, dev="cuda") -> torch.Tensor:
    """Apply a local op fn (support <= halo voxels) to a large [Z,Y,X] volume in z-slabs, so only one slab plus halo
    is on the GPU at a time. Identical to fn(vol) (fn's own edge padding still applies at the true volume edges)."""
    Z = vol.shape[0]
    out = torch.empty(vol.shape, dtype=torch.float32, device=out_device)
    for z0 in range(0, Z, slab):
        z1 = min(Z, z0 + slab)
        a, b = max(0, z0 - halo), min(Z, z1 + halo)
        r = fn(vol[a:b].to(dev, torch.float32))
        out[z0:z1] = r[z0 - a : z0 - a + (z1 - z0)].to(out_device)
    return out


def hi_clean_halo(p: CTParams) -> int:
    return int(math.ceil(3.5 * max(p.psf_sigma_um, p.fringe_sigma_um) / VOX_HI)) + 2


def hi_clean(rho: torch.Tensor, p: CTParams) -> torch.Tensor:
    """Noise-free 2.4 um image (float grey levels) from density at 2.4 um."""
    img = p.air + p.gain * gauss3(rho, p.psf_sigma_um / VOX_HI)
    if p.fringe_alpha:
        img = img - (p.fringe_alpha / VOX_HI**2) * laplacian(gauss3(rho, p.fringe_sigma_um / VOX_HI))
    return img


def hi_scan(clean: torch.Tensor, p: CTParams, gen: torch.Generator) -> torch.Tensor:
    n = corr_noise(clean.shape, p.n2_std, p.n2_s1_um / VOX_HI, p.n2_s2_um / VOX_HI, p.n2_w2, gen, clean.device)
    return (clean + n).round().clamp(0, 255)


def resample(b: torch.Tensor, vox_um: float, out_shape=None, offset_hi=None) -> torch.Tensor:
    """Trilinear samples of a 2.4 um grid at spacing vox_um. Output voxel (i,j,k) sits at offset_hi + (i,j,k) * vox_um /
    VOX_HI (zyx, in 2.4 voxels). Default: centred, as many output voxels as fit."""
    step = vox_um / VOX_HI
    n_in = torch.tensor(b.shape, dtype=torch.float32)
    if out_shape is None:
        out_shape = tuple(int(v) for v in torch.floor((n_in - 1) / step) + 1)
    if offset_hi is None:
        offset_hi = ((n_in - 1) - (torch.tensor(out_shape, dtype=torch.float32) - 1) * step) / 2
    dev = b.device
    axes = [torch.arange(n, device=dev, dtype=torch.float32) * step + float(o) for n, o in zip(out_shape, offset_hi)]
    gz, gy, gx = torch.meshgrid(*axes, indexing="ij")
    # grid_sample wants (x, y, z) normalized to [-1, 1] with align_corners=True
    norm = lambda g, n: 2 * g / (n - 1) - 1  # noqa: E731
    grid = torch.stack([norm(gx, b.shape[2]), norm(gy, b.shape[1]), norm(gz, b.shape[0])], -1)
    return F.grid_sample(b[None, None], grid[None], mode="bilinear", align_corners=True)[0, 0]


def lo_scan(clean_hi: torch.Tensor, p: CTParams, gen: torch.Generator, out_shape=None, offset_hi=None,
            noise: bool = True, preblurred: bool = False) -> torch.Tensor:
    """7.91 um scan from the clean 2.4 um image (grid placement as in resample())."""
    b = clean_hi if preblurred else gauss3(clean_hi, p.blur_lo_vox_hi)
    lo = p.lo_a * resample(b, VOX_LO, out_shape, offset_hi) + p.lo_b
    if p.lo_fringe_alpha:
        lo = lo - (p.lo_fringe_alpha / VOX_LO**2) * laplacian(gauss3(lo, p.lo_fringe_sigma_um / VOX_LO))
    if noise:
        lo = lo + corr_noise(lo.shape, p.n7_std, p.n7_s1_um / VOX_LO, p.n7_s2_um / VOX_LO, p.n7_w2, gen, lo.device)
    return lo.round().clamp(0, 255)


@dataclass
class ScanStage:
    """An ESRF scan simulated straight from density (no DLS 2.4 um stage in between). fringe_alpha may be negative."""
    vox_um: float = 9.362
    air: float = 51.0
    gain: float = 80.0  # grey levels per unit density
    psf_sigma_um: float = 8.0  # applied on the 2.4 grid before resampling
    fringe_alpha: float = 300.0  # grey * um^2; > 0 gives the dark air-side fringe and bright inner rim
    fringe_sigma_um: float = 12.0
    n_std: float = 3.8
    n_s1_um: float = 9.0
    n_s2_um: float = 25.0
    n_w2: float = 0.2

    @classmethod
    def load(cls, key: str, path: Path = HERE / "ct_params.json") -> "ScanStage":
        d = json.loads(path.read_text())["stages"][key]
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})

    def save(self, key: str, path: Path = HERE / "ct_params.json", extra: dict | None = None) -> None:
        d = json.loads(path.read_text()) if path.exists() else {}
        d.setdefault("stages", {})[key] = asdict(self) | (extra or {})
        path.write_text(json.dumps(d, indent=1))


def stage_halo(st: ScanStage) -> int:
    """2.4 um voxels of margin the stage needs around its output grid."""
    return int(math.ceil(3.5 * (st.psf_sigma_um + abs(st.fringe_sigma_um)) / VOX_HI)) + 2


def stage_scan(rho: torch.Tensor, st: ScanStage, gen: torch.Generator, out_shape=None, offset_hi=None,
               noise: bool = True, preblurred: bool = False) -> torch.Tensor:
    """ESRF scan from density on the 2.4 grid (grid placement as in resample()). preblurred: rho is already G_psf * rho."""
    b = rho if preblurred else gauss3(rho, st.psf_sigma_um / VOX_HI)
    img = st.air + st.gain * resample(b, st.vox_um, out_shape, offset_hi)
    if st.fringe_alpha:
        img = img - (st.fringe_alpha / st.vox_um**2) * laplacian(gauss3(img, st.fringe_sigma_um / st.vox_um))
    if noise:
        img = img + corr_noise(img.shape, st.n_std, st.n_s1_um / st.vox_um, st.n_s2_um / st.vox_um, st.n_w2, gen,
                               img.device)
    return img.round().clamp(0, 255)
