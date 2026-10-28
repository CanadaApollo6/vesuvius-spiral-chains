"""W1: pull small ESRF ~9 um crops from Grand Prize scrolls and measure forward-model target statistics.

  uv run python scripts/w1_real9.py fetch     # pick sites on cached L4 overviews, stream 256^3 L0 crops -> data/real9/
  uv run python scripts/w1_real9.py measure   # stats per crop -> data/real9/stats.json

Sites: "edge" crops straddle the scroll's outer surface (papyrus + air inside the scan mask, for far-air noise and
edge profiles); "interior" crops are fully inside the scroll. Masked voxels (value 0) are excluded from all stats.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "forward"))
from measure import measure_masked as stats  # noqa: E402

OUT = REPO / "data" / "real9"
VOLS = {
    "PHerc0826": ("s3://vesuvius-challenge-open-data/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc1447": ("s3://vesuvius-challenge-open-data/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", 8.640),
}
N = 256
L4 = 16  # L0 voxels per L4 voxel


def pick_sites(a: np.ndarray, rng: np.random.Generator, n_edge=3, n_int=2):
    """a = L4 overview. Windows of 16^3 L4 vox (= 256^3 L0)."""
    w = N // L4
    valid = a > 0
    mat = a > 80
    air = valid & (a < 65)
    dist_air = ndi.distance_transform_edt(~mat)  # L4 vox to nearest material
    far_air = air & (dist_air >= 1.5)  # >= ~220 um from material at L4 precision
    f_valid = ndi.uniform_filter(valid.astype(np.float32), w, mode="constant")
    f_mat = ndi.uniform_filter(mat.astype(np.float32), w, mode="constant")
    f_far = ndi.uniform_filter(far_air.astype(np.float32), w, mode="constant")
    zlo, zhi = int(0.15 * a.shape[0]), int(0.85 * a.shape[0])
    m = np.zeros(a.shape, bool)
    m[zlo:zhi, w:-w, w:-w] = True
    edge = m & (f_valid > 0.97) & (f_mat > 0.3) & (f_far > 0.15)
    inter = m & (f_valid > 0.999) & (f_mat > 0.85)
    out = []
    for kind, cand, n in (("edge", edge, n_edge), ("interior", inter, n_int)):
        idx = np.argwhere(cand)
        rng.shuffle(idx)
        chosen = []
        for c in idx:
            if all(np.abs(c - d).max() > 3 * w for d in chosen):
                chosen.append(c)
            if len(chosen) == n:
                break
        for c in chosen:
            out.append((kind, (c * L4 - N // 2).astype(int), float(f_mat[tuple(c)]), float(f_far[tuple(c)])))
    return out


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(18)
    for scroll, (url, vox) in VOLS.items():
        g = zarr.open_group(url, mode="r", storage_options={"anon": True})
        cache = OUT / f"_{scroll}_L4.npy"
        a = np.load(cache) if cache.exists() else np.asarray(g["4"])
        np.save(cache, a)
        for i, (kind, o, fm, ff) in enumerate(pick_sites(a, rng)):
            t0 = time.time()
            crop = np.asarray(g["0"][tuple(slice(int(x), int(x) + N) for x in o)])
            name = f"{scroll}_{kind}{i}_z{o[0]}_y{o[1]}_x{o[2]}"
            np.save(OUT / f"{name}.npy", crop)
            prov = {"scroll": scroll, "volume": url, "zarr_level": "0", "voxel_um": vox, "origin_zyx": o.tolist(),
                    "shape": [N] * 3, "dtype": str(crop.dtype), "kind": kind, "L4_material_frac": fm, "L4_far_air_frac": ff,
                    "site_seed": 18, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1)}
            (OUT / f"{name}.json").write_text(json.dumps(prov, indent=1))
            print(name, prov["seconds"], "s", flush=True)


def measure():
    rows = {}
    for f in sorted(OUT.glob("*.npy")):
        if f.name.startswith("_"):
            continue
        prov = json.loads(f.with_suffix(".json").read_text())
        s = stats(np.load(f), prov["voxel_um"])
        rows[f.stem] = s
        keys = ("valid_frac", "otsu", "mat_mean", "mat_std", "air30_mean", "air90_n", "air90_mean", "air90_noise_std",
                "air60_noise_std", "air30_noise_std", "edge_10_90_um", "fringe_depth", "fringe_pos_um", "near_air_mean_5_15um")
        print(f.stem, {k: (round(s[k], 2) if isinstance(s.get(k), float) else s.get(k)) for k in keys}, flush=True)
        if "air90_acf" in s:
            print("   air90 acf", np.round(s["air90_acf"], 3).tolist())
    for f in sorted((REPO / "data" / "eval" / "surface_labels_2026_07_sample").glob("p0139_*.npz")):
        v9 = np.load(f)["ct9"]
        s = stats(v9, 9.362)
        rows[f.stem + "_ct9"] = s
        print(f.stem + "_ct9", {k: (round(s[k], 2) if isinstance(s.get(k), float) else s.get(k)) for k in keys}, flush=True)
    (OUT / "stats.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    {"fetch": fetch, "measure": measure}[sys.argv[1]]()
