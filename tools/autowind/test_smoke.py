"""autowind smoke tests (< 1 min on 8+ cores).

  uv run --with pytest pytest -q tools/autowind/test_smoke.py
  uv run python tools/autowind/test_smoke.py          # same tests without pytest

1. test_synthetic_spiral: a synthetic Archimedean-spiral surface mask (192 x 640 x 640, pitch 16 vox, 15 turns)
   written as an OME-Zarr with level '0' only, plus a villa umbilicus.json and a constant grad_mag. Runs the full CLI
   (`chains ... --raise gm`): builds the extent cache from level 0, solves 9 crops from scratch, reassembles, raises.
   Checks the villa JSON layout and that the chains count +1 per sheet (the true answer).
2. test_cached_crop_p1667: re-solves one cached PHerc1667 crop through the generic inputs (mask path, umbilicus.json,
   worker initializer) and requires the checkpoint to be identical to the one behind band_m7_p1667_ray_c. Skipped if
   the private cache is absent.
"""

import json
import os
import pickle
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from autowind import cli, core  # noqa: E402  (first: sets single-threaded BLAS before numpy loads)
from autowind.presets import PRESETS  # noqa: E402

import numpy as np  # noqa: E402

PITCH, N, Z, R0, R1 = 16.0, 640, 192, 40.0, 290.0


def make_spiral(root: Path) -> tuple[Path, Path, Path]:
    import zarr
    c = N / 2
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    r, th = np.hypot(yy - c, xx - c), np.arctan2(yy - c, xx - c)
    s = (r - R0 - PITCH * th / (2 * np.pi)) / PITCH                # fractional turn index along the radius
    sheet = (np.abs(s - np.round(s)) * PITCH < 0.75) & (r >= R0) & (r <= R1)
    mask = root / "spiral_mask.zarr"
    g = zarr.open_group(str(mask), mode="w")
    g.create_array("0", data=np.broadcast_to((sheet * 255).astype(np.uint8), (Z, N, N)).copy(), chunks=(64, 128, 128))
    g.attrs["multiscales"] = [{"axes": [{"name": a, "type": "space"} for a in "zyx"],
                               "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0] * 3}]}]}]
    umb = root / "umbilicus.json"
    umb.write_text(json.dumps({"control_points": [{"x": c, "y": c, "z": 0.0}, {"x": c, "y": c, "z": Z - 1.0}]}))
    gm = root / "grad_mag.zarr"                                      # lasagna-style density, 1 winding per pitch
    h = zarr.open_group(str(gm), mode="w")
    r4 = np.hypot(*(np.mgrid[0:N // 4, 0:N // 4] * 4.0 - c))
    h.create_array("2", data=np.broadcast_to(np.where(r4 <= R1, round(1000 / PITCH), 0).astype(np.uint8),
                                             (Z // 4, N // 4, N // 4)).copy())
    return mask, umb, gm


def test_synthetic_spiral():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mask, umb, gm = make_spiral(root)
        out = root / "out"
        rc = cli.main(["chains", "--mask", str(mask), "--umbilicus", str(umb), "--z0", "90", "--z1", "100",
                       "--voxel-um", "9.5", "--dtheta", "120", "--raise", "gm", "--grad-mag", str(gm),
                       "--workers", str(min(9, os.cpu_count() or 1)), "--out", str(out)])
        assert rc == 0
        run = json.loads((out / "autowind_run.json").read_text())
        assert run["crops"]["planned"] == 9 and run["crops"]["solved"] == 9, run["crops"]
        for d in (out / "noraise", out):
            rw = json.loads((d / "relative_windings.json").read_text())
            assert rw["vc_pointcollections_json_version"] == "1"
            cols = list(rw["collections"].values())
            assert len(cols) >= 3, len(cols)
            for col in cols:
                pts = [col["points"][k] for k in sorted(col["points"], key=int)]
                w = [p["wind_a"] for p in pts]
                assert w[0] == 0 and all(b >= a for a, b in zip(w, w[1:])), w
                assert all(len(p["p"]) == 3 for p in pts)
        ch = json.loads((out / "noraise" / "chains.json").read_text())
        steps = np.concatenate([np.diff(c["wind"]) for c in ch])
        assert (steps == 1).mean() >= 0.95, np.bincount(steps)
        assert max(len(c["pts"]) for c in ch) >= 10                  # long chains across the crops of a ray
        chg = json.loads((out / "chains.json").read_text())
        assert all(len(c["gm_p"]) == len(c["pts"]) - 1 for c in chg)
    print(f"synthetic spiral ok: {len(ch)} chains, {len(steps)} steps, {(steps == 1).mean():.1%} = +1 "
          f"({time.time() - t0:.0f}s)")


P1667_CROP = "6624_195_12.pkl"  # an outer crop of band_m7_p1667 (5 solved crossings; fast to solve)


def test_cached_crop_p1667():
    pre = PRESETS["p1667"]
    ref = Path(pre["crops_ro"][0]) / P1667_CROP
    if not (ref.exists() and Path(pre["mask"]).exists()):
        try:
            import pytest
            pytest.skip("private PHerc1667 cache not present")
        except ImportError:
            print("skip: private PHerc1667 cache not present")
            return
    t0 = time.time()
    o = cli.resolve(cli.build_parser().parse_args(["chains", "--preset", "p1667", "--out", "/nonexistent"]))
    cfg = dict(mask=str(o.mask), work=str(o.work), shape=list(core.mask_shape(o.mask)),
               umbilicus_xyz=core.load_umbilicus(o.umbilicus).tolist(), seg=o.seg)
    core.configure(cfg)
    _, crops = core.plan(o, cfg)
    c = [c for c in crops if core._key(c) == P1667_CROP]
    assert len(c) == 1
    with tempfile.TemporaryDirectory() as td:
        core.solve(c, cfg, Path(td), workers=1, io_threads=4, log=lambda s: None)
        got = (Path(td) / P1667_CROP).read_bytes()
    want = ref.read_bytes()
    assert got == want or repr(pickle.loads(got)) == repr(pickle.loads(want))
    print(f"cached PHerc1667 crop {P1667_CROP} re-solved identically ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    test_synthetic_spiral()
    test_cached_crop_p1667()
