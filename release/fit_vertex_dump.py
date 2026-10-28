"""Per-vertex winding residuals of a PHerc0139 spiral fit, exactly as eval/spiral_eval.py scores them, saved for figures.

For every human vertex in the cached reference sample (z 3000-4500, stride 2) this stores the fit's nearest-surface
winding j (within --tol), and the seam- and global-offset residuals (0 = right winding). Surfaces are streamed one at a
time, so peak RAM stays ~2-3 GB instead of the scorer's ~15 GB; the result is identical (nearest = min over surfaces).

  uv run python release/fit_vertex_dump.py --fit data/scratch/p0139_spiral/arm19/meshes_step4 --name arm19
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import spiral_eval as se  # noqa: E402

OUT = REPO / "data" / "scratch" / "fit_figures" / "vertex"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--z0", type=float, default=3000)
    ap.add_argument("--z1", type=float, default=4500)
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--scroll", choices=list(se.SCROLLS), default="p0139",
                    help="reference sample of spiral_eval --scroll (run spiral_eval once first to build it)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    se.set_scroll(a.scroll)
    V = dict(np.load(se.ref_path(a.z0, a.z1), allow_pickle=False))
    zlo, zhi = a.z0 - a.tol - 2, a.z1 + a.tol + 2
    best_d = np.full(len(V["k"]), np.inf)
    best_l = np.full(len(V["k"]), se.NOLAB, np.int64)
    dirs = sorted({p.parent for p in a.fit.rglob("x.tif") if (p.parent / "z.tif").exists()})
    for i, d in enumerate(dirs):
        g, meta = se.read_tifxyz(d)
        w = meta.get("winding", meta.get("winding_index"))
        if w is None:
            w = re.findall(r"w(-?\d+)", d.name)[-1]
        g = se.crop_band(g, zlo, zhi)
        if not g.size:
            continue
        S = se.Surfaces([{"label": int(w), "grid": g}], zlo, zhi)
        lab, dist = S.query(V["pts"], a.tol)
        upd = dist < best_d
        best_d[upd], best_l[upd] = dist[upd], lab[upd]
        if i % 40 == 0:
            print(f"  {i}/{len(dirs)} {d.name} ({time.time() - t0:.0f}s)", flush=True)
    best_l[best_d > a.tol] = se.NOLAB
    vs = se.vertex_scores(V, best_l)
    m = vs["matched"]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out_dir / f"{a.name}.npz", j=best_l.astype(np.int32), matched=m,
                        res_s=np.where(m, vs["res_s"], 0).astype(np.int16),
                        res_g=np.where(m, vs["res_g"], 0).astype(np.int16), dist=best_d.astype(np.float32))
    info = {"fit": str(a.fit), "sign": vs["sign"], "global_offset": vs["global_offset"], "zbin": vs["zbin"],
            "abin": vs["abin"], "bin_off": {str(k): v for k, v in vs["bin_off"].items()},
            "wrong_area_seam": float((m & (vs["res_s"] != 0)).sum() / m.sum() * 100),
            "wrong_area_global": float((m & (vs["res_g"] != 0)).sum() / m.sum() * 100),
            "unmatched": float((~m).mean() * 100), "seconds": round(time.time() - t0, 1)}
    (a.out_dir / f"{a.name}.json").write_text(json.dumps(info, indent=1))
    print({k: v for k, v in info.items() if k != "bin_off"})


if __name__ == "__main__":
    main()
