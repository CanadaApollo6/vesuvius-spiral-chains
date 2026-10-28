"""autowind command line.

  uv run python -m autowind chains --mask <surface-mask OME-Zarr> --umbilicus <umbilicus.json> \
      --z0 6500 --z1 8000 --voxel-um 9.596 --raise off --out <dir>
  uv run python -m autowind chains --preset p1667 --out <dir>          # inputs of a published chain set
  uv run python -m autowind presets

Writes <out>/relative_windings.json (villa point collections: one unattached strip per chain, cumulative wind_a),
<out>/chains.json (per chain: z, theta, pts [z,y,x], wind, r, perr per step, rl per point; gm_p per step after
--raise gm), <out>/steps.csv (every candidate step: dr, p_err, second-crop agreement, kept, bridged),
<out>/summary.json and <out>/autowind_run.json (options, crop counts, timing, peak RSS, input sha256).
See release/AUTOWIND.md.
"""

import argparse
import json
import sys
from pathlib import Path

from . import core
from .presets import PRESETS

PRESET_KEYS = ("mask", "umbilicus", "umbilicus_transform", "z0", "z1", "voxel_um", "work", "crops_ro", "grad_mag",
               "grad_mag_extra", "shape_hint")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="autowind", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chains", help="relative-winding chains from a surface mask", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--preset", choices=sorted(PRESETS), help="fill the inputs below from a preset (explicit flags win)")
    g.add_argument("--mask", help="surface-mask OME-Zarr group (local path or s3:// URL, read anonymously); level '0' "
                                  "is the working frame, > 0 = sheet")
    g.add_argument("--umbilicus", help="umbilicus in villa format (umbilicus.json control points, or Thaumato .txt)")
    g.add_argument("--umbilicus-transform", help="optional JSON with a 3x4 'transformation_matrix' mapping the "
                                                 "umbilicus xyz into the mask frame")
    g.add_argument("--z0", type=float, help="band start (mask level-0 voxels)")
    g.add_argument("--z1", type=float, help="band end (exclusive)")
    g.add_argument("--voxel-um", type=float, help="voxel size of mask level 0 (um); recorded, warns outside 8.5-10.5")
    g.add_argument("--raise", dest="raise_", choices=["off", "gm", "auto"], default="off",
                   help="grad_mag step raise; 'gm' only for tightly packed sheets; 'auto' = label-free rule: gm at "
                        "--auto-gm-thr iff the crops' sheet packing S_pack < --spack-thr, else off (release/AUTOWIND.md)")
    g.add_argument("--grad-mag", help="lasagna grad_mag OME-Zarr in the mask frame, level '2' = every 4 voxels (--raise gm)")
    g.add_argument("--grad-mag-extra", action="append", help="fallback grad_mag stores for z rows missing in --grad-mag")
    g.add_argument("--out", required=True)
    g = p.add_argument_group("caches and compute")
    g.add_argument("--work", help="cache dir: mask chunks (m7_L0/), extent and grad_mag slices (default <out>/work)")
    g.add_argument("--crops", help="crop checkpoint dir, written (default <out>/crops)")
    g.add_argument("--crops-ro", action="append", help="extra read-only crop checkpoint dirs to reuse")
    g.add_argument("--no-solve", action="store_true", help="fail instead of solving crops that are not cached")
    g.add_argument("--workers", type=int, default=6, help="crop-solve processes (~1-1.5 GB RSS each)")
    g.add_argument("--io-threads", type=int, default=16)
    g = p.add_argument_group("recipe (defaults = the confirmed PHerc1667 recipe; changing them leaves the validated setting)")
    for k, v in core.RECIPE.items():
        if k == "gm_thr":
            continue
        g.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    g.add_argument("--conf-json", default=str(core.CONF_JSON), help="step-confidence model")
    g.add_argument("--gm-model", default=str(core.GM_JSON), help="grad_mag raise model")
    g.add_argument("--gm-thr", type=float, default=core.RECIPE["gm_thr"], help="raise a 1-step to 2 where P >= this")
    g.add_argument("--spack-thr", type=float, default=21.0, help="--raise auto: raise iff S_pack (vox) < this")
    g.add_argument("--auto-gm-thr", type=float, default=0.263, help="--raise auto: gm threshold when it raises")
    sub.add_parser("presets", help="list presets")
    return ap


def resolve(o: argparse.Namespace) -> argparse.Namespace:
    pre = PRESETS.get(o.preset or "", {})
    for k in PRESET_KEYS:
        if getattr(o, k, None) in (None, []) and k in pre:
            setattr(o, k, pre[k])
        elif not hasattr(o, k):
            setattr(o, k, None)
    miss = [k for k in ("mask", "umbilicus", "z0", "z1", "voxel_um") if getattr(o, k) is None]
    if miss:
        raise SystemExit(f"autowind chains: missing --{', --'.join(m.replace('_', '-') for m in miss)} (or --preset)")
    return o


def main(argv=None) -> int:
    o = build_parser().parse_args(argv)
    if o.cmd == "presets":
        print(json.dumps({k: {kk: (str(vv) if isinstance(vv, Path) else [str(x) for x in vv] if isinstance(vv, list) else vv)
                              for kk, vv in v.items()} for k, v in PRESETS.items()}, indent=1))
        return 0
    core.run_chains(resolve(o), log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
