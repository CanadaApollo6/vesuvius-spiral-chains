"""autowind core: a thin, behaviour-preserving driver around tools/auto_windings.py.

The algorithm (crop planning, frozen patch-graph solve, ray reassembly, step-confidence model, grad_mag raise) lives in
tools/auto_windings.py and eval/p0139_graph.py and is NOT modified here: that file is sha-pinned by the PHerc1667
pre-registration (notes/2026-10-22-confirmation-region.md section 11). This module only replaces the scroll-specific
module globals of auto_windings (mask URL, cache dir, volume shape, umbilicus, grad_mag store) with user inputs, in
the main process and in every worker process (pool initializer), and orchestrates plan -> solve -> assemble -> raise.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import resource
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "tools", REPO / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import auto_windings as aw  # noqa: E402  (sets OMP/BLAS thread env defaults on import)

CONF_JSON = REPO / "tools" / "auto_windings_conf_clean.json"     # leakage-free PHerc0139 refit (confirmed setting)
GM_JSON = REPO / "tools" / "auto_windings_gm_z4500_clean.json"   # fitted on PHerc0139 z 4500-6000 only

# The confirmed recipe (PHerc1667 M2 = band_m7_p1667_ray_c): ray reassembly, certifier does not cut (--conf 1.0),
# second-crop disagreements split chains, unscorable steps bridged as +1, steps above 2 never emitted.
RECIPE = dict(dtheta=15.0, theta0=0.0, zoffs="-64,0,64", stride=96.0, seg=144.0, rmin=48.0, conf=1.0,
              require_agree=1, max_step=2, bridge=1, max_len=0, gm_thr=0.18)


# ------------------------------------------------------------------------------------------------- inputs
def load_umbilicus(path: str | Path, transform: str | Path | None = None) -> np.ndarray:
    """Umbilicus control points as an (n, 3) xyz array in the mask frame, sorted by z.

    Formats (as villa volume-cartographer/scripts/spiral/umbilicus.py reads them):
      *.json  {"control_points": [{"x":, "y":, "z":}, ...]}  (villa umbilicus.json)
      other   Thaumato text, one "y, z, x" integer triple per line with a +500 offset (read_thaumato_umbilicus)
    transform: optional JSON with "transformation_matrix" (3x4 or 4x4) mapping the umbilicus xyz into the mask frame
    (p_mask = A p + t), e.g. a published 2.399 um umbilicus onto a 9.362 um mask.
    Interpolation is linear in z and clamps outside the control-point range (auto_windings / eval/spiral_eval); villa's
    interp1d extrapolates instead, which only differs outside [z_min, z_max] of the control points."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        cp = json.loads(path.read_text())["control_points"]
        p = np.array([[q["x"], q["y"], q["z"]] for q in cp], float)
    else:
        rows = [tuple(map(int, ln.strip().replace(" ", "").split(","))) for ln in path.read_text().splitlines() if ln.strip()]
        p = (np.asarray(rows)[:, [2, 0, 1]] - 500).astype(float)  # villa: xyzs = rows[:, [2, 0, 1]] - 500
    if transform:
        M = np.array(json.loads(Path(transform).read_text())["transformation_matrix"], float)[:3]
        p = p @ M[:, :3].T + M[:, 3]
    return p[np.argsort(p[:, 2])]


def umbilicus_fn(xyz: np.ndarray):
    """z -> (x, y), vectorised, identical to auto_windings.umbilicus_9p362() for the same control points."""
    Z, Y, X = xyz[:, 2], xyz[:, 1], xyz[:, 0]
    return lambda z: np.stack([np.interp(z, Z, X), np.interp(z, Z, Y)], -1)


def write_umbilicus(xyz: np.ndarray, out: Path) -> None:
    """The umbilicus actually used (mask frame), villa umbilicus.json format (as auto_windings band writes it)."""
    out.write_text(json.dumps({"control_points": [{"x": round(float(x), 2), "y": round(float(y), 2),
                                                   "z": round(float(z), 2), "score": 100} for x, y, z in xyz]}, indent=1))


def _open_group(url: str):
    import zarr
    so = {"storage_options": {"anon": True}} if "://" in str(url) else {}
    return zarr.open_group(str(url), mode="r", **so)


def _level_scale(g, name: str) -> float | None:
    """Isotropic scale of dataset `name` relative to level 0 from OME-Zarr multiscales metadata (None if absent)."""
    try:
        ms = g.attrs.get("multiscales") or g.attrs.get("ome", {}).get("multiscales")
        ds = {d["path"]: d for d in ms[0]["datasets"]}
        sc = lambda d: [t for t in d["coordinateTransformations"] if t["type"] == "scale"][0]["scale"]
        s0, s = sc(ds["0"]), sc(ds[name])
        r = [b / a for a, b in zip(s0[-3:], s[-3:])]
        return float(r[0]) if max(r) - min(r) < 1e-6 else None
    except Exception:
        return None


def mask_shape(url: str) -> tuple[int, int, int]:
    return tuple(int(v) for v in _open_group(url)["0"].shape)


# ------------------------------------------------------------------------------------------------- configuration
def configure(cfg: dict) -> None:
    """Point auto_windings' module globals at the user's inputs (main process and every worker)."""
    aw.M7_URL = str(cfg["mask"])
    aw.CACHE = Path(cfg["work"])
    aw.M7_SHAPE = tuple(cfg["shape"])
    fn = umbilicus_fn(np.asarray(cfg["umbilicus_xyz"], float))
    aw.umbilicus_9p362 = lambda: fn
    if cfg.get("grad_mag"):
        aw.GM_ZARR = Path(cfg["grad_mag"])
        aw.GM_ZARR_EXTRA = [Path(p) for p in cfg.get("grad_mag_extra", [])]


def recipe_args(o: Namespace) -> Namespace:
    """The argparse namespace auto_windings.plan_band / assemble expect (band subcommand flags)."""
    return Namespace(z=[float(o.z0), float(o.z1)], zoffs=o.zoffs, dtheta=o.dtheta, theta0=o.theta0, stride=o.stride,
                     seg=o.seg, rmin=o.rmin, max_len=o.max_len, conf=o.conf, require_agree=o.require_agree,
                     max_step=o.max_step, bridge=o.bridge, conf_json=str(o.conf_json))


def ensure_extent_cache(args: Namespace, url: str) -> None:
    """auto_windings.radial_extent reads the mask's 8x level ("3") over the band, cached as <work>/l3_<a>_<b>.npy.
    If the store has no 8x level "3", build that cache from level 0 (8^3 any-pooling, one level-3 slice at a time)."""
    pl, _ = aw.probes(args)
    zs = [z for _, zz in pl for z in zz]
    zmin, zmax = min(zs) // 8, max(zs) // 8 + 1
    f = aw.CACHE / f"l3_{zmin}_{zmax}.npy"
    if f.exists():
        return
    g = _open_group(url)
    names = set(g.array_keys()) if hasattr(g, "array_keys") else set(g)
    if "3" in names and _level_scale(g, "3") in (8.0, None):
        return  # radial_extent reads level "3" itself
    a0 = g["0"]
    Z, Y, X = a0.shape
    ny, nx = -(-Y // 8), -(-X // 8)
    v = np.zeros((zmax + 1 - zmin, ny, nx), bool)
    for i, zi in enumerate(range(zmin, zmax + 1)):
        if zi * 8 >= Z:
            break
        s = np.asarray(a0[zi * 8:min(zi * 8 + 8, Z)]) > 0
        s = s.any(0)
        s = np.pad(s, ((0, ny * 8 - Y), (0, nx * 8 - X)))
        v[i] = s.reshape(ny, 8, nx, 8).any((1, 3))
    print(f"no 8x level '3' in {url}: built {f.name} from level 0", flush=True)
    np.save(f, v)


# ------------------------------------------------------------------------------------------------- pipeline
def _key(c: dict) -> str:
    return f"{c['zc']}_{c['theta']:g}_{c['k']}.pkl"  # auto_windings band checkpoint name


def _same_crop(a: dict, b: dict) -> bool:
    """Cached checkpoint `a` was solved for planned crop `b` (older checkpoints may lack the origin "o")."""
    return (a["zc"], a["k"], list(a["zs"])) == (b["zc"], b["k"], list(b["zs"])) and \
        list(a.get("o", b["o"])) == list(b["o"]) and abs(a["theta"] - b["theta"]) < 1e-9 and abs(a["rk"] - b["rk"]) < 1e-6


def plan(o: Namespace, cfg: dict) -> tuple[Namespace, list[dict]]:
    args = recipe_args(o)
    ensure_extent_cache(args, cfg["mask"])
    umb = aw.umbilicus_9p362()
    crops = aw.plan_band(args, None, umb)  # src is unused by plan_band / radial_extent
    for c in crops:  # crop origin, exactly as auto_windings.band
        th = np.radians(c["theta"])
        cx, cy = umb(c["zc"])
        c["o"] = np.rint(np.array([c["zc"], cy + c["rk"] * np.sin(th), cx + c["rk"] * np.cos(th)]) - aw.CROP / 2
                         ).astype(int).tolist()
    return args, crops


def solve(crops: list[dict], cfg: dict, crops_dir: Path, workers: int, io_threads: int, log=print) -> None:
    """Solve crops without a checkpoint in crops_dir (prefetch chunks with threads, solve with processes), as
    auto_windings.band does; per-crop pickles (c, result) are fsynced so an interrupted run resumes."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = aw.M7Chunks(url=cfg["mask"], cache=Path(cfg["work"]) / "m7_L0")
    need = sorted({ijk for c in crops for ijk in src.chunks_of(tuple(c["o"]))})
    with ThreadPoolExecutor(io_threads) as ex:
        for k, _ in enumerate(ex.map(lambda ijk: src.chunk(*ijk) is None, need)):
            if k % 200 == 0:
                log(f"chunks {k}/{len(need)} ({time.time() - t0:.0f}s)")
    log(f"{len(need)} chunks cached ({time.time() - t0:.0f}s)")
    args_d = dict(cache=str(Path(cfg["work"]) / "m7_L0"), seg=cfg["seg"], url=str(cfg["mask"]))
    for attempt in range(6):
        todo = [c for c in crops if not (crops_dir / _key(c)).exists()]
        if not todo:
            return
        log(f"attempt {attempt}: {len(todo)} crops to solve on {workers} workers")
        try:
            # spawn: workers start clean with the single-threaded BLAS env even if the caller (e.g. pytest) loaded
            # numpy first; a forked worker would inherit a multi-threaded BLAS (last-ulp differences)
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=configure,
                                     initargs=(cfg,)) as ex:
                for k, (c, r) in enumerate(ex.map(aw._band_crop, [(c, args_d) for c in todo], chunksize=1)):
                    f = crops_dir / _key(c)
                    tmp = f.with_suffix(".tmp")
                    with open(tmp, "wb") as fh:
                        fh.write(pickle.dumps((c, r)))
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, f)
                    if k % 100 == 0:
                        log(f"crops {k}/{len(todo)} ({time.time() - t0:.0f}s)")
        except BrokenProcessPool as e:
            log(f"pool broken ({e}); restarting")
    raise RuntimeError(f"crops still unsolved after 6 attempts in {crops_dir}")


def sha256(p: Path) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def run_chains(o: Namespace, log=print) -> dict:
    """plan -> (solve missing crops) -> assemble (ray reassembly + certifier) -> optional grad_mag raise."""
    t0 = time.time()
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(o.work or out / "work")
    work.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(o.crops or out / "crops")
    ro_dirs = [Path(d) for d in (o.crops_ro or [])]
    xyz = load_umbilicus(o.umbilicus, o.umbilicus_transform)
    try:
        shape = mask_shape(o.mask)
    except Exception as e:  # only needed to solve crops; cached crops can be reassembled offline
        shape = tuple(o.shape_hint) if getattr(o, "shape_hint", None) else None
        log(f"mask not readable ({type(e).__name__}: {e}); continuing with cached crops only")
    cfg = dict(mask=str(o.mask), work=str(work), shape=list(shape or (0, 0, 0)), umbilicus_xyz=xyz.tolist(),
               grad_mag=str(o.grad_mag) if o.grad_mag else None, grad_mag_extra=[str(p) for p in o.grad_mag_extra or []],
               seg=o.seg)
    configure(cfg)
    if not 8.5 <= o.voxel_um <= 10.5:
        log(f"WARNING: --voxel-um {o.voxel_um}: the frozen graph (192^3 crops, block 12, radius 40) and both models "
            "were only validated at 9.362 and 9.596 um; consider resampling the mask to ~9.5 um")
    write_umbilicus(xyz, out / "umbilicus_maskframe.json")

    args, crops = plan(o, cfg)
    log(f"{len(crops)} crops planned ({time.time() - t0:.0f}s)")
    have, missing = {}, []
    for c in crops:
        f = next((d / _key(c) for d in [crops_dir] + ro_dirs if (d / _key(c)).exists()), None)
        if f is None:
            missing.append(c)
            continue
        cc, r = pickle.loads(f.read_bytes())
        if not _same_crop(cc, c):
            raise RuntimeError(f"cached crop {f} was solved for different inputs (umbilicus / band / recipe): "
                               f"cached o={cc['o']} rk={cc['rk']}, planned o={c['o']} rk={c['rk']}; use a fresh --crops")
        have[_key(c)] = (cc, r)
    n_reused = len(have)
    if missing:
        if o.no_solve:
            raise RuntimeError(f"{len(missing)} of {len(crops)} crops not cached and --no-solve given")
        if shape is None:
            raise RuntimeError(f"{len(missing)} crops need solving but the mask {o.mask} cannot be opened")
        t_s = time.time()
        solve(missing, cfg, crops_dir, o.workers, o.io_threads, log)
        log(f"solved {len(missing)} crops in {time.time() - t_s:.0f}s")
        for c in missing:
            have[_key(c)] = pickle.loads((crops_dir / _key(c)).read_bytes())
    # crop order = sorted checkpoint file names, as `auto_windings.py band --reassemble` (glob order matters for ties)
    results = [have[k] for k in sorted(have)]
    t_solve = time.time() - t0

    mode, gm_thr, pack = o.raise_, o.gm_thr, None
    if mode == "auto":  # label-free count rule: S_pack from the solved crops decides raise / no raise (packing.py)
        from . import packing
        pack = packing.s_pack(results, o.z0, o.z1)
        mode = packing.decide(pack["S_pack"], o.spack_thr)
        gm_thr = o.auto_gm_thr
        pack.update(spack_thr=o.spack_thr, auto_gm_thr=o.auto_gm_thr, decision=mode,
                    gm_thr_used=gm_thr if mode == "gm" else None)
        (out / "packing.json").write_text(json.dumps(pack, indent=1))
        log(f"--raise auto: S_pack {pack['S_pack']} vox (95 % {pack['boot95']}) vs {o.spack_thr} -> "
            f"{'gm raise at ' + str(gm_thr) if mode == 'gm' else 'no raise'}")
    stage = out / "noraise" if mode == "gm" else out
    stage.mkdir(parents=True, exist_ok=True)
    aw.assemble(args, results, stage, timing=None)
    if mode == "gm":
        ch = json.loads((stage / "chains.json").read_text())
        a, b = aw._zrange(ch)
        f = aw.CACHE / f"gm_band_z{a // 4}_{b // 4 + 1}.npy"
        if not f.exists() and not (o.grad_mag and Path(o.grad_mag).exists()):
            raise RuntimeError(f"--raise gm needs --grad-mag (lasagna grad_mag OME-Zarr, level '2' = 4 mask voxels) "
                               f"or the cache {f}")
        aw.gmfix(Namespace(src=str(stage), model=str(o.gm_model), thr=gm_thr, out=str(out)))

    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    run = dict(autowind=__import__("autowind").__version__, argv=sys.argv,
               options=vars(o),
               mask_shape=list(shape) if shape else None, voxel_um=o.voxel_um,
               raise_applied=dict(mode=mode, gm_thr=gm_thr if mode == "gm" else None, packing=pack),
               crops=dict(planned=len(crops), reused=n_reused, solved=len(missing), dir=str(crops_dir),
                          read_only=[str(d) for d in ro_dirs]),
               seconds=dict(plan_and_solve=round(t_solve, 1), total=round(time.time() - t0, 1)),
               peak_rss_mb=dict(main=round(ru_self), largest_worker=round(ru_kids)),
               sha256={"relative_windings.json": sha256(out / "relative_windings.json"),
                       "chains.json": sha256(out / "chains.json"),
                       "umbilicus": sha256(Path(o.umbilicus)), "conf_json": sha256(Path(o.conf_json)),
                       "gm_model": sha256(Path(o.gm_model)) if mode == "gm" else None,
                       "tools/auto_windings.py": sha256(REPO / "tools" / "auto_windings.py"),
                       "eval/p0139_graph.py": sha256(REPO / "eval" / "p0139_graph.py")})
    (out / "autowind_run.json").write_text(json.dumps(run, indent=1, default=str))
    log(f"wrote {out / 'relative_windings.json'} ({run['seconds']['total']:.0f}s)")
    return run
