"""Automatic relative-winding annotations (relative_windings.json) for the villa spiral fitter, from surface masks.

  # site validation on the 87 PHerc0139 test crops (fits the step-confidence model on the tune half)
  uv run python tools/auto_windings.py validate --seg data/preds/m7_seg --out data/scratch/auto_windings/val_m7
  # automatic annotations for a z band (streams the bucket's m7 masks, caches chunks)
  uv run python tools/auto_windings.py band --z 3000 4500 --dtheta 15 --out data/scratch/auto_windings/band_m7
  # oracle chains from the human PHerc0139 meshes (same probes), and a corrupted negative control
  uv run python tools/auto_windings.py oracle --z 3000 4500 --dtheta 15 --out data/scratch/auto_windings/band_oracle
  uv run python tools/auto_windings.py corrupt --src data/scratch/auto_windings/band_oracle/relative_windings.json \
      --frac 0.1 --out data/scratch/auto_windings/band_corrupt10
  # compare auto chains with oracle chains (end-to-end step precision), check the JSON with the pinned loader
  uv run python tools/auto_windings.py compare --auto data/scratch/auto_windings/band_m7 --oracle data/scratch/auto_windings/band_oracle
  uv run --with tqdm --with einops python tools/auto_windings.py check data/scratch/auto_windings/band_m7/relative_windings.json

Method (notes/2026-10-20-auto-windings.md). A probe is a horizontal ray (fixed z, angle theta) from the umbilicus
outward. Along it, 192^3 crops are centred every --stride vox of radius; each crop is solved with the frozen patch
graph (eval/p0139_graph.py: mask dilated 1, block 12, radius 40, k 30) and the ray's mask crossings (runs of the
dilated mask, gaps < 2.5 vox merged) are read off in order with the solved winding of their patch. Consecutive
crossings give a step d = w_next - w_prev (sign fixed so + is outward); steps rounding to 0 merge the two crossings,
steps < 0 break the chain. Each step is taken from the crop most central to it, and checked against a second crop
when one holds both ends. A logistic step-confidence model (graph residual, non-integrality, degree, step size,
spacing, run length, grazing, multi-winding run; fitted on the tune half of the test sites) scores every step;
chains are split at steps below --conf, so only confident consecutive runs (>= 2 points) are emitted, with
wind_a = cumulative step (0, 1, 2, ... outward). Points are on the sheet (centre of the raw-mask run).

Coordinates: PHerc0139 volume 20250728140407 (9.362 um), point "p" = [x, y, z]. The umbilicus is published in the
2.399 um frame; umbilicus_9p362() maps every control point with the published affine (p9_xyz = A p2399_xyz + t,
data/eval/surface_labels_2026_07_sample/transform_0139_2p399_to_9p362.json) and interpolates linearly in z, as the
fitter's json_umbilicus_z_to_yx does.
Fitter note: pcl_unattached_pcl_min_point_spacing (default 16 vox) decimates strip points closer than that; sheet
pitch is ~10-15 vox here, so run the fit with pcl_unattached_pcl_min_point_spacing = 0.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")  # THP "always" + madvise: compaction stalls (2x sys time) in workers

import numpy as np
import tifffile
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from p0139_crossings import mask_normals, orient_normals  # noqa: E402
from p0139_graph import site_graph  # noqa: E402

M7_URL = ("s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/"
          "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr")
CACHE = REPO / "data" / "scratch" / "auto_windings"
UMB_2P399 = REPO / "data" / "eval" / "p0139_umbilicus" / "umbilicus_20260102150214.json"
TF_JSON = REPO / "data" / "eval" / "surface_labels_2026_07_sample" / "transform_0139_2p399_to_9p362.json"
CONF_JSON = Path(__file__).with_name("auto_windings_conf.json")
MESHES = REPO / "data" / "meshes" / "p0139"
MESH_TAG = "20260102150214-2.399um"
FROZEN = dict(block=12, radius=40.0, k=30, min_vox=20, dilate=1)
CROP, CORE = 192, 24
GAP, STEP = 2.5, 0.5          # run merge gap (vox, as the frozen signed count), ray sampling step
GRAZE_RAY = 0.3               # skip a ray in a crop if |dominant normal . ray| is below this
M7_SHAPE = (20974, 6621, 6621)  # bucket m7 L0 (9.362 frame); M7Chunks.arr() replaces it with the real shape
SCROLL = "p0139"


def set_scroll(name: str) -> None:
    """--scroll p1667: PHerc1667 confirmation band in its fit frame (20251217075048 level 2, 9.596 um, p = p_2399 / 4).
    Local m7 mask (spiral/predict_band.py, levels 0-3), lasagna grad_mag, the human-mesh umbilicus and the merged
    human mesh labels (eval/spiral_eval.py --scroll p1667); every cache under data/scratch/auto_windings/p1667/.
    The method itself (crops, frozen patch graph, confidence and gm models, thresholds) is unchanged. p0139 = no-op."""
    global SCROLL, M7_URL, CACHE, M7_SHAPE, GM_ZARR, GM_ZARR_EXTRA
    assert name in ("p0139", "p1667"), name
    import spiral_eval as se
    se.set_scroll(name)
    if name == SCROLL:
        return
    assert SCROLL == "p0139", "set_scroll is one-way (p0139 -> p1667)"
    SCROLL = name
    M7_URL = str(REPO / "data" / "p1667" / "masks" / "m7_L2_th0.2.zarr")
    CACHE = REPO / "data" / "scratch" / "auto_windings" / "p1667"
    CACHE.mkdir(parents=True, exist_ok=True)
    M7_SHAPE = (9269, 3808, 3808)
    GM_ZARR = REPO / "data" / "p1667" / "inputs" / "PHerc1667_grad_mag.ome.zarr"  # identity resample, level 2 = 4 vox
    GM_ZARR_EXTRA = []


FEATURES = ["l_res", "l_dfrac", "logdeg", "step2", "gap", "runlen", "graze", "multi", "l_ray_dfrac", "l_ray_res",
            "l_ray_resmax", "ray_dr2"]


# ------------------------------------------------------------------------------------------------- geometry
def affine_2p399_to_9p362(path: Path = TF_JSON) -> tuple[np.ndarray, np.ndarray]:
    """(A, t) with p9_xyz = A @ p2399_xyz + t; eval/spiral_eval.affine (shared helper)."""
    from spiral_eval import affine
    return affine()


def umbilicus_9p362():
    """z -> (x, y) of the PHerc0139 umbilicus in the 9.362 frame, from eval/spiral_eval.Umbilicus (2.399 control
    points mapped whole with p9 = A p + t, linear in z). Vectorised; clamps outside the control-point range."""
    from spiral_eval import Umbilicus
    u = Umbilicus()
    return lambda z: np.stack([np.interp(z, u.z, u.x), np.interp(z, u.z, u.y)], -1)


def write_umbilicus_9p362(out: Path) -> None:
    if SCROLL == "p1667":  # already in the fit frame
        import spiral_eval as se
        out.write_text((se.P1667 / "umbilicus_p1667_L2.json").read_text())
        return
    A, t = affine_2p399_to_9p362()
    cp = json.loads(UMB_2P399.read_text())["control_points"]
    q = np.array([[p["x"], p["y"], p["z"]] for p in cp], float) @ A.T + t
    q = q[np.argsort(q[:, 2])]
    out.write_text(json.dumps({"control_points": [{"x": round(float(x), 2), "y": round(float(y), 2),
                                                   "z": round(float(z), 2), "score": 100} for x, y, z in q]}, indent=1))


# ------------------------------------------------------------------------------------------------- mask source
class M7Chunks:
    """Streams the bucket's m7 surface prediction (L0, 192^3 chunks); each chunk is cached as packed bits."""

    def __init__(self, url: str | None = None, cache: Path | None = None, level: str = "0"):
        url, cache = url or M7_URL, cache or CACHE / "m7_L0"  # module globals at call time (set_scroll)
        self.url, self.cache, self.level = url, Path(cache), level
        self.cache.mkdir(parents=True, exist_ok=True)
        self._a = None
        self.shape = M7_SHAPE
        self.ch = 192

    def arr(self):
        if self._a is None:
            import zarr
            so = {"storage_options": {"anon": True}} if "://" in str(self.url) else {}  # local zarrs: plain path
            self._a = zarr.open_group(str(self.url), mode="r", **so)[self.level]
            self.shape = self._a.shape
        return self._a

    def chunk(self, cz: int, cy: int, cx: int) -> np.ndarray:
        f = self.cache / f"{cz}_{cy}_{cx}.npz"
        c = self.ch
        if f.exists():
            try:
                with np.load(f) as z:
                    return np.unpackbits(z["b"], count=c ** 3).reshape(c, c, c).astype(bool)
            except Exception:
                pass
        a = self.arr()
        sl = tuple(slice(i * c, min((i + 1) * c, n)) for i, n in zip((cz, cy, cx), self.shape))
        m = np.zeros((c, c, c), bool)
        if all(s.stop > s.start for s in sl):
            v = np.asarray(a[sl]) > 0
            m[:v.shape[0], :v.shape[1], :v.shape[2]] = v
        tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, b=np.packbits(m.ravel()))
        os.replace(tmp, f)
        return m

    def chunks_of(self, o, shape=(CROP, CROP, CROP)) -> list[tuple[int, int, int]]:
        c = self.ch
        rng = [range(max(0, o[a] // c), (o[a] + shape[a] - 1) // c + 1) for a in range(3)]
        return [(i, j, k) for i in rng[0] for j in rng[1] for k in rng[2]]

    def crop(self, o, shape=(CROP, CROP, CROP)) -> np.ndarray:
        c = self.ch
        out = np.zeros(shape, bool)
        for (i, j, k) in self.chunks_of(o, shape):
            m = self.chunk(i, j, k)
            lo = np.array([i, j, k]) * c
            a0 = np.maximum(lo, o)
            a1 = np.minimum(lo + c, np.array(o) + shape)
            if (a1 <= a0).any():
                continue
            out[tuple(slice(s - oo, e - oo) for s, e, oo in zip(a0, a1, o))] = \
                m[tuple(slice(s - ll, e - ll) for s, e, ll in zip(a0, a1, lo))]
        return out


# ------------------------------------------------------------------------------------------------- per crop
def solve_crop(raw: np.ndarray) -> dict | None:
    """Frozen patch-graph solve of one crop (raw = binary surface mask)."""
    mask = ndi.binary_dilation(raw, iterations=FROZEN["dilate"]) if FROZEN["dilate"] else raw.copy()
    if mask.sum() < 2000:
        return None
    nrm = orient_normals(mask, mask_normals(mask))
    dvec = nrm[:, mask].mean(1)
    dvec /= max(np.linalg.norm(dvec), 1e-9)
    L, w, ne, rnode, deg = site_graph(mask, nrm, FROZEN["block"], FROZEN["min_vox"], FROZEN["k"], FROZEN["radius"],
                                      np.random.default_rng(1), 0.0, edges=True)
    if L.max() < 2:
        return None
    return dict(raw=raw, mask=mask, nrm=nrm, L=L, w=w, rnode=rnode, deg=deg, d=dvec)


def ray_crossings(sol: dict, p0: np.ndarray, u: np.ndarray, t0: float, t1: float) -> list[dict] | None:
    """Mask crossings along p0 + u t (local zyx), t in [t0, t1]; winding sign fixed so + is along u.
    Runs touching the segment ends are dropped (truncated). None if the ray grazes the crop's sheets."""
    s = float(sol["d"] @ u)
    if abs(s) < GRAZE_RAY or t1 - t0 < 4:
        return None
    sg = np.sign(s)
    t = np.arange(t0, t1 + 1e-9, STEP)
    q = np.rint(p0[None] + t[:, None] * u[None]).astype(int)
    shp = np.array(sol["mask"].shape)
    ok = ((q >= 0) & (q < shp)).all(1)
    q = np.clip(q, 0, shp - 1)
    qi = tuple(q.T)
    on = sol["mask"][qi] & ok
    raw = sol["raw"][qi] & ok
    lab = sol["L"][qi]
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) * STEP > GAP)  # gap >= GAP vox -> new run (as count_signed min_gap)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    w, rnode, deg, nrm = sol["w"], sol["rnode"], sol["deg"], sol["nrm"]
    out = []
    for a, b in zip(starts, ends):
        if a == 0 or b == len(t) - 1:
            continue
        seg = np.arange(a, b + 1)
        rr = seg[raw[seg]]
        c = int(np.rint(rr.mean())) if len(rr) else int((a + b) // 2)
        ll = lab[seg]
        nz = seg[ll > 0]
        if len(nz) == 0:
            out.append(dict(t=float(t[c]), pos=q[c].astype(float), patch=0, runlen=float((b - a) * STEP)))
            continue
        k = nz[np.argmin(np.abs(nz - c))]
        pt = int(lab[k])
        wv = w[lab[nz] - 1] * sg
        n = nrm[(slice(None),) + tuple(q[c])]
        out.append(dict(t=float(t[c]), pos=p0 + t[c] * u, patch=pt, w=float(w[pt - 1] * sg),
                        res=float(rnode[pt - 1]), deg=int(deg[pt - 1]), runlen=float((b - a) * STEP),
                        multi=int(len(set(np.rint(wv).astype(int).tolist())) > 1), graze=float(abs(n @ u))))
    segment_stats(out)
    return out


def frac(v: float) -> float:
    return abs(v - round(v))


def step_feats(a: dict, b: dict) -> dict:
    """Step a -> b features. res: max node mean L1 residual; dfrac: non-integrality of the winding difference;
    ray_*: the same signals over all steps of this crop's ray segment (attached by segment_stats)."""
    d = b["w"] - a["w"]
    lg = lambda v: float(np.log(v + 1e-3))
    sg = a.get("seg", {})
    res = max(a["res"], b["res"])
    return dict(d=d, dr=int(round(d)), res=res, frac=max(frac(a["w"]), frac(b["w"])), dfrac=frac(d),
                l_res=lg(res), l_dfrac=lg(frac(d)), logdeg=float(np.log1p(min(a["deg"], b["deg"]))),
                step2=int(round(d) >= 2), gap=b["t"] - a["t"], runlen=max(a["runlen"], b["runlen"]),
                graze=min(a["graze"], b["graze"]), multi=max(a["multi"], b["multi"]),
                l_ray_dfrac=lg(sg.get("dfrac", 0.0)), l_ray_res=lg(sg.get("res", 0.0)),
                l_ray_resmax=lg(sg.get("resmax", 0.0)), ray_dr2=sg.get("dr2", 0.0))


def segment_stats(cr: list[dict]) -> None:
    """Attach ray-segment aggregates (mean dfrac, mean / max residual, share of steps >= 2) to every crossing."""
    _, st = chain_steps(cr)
    st = [x for x in st if x.get("dr") is not None and x["dr"] >= 1]
    seg = dict(dfrac=float(np.mean([x["dfrac"] for x in st])) if st else 0.0,
               res=float(np.mean([x["res"] for x in st])) if st else 0.0,
               resmax=float(np.max([x["res"] for x in st])) if st else 0.0,
               dr2=float(np.mean([x["dr"] >= 2 for x in st])) if st else 0.0, n=len(st))
    for c in cr:
        c["seg"] = seg


def chain_steps(cr: list[dict]) -> tuple[list[dict], list[dict]]:
    """Crossings (sorted by t) -> (kept crossings, steps between consecutive kept crossings). Steps rounding to 0
    merge (the later crossing is dropped); unsolved crossings (no patch) stay and break the chain."""
    keep: list[dict] = []
    for c in cr:
        if keep and "w" in c and "w" in keep[-1] and round(c["w"] - keep[-1]["w"]) == 0:
            continue
        keep.append(c)
    steps = []
    for a, b in zip(keep[:-1], keep[1:]):
        if "w" not in a or "w" not in b:
            steps.append(dict(dr=None))
        else:
            steps.append(step_feats(a, b))
    return keep, steps


# ------------------------------------------------------------------------------------------------- confidence
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    beta = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ (p - y) + l2 * np.r_[0, beta[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(beta))
        beta -= np.linalg.solve(H, g)
    return dict(features=FEATURES, mu=mu.tolist(), sd=sd.tolist(), beta=beta.tolist())


def p_err(model: dict, rows: list[dict]) -> np.ndarray:
    X = np.array([[r[f] for f in model["features"]] for r in rows], float)
    Z = np.c_[np.ones(len(X)), (X - np.array(model["mu"])) / np.array(model["sd"])]
    return 1 / (1 + np.exp(-Z @ np.array(model["beta"])))


def valid_step(s: dict) -> bool:
    return s.get("dr") is not None and s["dr"] >= 1


# ------------------------------------------------------------------------------------------------- validation
def _site_rows(args_tuple):
    s, seg_dir, zoffs, loffs = args_tuple
    raw = tifffile.imread(Path(seg_dir) / f"{s['stem']}.tif") > 0
    sol = solve_crop(raw)
    if sol is None:
        return []
    surf = tifffile.imread(s["surfaces"]).astype(np.int32)
    wind = {int(k): v["winding"] for k, v in s["instances"].items()}
    present = set(wind.values())
    lab_pts = np.argwhere(surf > 0)
    lab_ids = surf[tuple(lab_pts.T)]
    from scipy.spatial import cKDTree
    tree = cKDTree(lab_pts)
    umb = umbilicus_9p362()
    o = np.array(s["origin_zyx"], float)
    rows = []
    for zl in zoffs:
        zg = o[0] + zl
        cx, cy = umb(zg)
        cen = o[1:] + CROP / 2  # (y, x) global
        ur = np.array([cen[0] - cy, cen[1] - cx])
        ur /= np.linalg.norm(ur)
        theta = float(np.degrees(np.arctan2(ur[0], ur[1])) % 360)
        v = np.array([-ur[1], ur[0]])
        for lo in loffs:
            c_loc = np.array([zl, CROP / 2 + lo * v[0], CROP / 2 + lo * v[1]])
            u = np.array([0.0, ur[0], ur[1]])
            # clip t so that the ray stays inside the core box [CORE, CROP - CORE)
            tlo, thi = -1e9, 1e9
            for a in (1, 2):
                if abs(u[a]) < 1e-9:
                    if not (CORE <= c_loc[a] < CROP - CORE):
                        tlo, thi = 1, 0
                    continue
                ta, tb = (CORE - c_loc[a]) / u[a], (CROP - CORE - 1 - c_loc[a]) / u[a]
                tlo, thi = max(tlo, min(ta, tb)), min(thi, max(ta, tb))
            if thi - tlo < 20:
                continue
            cr = ray_crossings(sol, c_loc, u, tlo, thi)
            if not cr:
                continue
            keep, steps = chain_steps(cr)
            # truth: winding of the nearest labelled surface voxel within 3 vox (ambiguous if another instance
            # is within 1 vox of that distance)
            tw = []
            for c in keep:
                dd, ii = tree.query(c["pos"], k=12, distance_upper_bound=4.5)
                okk = np.isfinite(dd)
                if not okk.any() or dd[0] > 3:
                    tw.append(None)
                    continue
                ids = lab_ids[ii[okk]]
                d0, id0 = dd[0], ids[0]
                other = dd[okk][ids != id0]
                tw.append(None if (len(other) and other[0] - d0 < 1.0) else wind[int(id0)])
            ray = f"{s['stem']}|{zl}|{lo}"
            for k, st in enumerate(steps):
                ta, tb = tw[k], tw[k + 1]
                truth = None if ta is None or tb is None else tb - ta
                skip = truth is not None and abs(truth) >= 2 and any(
                    w not in present for w in range(min(ta, tb) + 1, max(ta, tb)))
                rows.append(dict(stem=s["stem"], cls=s["cls"], ray=ray, k=k, z=zg, theta=theta, lateral=lo,
                                 truth=truth, skip=int(skip), wa=ta, wb=tb, **{kk: vv for kk, vv in st.items()}))
    return rows


def validate(args) -> None:
    from p0139_sites import ROOT, load_sites
    sites = sorted(load_sites(ROOT), key=lambda s: s["stem"])[: args.limit or None]
    if args.exclude_z:  # drop sites whose 192^3 crop overlaps [z0, z1) (leakage-free confidence model)
        za, zb = args.exclude_z
        sites = [s for s in sites if not (s["origin_zyx"][0] < zb and s["origin_zyx"][0] + CROP > za)]
        print(f"{len(sites)} sites outside z [{za:g}, {zb:g})", flush=True)
    half = {s["stem"]: k % 2 for k, s in enumerate(sites)}
    zoffs = [int(v) for v in args.zoffs.split(",")]
    loffs = [int(v) for v in args.loffs.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csvf = Path(args.rows_csv) if args.rows_csv else out.with_suffix(".csv")
    t0 = time.time()
    if csvf.exists() and not args.redo:
        rows = list(csv.DictReader(csvf.open()))
        for r in rows:
            for kk, vv in r.items():
                if kk in ("stem", "cls", "ray"):
                    continue
                r[kk] = None if vv in ("", "None") else float(vv)
        rows = [r for r in rows if r["stem"] in half]
    else:
        rows = []
        with ProcessPoolExecutor(args.workers) as ex:
            for k, rr in enumerate(ex.map(_site_rows, [(s, args.seg, zoffs, loffs) for s in sites])):
                rows += rr
                print(f"{k + 1}/{len(sites)} {sites[k]['stem']} steps {len(rr)} ({time.time() - t0:.0f}s)", flush=True)
        cols = sorted({kk for r in rows for kk in r}, key=lambda c: (c not in ("stem", "cls", "ray", "k"), c))
        with csvf.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
    for r in rows:
        r["half"] = half[r["stem"]]
    summ = analyse(rows, args, out)
    summ["seconds"] = round(time.time() - t0, 1)
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))


def analyse(rows, args, out) -> dict:
    """Coverage / precision of steps vs the human windings. The confidence model is scored out of fold (fit on one
    site half, predict the other, both ways); the frozen model (tools/auto_windings_conf.json, --fit) is refit on all
    sites, with the threshold = the largest out-of-fold p_err cut whose cumulative precision reaches --target."""
    sc = [r for r in rows if r["truth"] is not None]           # scorable: both ends carry an unambiguous label
    cand = [r for r in sc if r["dr"] is not None and r["dr"] >= 1]
    for r in sc:
        r["ok"] = int(r["dr"] is not None and r["dr"] == r["truth"])
        lo, hi = min(r["wa"], r["wb"]), max(r["wa"], r["wb"])
        r["band3337"] = int(hi >= 33 and lo <= 37)
    X = lambda rr: np.array([[r[f] for f in FEATURES] for r in rr], float)
    y = lambda rr: np.array([1 - r["ok"] for r in rr], float)
    allc = [r for r in rows if r["dr"] is not None and r["dr"] >= 1]  # p_err for unscorable steps too (chains)
    conf_json = Path(args.conf_json)
    if args.fit or not conf_json.exists():
        for h in (0, 1):
            tr = [r for r in cand if r["half"] == h]
            te = [r for r in allc if r["half"] != h]
            for r, p in zip(te, p_err(fit_logistic(X(tr), y(tr)), te)):
                r["p_err"] = float(p)
        model = fit_logistic(X(cand), y(cand))
    else:
        model = json.loads(conf_json.read_text())["model"]
        for r, p in zip(allc, p_err(model, allc)):
            r["p_err"] = float(p)
    for r in allc:
        if r["dr"] > args.max_step:  # hard rule: steps above --max-step are never emitted
            r["p_err"] = 1.0
    sec = json.loads((CACHE / "sector_2167.json").read_text()) if (CACHE / "sector_2167.json").exists() else None
    for r in sc:
        r["sector"] = int(in_sector(r["theta"], r["z"], sec))
    summ = {"seg": args.seg, "sites": len({r["stem"] for r in rows}), "rays": len({r["ray"] for r in rows}),
            "steps_all": len(rows), "steps_scorable": len(sc), "steps_candidate_dr_ge_1": len(cand),
            "raw_precision_scorable": round(100 * np.mean([r["ok"] for r in sc]), 2)}

    def curve(rs, base_n):
        rs = sorted(rs, key=lambda r: r["p_err"])
        okc = np.cumsum([r["ok"] for r in rs])
        out_ = []
        for cov in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            k = max(1, int(round(cov * len(rs))))
            out_.append(dict(cand_frac=cov, thr=round(rs[k - 1]["p_err"], 4), kept=k,
                             coverage=round(100 * k / max(base_n, 1), 2), precision=round(100 * okc[k - 1] / k, 2)))
        return out_

    def thr_for(rs, target):
        rs = sorted(rs, key=lambda r: r["p_err"])
        prec = np.cumsum([r["ok"] for r in rs]) / np.arange(1, len(rs) + 1)
        good = np.flatnonzero(prec[19:] >= target / 100) + 19  # at least 20 kept steps
        return float(rs[good[-1]]["p_err"]) if len(good) else 0.0

    def at(rs, base, thr):
        k = [r for r in rs if r["p_err"] <= thr]
        res = dict(kept=len(k), scorable=len(base), coverage=round(100 * len(k) / max(len(base), 1), 2),
                   precision=round(100 * np.mean([r["ok"] for r in k]), 2) if k else None)
        stems = sorted({r["stem"] for r in k})
        if len(stems) > 2:
            si = np.array([stems.index(r["stem"]) for r in k])
            okv = np.array([r["ok"] for r in k], float)
            bt = np.random.default_rng(1).integers(0, len(stems), (1000, len(stems)))
            num = np.bincount(si, okv, len(stems))[bt].sum(1) / np.maximum(np.bincount(si, None, len(stems))[bt].sum(1), 1)
            res["precision_ci95"] = [round(100 * float(np.percentile(num, q)), 2) for q in (2.5, 97.5)]
        return res

    nb = [r for r in cand if not r["band3337"]]
    summ["oof_curve_all"] = curve(cand, len(sc))
    summ["oof_curve_outside_w033_w037"] = curve(nb, len([r for r in sc if not r["band3337"]]))
    summ["thresholds"] = {}
    for target in sorted({args.target, 98.0, 99.0}):
        thr = thr_for(cand, target)
        thr_nb = thr_for(nb, target)
        summ["thresholds"][str(target)] = {
            "thr_all": round(thr, 4), "all": at(cand, sc, thr),
            "all_no_skip_labels": at([r for r in cand if not r["skip"]], [r for r in sc if not r["skip"]], thr),
            "outside_2167_sector": at([r for r in cand if not r["sector"]], [r for r in sc if not r["sector"]], thr),
            "inside_2167_sector": at([r for r in cand if r["sector"]], [r for r in sc if r["sector"]], thr),
            "outside_w033_w037_same_thr": at(nb, [r for r in sc if not r["band3337"]], thr),
            "touching_w033_w037_same_thr": at([r for r in cand if r["band3337"]], [r for r in sc if r["band3337"]], thr),
            "thr_outside_w033_w037": round(thr_nb, 4),
            "outside_w033_w037": at(nb, [r for r in sc if not r["band3337"]], thr_nb),
            "by_dr": {int(d): at([r for r in cand if r["dr"] == d], [r for r in sc if r["dr"] == d], thr)
                      for d in sorted({r["dr"] for r in cand})},
            "by_cls": {c: at([r for r in cand if r["cls"] == c], [r for r in sc if r["cls"] == c], thr)
                       for c in sorted({r["cls"] for r in cand})},
            "chains": emit_chains_from_rows([r for r in rows if r["dr"] is not None], thr),
        }
    chosen = summ["thresholds"][str(args.target)]["thr_all"]
    summ["chosen_thr"] = chosen
    summ["model"] = model
    if args.fit:
        n_sites = len({r["stem"] for r in rows})
        excl = f", sites whose crop overlaps z {args.exclude_z} excluded" if args.exclude_z else ""
        conf_json.write_text(json.dumps({"model": model, "thr": chosen, "target_precision": args.target,
                                         "fitted_on": f"{args.seg}, all {n_sites} sites{excl} (threshold from 2-fold out-of-fold)",
                                         "frozen_graph": FROZEN}, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "model"}, indent=1))
    return summ


def emit_chains_from_rows(rows, thr) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["ray"], []).append(r)
    lens, full = [], []
    for ray, rr in by.items():
        rr.sort(key=lambda r: r["k"])
        cur = []
        prev_k = None
        for r in rr:
            good = r.get("p_err") is not None and r["p_err"] <= thr and r["dr"] >= 1
            if good and (prev_k is None or r["k"] == prev_k + 1) and cur:
                cur.append(r)
            elif good:
                if cur:
                    lens.append(len(cur) + 1)
                    full.append(all(x.get("ok", 1) for x in cur))
                cur = [r]
            else:
                if cur:
                    lens.append(len(cur) + 1)
                    full.append(all(x.get("ok", 1) for x in cur))
                cur = []
            prev_k = r["k"]
        if cur:
            lens.append(len(cur) + 1)
            full.append(all(x.get("ok", 1) for x in cur))
    return dict(n=len(lens), median_points=float(np.median(lens)) if lens else 0, fully_correct=round(100 * np.mean(full), 2) if full else None)


# ------------------------------------------------------------------------------------------------- #2167 sector
def load_meshes9(zlo: float, zhi: float, names=None) -> dict:
    """{winding: (H, W, 3) zyx 9.362 grid restricted to rows/cols touching [zlo, zhi]} for the PHerc0139 segments
    (published affine, no per-site shift). Title segment and segments without a 2.399 mesh are skipped.
    --scroll p1667: the merged human mesh labels (one surface per winding k, fit frame)."""
    if SCROLL == "p1667":
        import spiral_eval as se
        return {s["label"]: [(s["name"], s["grid"])] for s in se.p1667_segments(zlo - 10, zhi + 10)}
    A, t = affine_2p399_to_9p362()
    out = {}
    import re
    for d in sorted(MESHES.iterdir()):
        m = re.search(r"-w(\d+)_", d.name)
        if not m or not (d / MESH_TAG / "x.tif").exists():
            continue
        g = np.stack([tifffile.imread(d / MESH_TAG / f"{c}.tif") for c in "xyz"], -1).astype(np.float64)
        ok = (g > 0).all(-1) & np.isfinite(g).all(-1)
        p = g @ A.T + t
        p[~ok] = np.nan
        z = p[..., 2]
        sel = np.isfinite(z) & (z >= zlo - 10) & (z <= zhi + 10)
        if not sel.any():
            continue
        ii, jj = np.nonzero(sel)
        sub = p[max(ii.min() - 1, 0):ii.max() + 2, max(jj.min() - 1, 0):jj.max() + 2][..., ::-1].copy()
        w = int(m.group(1))
        out.setdefault(w, []).append((d.name, sub))
    return out


def sector(args) -> None:
    """villa #2167 gap per 50-z slab (largest empty angular run of w035), via eval/spiral_eval.gap_sectors."""
    from spiral_eval import Umbilicus, gap_sectors, human_segments
    z0, z1 = 1700, 8400  # covers the test sites and the band
    gaps = gap_sectors(human_segments(z0, z1), Umbilicus(), z0, z1)
    (CACHE / "sector_2167.json").write_text(json.dumps({"source": "eval/spiral_eval.gap_sectors", "gaps": gaps,
                                                        "margin_deg": 10.0}, indent=1))
    band = [g for g in gaps if 3000 <= g["z0"] < 4500]
    print(f"{len(gaps)} slabs with a gap; z 3000-4500: start {min(g['start'] for g in band):.0f}-"
          f"{max(g['start'] for g in band):.0f}, width {min(g['width'] for g in band):.0f}-{max(g['width'] for g in band):.0f}")


def in_sector(theta: float, z: float, sec: dict, margin: float = 10.0) -> bool:
    if not sec or "gaps" not in sec:
        return False
    for g in sec["gaps"]:
        if g["z0"] <= z < g["z1"]:
            return bool((theta - (g["start"] - margin)) % 360.0 < g["width"] + 2 * margin)
    return False


# ------------------------------------------------------------------------------------------------- band (auto)
def probes(args):
    zc_all = [c * 192 + 96 for c in range(int(args.z[0]) // 192 - 1, int(args.z[1]) // 192 + 2)]
    zoffs = [int(v) for v in args.zoffs.split(",")]
    thetas = np.arange(0, 360, args.dtheta) + args.theta0
    out = []
    for zc in zc_all:
        zs = [zc + dz for dz in zoffs if args.z[0] <= zc + dz < args.z[1]]
        if zs:
            out.append((zc, zs))
    return out, thetas


def radial_extent(zs, thetas, src: M7Chunks, umb) -> dict:
    """Per (z, theta): (r_lo, r_hi) of the m7 mask along the ray, from level 3 (8x) of the prediction."""
    import zarr
    zmin, zmax = min(zs) // 8, max(zs) // 8 + 1
    f = CACHE / f"l3_{zmin}_{zmax}.npy"  # from the bucket m7 level 3; reused for other masks so the rays stay identical
    if f.exists():
        v = np.load(f)
    else:
        so = {"storage_options": {"anon": True}} if "://" in str(M7_URL) else {}  # p1667: local mask, level 3 = 8x
        a3 = zarr.open_group(str(M7_URL), mode="r", **so)["3"]
        v = np.asarray(a3[zmin:zmax + 1]) > 0
        np.save(f, v)
    ext = {}
    r = np.arange(0, 4000, 4.0)
    for z in zs:
        cx, cy = umb(z)
        sl = v[min(int(round(z / 8)) - zmin, v.shape[0] - 1)]
        sl = ndi.binary_dilation(sl, iterations=1)
        for th in thetas:
            x = (cx + r * np.cos(np.radians(th))) / 8
            y = (cy + r * np.sin(np.radians(th))) / 8
            ok = (x >= 0) & (y >= 0) & (x < sl.shape[1] - 0.5) & (y < sl.shape[0] - 0.5)
            on = np.zeros(len(r), bool)
            on[ok] = sl[np.rint(y[ok]).astype(int), np.rint(x[ok]).astype(int)]
            ext[(int(z), float(th))] = (float(r[on][0]), float(r[on][-1])) if on.any() else None
    return ext


def plan_band(args, src, umb):
    pl, thetas = probes(args)
    allz = [z for _, zs in pl for z in zs]
    ext = radial_extent(allz, thetas, src, umb)
    crops = []
    for zc, zs in pl:
        for th in thetas:
            e = [ext[(z, float(th))] for z in zs if ext[(z, float(th))]]
            if not e:
                continue
            r_lo = max(min(x[0] for x in e), args.rmin)
            r_hi = max(x[1] for x in e)
            r_k = r_lo + args.stride / 2
            k = 0
            while r_k - args.stride / 2 < r_hi:
                crops.append(dict(zc=zc, zs=zs, theta=float(th), k=k, rk=float(r_k)))
                r_k += args.stride
                k += 1
    return crops


def _band_crop(job):
    """Solve one crop and return, per ray z, its crossings (global radius) with patch windings, plus the step
    features for every consecutive pair inside the crop."""
    c, args_d = job
    if args_d.get("scroll", "p0139") != SCROLL:  # spawned workers start as p0139
        set_scroll(args_d["scroll"])
    src = M7Chunks(url=args_d.get("url", M7_URL), cache=Path(args_d["cache"]))
    umb = umbilicus_9p362()
    th = np.radians(c["theta"])
    ur = np.array([np.sin(th), np.cos(th)])  # (y, x)
    cx, cy = umb(c["zc"])
    cen = np.array([c["zc"], cy + c["rk"] * ur[0], cx + c["rk"] * ur[1]])
    o = np.rint(cen - CROP / 2).astype(int)
    if (o[1:] + CROP < 0).any() or (o[1:] > np.array(src.shape[1:])).any():
        return c, None
    raw = src.crop(tuple(o))
    sol = solve_crop(raw)
    if sol is None:
        return c, {}
    res = {}
    half = args_d["seg"] / 2
    for z in c["zs"]:
        cxz, cyz = umb(z)
        p0 = np.array([z, cyz, cxz], float) - o  # local position of the ray origin (r = 0)
        u = np.array([0.0, ur[0], ur[1]])
        tlo, thi = c["rk"] - half, c["rk"] + half
        for a in (1, 2):
            if abs(u[a]) > 1e-9:
                ta, tb = (CORE - p0[a]) / u[a], (CROP - CORE - 1 - p0[a]) / u[a]
                tlo, thi = max(tlo, min(ta, tb)), min(thi, max(ta, tb))
            elif not (CORE <= p0[a] < CROP - CORE):
                tlo, thi = 1, 0
        if not (CORE <= p0[0] < CROP - CORE) or thi - tlo < 20:
            continue
        cr = ray_crossings(sol, p0, u, tlo, thi)
        if cr is None:
            continue
        for x in cr:
            x["gpos"] = (x.pop("pos") + o).tolist()
        res[z] = dict(cr=cr, tc=(tlo + thi) / 2, tlo=tlo, thi=thi)
    return c, res


def band(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = M7Chunks(url=args.mask_url, cache=Path(args.cache))
    src.arr()
    umb = umbilicus_9p362()
    write_umbilicus_9p362(out / "umbilicus_9p362.json")
    t0 = time.time()
    crops = plan_band(args, src, umb)
    if args.limit:
        crops = crops[: args.limit]
    print(f"{len(crops)} crops planned ({time.time() - t0:.0f}s)", flush=True)
    # prefetch chunks (I/O bound, threads), then solve crops (CPU bound, processes)
    need = set()
    for c in crops:
        th = np.radians(c["theta"])
        cx, cy = umb(c["zc"])
        o = np.rint(np.array([c["zc"], cy + c["rk"] * np.sin(th), cx + c["rk"] * np.cos(th)]) - CROP / 2).astype(int)
        c["o"] = o.tolist()
        need.update(src.chunks_of(tuple(o)))
    need = sorted(need)
    with ThreadPoolExecutor(args.io_threads) as ex:
        for k, _ in enumerate(ex.map(lambda ijk: src.chunk(*ijk) is None, need)):
            if k % 200 == 0:
                print(f"chunks {k}/{len(need)} ({time.time() - t0:.0f}s)", flush=True)
    t_fetch = time.time() - t0
    print(f"{len(need)} chunks cached ({t_fetch:.0f}s)", flush=True)
    args_d = dict(cache=str(args.cache), seg=args.seg, url=str(args.mask_url))
    if SCROLL != "p0139":
        args_d["scroll"] = SCROLL
    import pickle
    from concurrent.futures.process import BrokenProcessPool
    cdir = out / "crops"
    cdir.mkdir(exist_ok=True)
    key = lambda c: cdir / f"{c['zc']}_{c['theta']:g}_{c['k']}.pkl"
    for attempt in range(6):  # per-crop checkpoints; a killed worker (memory pressure) only costs a pool restart
        todo = sorted([c for c in crops if not key(c).exists()],  # coarse angles first: a usable subset early
                      key=lambda c: (round(c["theta"] / args.dtheta) % 2, c["zc"], c["theta"], c["k"]))
        if not todo:
            break
        print(f"attempt {attempt}: {len(todo)} crops to solve", flush=True)
        try:
            with ProcessPoolExecutor(args.workers) as ex:
                for k, (c, r) in enumerate(ex.map(_band_crop, [(c, args_d) for c in todo], chunksize=1)):
                    tmp = key(c).with_suffix(".tmp")
                    with open(tmp, "wb") as fh:  # fsync: unsynced files came back empty after a hard reboot
                        fh.write(pickle.dumps((c, r)))
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, key(c))
                    if k % 100 == 0:
                        print(f"crops {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        except BrokenProcessPool as e:
            print(f"pool broken ({e}); restarting", flush=True)
    results = [pickle.loads(key(c).read_bytes()) for c in crops if key(c).exists()]
    t_solve = time.time() - t0 - t_fetch
    assemble(args, results, out, timing=dict(fetch_s=round(t_fetch), solve_s=round(t_solve), crops=len(crops),
                                             chunks=len(need), workers=args.workers))


def assemble(args, results, out: Path, timing=None) -> None:
    """Per ray: dedupe crossings across crops (nearest segment centre), take each step from the most central crop holding both ends,
    check a second crop, score with the confidence model and split into confident chains."""
    conf = json.loads(Path(getattr(args, "conf_json", CONF_JSON)).read_text())
    model, thr = conf["model"], args.conf if args.conf is not None else conf["thr"]
    rays = {}
    for c, res in results:
        if not res:
            continue
        for z, seg in res.items():
            rays.setdefault((z, c["theta"]), []).append((seg["tc"], seg["cr"]))
    chains, steps_out = [], []
    n_cand = n_keep = n_disagree = n_nocrop = n_bridge = 0
    for (z, th), lst in sorted(rays.items()):
        lst.sort(key=lambda x: x[0])
        # one crossing per physical crossing: cluster by radius (<= 2 vox), keep the copy from the crop whose ray
        # segment centre is nearest
        allx = sorted([(x["t"], abs(x["t"] - tc), x) for tc, cr in lst for x in cr], key=lambda v: v[0])
        own, cl = [], []
        for v in allx + [(1e18, 0, None)]:
            if cl and v[0] - cl[-1][0] > 2:
                own.append(min(cl, key=lambda q: q[1])[2])
                cl = []
            cl.append(v)

        def pair(a, b):
            hold = []
            for tc, cr in lst:
                ia = [x for x in cr if abs(x["t"] - a["t"]) <= 2 and "w" in x]
                ib = [x for x in cr if abs(x["t"] - b["t"]) <= 2 and "w" in x]
                if ia and ib:
                    hold.append((abs((a["t"] + b["t"]) / 2 - tc), ia[0], ib[0]))
            hold.sort(key=lambda h: h[0])
            return hold
        seq = []
        for x in own:
            if seq:
                h = pair(seq[-1], x)
                if h and round(h[0][2]["w"] - h[0][1]["w"]) == 0:
                    continue
            seq.append(x)
        st = []
        for a, b in zip(seq[:-1], seq[1:]):
            h = pair(a, b)
            if not h:
                st.append(None)
                n_nocrop += 1
                continue
            f = step_feats(h[0][1], h[0][2])
            f["agree"] = None if len(h) < 2 else int(round(h[1][2]["w"] - h[1][1]["w"]) == f["dr"])
            st.append(f)
        pe = [None if (s is None or s["dr"] < 1 or s["dr"] > args.max_step) else float(p_err(model, [s])[0]) for s in st]
        # --bridge (long chains): a step with no common crop, a negative step or one above --max-step counts +1
        # (consecutive distinct crossings on a radial ray are one winding apart unless a sheet was missed)
        br = [bool(getattr(args, "bridge", 0)) and p is None for p in pe]
        good = []
        for s, p, b in zip(st, pe, br):
            g = p is not None and p <= thr
            if g:
                n_cand += 1
            if g and args.require_agree and s["agree"] == 0:
                g = False
                n_disagree += 1
            g = g or b
            n_bridge += int(b)
            good.append(g)
            if g:
                n_keep += 1
        for k, (s, p, g) in enumerate(zip(st, pe, good)):
            steps_out.append(dict(z=z, theta=th, k=k, r_a=seq[k]["t"], r_b=seq[k + 1]["t"],
                                  dr=None if s is None else s["dr"], p_err=p, agree=None if s is None else s["agree"],
                                  kept=int(g), bridged=int(br[k])))
        # split into runs of good steps
        cur = None
        for k, g in enumerate(good + [False]):
            if g:
                if cur is None:
                    cur = dict(z=z, theta=th, pts=[seq[k]["gpos"]], wind=[0], r=[seq[k]["t"]], perr=[],
                               rl=[seq[k].get("runlen")])
                cur["pts"].append(seq[k + 1]["gpos"])
                cur["rl"].append(seq[k + 1].get("runlen"))
                cur["wind"].append(cur["wind"][-1] + (1 if br[k] else st[k]["dr"]))
                cur["r"].append(seq[k + 1]["t"])
                cur["perr"].append(1.0 if br[k] else pe[k])
            elif cur is not None:
                chains.append(cur)
                cur = None
    chains = split_max(chains, args.max_len)
    write_json(chains, out / "relative_windings.json", prefix="auto")
    with (out / "steps.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(steps_out[0].keys()) if steps_out else ["z"])
        wr.writeheader()
        wr.writerows(steps_out)
    summ = dict(rays=len(rays), chains=len(chains), points=sum(len(c["pts"]) for c in chains),
                median_chain_points=float(np.median([len(c["pts"]) for c in chains])) if chains else 0,
                max_chain_points=max((len(c["pts"]) for c in chains), default=0),
                steps_total=len(steps_out), steps_confident=n_cand, steps_kept=n_keep,
                steps_dropped_crop_disagree=n_disagree, steps_no_common_crop=n_nocrop, steps_bridged=n_bridge, thr=thr,
                median_winding_span=float(np.median([c["wind"][-1] - c["wind"][0] for c in chains])) if chains else 0,
                require_agree=bool(args.require_agree), max_len=args.max_len, timing=timing,
                conf_json=str(getattr(args, "conf_json", CONF_JSON)))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "chains.json").write_text(json.dumps(chains))
    print(json.dumps(summ, indent=1))


def split_max(chains, max_len: int):
    if not max_len:
        return chains
    out = []
    for c in chains:
        n = len(c["pts"])
        if n <= max_len:
            out.append(c)
            continue
        s = 0
        while s < n - 1:
            e = min(s + max_len, n)
            w0 = c["wind"][s]
            out.append(dict(c, pts=c["pts"][s:e], wind=[w - w0 for w in c["wind"][s:e]], r=c["r"][s:e],
                            perr=c.get("perr", [])[s:e - 1], **({"rl": c["rl"][s:e]} if "rl" in c else {})))
            s = e - 1  # share the boundary point
    return out


def write_json(chains, path: Path, prefix: str) -> None:
    cols, pid = {}, 1
    for ci, c in enumerate(chains, 1):
        pts = {}
        for p, w in zip(c["pts"], c["wind"]):
            z, y, x = p
            pts[str(pid)] = {"p": [round(float(x), 2), round(float(y), 2), round(float(z), 2)], "wind_a": int(w),
                             "creation_time": 0}
            pid += 1
        cols[str(ci)] = {"name": f"{prefix}_z{int(c['z'])}_t{c['theta']:g}_{ci}", "points": pts,
                         "metadata": {}, "color": [0.0, 0.0, 0.0]}
    path.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": cols}))


# ------------------------------------------------------------------------------------------------- oracle
def mesh_ray_crossings(meshes_z: list, z0: float, cxy, thetas) -> dict:
    """Crossings of the plane z = z0 polylines of each segment with rays from (cx, cy): {theta: [(r, w, (z,y,x))]}.
    meshes_z: [(w, tri_a, tri_b, tri_c)] triangles (n, 3) zyx already filtered to straddle z0."""
    cx, cy = cxy
    out = {float(th): [] for th in thetas}
    for w, P0, P1, P2 in meshes_z:
        # intersect each triangle with the plane z = z0 -> segment (q1, q2) in xy
        V = np.stack([P0, P1, P2], 1)  # (n, 3, 3)
        dz = V[..., 0] - z0
        segs = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            s = (dz[:, a] * dz[:, b]) < 0
            tt = np.where(s, dz[:, a] / np.where(s, dz[:, a] - dz[:, b], 1), np.nan)
            segs.append(V[:, a, 1:] + tt[:, None] * (V[:, b, 1:] - V[:, a, 1:]))
        S = np.stack(segs, 1)  # (n, 3, 2) yx, two finite per crossing triangle
        fin = np.isfinite(S[..., 0])
        keep = fin.sum(1) == 2
        S, fin = S[keep], fin[keep]
        if len(S) == 0:
            continue
        o = np.argsort(~fin, axis=1, kind="stable")[:, :2]
        Q1 = S[np.arange(len(S)), o[:, 0]]
        Q2 = S[np.arange(len(S)), o[:, 1]]
        for th in thetas:
            u = np.array([np.sin(np.radians(th)), np.cos(np.radians(th))])
            v = np.array([-u[1], u[0]])
            c = np.array([cy, cx])
            l1, l2 = (Q1 - c) @ v, (Q2 - c) @ v
            hit = (l1 * l2) <= 0
            if not hit.any():
                continue
            a = l1[hit] / np.where(l1[hit] - l2[hit] == 0, 1, l1[hit] - l2[hit])
            X = Q1[hit] + a[:, None] * (Q2[hit] - Q1[hit])
            r = (X - c) @ u
            X, r = X[r > 0], r[r > 0]
            if len(r) == 0:
                continue
            # one crossing per cluster of radii (a segment may cross a ray more than once)
            o2 = np.argsort(r)
            r, X = r[o2], X[o2]
            br = np.flatnonzero(np.diff(r) > 5) + 1
            for g in np.split(np.arange(len(r)), br):
                out[float(th)].append((float(np.median(r[g])), w, (z0, float(np.median(X[g, 0])), float(np.median(X[g, 1])))))
    return out


def mesh_tris(meshes: dict):
    tris = []
    for w, lst in meshes.items():
        for name, p in lst:
            a, b, c, d = p[:-1, :-1], p[1:, :-1], p[:-1, 1:], p[1:, 1:]
            ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & np.isfinite(c).all(-1) & np.isfinite(d).all(-1)
            A, B, C, D = a[ok], b[ok], c[ok], d[ok]
            tris.append((w, np.r_[A, B], np.r_[B, D], np.r_[C, C]))
    return tris


def oracle_rays(args) -> dict:
    """{(z, theta): [(r, w, zyx)]} from the human meshes for the band probes (same grid as band mode)."""
    umb = umbilicus_9p362()
    pl, thetas = probes(args)
    zs = sorted({z for _, zz in pl for z in zz})
    meshes = load_meshes9(min(zs), max(zs))
    tris = mesh_tris(meshes)
    zmin = [(w, np.minimum(np.minimum(P0[:, 0], P1[:, 0]), P2[:, 0]), np.maximum(np.maximum(P0[:, 0], P1[:, 0]), P2[:, 0]))
            for w, P0, P1, P2 in tris]
    rays = {}
    for z0 in zs:
        mz = []
        for (w, P0, P1, P2), (_, lo, hi) in zip(tris, zmin):
            s = (lo <= z0) & (hi >= z0)
            if s.any():
                mz.append((w, P0[s], P1[s], P2[s]))
        cr = mesh_ray_crossings(mz, z0, umb(z0), thetas)
        for th, lst in cr.items():
            lst.sort()
            # drop coincident crossings of different windings (label conflicts, e.g. w041/w042, w045/w046)
            bad = set()
            for i in range(len(lst) - 1):
                if lst[i + 1][0] - lst[i][0] < 1.5 and lst[i + 1][1] != lst[i][1]:
                    bad |= {i, i + 1}
            rays[(int(z0), th)] = [x for i, x in enumerate(lst) if i not in bad]
    return rays


def oracle(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rays = oracle_rays(args)
    chains, nflip = [], 0
    for (z, th), lst in sorted(rays.items()):
        cur = None
        for i, (r, w, p) in enumerate(lst):
            if cur is not None and w - cur["w_abs"][-1] >= 1:
                cur["pts"].append(p)
                cur["w_abs"].append(w)
                cur["r"].append(r)
                continue
            if cur is not None:
                nflip += int(w - cur["w_abs"][-1] < 1)
                if len(cur["pts"]) >= 2:
                    chains.append(cur)
            cur = dict(z=z, theta=th, pts=[p], w_abs=[w], r=[r])
        if cur is not None and len(cur["pts"]) >= 2:
            chains.append(cur)
    for c in chains:
        c["wind"] = [w - c["w_abs"][0] for w in c["w_abs"]]
    chains = split_max(chains, args.max_len)
    write_json(chains, out / "relative_windings.json", prefix="oracle")
    (out / "chains.json").write_text(json.dumps(chains))
    (out / "rays.json").write_text(json.dumps({f"{z}|{th}": v for (z, th), v in rays.items()}))
    summ = dict(rays=len(rays), chains=len(chains), points=sum(len(c["pts"]) for c in chains),
                breaks_nonincreasing=nflip, seconds=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def corrupt(args) -> None:
    """Negative control: flip a fraction of the steps of every chain by +-1 (the later points shift with it)."""
    d = json.loads(Path(args.src).read_text())
    rng = np.random.default_rng(args.seed)
    n = nf = 0
    for col in d["collections"].values():
        items = sorted(col["points"].items(), key=lambda kv: int(kv[0]))
        w = [p["wind_a"] for _, p in items]
        steps = np.diff(w)
        flip = rng.random(len(steps)) < args.frac
        steps = steps + flip * rng.choice([-1, 1], len(steps))
        n += len(steps)
        nf += int(flip.sum())
        neww = np.r_[0, np.cumsum(steps)]
        neww -= neww.min()  # keep wind_a >= 0
        for (_, p), v in zip(items, neww):
            p["wind_a"] = int(v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "relative_windings.json").write_text(json.dumps(d))
    src_chains = Path(args.src).with_name("chains.json")
    if src_chains.exists():  # same order as the collections (write_json enumerates chains)
        ch = json.loads(src_chains.read_text())
        for c, (_, col) in zip(ch, sorted(d["collections"].items(), key=lambda kv: int(kv[0]))):
            c["wind"] = [p["wind_a"] for _, p in sorted(col["points"].items(), key=lambda kv: int(kv[0]))]
        (out / "chains.json").write_text(json.dumps(ch))
    s = dict(src=str(args.src), frac=args.frac, seed=args.seed, steps=n, flipped=nf)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(s)


# ------------------------------------------------------------------------------------------------- compare / check
# ------------------------------------------------------------------------------------------------- grad_mag fix
GM_ZARR = REPO / "data" / "p0139" / "inputs" / "PHerc0139_grad_mag.ome.zarr"  # lasagna density, 9.362 frame, 4 vox
GM_ZARR_EXTRA = [CACHE / "lasagna_z4500" / "PHerc0139_grad_mag.ome.zarr"]  # prep_inputs.py --z0 4500 --z1 6000
GM_FEATURES = ["I", "Ir", "Lr", "logL", "Iloc", "logperr", "bridged"]
GM_JSON = Path(__file__).with_name("auto_windings_gm.json")


def gm_volume(z0=2990, z1=4510):  # cached per z range
    import zarr
    a, b = z0 // 4, z1 // 4 + 1
    f = CACHE / f"gm_band_z{a}_{b}.npy"
    if not f.exists():
        v = np.asarray(zarr.open_group(str(GM_ZARR), mode="r")["2"][a:b])
        for extra in GM_ZARR_EXTRA:  # rows absent from the primary store (written by the same prep_inputs.py resample)
            empty = np.flatnonzero(~v.reshape(len(v), -1).any(1))
            if len(empty) and extra.exists():
                e = zarr.open_group(str(extra), mode="r")["2"]
                for i in empty:
                    v[i] = np.asarray(e[a + i])
        np.save(f, v)
    return np.load(f, mmap_mode="r"), a


def _zrange(ch):
    z = [p[0] for c in ch for p in c["pts"]]
    return int(min(z)) - 10, int(max(z)) + 10


def gm_integral(G, za, pa, pb):
    """Windings crossed between two points: line integral of the lasagna density (u8 / 1000 per voxel, the
    fitter's encoding with factor 1), nearest sample on the 4-vox grid, 1-vox steps. Returns (integral, covered)."""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    L = float(np.linalg.norm(pb - pa))
    n = max(int(np.ceil(L)), 1)
    q = pa[None] + ((np.arange(n) + 0.5) / n)[:, None] * (pb - pa)[None]
    i = np.rint(q / 4).astype(int)
    i[:, 0] -= za
    ok = ((i >= 0) & (i < np.array(G.shape))).all(1)
    v = np.zeros(n)
    v[ok] = G[i[ok, 0], i[ok, 1], i[ok, 2]]
    return v.sum() / 1000 * L / n, float((v > 0).mean()), L


def gm_chain_features(c, G, za, W=5):
    """Per step of one chain: GM_FEATURES rows (+ coverage). Local ratios use the +-W neighbouring single steps."""
    n = len(c["pts"]) - 1
    I, cov, L = map(np.array, zip(*[gm_integral(G, za, c["pts"][k], c["pts"][k + 1]) for k in range(n)])) if n else ([], [], [])
    d = np.diff(c["wind"])
    pe = np.array(c.get("perr") or [1.0] * n, float)
    rows = []
    for j in range(n):
        nb = [i for i in range(max(0, j - W), min(n, j + W + 1)) if i != j and d[i] == 1]
        il = float(np.median(I[nb])) if nb else 1.2
        if not il > 0:
            il = 1.2
        rows.append(dict(I=I[j], Ir=I[j] / il, Lr=L[j] / np.median(L[nb]) if nb else 1.0,
                         logL=np.log(max(L[j], 1e-3)), Iloc=il, logperr=np.log(np.clip(pe[j], 1e-4, 1)),
                         bridged=float(pe[j] >= 1), cov=cov[j]))
    return rows


def _oracle_tw(c, rays, tol=4.0):
    orc = rays.get((int(c["z"]), float(c["theta"])), [])
    rr = np.array([x[0] for x in orc]) if orc else np.zeros(0)
    tw = []
    for r in c["r"]:
        if len(rr) == 0:
            tw.append(None)
            continue
        dd = np.abs(rr - r)
        i = int(np.argmin(dd))
        s = np.partition(dd, 1)[1] if len(dd) > 1 else 1e9
        tw.append(orc[i][1] if dd[i] <= tol and s - dd[i] >= 1 else None)
    return tw


def gmfit(args) -> None:
    """Fit P(true step = 2 | chain step = 1, grad_mag features) on oracle-matched steps of the rays whose angle index
    has parity --half (the other half stays held out); the threshold makes the mean signed long-range error zero on the
    fitting half."""
    ch = json.loads((Path(args.src) / "chains.json").read_text())
    G, za = gm_volume(*_zrange(ch))
    rays = {(int(k.split("|")[0]), float(k.split("|")[1])): v
            for k, v in json.loads((Path(args.oracle) / "rays.json").read_text()).items()}
    X, Y, fit_ch = [], [], []
    for c in ch:
        if args.half >= 0 and int(round(c["theta"] / args.dtheta)) % 2 != args.half:
            continue
        f = gm_chain_features(c, G, za)
        tw = _oracle_tw(c, rays)
        fit_ch.append((c, f, tw))
        for k, r in enumerate(f):
            if c["wind"][k + 1] - c["wind"][k] != 1 or tw[k] is None or tw[k + 1] is None:
                continue
            t = tw[k + 1] - tw[k]
            if t in (1, 2):
                X.append([r[g] for g in GM_FEATURES])
                Y.append(float(t == 2))
    m = fit_logistic(np.array(X), np.array(Y))
    m["features"] = GM_FEATURES

    def drift(thr, sel=None):
        e = []
        for c, f, tw in (sel if sel is not None else fit_ch):
            P = p_err(m, f) if f else np.zeros(0)
            d = np.diff(c["wind"]) + ((np.diff(c["wind"]) == 1) & (P >= thr))
            w = np.r_[0, np.cumsum(d)]
            ok = [k for k in range(len(tw)) if tw[k] is not None]
            for a in range(len(ok)):
                for b in range(a + 1, len(ok)):
                    t = tw[ok[b]] - tw[ok[a]]
                    if 10 <= t <= 20:
                        e.append(w[ok[b]] - w[ok[a]] - t)
        return float(np.mean(e)) if e else 0.0
    grid = np.round(np.arange(0.05, 0.6, 0.01), 2)
    dr = {float(t): drift(t) for t in grid}
    thr0 = min(dr, key=lambda t: abs(dr[t]))
    # robustness: zero-drift threshold per z slab (same model)
    slabs = {}
    zs = sorted({int(c["z"]) for c, _, _ in fit_ch})
    for s0 in range(zs[0] - zs[0] % args.slab, zs[-1] + 1, args.slab):
        sel = [x for x in fit_ch if s0 <= x[0]["z"] < s0 + args.slab]
        if sel:
            d_s = {float(t): drift(t, sel) for t in grid}
            slabs[f"{s0}-{s0 + args.slab}"] = dict(thr_zero_drift=min(d_s, key=lambda t: abs(d_s[t])),
                                                    drift_at_band_thr=round(d_s[thr0], 3), chains=len(sel))
    Path(args.model).write_text(json.dumps(dict(model=m, thr_zero_drift=thr0, drift_by_thr=dr, fitted_on=str(args.src),
                                                oracle=str(args.oracle), half=args.half, dtheta=args.dtheta, n=len(Y),
                                                pos=int(sum(Y)), slabs=slabs), indent=1))
    print(json.dumps(dict(n=len(Y), pos=int(sum(Y)), thr_zero_drift=thr0, drift_at_0p2=dr.get(0.2), slabs=slabs,
                          beta=[round(b, 3) for b in m["beta"]]), indent=1))


def gmfix(args) -> None:
    """Apply the grad_mag rule: a chain step of 1 becomes 2 where P >= --thr (default: zero-drift threshold)."""
    conf = json.loads(Path(args.model).read_text())
    m, thr = conf["model"], args.thr if args.thr is not None else conf["thr_zero_drift"]
    ch = json.loads((Path(args.src) / "chains.json").read_text())
    G, za = gm_volume(*_zrange(ch))
    n_up = n1 = 0
    for c in ch:
        f = gm_chain_features(c, G, za)
        if not f:
            continue
        P = p_err(m, f)
        d = np.diff(c["wind"])
        up = (d == 1) & (P >= thr)
        n_up += int(up.sum())
        n1 += int((d == 1).sum())
        c["wind"] = np.r_[0, np.cumsum(d + up)].astype(int).tolist()
        c["gm_p"] = [round(float(x), 4) for x in P]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(ch, out / "relative_windings.json", prefix="autogm")
    (out / "chains.json").write_text(json.dumps(ch))
    s = dict(src=str(args.src), thr=thr, steps_raised_1_to_2=n_up, single_steps=n1, chains=len(ch),
             points=sum(len(c["pts"]) for c in ch), median_chain_points=float(np.median([len(c["pts"]) for c in ch])),
             median_winding_span=float(np.median([c["wind"][-1] for c in ch])), model=str(args.model))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def relabel(args) -> None:
    """Oracle labels on another chain layout: keep each chain's points and splits, replace its windings by the human
    mesh winding of the crossing within --tol on the same ray (unique by >= 1 vox), drop unmatched points, drop a point
    repeating the previous winding, split where the mesh winding decreases."""
    src = json.loads((Path(args.src) / "chains.json").read_text())
    rays = {(int(k.split("|")[0]), float(k.split("|")[1])): v
            for k, v in json.loads((Path(args.oracle) / "rays.json").read_text()).items()}
    out_ch, n_pts, n_match, n_dup, n_split = [], 0, 0, 0, 0
    for c in src:
        orc = rays.get((int(c["z"]), float(c["theta"])), [])
        rr = np.array([x[0] for x in orc]) if orc else np.zeros(0)
        cur = None
        pieces = []
        for p_, r in zip(c["pts"], c["r"]):
            n_pts += 1
            if len(rr) == 0:
                continue
            d = np.abs(rr - r)
            i = int(np.argmin(d))
            second = np.partition(d, 1)[1] if len(d) > 1 else 1e9
            if not (d[i] <= args.tol and second - d[i] >= 1.0):
                continue
            n_match += 1
            w = int(orc[i][1])
            if cur and w == cur["w_abs"][-1]:
                n_dup += 1
                continue
            if cur and w < cur["w_abs"][-1]:
                n_split += 1
                pieces.append(cur)
                cur = None
            if cur is None:
                cur = dict(z=c["z"], theta=c["theta"], pts=[], w_abs=[], r=[])
            cur["pts"].append(p_)
            cur["w_abs"].append(w)
            cur["r"].append(r)
        if cur:
            pieces.append(cur)
        for q in pieces:
            if len(q["pts"]) >= 2:
                q["wind"] = [w - q["w_abs"][0] for w in q["w_abs"]]
                out_ch.append(q)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out_ch, out / "relative_windings.json", prefix="orc_on")
    (out / "chains.json").write_text(json.dumps(out_ch))
    s = dict(src=str(args.src), oracle=str(args.oracle), tol=args.tol, src_chains=len(src), src_points=n_pts,
             points_matched=n_match, dropped_duplicate=n_dup, splits_decreasing=n_split, chains=len(out_ch),
             points=sum(len(c["pts"]) for c in out_ch),
             median_chain_points=float(np.median([len(c["pts"]) for c in out_ch])) if out_ch else 0,
             median_winding_span=float(np.median([c["wind"][-1] for c in out_ch])) if out_ch else 0)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def split(args) -> None:
    """Cut every chain into consecutive pieces of --n points (consecutive pieces share their boundary point, so every
    step is kept; only the long-range link is lost)."""
    src = json.loads((Path(args.src) / "chains.json").read_text())
    for c in src:
        c.setdefault("perr", [])
    out_ch = split_max(src, args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out_ch, out / "relative_windings.json", prefix=f"len{args.n}")
    (out / "chains.json").write_text(json.dumps(out_ch))
    s = dict(src=str(args.src), n=args.n, src_chains=len(src), chains=len(out_ch), points=sum(len(c["pts"]) for c in out_ch),
             median_chain_points=float(np.median([len(c["pts"]) for c in out_ch])),
             median_winding_span=float(np.median([c["wind"][-1] - c["wind"][0] for c in out_ch])))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def compare(args) -> None:
    """End-to-end: every consecutive step of the auto chains whose two points match an oracle (mesh) crossing on the
    same ray within --tol vox; correct if the wind_a difference equals the mesh winding difference."""
    auto = json.loads((Path(args.auto) / "chains.json").read_text())
    rays = {tuple([int(k.split("|")[0]), float(k.split("|")[1])]): v
            for k, v in json.loads((Path(args.oracle) / "rays.json").read_text()).items()}
    sec = json.loads((CACHE / "sector_2167.json").read_text()) if (CACHE / "sector_2167.json").exists() else None
    tot = {k: [0, 0] for k in ("all", "in_2167", "out_2167", "clean", "touch_w033_w037", "touch_blacklist",
                               "clean_out_w033_w037")}
    BL = {41, 42, 45, 46} if SCROLL == "p0139" else set()  # p0139 coincident segment pairs (notes/2026-10-18-p0139-testset.md)
    kmax = getattr(args, "kmax", None)  # optional inner subset: both oracle windings < kmax
    if kmax is not None:
        tot[f"inner_k_lt_{kmax}"] = [0, 0]
        lag_in = {k: [0, 0] for k in ("1-2", "3-5", "6-10", "11-20", "21-999")}
        drift_in = {"3-6": [], "10-20": []}
    matched_pts = all_pts = 0
    by_dr = {}
    LAGS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
    lag = {f"{a}-{b}": [0, 0] for a, b in LAGS}  # all matched point pairs in a chain, by true winding distance
    err = {"under_contact": 0, "under_spaced": 0, "under_label_gap": 0, "over_duplicate": 0, "over_other": 0}
    err_rays, err_theta, n_theta = {}, {}, {}
    for c in auto:
        orc = rays.get((int(c["z"]), float(c["theta"])), [])
        rr = np.array([x[0] for x in orc]) if orc else np.zeros(0)
        tw = []
        for r in c["r"]:
            all_pts += 1
            if len(rr) == 0:
                tw.append(None)
                continue
            d = np.abs(rr - r)
            i = int(np.argmin(d))
            second = np.partition(d, 1)[1] if len(d) > 1 else 1e9
            if d[i] <= args.tol and second - d[i] >= 1.0:
                tw.append(orc[i][1])
                matched_pts += 1
            else:
                tw.append(None)
        ins = in_sector(c["theta"], c["z"], sec)
        mi = [k for k in range(len(c["r"])) if tw[k] is not None]
        for ia in range(len(mi)):
            for ib in range(ia + 1, len(mi)):
                a, b = mi[ia], mi[ib]
                tdw = abs(tw[b] - tw[a])
                for lo_, hi_ in LAGS:
                    if lo_ <= tdw <= hi_:
                        lag[f"{lo_}-{hi_}"][0] += int(c["wind"][b] - c["wind"][a] == tw[b] - tw[a])
                        lag[f"{lo_}-{hi_}"][1] += 1
                        if kmax is not None and max(tw[a], tw[b]) < kmax:
                            lag_in[f"{lo_}-{hi_}"][0] += int(c["wind"][b] - c["wind"][a] == tw[b] - tw[a])
                            lag_in[f"{lo_}-{hi_}"][1] += 1
                if kmax is not None and max(tw[a], tw[b]) < kmax:
                    t_ = tw[b] - tw[a]  # drift as gmfit / band2_diag: chain minus true winding distance
                    for key_, (lo_, hi_) in (("3-6", (3, 6)), ("10-20", (10, 20))):
                        if lo_ <= t_ <= hi_:
                            drift_in[key_].append(c["wind"][b] - c["wind"][a] - t_)
        for k in range(len(c["r"]) - 1):
            if tw[k] is None or tw[k + 1] is None:
                continue
            dr = c["wind"][k + 1] - c["wind"][k]
            ok = int(dr == tw[k + 1] - tw[k])
            n_theta[c["theta"]] = n_theta.get(c["theta"], 0) + 1
            if not ok:
                ra, rb = c["r"][k], c["r"][k + 1]
                mid = sorted(x for x in rr if ra + 1 < x < rb - 1)
                if dr < tw[k + 1] - tw[k]:
                    if not mid:
                        err["under_label_gap"] += 1
                    else:
                        sp = np.diff([ra] + mid + [rb])
                        err["under_contact" if sp.min() <= 8 else "under_spaced"] += 1
                else:
                    err["over_duplicate" if tw[k + 1] == tw[k] else "over_other"] += 1
                err_rays[(c["z"], c["theta"])] = err_rays.get((c["z"], c["theta"]), 0) + 1
                err_theta[c["theta"]] = err_theta.get(c["theta"], 0) + 1
            lo, hi = min(tw[k], tw[k + 1]), max(tw[k], tw[k + 1])
            bl = any(lo <= w <= hi for w in BL)
            band = hi >= 33 and lo <= 37
            keys = ["all", "in_2167" if ins else "out_2167"]
            keys.append("touch_blacklist" if bl else "clean")
            if band:
                keys.append("touch_w033_w037")
            if not bl and not band:
                keys.append("clean_out_w033_w037")
            if kmax is not None and hi < kmax:
                keys.append(f"inner_k_lt_{kmax}")
            for key in keys:
                tot[key][0] += ok
                tot[key][1] += 1
            by_dr.setdefault(dr, [0, 0])
            by_dr[dr][0] += ok
            by_dr[dr][1] += 1
    summ = {k: dict(steps=v[1], precision=round(100 * v[0] / max(v[1], 1), 2)) for k, v in tot.items()}
    summ["by_dr"] = {int(k): dict(steps=v[1], precision=round(100 * v[0] / max(v[1], 1), 2)) for k, v in sorted(by_dr.items())}
    summ["points_matched"] = f"{matched_pts}/{all_pts}"
    summ["tol"] = args.tol
    summ["pair_precision_by_true_winding_distance"] = {k: dict(pairs=v[1], precision=round(100 * v[0] / max(v[1], 1), 2))
                                                       for k, v in lag.items()}
    if kmax is not None:
        summ[f"pair_precision_by_true_winding_distance_k_lt_{kmax}"] = {
            k: dict(pairs=v[1], precision=round(100 * v[0] / max(v[1], 1), 2)) for k, v in lag_in.items()}
        summ[f"drift_k_lt_{kmax}"] = {k: dict(pairs=len(v), mean=round(float(np.mean(v)), 3) if v else None,
                                              within_1=round(100 * float(np.mean(np.abs(v) <= 1)), 1) if v else None)
                                      for k, v in drift_in.items()}
    summ["errors"] = err
    ne = sum(err_rays.values())
    if ne:  # clustering: share of errors on the 10 % worst rays / precision by angle
        top = sorted(err_rays.values(), reverse=True)[:max(1, len(rays) // 10)]
        summ["errors_share_on_worst_10pct_rays"] = round(100 * sum(top) / ne, 1)
        summ["precision_by_theta"] = {f"{th:g}": round(100 * (1 - err_theta.get(th, 0) / n), 1) for th, n in sorted(n_theta.items())}
    summ["chains"] = len(auto)
    summ["median_chain_points"] = float(np.median([len(c["r"]) for c in auto])) if auto else 0
    summ["median_winding_span"] = float(np.median([c["wind"][-1] - c["wind"][0] for c in auto])) if auto else 0
    Path(args.auto, f"compare_{Path(args.oracle).name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def check(args) -> None:
    """Parse with the pinned fitter loader and replicate its unattached-strip preparation."""
    spiral = REPO / "data" / "ext" / "villa-0fb38c45" / "volume-cartographer" / "scripts" / "spiral"
    sys.path.insert(0, str(spiral))
    from point_collection import load_point_collection, normalise_pcl_winding_annotations

    def _decimate_ordered_points_min_spacing(points, min_spacing, return_indices=False, force_keep=None):
        # verbatim logic of spiral_helpers.py:89 (that module imports vc3d_fiber_format, not installed here)
        force_keep = set(force_keep or ())
        if min_spacing <= 0 or len(points) <= 1:
            keep = list(range(len(points)))
            return (points, np.asarray(keep, dtype=np.int64)) if return_indices else points
        keep = [0]
        last_kept = points[0]
        for i in range(1, len(points)):
            if i in force_keep or np.linalg.norm(points[i] - last_kept) >= min_spacing:
                keep.append(i)
                last_kept = points[i]
        return (points[keep], np.asarray(keep, dtype=np.int64)) if return_indices else points[keep]
    cols = load_point_collection(str(args.json))
    assert cols is not None, "loader returned None"
    for pcl in cols.values():
        for p in pcl["points"].values():
            p["zyx"] = np.array([p["p"][2], p["p"][1], p["p"][0]], dtype=np.float32)  # fit_spiral.py:1219
    normalise_pcl_winding_annotations(cols)
    n_pts = sum(len(c["points"]) for c in cols.values())
    bad_order = dropped16 = 0
    dth_max = 0.0
    umb = umbilicus_9p362()
    for c in cols.values():
        items = sorted(c["points"].items(), key=lambda kv: int(kv[0]))
        w = np.array([p["winding_annotation"] for _, p in items])
        zyx = np.stack([p["zyx"] for _, p in items])
        bad_order += int((np.diff(w) < 0).any())
        _, keep = _decimate_ordered_points_min_spacing(zyx, 16.0, return_indices=True, force_keep={len(zyx) - 1})
        dropped16 += len(zyx) - len(keep)
        cxy = umb(zyx[:, 0])
        th = np.unwrap(np.arctan2(zyx[:, 1] - cxy[:, 1], zyx[:, 2] - cxy[:, 0]))
        dth_max = max(dth_max, float(np.abs(np.diff(th)).max()) if len(th) > 1 else 0.0)
    s = dict(collections=len(cols), points=n_pts, all_annotated=all(c["has_winding_annotations"] for c in cols.values()),
             chains_with_decreasing_wind_a=bad_order, max_consecutive_dtheta_rad=round(dth_max, 4),
             points_dropped_by_default_min_spacing_16=dropped16, frac_dropped_16=round(dropped16 / max(n_pts, 1), 3))
    print(json.dumps(s, indent=1))


# ------------------------------------------------------------------------------------------------- CLI
def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)  # --scroll first: the CLI defaults below follow it
    pre.add_argument("--scroll", default="p0139", choices=["p0139", "p1667"])
    sc, rest = pre.parse_known_args()
    if sc.scroll != "p0139":  # default: no-op (leaves an importer's spiral_eval.set_scroll alone)
        set_scroll(sc.scroll)
    sys.argv = [sys.argv[0]] + rest
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def band_opts(p):
        p.add_argument("--z", type=float, nargs=2, default=[3000, 4500], help="z band (9.362 vox)")
        p.add_argument("--zoffs", default="-64,0,64", help="ray z offsets from each 192-chunk layer centre (|.| <= 72)")
        p.add_argument("--dtheta", type=float, default=15.0, help="probe angle spacing (deg)")
        p.add_argument("--theta0", type=float, default=0.0)
        p.add_argument("--max-len", type=int, default=0, help="max points per chain (0 = no cap; split chains share a point)")
        p.add_argument("--out", required=True)

    p = sub.add_parser("band", help="automatic annotations from m7 masks")
    band_opts(p)
    p.add_argument("--stride", type=float, default=96.0, help="crop spacing along the ray (vox)")
    p.add_argument("--seg", type=float, default=144.0, help="ray segment length read in each crop (vox, <= 144)")
    p.add_argument("--rmin", type=float, default=48.0, help="start of the rays (vox from the umbilicus)")
    p.add_argument("--conf", type=float, default=None, help="max step error probability (default: tools/auto_windings_conf.json thr)")
    p.add_argument("--require-agree", type=int, default=1, help="drop steps where a second crop disagrees")
    p.add_argument("--max-step", type=int, default=1, help="never emit steps larger than this (winding units)")
    p.add_argument("--bridge", type=int, default=0, help="long chains: count unscorable steps (no common crop, negative, > max-step) as +1 instead of cutting")
    p.add_argument("--cache", default=str(CACHE / "m7_L0"))
    p.add_argument("--mask-url", default=M7_URL, help="mask zarr group (bucket URL or local path; level 0 used); pair with its own --cache")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--io-threads", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reassemble", action="store_true", help="reuse solved crops (change --conf etc. only)")
    p.add_argument("--from-crops", default=None, help="crop checkpoints dir for --reassemble (default <out>/crops)")
    p.add_argument("--conf-json", default=str(CONF_JSON), help="step-confidence model json (default tools/auto_windings_conf.json)")

    p = sub.add_parser("oracle", help="the same chains from the human meshes (true winding indices)")
    band_opts(p)

    p = sub.add_parser("corrupt", help="negative control: flip a fraction of steps by +-1")
    p.add_argument("--src", required=True)
    p.add_argument("--frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("gmfit", help="fit the grad_mag step-raise rule on one half of the rays")
    p.add_argument("--src", required=True)
    p.add_argument("--oracle", default=str(CACHE / "band_oracle"))
    p.add_argument("--half", type=int, default=0, help="angle-index parity to fit on; -1 = all rays")
    p.add_argument("--dtheta", type=float, default=15.0)
    p.add_argument("--slab", type=int, default=500, help="z slab size for the per-slab zero-drift threshold")
    p.add_argument("--model", default=str(GM_JSON))

    p = sub.add_parser("gmfix", help="apply the grad_mag step-raise rule to a chain set")
    p.add_argument("--src", required=True)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--model", default=str(GM_JSON))
    p.add_argument("--out", required=True)

    p = sub.add_parser("relabel", help="oracle windings on another chain layout (same points and splits)")
    p.add_argument("--src", required=True)
    p.add_argument("--oracle", default=str(CACHE / "band_oracle"))
    p.add_argument("--tol", type=float, default=4.0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("split", help="cut chains into pieces of --n points")
    p.add_argument("--src", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", required=True)

    p = sub.add_parser("validate", help="step precision vs human windings on the 87 test-site crops")
    p.add_argument("--seg", default=str(REPO / "data" / "preds" / "m7_seg"))
    p.add_argument("--out", required=True)
    p.add_argument("--zoffs", default="36,66,96,126,156")
    p.add_argument("--loffs", default="-60,-30,0,30,60")
    p.add_argument("--target", type=float, default=98.0, help="out-of-fold step precision target for the threshold")
    p.add_argument("--fit", action="store_true", help="fit the confidence model and write --conf-json")
    p.add_argument("--conf-json", default=str(CONF_JSON), help="confidence model json written by --fit / read otherwise")
    p.add_argument("--exclude-z", type=float, nargs=2, default=None, help="drop sites whose crop overlaps z [z0, z1)")
    p.add_argument("--rows-csv", default=None, help="cached step rows csv to reuse (default <out>.csv)")
    p.add_argument("--max-step", type=int, default=1, help="never emit steps larger than this (winding units)")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("compare", help="auto chains vs oracle (mesh) crossings on the same rays")
    p.add_argument("--auto", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--tol", type=float, default=4.0)
    p.add_argument("--kmax", type=int, default=None, help="also report steps / pairs whose oracle windings are all < kmax")

    p = sub.add_parser("check", help="parse a relative_windings.json with the pinned fitter loader")
    p.add_argument("json")

    sub.add_parser("sector", help="villa #2167 sector (w035 missing between w034 and w036)")
    args = ap.parse_args()
    if args.cmd == "band" and args.reassemble:
        import pickle
        src_dir = Path(args.from_crops or Path(args.out) / "crops")
        res = [pickle.loads(f.read_bytes()) for f in sorted(src_dir.glob("*.pkl"))]
        res = [r for r in res if abs(((r[0]["theta"] - args.theta0) / args.dtheta) - round((r[0]["theta"] - args.theta0) / args.dtheta)) < 1e-6]
        Path(args.out).mkdir(parents=True, exist_ok=True)
        prev = Path(args.out) / "summary.json"
        assemble(args, res, Path(args.out), timing=json.loads(prev.read_text()).get("timing") if prev.exists() else None)
        return
    {"band": band, "oracle": oracle, "corrupt": corrupt, "validate": validate, "compare": compare, "check": check, "relabel": relabel, "split": split, "gmfit": gmfit, "gmfix": gmfix,
     "sector": sector}[args.cmd](args)


if __name__ == "__main__":
    main()
