"""Flattened CT renders of the same PHerc0139 windings from the human meshes and three spiral fits (arm0 baseline, arm19
automatic annotations, arm2 human annotations), z 3000-4500. Fits went through the render stages on a pod
(spiral/render_compare.sh: s05 snap -> s06 lasagna flatten -> s06b montage -> s07 VC3D render, 66 layers);
human meshes through s06b + s07 only. Each render (mean of the 7 middle layers) is resampled onto a common
angle x z canvas using the rendered surface's own xyz, so the strips line up column by column.

Outlines on fit strips (the rendered, snapped surface itself, checked against the human meshes, blacklisted ones left
out): red = on a human sheet with a different number than the fit gives it (seam-offset mapping of spiral_eval);
yellow = no human sheet within 4 vox.

  uv run python release/fig_renders.py --overview release/figures/fit_render_strips.png \
      --zoom release/figures/fit_render_zoom.png --zoom-win 0 45 3300 3900 --zoom-win 285 330 3500 4100
  uv run python release/fig_renders.py --set w031_039 --overview release/figures/fit_render_strips_w031_w039.png --zoom /dev/null

--scroll p1667 (PHerc1667 z 6500-8000, fit frame L2 of 20251217075048): the render chain above is PHerc0139-
specific (m7 L0 snap, pod VC3D), so this mode renders locally and directly: each surface (fit winding *_spliced, or
human winding k of the labelled merged mesh, confirm_region/p1667/oracle_z6500) is upsampled to ~1 vox, the CT (zarr
level 2, 128^3 chunks streamed from the open-data bucket and cached under data/scratch/fit_figures/p1667_ct_L2) is
sampled at -3..+3 vox along the surface normal (trilinear) and averaged (7 layers, as mean7 above), then put on the
same angle x z canvas. No snap / lasagna flatten: the fit surface is rendered where the fit put it. Rows per human
winding k: human k, then for each arm the fit winding j holding most of k's matched vertices (vertex dump).
  uv run python release/fig_renders.py --scroll p1667 --human-k 22 27 --overview release/figures/p1667_render_strips.png \
      --zoom release/figures/p1667_render_zoom.png --zoom-win 20 60 7000 7500
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import binary_opening, gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_traces import ORACLE, HumanFrame, blacklisted_windings, save_png, se  # noqa: E402
import fit_traces as ft  # noqa: E402
ARMS_DIR = ft.ARMS

REPO = Path(__file__).resolve().parents[1]
RENDERS = REPO / "data/scratch/fit_figures/renders"
DA, DZ = 0.25, 4.0          # canvas cell: 0.25 deg x 4 z (about 4.5 x 4 vox at r ~ 1000)
ZR = (3000, 4500)
SETS = {  # (label, source, winding name in layout, human winding it lies on or None)
    "w029_030": [
        ("human w029", "human", "w029", 29), ("arm0 j47", "arm0", "w047_spliced", 29),
        ("arm19 j42", "arm19", "w042_spliced", 29), ("arm2 j46", "arm2", "w046_spliced", 29),
        ("arm0 j48 (extra turn)", "arm0", "w048_spliced", None),
        ("human w030", "human", "w030", 30), ("arm0 j49", "arm0", "w049_spliced", 30),
        ("arm19 j43", "arm19", "w043_spliced", 30), ("arm2 j47", "arm2", "w047_spliced", 30)],
    "w031_039": [
        ("human w031", "human", "w031", 31), ("arm0 j50", "arm0", "w050_spliced", 31),
        ("arm19 j44", "arm19", "w044_spliced", 31),
        ("human w039", "human", "w039", 39), ("arm0 j57", "arm0", "w057_spliced", 39),
        ("arm0 j58", "arm0", "w058_spliced", 39), ("arm19 j52", "arm19", "w052_spliced", 39)],
}
ROWS = SETS["w029_030"]


def load_winding(src: str, name: str, stride: int = 2):
    """Pixels of one winding's render: intensity, xyz (zyx) per pixel (subsampled by stride), from its montage."""
    mid = RENDERS / src / "mid"
    lay = json.loads((mid / "layout.json").read_text())
    for m, info in lay["montages"].items():
        for e in info["windings"]:
            if e["name"] == name:
                img = tifffile.imread(mid / f"{m}_mean7.tif").astype(np.float32)
                g = {c: tifffile.imread(mid / f"{m}.tifxyz" / f"{c}.tif").astype(np.float32) for c in "xyz"}
                step = info["step_vox"]
                r0, c0 = e["row0"], e["col0"]
                h, w = e["shape"]
                pr = np.arange(int(r0 * step), int((r0 + h - 1) * step) + 1, stride)
                pc = np.arange(int(c0 * step), int((c0 + w - 1) * step) + 1, stride)
                pr = pr[pr < img.shape[0]]
                pc = pc[pc < img.shape[1]]
                R, C = np.meshgrid(pr, pc, indexing="ij")
                gr, gc = R / step, C / step
                valid = (g["x"] > -0.5) & (g["y"] > -0.5) & (g["z"] > -0.5)
                ok = map_coordinates(valid.astype(np.float32), [gr, gc], order=1) > 0.999
                xyz = np.stack([map_coordinates(g[c], [gr, gc], order=1) for c in "zyx"], -1)
                return img[R, C], xyz, ok, (R, C)
    raise KeyError(f"{src}/{name} not in layout")


def to_canvas(val, zyx, ok, umb, human: bool, da=DA, dz=DZ, ar=(0.0, 360.0), zr=ZR):
    ang = umb.angle(zyx)
    if human:  # human meshes span 365-400 deg: keep the first 360 deg along the grid columns
        colang = np.array([np.nanmedian(np.where(ok[:, i], ang[:, i], np.nan)) if ok[:, i].any() else np.nan
                           for i in range(ang.shape[1])])
        f = np.isfinite(colang)
        un = np.full_like(colang, np.nan)
        un[f] = np.degrees(np.unwrap(np.radians(colang[f])))
        un -= np.nanmin(un[:max(f.sum() // 20, 1)]) if f.any() else 0
        ok = ok & (np.abs(un) < 360)[None, :]
    ia = ((ang - ar[0]) // da).astype(int)
    iz = ((zyx[..., 0] - zr[0]) // dz).astype(int)
    shape = (int((zr[1] - zr[0]) / dz), int((ar[1] - ar[0]) / da))
    ok = ok & (iz >= 0) & (iz < shape[0]) & (ia >= 0) & (ia < shape[1])
    acc = np.zeros(shape + (5,))
    idx = (iz[ok], ia[ok])
    for k, v in enumerate([val[ok], zyx[ok][:, 0], zyx[ok][:, 1], zyx[ok][:, 2], np.ones(ok.sum())]):
        np.add.at(acc[..., k], idx, v)
    n = acc[..., 4]
    with np.errstate(invalid="ignore"):
        out = acc[..., :4] / n[..., None]
    out[n == 0] = np.nan
    return out  # (..., [intensity, z, y, x])


def overlap_zone(k: int, umb) -> tuple[float, float]:
    """Angular range where human mesh w<k> covers the same angle twice (it spans 365-400 deg): there the turn's start and
    the previous turn's end are both labelled, so 'which human sheet' is ambiguous. (start, end) deg, may wrap."""
    d = next(ORACLE.glob(f"w{k:03d}_s*"))
    g, _ = se.read_tifxyz(d)
    g = g[::4, ::4]
    col = np.array([np.nanmedian(umb.angle(g[:, i][np.isfinite(g[:, i]).all(-1)])) if np.isfinite(g[:, i]).all(-1).any()
                    else np.nan for i in range(g.shape[1])])
    f = np.isfinite(col)
    un = np.degrees(np.unwrap(np.radians(col[f])))
    span = abs(un[-1] - un[0])
    a0 = col[f][0]
    return float(a0), float(a0 + np.sign(un[-1] - un[0]) * (span - 360.0))


class ChunkCT:
    """Nearest-voxel CT lookup from a remote zarr level, 128^3 chunks fetched in parallel and cached on disk (.npy)."""

    def __init__(self, url: str, level: int, cache: Path, workers: int = 24):
        import zarr
        self.a = zarr.open(url, mode="r", storage_options={"anon": True})[str(level)]
        self.c = self.a.chunks
        self.cache = cache / str(level)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.mem = {}
        self.fetched = 0

    def _load(self, key):
        f = self.cache / ("%d_%d_%d.npy" % key)
        if f.exists():
            return key, np.load(f)
        sl = tuple(slice(k * c, min((k + 1) * c, n)) for k, c, n in zip(key, self.c, self.a.shape))
        blk = np.asarray(self.a[sl])
        np.save(f, blk)
        self.fetched += 1
        return key, blk

    def _gather(self, p: np.ndarray) -> np.ndarray:
        """Integer voxels (N, 3) zyx -> float32 values (0 outside the volume)."""
        from concurrent.futures import ThreadPoolExecutor
        shp = np.array(self.a.shape)
        inb = (p >= 0).all(1) & (p < shp).all(1)
        out = np.zeros(len(p), np.float32)
        c = np.array(self.c)
        idx = np.nonzero(inb)[0]
        ck = p[idx] // c
        key = (ck[:, 0] * 4096 + ck[:, 1]) * 4096 + ck[:, 2]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.nonzero(np.r_[True, ks[1:] != ks[:-1]])[0]
        bounds = np.r_[starts, len(ks)]
        keys = [tuple(int(v) for v in ck[order[i]]) for i in starts]
        need = [k for k in keys if k not in self.mem]
        with ThreadPoolExecutor(self.workers) as ex:
            for k, blk in ex.map(self._load, need):
                self.mem[k] = blk
        loc = p[idx] - ck * c
        for i, k in enumerate(keys):
            sel = order[bounds[i]:bounds[i + 1]]
            l = loc[sel]
            out[idx[sel]] = self.mem[k][l[:, 0], l[:, 1], l[:, 2]]
        return out

    def sample(self, zyx: np.ndarray) -> np.ndarray:
        """Trilinear CT value at zyx (N, 3) float (nearest-voxel sampling terraces inclined surfaces)."""
        p0 = np.floor(zyx).astype(np.int64)
        f = (zyx - p0).astype(np.float32)
        out = np.zeros(len(p0), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dx else 1 - f[:, 2]))
                    out += w * self._gather(p0 + np.array([dz, dy, dx]))
        return out

    def drop(self) -> None:
        self.mem.clear()


def direct_render(d: Path, ct: ChunkCT, zr, layers: int = 7, stride: int = 1):
    """CT mean of `layers` normal offsets around a tifxyz surface (zyx frame = CT level). Returns (val, zyx, ok) on the
    surface grid upsampled to ~1 vox (then subsampled by stride)."""
    from scipy.ndimage import zoom
    g, _ = se.read_tifxyz(d)
    g = se.crop_band(g, zr[0] - 8, zr[1] + 8)
    fin = np.isfinite(g).all(-1)
    st = np.nanmedian(np.linalg.norm(np.diff(g, axis=1), axis=-1))
    up = max(1, int(round(st / stride)))
    okz = zoom(fin.astype(np.float32), up, order=1) > 0.999
    G = np.stack([zoom(np.where(fin, g[..., i], 0), up, order=1) for i in range(3)], -1)
    G[~okz] = np.nan
    gr = np.gradient(G, axis=0)
    gc = np.gradient(G, axis=1)
    n = np.cross(gr, gc)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    ok = okz & np.isfinite(n).all(-1)
    P, N = G[ok], n[ok]
    acc = np.zeros(len(P))
    for k in range(layers):
        acc += ct.sample(P + (k - (layers - 1) / 2) * N)
    val = np.full(ok.shape, np.nan, np.float32)
    val[ok] = acc / layers
    return val, G, ok


def fit_winding_for(arm: str, k: int, cache: Path) -> tuple[int, float]:
    """Fit winding j holding most of human winding k's matched vertices, and that share."""
    V = np.load(cache)
    vd = next(dd for dd in ft.VDUMPS if (dd / f"{arm}.npz").exists())
    D = np.load(vd / f"{arm}.npz")
    m = (V["k"] == k) & D["matched"]
    v, c = np.unique(D["j"][m], return_counts=True)
    return int(v[np.argmax(c)]), float(c.max() / max(m.sum(), 1))


def main_p1667(a) -> None:
    ft.use_scroll("p1667")
    zr = (6500, 8000)
    cache = se.ref_path(*zr)
    umb = se.Umbilicus()
    ct = ChunkCT(ft.CT_URL, ft.CT_BASE, REPO / "data/scratch/fit_figures/p1667_ct_L2")
    titles = dict(zip(a.arms, a.arm_titles or a.arms))
    rows = []   # (label, src_dir, human k, arm or None, j)
    for k in a.human_k:
        rows.append((f"human w{k:03d}", next(ft.ORACLE.glob(f"w{k:03d}_s*")), k, None, None))
        for arm in a.arms:
            j, share = fit_winding_for(arm, k, cache)
            d = ARMS_DIR / arm / "meshes_step4" / f"w{j:03d}_spliced"
            rows.append((f"{titles[arm]} j{j}", d, k, arm, j))
            print(f"k{k} {arm}: j{j} holds {100 * share:.0f} % of k's matched vertices", flush=True)
    ks = sorted(set(a.human_k))
    items = [it for it in se.discover_fit(ft.ORACLE, zr[0] - 10, zr[1] + 10) if ks[0] - 5 <= it["label"] <= ks[-1] + 5]
    H = se.Surfaces(items, zr[0] - 10, zr[1] + 10)
    da, dz = 0.15, 2.0
    acols = (np.arange(int(360 / da)) + 0.5) * da
    amb = np.zeros(len(acols), bool)   # human label cut (~145 deg) .. fit seam (~175 deg): label identity is convention
    for lo_, hi_ in a.amb_span:
        amb |= (acols >= lo_) & (acols <= hi_)
    canv, marks, stats = {}, {}, {}
    pk = REPO / "data/scratch/fit_figures" / f"p1667_render_cache_{'_'.join(map(str, a.human_k))}.pkl"
    if a.reuse and pk.exists():
        import pickle
        canv, marks, stats = pickle.load(open(pk, "rb"))
    for lab, d, k, arm, j in rows:
        if lab in canv:
            continue
        val, zyx, ok = direct_render(d, ct, zr)
        c = to_canvas(val, zyx, ok & np.isfinite(val), umb, False, da=da, dz=dz, zr=zr)
        canv[lab] = c[..., 0]
        if arm is not None:
            f = np.isfinite(c[..., 1]) & ~amb[None, :]
            pts = c[..., 1:][f].astype(np.float64)
            hl, _ = H.query(pts, 4.0)
            kf = HumanFrame(arm).k(np.full(len(pts), j), pts)
            on = hl != se.NOLAB
            other = np.zeros(f.shape, bool)
            unk = np.zeros(f.shape, bool)
            other[f] = on & (hl != k)
            unk[f] = ~on
            marks[lab] = (other, unk)
            stats[lab] = {"human_k": k, "on_k": round(float((hl == k).mean()), 3),
                          "on_other": round(float(other[f].mean()), 3), "off_sheet": round(float(unk[f].mean()), 3),
                          "numbered_as": int(np.median(kf)), "numbered_right": round(float((hl == kf).mean()), 3)}
        print(lab, d.name, stats.get(lab, ""), f"(chunks fetched so far {ct.fetched})", flush=True)
    import pickle
    pickle.dump((canv, marks, stats), open(pk, "wb"))
    r = 800.0  # nominal radius (vox) of the shown windings, for the aspect of the angle axis
    asp = 1.0 / (np.pi / 180 * r)

    def draw(ax, im, msk, extent, sig, lw, aspect=asp):
        v = im[np.isfinite(im)]
        lo, hi = np.percentile(v, [1, 99.5]) if len(v) else (0, 1)
        ax.imshow(np.where(np.isfinite(im), im, lo), cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=extent,
                  aspect=aspect, interpolation="bilinear")
        for m0, col in zip(msk or (), ("#ff2d2d", "#ffd400")):
            # unsnapped fit vs human mesh at 4 vox is speckled: outline only regions that persist at ~1 deg x 10 z
            m = gaussian_filter(binary_opening(m0, iterations=2).astype(float), sig)
            if m.max() > 0.5:
                ax.contour(np.linspace(extent[0], extent[1], m.shape[1]), np.linspace(extent[2], extent[3], m.shape[0]),
                           m, [0.5], colors=col, linewidths=lw)

    fig, axs = plt.subplots(len(rows), 1, figsize=(12.5, 1.85 * len(rows) + 0.9), constrained_layout=True)
    for ax, (lab, d, k, arm, j) in zip(axs, rows):
        draw(ax, canv[lab], marks.get(lab), [0, 360, zr[0], zr[1]], (5, 6), 0.9, aspect="auto")
        for lo_, hi_ in a.amb_span:
            ax.axvspan(lo_, hi_, color="#5a8fd8", alpha=0.35, lw=0)
        s = stats.get(lab)
        if s:
            txt = (f"{lab}: on human sheet w{k:03d} {100 * s['on_k']:.0f} %, on another sheet {100 * s['on_other']:.0f} %, "
                   f"on none {100 * s['off_sheet']:.0f} %   |   numbered w{s['numbered_as']:03d} by its fit "
                   f"(right number {100 * s['numbered_right']:.0f} %)")
        else:
            txt = f"{lab} (labelled human mesh, reference)"
        ax.set_title(txt, fontsize=8.5, loc="left")
        ax.set_yticks([6500, 7250, 8000])
        ax.set_xticks(range(0, 361, 30))
        ax.tick_params(labelsize=7)
    axs[-1].set_xlabel("angle around the umbilicus (deg); y = z (L2 slice, 9.596 um); axes not to scale.\nBlue: between the "
                       "human label cut and the fits' seam a winding continues onto the next human label by convention "
                       "(scored right with seam offsets); left out of the percentages", fontsize=9)
    fig.suptitle(f"PHerc1667 z {zr[0]}-{zr[1]}: CT rendered along human windings {', '.join(map(str, ks))} and the fit "
                 "windings lying on them (direct render: mean of 7 layers across the surface, no snap/flatten).\n"
                 "Outlines on fit rows: red = on a different human sheet than the row's (a sheet jump); yellow = on no "
                 "human sheet within 4 vox.", fontsize=9)
    save_png(fig, a.overview, 100)
    print(a.overview)
    out = REPO / "data/scratch/fit_figures" / "render_stats_p1667.json"
    json.dump({"rows": stats, "human_k": ks, "arms": a.arms}, open(out, "w"), indent=1)
    if not a.zoom_win or str(a.zoom) == "/dev/null":
        return
    zrows = rows if not a.zoom_rows_p1667 else [r_ for r_ in rows if r_[0] in a.zoom_rows_p1667]
    nw, nr = len(a.zoom_win), len(zrows)
    fig, axs = plt.subplots(nw, nr, figsize=(3.6 * nr, 3.3 * nw), constrained_layout=True, squeeze=False)
    for wi, (a0, a1, z0, z1) in enumerate(a.zoom_win):
        for ri, (lab, d, k, arm, j) in enumerate(zrows):
            val, zyx, ok = direct_render(d, ct, (z0, z1), stride=0.5)   # 0.5 vox grid: no empty ~1 vox bins
            c = to_canvas(val, zyx, ok & np.isfinite(val), umb, False, da=0.07, dz=1.0, ar=(a0, a1), zr=(z0, z1))
            msk = None
            if arm is not None:
                f = np.isfinite(c[..., 1])
                pts = c[..., 1:][f].astype(np.float64)
                hl, _ = H.query(pts, 4.0)
                other = np.zeros(f.shape, bool)
                unk = np.zeros(f.shape, bool)
                other[f] = (hl != se.NOLAB) & (hl != k)
                unk[f] = hl == se.NOLAB
                msk = (other, unk)
            ax = axs[wi, ri]
            draw(ax, c[..., 0], msk, [a0, a1, z0, z1], 2.5, 0.8)
            ax.set_title(lab, fontsize=8.5)
            ax.tick_params(labelsize=7)
        axs[wi, 0].set_ylabel(f"z  ({a0:.0f}-{a1:.0f} deg)", fontsize=8)
    fig.suptitle("PHerc1667 zoom (~1 vox/px), same window on every surface. red = on another human sheet, "
                 "yellow = between sheets", fontsize=10)
    save_png(fig, a.zoom, 110)
    print(a.zoom, f"chunks fetched {ct.fetched}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overview", type=Path, required=True)
    ap.add_argument("--zoom", type=Path, required=True)
    ap.add_argument("--zoom-win", type=float, nargs=4, action="append", default=[],
                    metavar=("A0", "A1", "Z0", "Z1"), help="zoom window (deg, deg, z, z); repeatable")
    ap.add_argument("--zoom-rows", nargs="+", default=["human w030", "arm0 j48 (extra turn)", "arm0 j49", "arm19 j43",
                                                       "arm2 j47"])
    ap.add_argument("--set", choices=list(SETS), default="w029_030")
    ap.add_argument("--scroll", choices=["p0139", "p1667"], default="p0139")
    ap.add_argument("--human-k", type=int, nargs="+", default=[22, 27], help="p1667: human windings to show")
    ap.add_argument("--arms", nargs="+", default=["c1667_base_fig", "c1667_m2s1_fig"], help="p1667 fit arms")
    ap.add_argument("--arm-titles", nargs="+", default=["baseline", "M2"])
    ap.add_argument("--zoom-rows-p1667", nargs="*", default=None)
    ap.add_argument("--reuse", action="store_true", help="p1667: reuse cached overview canvases")
    ap.add_argument("--amb-span", type=float, nargs=2, action="append", default=None, metavar=("A0", "A1"),
                    help="p1667: reference-ambiguous angle span(s), shaded and left out (default 143 178)")
    a = ap.parse_args()
    if a.scroll == "p1667":
        a.amb_span = a.amb_span or [(143.0, 178.0)]
        return main_p1667(a)
    global ROWS
    ROWS = SETS[a.set]
    ks = sorted({k for *_, k in ROWS if k is not None})
    umb = se.Umbilicus()
    black = blacklisted_windings()
    items = [it for it in se.discover_fit(ORACLE, ZR[0] - 10, ZR[1] + 10) if ks[0] - 5 <= it["label"] <= ks[-1] + 5
             and it["label"] not in black]
    H = se.Surfaces(items, ZR[0] - 10, ZR[1] + 10)
    zones = {k: overlap_zone(k, umb) for k in sorted({x + d for x in ks for d in (-1, 0, 1)})}
    spans = [(min(v) - 2, max(v) + 2) for v in zones.values()] + [(171.0, 178.0)]  # + the fits' seam (173-176 deg)
    acols = (np.arange(int(360 / DA)) + 0.5) * DA
    amb = np.zeros(len(acols), bool)
    for lo_, hi_ in spans:
        amb |= (acols >= lo_) & (acols <= hi_)
    print("reference-ambiguous zones (human overlap + fit seam):", {k: [round(x, 1) for x in v] for k, v in zones.items()})
    canv, marks, stats = {}, {}, {}
    for lab, src, name, k in ROWS:
        val, zyx, ok, _ = load_winding(src, name)
        c = to_canvas(val, zyx, ok, umb, src == "human")
        canv[lab] = c[..., 0]
        if src != "human":
            f = np.isfinite(c[..., 1]) & ~amb[None, :]
            pts = c[..., 1:][f].astype(np.float64)
            hl, _ = H.query(pts, 4.0)
            j = int(name[1:4])
            kf = HumanFrame(src).k(np.full(len(pts), j), pts)
            on = hl != se.NOLAB
            v, cnt = np.unique(hl[on], return_counts=True)
            main_k = int(v[np.argmax(cnt)])
            other = np.zeros(f.shape, bool)
            unk = np.zeros(f.shape, bool)
            other[f] = on & (hl != main_k)
            unk[f] = ~on
            marks[lab] = (other, unk)
            top2 = sorted(zip(cnt, v), reverse=True)[:2]
            stats[lab] = {"main_sheet": main_k, "on_main": round(float((hl == main_k).mean()), 3),
                          "on_other": round(float(other[f].mean()), 3), "off_sheet": round(float(unk[f].mean()), 3),
                          "second_sheet": int(top2[1][1]) if len(top2) > 1 else None,
                          "numbered_as": int(np.median(kf)),
                          "numbered_right": round(float((hl == kf).mean()), 3)}
        print(lab, stats.get(lab, ""), flush=True)
    r = 1000.0  # nominal radius (vox) for the aspect of the angle axis
    asp = 1.0 / (np.pi / 180 * r)  # 1 deg = r*pi/180 vox along the sheet, 1 z = 1 vox

    def draw(ax, im, msk, extent, sig, lw):
        v = im[np.isfinite(im)]
        lo, hi = np.percentile(v, [1, 99.5]) if len(v) else (0, 1)
        ax.imshow(np.nan_to_num(im, nan=0), cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=extent,
                  aspect=asp, interpolation="bilinear")
        for m0, col in zip(msk or (), ("#ff2d2d", "#ffd400")):
            m = gaussian_filter(binary_opening(m0, iterations=1).astype(float), sig)
            if m.max() > 0.5:
                ax.contour(np.linspace(extent[0], extent[1], m.shape[1]), np.linspace(extent[2], extent[3], m.shape[0]),
                           m, [0.5], colors=col, linewidths=lw)

    fig, axs = plt.subplots(len(ROWS), 1, figsize=(11.5, 2.05 * len(ROWS)), constrained_layout=True)
    for ax, (lab, src, name, k) in zip(axs, ROWS):
        draw(ax, canv[lab], marks.get(lab), [0, 360, ZR[0], ZR[1]], 1.5, 0.9)
        for lo_, hi_ in spans:
            ax.axvspan(lo_, hi_, color="#5a8fd8", alpha=0.35, lw=0)
        s = stats.get(lab)
        if s:
            txt = (f"{lab}: on its main human sheet w{s['main_sheet']:03d} {100 * s['on_main']:.0f} %, on another sheet "
                   f"{100 * s['on_other']:.0f} % (mostly w{s['second_sheet']:03d}), on none {100 * s['off_sheet']:.0f} %"
                   f"   |   numbered w{s['numbered_as']:03d} by its fit")
        else:
            txt = f"{lab} (reference)"
        ax.set_title(txt, fontsize=9, loc="left")
        ax.set_yticks([3000, 3750, 4500])
        ax.set_xticks(range(0, 361, 30))
        ax.tick_params(labelsize=7)
    axs[-1].set_xlabel("angle around the umbilicus (deg); y = z. Blue: human meshes overlap themselves / fit seam,\n"
                       "sheet identity ambiguous, left out of the percentages", fontsize=9)
    fig.suptitle("PHerc0139 z 3000-4500: flattened renders (snap + lasagna flatten + VC3D, mean of 7 middle layers) of "
                 f"human windings {', '.join(map(str, ks))} and the fit windings lying on them.\nOutlines: red = the surface leaves its main "
                 "human sheet for another one (a sheet jump); yellow = on no human sheet within 4 vox (between sheets / "
                 "in a gap).", fontsize=9)
    save_png(fig, a.overview, 100)
    print(a.overview)
    json.dump({"ambiguous_zones_deg": spans, "rows": stats}, open(RENDERS / f"render_stats_{a.set}.json", "w"), indent=1)
    if not a.zoom_win:
        return
    # zoom: native-resolution canvas (0.06 deg x 1 z ~ 1 x 1 vox) per window
    rows = [next(r for r in ROWS if r[0] == lab) for lab in a.zoom_rows]
    nw, nr = len(a.zoom_win), len(rows)
    w0 = a.zoom_win[0]
    pw = 2.9 * (w0[1] - w0[0]) * np.pi / 180 * r / max(w0[3] - w0[2], 1)
    fig, axs = plt.subplots(nw, nr, figsize=(min(pw, 4.2) * nr, 3.4 * nw), constrained_layout=True, squeeze=False)
    for wi, (a0, a1, z0, z1) in enumerate(a.zoom_win):
        for ri, (lab, src, name, k) in enumerate(rows):
            val, zyx, ok, _ = load_winding(src, name, stride=1)
            c = to_canvas(val, zyx, ok, umb, src == "human", da=0.06, dz=1.0, ar=(a0, a1), zr=(z0, z1))
            msk = None
            if src != "human":
                f = np.isfinite(c[..., 1])
                pts = c[..., 1:][f].astype(np.float64)
                hl, _ = H.query(pts, 4.0)
                other = np.zeros(f.shape, bool)
                unk = np.zeros(f.shape, bool)
                other[f] = (hl != se.NOLAB) & (hl != stats[lab]["main_sheet"])
                unk[f] = hl == se.NOLAB
                msk = (other, unk)
            ax = axs[wi, ri]
            draw(ax, c[..., 0], msk, [a0, a1, z0, z1], 2.5, 0.8)
            ax.set_title(lab, fontsize=9)
            ax.tick_params(labelsize=7)
        axs[wi, 0].set_ylabel(f"z  ({a0:.0f}-{a1:.0f} deg)", fontsize=8)
    fig.suptitle("Zoom, native resolution (~1 vox/px): same windows on every surface. red = jump to another human sheet, "
                 "yellow = between sheets", fontsize=10)
    save_png(fig, a.zoom, 110)
    print(a.zoom)


if __name__ == "__main__":
    main()
