"""autowind mask->CT offset correction of finished chain sets (mask-offset test, notes/2026-10-24-mask-offset-prereg.md).

New, optional and separate: no existing code path changes, so every published chain set still reproduces byte for byte.

  uv run python -m autowind.offset --chains <dir with chains.json + relative_windings.json> --mode ctpeak \
      --ct <CT OME-Zarr group, local path or s3:// URL> --ct-level <level in the mask frame> --out <dir>

Why: the m7 surface mask marks one face of each sheet (its training labels are thin bands on the recto skin, about
1-2 vox off the CT intensity peak), so the chain points (centres of m7 runs along the ray) sit about 1.6-1.9 vox
inward of the CT sheet centre on PHerc0139 and PHercParis4 (notes/2026-10-24-fiber-contacts-gate.md, L4).

Modes:
  off     (default) relative_windings.json and chains.json are copied byte for byte.
  ctpeak  every chain point p is moved along its own ray (horizontal, from the umbilicus; direction taken from the
          chain's first and last point) to the CT intensity peak nearest in height, label-free:
            profile: CT at p + t u, t in [-(W+2), W+2] every 0.25 vox, trilinear, mean of 9 parallel lines
                     (z +-1, horizontal perpendicular +-1 vox), Gaussian-smoothed (sigma --sigma vox);
            allowed: |t| <= W (--window, default 4) and the new radius stays inside the midpoints to the chain's
                     neighbouring points minus 0.5 vox (so the order along the ray is kept and a point never moves
                     onto its neighbour's sheet);
            shift:   the highest local maximum of the smoothed profile inside the allowed range (parabolic sub-sample
                     refinement); no local maximum -> no shift.
          Only "pts" (and "r") change; "wind", step values and chain membership are unchanged. Per-point shifts go to
          offset_shifts.json. With --window 0 the output is byte-identical to the input.
  field   (the mask-offset gate's arm) the local systematic offset, robust to CT noise: for each point, the raw CT
          profiles (as ctpeak, unsmoothed) of all chain points within --nb-dz vox in z (default 64 = the adjacent ray
          levels), --nb-dtheta degrees (default 15 = the adjacent rays) and --nb-dr vox in radius (default 48) are
          averaged, each aligned on its own chain point. The shift is the centre a of the window [a - w/2, a + w/2]
          with the highest mean CT, w = the point's m7 run length along the ray (rl, clipped to [2, 8]; 4 if absent),
          searched over |a| <= W, then clipped to the neighbour midpoints as in ctpeak. This is the 1-D analogue,
          along the ray, of the block registration that found the offset (maximise CT inside the shifted mask).
  radial  every point is moved outward along its ray by the constant --shift vox (the measured block mean), clipped
          to the same neighbour midpoints.
The CT must be in the mask frame (same grid as mask level 0): PHercParis4 / PHerc1667 m7-L2 = CT level 2 of the
2.4 um volume; PHerc0139 m7-L0 = CT level 0 of the 9.362 um volume. Chunks are cached as raw npy (--ct-cache).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import core  # noqa: F401  (puts tools/ and eval/ on sys.path)
import auto_windings as aw  # noqa: E402  (frozen; imported, never modified)

OFF = (-1.0, 0.0, 1.0)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class CTChunks:
    """Chunk-cached reader of one CT level (raw npy per chunk, key '%d_%d_%d.npy', zero outside the array)."""

    def __init__(self, url: str, level: str, cache: Path):
        import zarr
        so = {"storage_options": {"anon": True}} if str(url).startswith("s3://") else {}
        self.a = zarr.open_group(str(url), mode="r", **so)[str(level)]
        self.ch = np.array(self.a.chunks, int)
        self.shape = self.a.shape
        self.dir = Path(cache)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: dict = {}
        self.lock = threading.Lock()

    def get(self, key):
        m = self.mem.get(key)
        if m is not None:
            return m
        f = self.dir / ("%d_%d_%d.npy" % key)
        m = None
        if f.exists():
            try:
                m = np.load(f)
            except Exception:
                m = None
        if m is None:
            m = np.zeros(tuple(self.ch), np.uint8)
            sl = tuple(slice(k * c, min((k + 1) * c, n)) for k, c, n in zip(key, self.ch, self.shape))
            if min(key) >= 0 and all(s.stop > s.start for s in sl):
                v = np.asarray(self.a[sl])
                m[:v.shape[0], :v.shape[1], :v.shape[2]] = v
            tmp = f.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp.npy")
            np.save(tmp, m)
            os.replace(tmp, f)
        with self.lock:
            if len(self.mem) > 600:
                self.mem.pop(next(iter(self.mem)))
            self.mem[key] = m
        return m

    def prefetch(self, keys, threads=16):
        todo = [k for k in keys if not (self.dir / ("%d_%d_%d.npy" % k)).exists()]

        def f(k):
            self.get(k)
            with self.lock:
                self.mem.pop(k, None)
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(f, todo))
        return len(todo)

    def gather(self, qi: np.ndarray) -> np.ndarray:
        """Integer voxel coordinates (n, 3) -> values (float32)."""
        ci = np.floor_divide(qi, self.ch)
        out = np.zeros(len(qi), np.float32)
        keys, inv = np.unique(ci, axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
        for k, key in enumerate(keys):
            sel = order[bounds[k]:bounds[k + 1]]
            m = self.get(tuple(int(x) for x in key))
            loc = qi[sel] - key * self.ch
            out[sel] = m[loc[:, 0], loc[:, 1], loc[:, 2]]
        return out

    def trilinear(self, q: np.ndarray) -> np.ndarray:
        f = np.floor(q)
        w = q - f
        f = f.astype(np.int64)
        out = np.zeros(len(q), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ww = ((w[:, 0] if dz else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) *
                          (w[:, 2] if dx else 1 - w[:, 2]))
                    out += ww.astype(np.float32) * self.gather(f + np.array([dz, dy, dx]))
        return out


def ray_dirs(ch: list[dict]) -> list[np.ndarray | None]:
    """Unit ray direction (zyx, z = 0) per chain from its first and last point; None for single-point chains."""
    out = []
    for c in ch:
        P = np.asarray(c["pts"], float)
        if len(P) < 2:
            out.append(None)
            continue
        d = P[-1] - P[0]
        d[0] = 0.0
        n = np.linalg.norm(d)
        out.append(d / n if n > 1e-6 else None)
    return out


def allowed_range(r: np.ndarray, i: int, window: float) -> tuple[float, float]:
    lo, hi = -window, window
    if i > 0:
        lo = max(lo, (r[i - 1] - r[i]) / 2 + 0.5)
    if i < len(r) - 1:
        hi = min(hi, (r[i + 1] - r[i]) / 2 - 0.5)
    return lo, hi


def profile_shifts(src: CTChunks, ch: list[dict], window: float, sigma: float, step: float = 0.25,
                   batch: int = 4000) -> list[dict]:
    """Per chain point: CT-peak shift along the ray (vox, + = outward) and diagnostics."""
    from scipy.ndimage import gaussian_filter1d
    U = ray_dirs(ch)
    t = np.arange(-(window + 2), window + 2 + 1e-9, step)
    jobs = []  # (chain, point, p, u)
    for ci, (c, u) in enumerate(zip(ch, U)):
        if u is None:
            continue
        for pi, p in enumerate(c["pts"]):
            jobs.append((ci, pi, np.asarray(p, float), u))
    res = {}
    for b0 in range(0, len(jobs), batch):
        J = jobs[b0:b0 + batch]
        P = np.array([j[2] for j in J])
        Uu = np.array([j[3] for j in J])
        V = np.stack([np.zeros(len(J)), -Uu[:, 2], Uu[:, 1]], 1)
        Q = []
        for dz in OFF:
            for dv in OFF:
                base = P + np.array([dz, 0, 0]) + dv * V
                Q.append(base[:, None, :] + t[None, :, None] * Uu[:, None, :])
        Q = np.stack(Q, 1)  # (n, 9, nt, 3)
        vals = src.trilinear(Q.reshape(-1, 3)).reshape(len(J), len(OFF) ** 2, len(t)).mean(1)
        sm = gaussian_filter1d(vals.astype(np.float64), sigma / step, axis=1, mode="nearest")
        for k, (ci, pi, _, _) in enumerate(J):
            res[(ci, pi)] = (vals[k], sm[k])
    out = []
    for ci, c in enumerate(ch):
        r = np.asarray(c["r"], float)
        rows = []
        for pi in range(len(c["pts"])):
            if (ci, pi) not in res:
                rows.append(dict(shift=0.0, status="no_ray"))
                continue
            raw, s = res[(ci, pi)]
            lo, hi = allowed_range(r, pi, window)
            i0 = int(np.argmin(np.abs(t)))
            row = dict(ct0=float(s[i0]), lo=float(lo), hi=float(hi))
            if hi < lo or window <= 0:
                row.update(shift=0.0, status="no_room")
                rows.append(row)
                continue
            # local maxima of the smoothed profile (strict interior of the sampled range) inside [lo, hi]
            pk = np.flatnonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])) + 1
            pk = pk[(t[pk] >= lo) & (t[pk] <= hi)]
            if len(pk) == 0:
                row.update(shift=0.0, status="no_peak")
                rows.append(row)
                continue
            j = int(pk[np.argmax(s[pk])])
            a, b, cc = s[j - 1], s[j], s[j + 1]
            den = a - 2 * b + cc
            off = 0.5 * (a - cc) / den if den < 0 else 0.0
            sh = float(np.clip(t[j] + off * step, lo, hi))
            row.update(shift=sh, status="peak", ct_peak=float(b), npk=int(len(pk)))
            rows.append(row)
        out.append(rows)
    return out


def raw_profiles(src: CTChunks, ch: list[dict], tmax: float, step: float = 0.25, batch: int = 4000):
    """Unsmoothed 9-line mean CT profile along the ray at every chain point: (t, {(chain, point): profile})."""
    U = ray_dirs(ch)
    t = np.arange(-tmax, tmax + 1e-9, step)
    jobs = [(ci, pi, np.asarray(p, float), u) for ci, (c, u) in enumerate(zip(ch, U)) if u is not None
            for pi, p in enumerate(c["pts"])]
    res = {}
    for b0 in range(0, len(jobs), batch):
        J = jobs[b0:b0 + batch]
        P = np.array([j[2] for j in J])
        Uu = np.array([j[3] for j in J])
        V = np.stack([np.zeros(len(J)), -Uu[:, 2], Uu[:, 1]], 1)
        Q = np.stack([(P + np.array([dz, 0, 0]) + dv * V)[:, None, :] + t[None, :, None] * Uu[:, None, :]
                      for dz in OFF for dv in OFF], 1)
        vals = src.trilinear(Q.reshape(-1, 3)).reshape(len(J), len(OFF) ** 2, len(t)).mean(1)
        for k, (ci, pi, _, _) in enumerate(J):
            res[(ci, pi)] = vals[k].astype(np.float64)
    return t, res


def field_shifts(src: CTChunks, ch: list[dict], window: float, nb_dz: float = 64.0, nb_dtheta: float = 15.0,
                 nb_dr: float = 48.0, step: float = 0.25) -> list[list[dict]]:
    t, prof = raw_profiles(src, ch, window + 4.5, step)
    # index points by ray (z, theta): sorted radii and their (chain, point) keys
    rays: dict = {}
    for ci, c in enumerate(ch):
        for pi, rr in enumerate(c["r"]):
            if (ci, pi) in prof:
                rays.setdefault((float(c["z"]), float(c["theta"]) % 360.0), []).append((float(rr), ci, pi))
    ray_r = {k: (np.array([x[0] for x in sorted(v)]), [x[1:] for x in sorted(v)]) for k, v in rays.items()}
    keys = list(ray_r)
    kz = np.array([k[0] for k in keys])
    kt = np.array([k[1] for k in keys])
    out = []
    for ci, c in enumerate(ch):
        r = np.asarray(c["r"], float)
        rows = []
        z0, th0 = float(c["z"]), float(c["theta"]) % 360.0
        dth = np.abs((kt - th0 + 180.0) % 360.0 - 180.0)
        nbr = [keys[i] for i in np.flatnonzero((np.abs(kz - z0) <= nb_dz) & (dth <= nb_dtheta + 1e-6))]
        rl = c.get("rl")
        for pi in range(len(c["pts"])):
            if (ci, pi) not in prof:
                rows.append(dict(shift=0.0, status="no_ray"))
                continue
            acc = np.zeros(len(t))
            n = 0
            for k in nbr:
                rr, ids = ray_r[k]
                a, b = np.searchsorted(rr, [r[pi] - nb_dr, r[pi] + nb_dr + 1e-9])
                for j in range(a, b):
                    acc += prof[ids[j]]
                    n += 1
            acc /= max(n, 1)
            w = float(np.clip(rl[pi], 2.0, 8.0)) if rl is not None and rl[pi] is not None else 4.0
            k = max(int(round(w / step)), 1) | 1  # odd number of samples
            box = np.convolve(acc, np.ones(k) / k, mode="same")
            lo, hi = allowed_range(r, pi, window)
            row = dict(n_profiles=n, width=w, lo=float(lo), hi=float(hi))
            if hi < lo or window <= 0:
                row.update(shift=0.0, status="no_room")
                rows.append(row)
                continue
            ok = np.flatnonzero((t >= lo) & (t <= hi) & (np.abs(t) <= window))
            j = int(ok[np.argmax(box[ok])])
            off = 0.0
            if 0 < j < len(t) - 1 and ok[0] < j < ok[-1]:
                a_, b_, c_ = box[j - 1], box[j], box[j + 1]
                den = a_ - 2 * b_ + c_
                off = 0.5 * (a_ - c_) / den if den < 0 else 0.0
            sh = float(np.clip(t[j] + off * step, lo, hi))
            row.update(shift=sh, status="field", gain=float(box[j] - box[int(np.argmin(np.abs(t)))]))
            rows.append(row)
        out.append(rows)
    return out


def apply_shifts(ch: list[dict], shifts: list[list[float]]) -> list[dict]:
    U = ray_dirs(ch)
    new = []
    for c, u, sh in zip(ch, U, shifts):
        c2 = dict(c)
        if u is not None and any(s != 0.0 for s in sh):
            pts, r = [], []
            for p, rr, s in zip(c["pts"], c["r"], sh):
                if s != 0.0:
                    q = np.asarray(p, float) + s * u
                    pts.append([float(q[0]), float(q[1]), float(q[2])])
                    r.append(float(rr) + s)
                else:
                    pts.append(p)
                    r.append(rr)
            c2["pts"], c2["r"] = pts, r
        new.append(c2)
    return new


def prefix_of(rw_path: Path) -> str:
    d = json.loads(Path(rw_path).read_text())
    name = next(iter(d["collections"].values()))["name"]
    return name.split("_z")[0]


def summarize(rows: list[list[dict]]) -> dict:
    flat = [x for rr in rows for x in rr]
    sh = np.array([x["shift"] for x in flat])
    st = {}
    for x in flat:
        st[x["status"]] = st.get(x["status"], 0) + 1
    pk = np.array([x["shift"] for x in flat if x["status"] in ("peak", "field", "radial")])
    q = (lambda a, p: float(np.percentile(a, p)) if len(a) else None)
    return dict(points=len(flat), status=st, shift_mean_all=float(sh.mean()) if len(sh) else None,
                peak_shift_mean=float(pk.mean()) if len(pk) else None, peak_shift_median=q(pk, 50),
                peak_shift_p10_p90=[q(pk, 10), q(pk, 90)],
                frac_outward=float((pk > 0.25).mean()) if len(pk) else None,
                frac_inward=float((pk < -0.25).mean()) if len(pk) else None,
                hist_peak_shift={f"{a:+.0f}": int(((pk >= a - 0.5) & (pk < a + 0.5)).sum()) for a in range(-4, 5)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autowind.offset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True, type=Path, help="dir with chains.json and relative_windings.json")
    ap.add_argument("--mode", default="off", choices=["off", "ctpeak", "field", "radial"])
    ap.add_argument("--ct", help="CT OME-Zarr group (local path or s3:// URL, anonymous) in the mask frame")
    ap.add_argument("--ct-level", default="0", help="CT level whose grid equals mask level 0")
    ap.add_argument("--ct-cache", type=Path, help="raw npy chunk cache (default <out>/ctcache)")
    ap.add_argument("--window", type=float, default=4.0, help="ctpeak: max |shift| (vox)")
    ap.add_argument("--sigma", type=float, default=1.0, help="ctpeak: profile smoothing (vox)")
    ap.add_argument("--nb-dz", type=float, default=64.0, help="field: neighbourhood half-height (vox)")
    ap.add_argument("--nb-dtheta", type=float, default=15.0, help="field: neighbourhood half-angle (deg)")
    ap.add_argument("--nb-dr", type=float, default=48.0, help="field: neighbourhood radial half-width (vox)")
    ap.add_argument("--shift", type=float, default=1.75, help="radial: constant outward shift (vox)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--zs", default=None, help="only chains at these z levels (comma list; diagnostics / pilots)")
    ap.add_argument("--measure-only", action="store_true", help="write offset_shifts.json + summary, no chain files")
    ap.add_argument("--out", required=True, type=Path)
    o = ap.parse_args(argv)
    src_dir, out = o.chains, o.out
    out.mkdir(parents=True, exist_ok=True)
    info = dict(offset="mask-offset", mode=o.mode, chains=str(src_dir),
                inputs={f: _sha(src_dir / f) for f in ("relative_windings.json", "chains.json")})
    if o.mode == "off":
        for f in ("relative_windings.json", "chains.json"):
            shutil.copyfile(src_dir / f, out / f)
        info["applied"] = "none: copied unchanged"
    else:
        ch = json.loads((src_dir / "chains.json").read_text())
        if o.zs:
            zs = {int(z) for z in o.zs.split(",")}
            ch = [c for c in ch if int(c["z"]) in zs]
        if o.mode in ("ctpeak", "field"):
            if not o.ct:
                ap.error(f"--mode {o.mode} needs --ct")
            src = CTChunks(o.ct, o.ct_level, o.ct_cache or (out / "ctcache"))
            # prefetch every chunk the profiles touch (threads; resumable via the chunk cache)
            need = set()
            for c, u in zip(ch, ray_dirs(ch)):
                if u is None:
                    continue
                P = np.asarray(c["pts"], float)
                for s in (-(o.window + 6), 0.0, o.window + 6):
                    for dz in (-2, 2):
                        for dv in (-2, 2):
                            q = P + s * u + np.array([dz, 0, 0]) + dv * np.array([0, -u[2], u[1]])
                            need |= {tuple(int(x) for x in k) for k in np.unique(np.floor_divide(np.floor(q).astype(int), src.ch), axis=0)}
            nf = src.prefetch(sorted(need), o.threads)
            info["ct"] = dict(url=o.ct, level=o.ct_level, chunks=len(need), fetched=nf)
            if o.mode == "ctpeak":
                rows = profile_shifts(src, ch, o.window, o.sigma)
                info.update(window=o.window, sigma=o.sigma)
            else:
                rows = field_shifts(src, ch, o.window, o.nb_dz, o.nb_dtheta, o.nb_dr)
                info.update(window=o.window, nb_dz=o.nb_dz, nb_dtheta=o.nb_dtheta, nb_dr=o.nb_dr)
        else:
            rows = []
            for c in ch:
                r = np.asarray(c["r"], float)
                rr = []
                for i in range(len(r)):
                    lo, hi = allowed_range(r, i, abs(o.shift) + 1)
                    rr.append(dict(shift=float(np.clip(o.shift, lo, hi)) if len(r) > 1 else 0.0,
                                   status="radial" if len(r) > 1 else "no_ray"))
                rows.append(rr)
            info["shift"] = o.shift
        info["applied"] = summarize(rows)
        shifts = [[x["shift"] for x in rr] for rr in rows]
        (out / "offset_shifts.json").write_text(json.dumps(
            [dict(z=c["z"], theta=c["theta"], shift=[round(s, 3) for s in sh], status=[x["status"] for x in rr])
             for c, sh, rr in zip(ch, shifts, rows)]))
        if not o.measure_only:
            if o.zs:
                ap.error("--zs is for --measure-only diagnostics")
            new = apply_shifts(ch, shifts)
            aw.write_json(new, out / "relative_windings.json", prefix=prefix_of(src_dir / "relative_windings.json"))
            (out / "chains.json").write_text(json.dumps(new))
    if (out / "relative_windings.json").exists() and not o.measure_only:
        info["sha256"] = {f: _sha(out / f) for f in ("relative_windings.json", "chains.json")}
    (out / "offset.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
