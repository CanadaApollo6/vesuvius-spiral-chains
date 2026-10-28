"""Figures of the CT audit of the PHercParis4 winding labels (notes/2026-10-24-paris4-label-audit.md).

  uv run python release/fig_label_audit_paris4.py check --band z7000      # blind visual-check panels + separate key
  uv run python release/fig_label_audit_paris4.py overview --band z7000   # verdict map + error clustering

z 7000 figures go to release/figures/label_audit_paris4_*.png. The z 9000 band is the HITL session band: its figures
go to data/scratch/label_audit/paris4/figures/ only (sealed until the session answers are frozen).

Blind check: 8 confidently flagged + 8 confidently ok (slab, 30 deg sector, label pair) cells per band, drawn with
seed 20261024, shown in shuffled order with no verdict. Each panel: a CT slice crop (L2) at the cell's best-covered
level with all label contours (k blue, k2 orange, others grey) and the representative ray, and the stacked CT
profiles of every ray of the cell aligned at the k crossing (k2 crossing marked). The key (cell -> verdict) is
written to a separate json that the reader opens only after recording the counts.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import winding_audit as wa  # noqa: E402

AUD = REPO / "data/scratch/label_audit/paris4"
SEALED = {"z9000", "ext_8500_9000", "ext_10500_10752"}
INK, INK2, BG = "#0b0b0b", "#52514e", "#fcfcfb"
CK, CK2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2})
SEED = 20261024


FIGDIR = None


def figdir(band):
    if FIGDIR:
        Path(FIGDIR).mkdir(parents=True, exist_ok=True)
        return Path(FIGDIR)
    d = AUD / "figures" if band in SEALED else REPO / "release/figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(band):
    A = json.loads((AUD / band / "audit.json").read_text())
    D = dict(np.load(AUD / band / "rays.npz"))
    P = dict(np.load(AUD / band / "pairs_rays.npz"))
    return A, D, P


# ------------------------------------------------------------------------------------------------------------ check
def pick(A, n=8):
    rows = [r for r in A["pairs"] if r["sector"] != "all" and r["conf"] >= 0.8]
    fl = [r for r in rows if r["verdict"] in ("duplicate", "overcount", "untraced", "order")]
    ok = [r for r in rows if r["verdict"] == "ok"]
    rng = np.random.default_rng(SEED)
    sf = [fl[i] for i in rng.choice(len(fl), min(n, len(fl)), replace=False)]
    so = [ok[i] for i in rng.choice(len(ok), min(n, len(ok)), replace=False)]
    cells = [("flagged", r) for r in sf] + [("unflagged", r) for r in so]
    order = rng.permutation(len(cells))
    return [cells[i] for i in order], len(fl), len(ok)


def cell_rays(r, A, D, P):
    edges = np.array(A["slab_edges"])
    s = int(np.searchsorted(edges, r["z"][0], side="right") - 1)
    c = int(round(r["sector"][0] / A["sector_deg"]))
    m = (P["slab"] == s) & (P["sec"] == c) & (P["k"] == r["k"]) & (P["k2"] == r["k2"])
    return np.nonzero(m)[0]


def ray_index(D):
    return {(int(l), int(i)): k for k, (l, i) in enumerate(zip(D["lev"], D["ia"]))}


def contours_at(z):
    pre = wa.PRESETS["PHercParis4"]
    out = []
    for m in wa.iter_meshes_paris4(pre):
        g = m["grid"]
        zz = g[..., 0]
        if not (np.nanmin(zz) <= z <= np.nanmax(zz)):
            continue
        sg = wa.contour_segments(g, np.array([z, z + 1e6]))
        sg = sg[sg[:, 0] == 0]
        if len(sg):
            out.append((m["label"], sg[:, 1:]))
    return out


def cmd_check(a):
    A, D, P = load(a.band)
    cells, nfl, nok = pick(A)
    ri = ray_index(D)
    ct = wa.open_zarr(wa.PRESETS["PHercParis4"]["ct"])
    prof_ct = D["ct"]
    key, panels = [], []
    cont_cache = {}
    rng = np.random.default_rng(SEED + 1)
    for j, (kind, r) in enumerate(cells):
        idx = cell_rays(r, A, D, P)
        i0 = int(rng.choice(idx))                          # one random ray of the cell for the slice crop
        il, ia = int(P["lev"][i0]), int(P["ia"][i0])
        z = float(D["levels"][il])
        uy, ux = D["umb"][il]
        ang = np.radians(D["angles"][ia])
        ra, rb = float(P["ra"][i0]), float(P["rb"][i0])
        rm = (ra + rb) / 2
        cy, cx = uy + rm * np.sin(ang), ux + rm * np.cos(ang)
        H = int(min(160, max(56, 0.75 * abs(rb - ra) + 28)))
        y0, x0 = int(cy) - H, int(cx) - H
        img = np.asarray(ct[int(round(z)), max(y0, 0):y0 + 2 * H, max(x0, 0):x0 + 2 * H]).astype(np.float32)
        if z not in cont_cache:
            cont_cache[z] = contours_at(z)
        # stacked profiles of all rays of the cell, aligned at the k crossing
        W0, W1 = -20.0, float(min(220.0, np.percentile(np.abs(P["gap"][idx]), 90) + 25.0))
        rows = []
        for i in idx[np.argsort(P["lev"][idx] * 1000 + P["ia"][idx])]:
            k = ri[(int(P["lev"][i]), int(P["ia"][i]))]
            t, m7, _ = wa.per_ray(D, int(P["lev"][i]), int(P["ia"][i]), k)
            o, n = D["off"][k], D["n"][k]
            xs = np.arange(W0, W1, wa.STEP)
            rows.append((np.interp(P["ra"][i] + xs, t, prof_ct[o:o + n].astype(np.float32), left=np.nan, right=np.nan),
                         float(P["rb"][i] - P["ra"][i]),
                         np.interp(P["ra"][i] + xs, t, (m7 > 0).astype(np.float32), left=0, right=0) > 0.5, xs))
        panels.append(dict(img=img, y0=max(y0, 0), x0=max(x0, 0), z=z, cy=cy, cx=cx, uy=uy, ux=ux, ang=ang, ra=ra,
                           rb=rb, k=r["k"], k2=r["k2"], rows=rows, W0=W0, W1=W1, cont=cont_cache[z], H=H))
        key.append({"panel": j + 1, "kind": kind, "z": r["z"], "sector": r["sector"], "k": r["k"], "k2": r["k2"],
                    "dk": r["dk"], "verdict": r["verdict"], "T": r["T"], "conf": r["conf"], "T_m7": r.get("T_m7"),
                    "T_ct": r.get("T_ct"), "ct_mode": r.get("ct_mode"), "level": z, "angle": float(D["angles"][ia]),
                    "rays": r["rays"]})
        print(f"panel {j + 1}: z {z:.0f} ang {D['angles'][ia]:.0f} k {r['k']}-{r['k2']} crop {2 * H}", flush=True)
    fd = figdir(a.band)
    per = 4
    for f0 in range(0, len(panels), per):
        fig, ax = plt.subplots(per, 2, figsize=(10.5, 4.6 * per), facecolor=BG,
                               gridspec_kw={"width_ratios": [1.25, 1]})
        for jj, (row, p) in enumerate(zip(ax, panels[f0:f0 + per])):
            a1, a2 = row
            im = p["img"]
            lo, hi = np.percentile(im[im > 0], [1, 99.5]) if (im > 0).any() else (0, 255)
            a1.imshow(im, cmap="gray", vmin=lo, vmax=hi, extent=(p["x0"], p["x0"] + im.shape[1],
                                                                 p["y0"] + im.shape[0], p["y0"]))
            for lab, sg in p["cont"]:
                col, lw = (CK, 1.4) if lab == p["k"] else (CK2, 1.4) if lab == p["k2"] else ("#c8c8c8", 0.5)
                inb = ((sg[:, 0] > p["y0"] - 5) & (sg[:, 0] < p["y0"] + im.shape[0] + 5) &
                       (sg[:, 1] > p["x0"] - 5) & (sg[:, 1] < p["x0"] + im.shape[1] + 5))
                for y_0, x_0, y_1, x_1 in sg[inb]:
                    a1.plot([x_0, x_1], [y_0, y_1], color=col, lw=lw, alpha=0.9)
            t = np.linspace(p["ra"] - 30, p["rb"] + 30, 2)
            a1.plot(p["ux"] + t * np.cos(p["ang"]), p["uy"] + t * np.sin(p["ang"]), color="#e34948", lw=0.6, ls="--")
            a1.set_xlim(p["x0"], p["x0"] + im.shape[1])
            a1.set_ylim(p["y0"] + im.shape[0], p["y0"])
            a1.set_title(f"panel {f0 + jj + 1}: z {p['z']:.0f}, k {p['k']} (blue) -> "
                         f"k {p['k2']} (orange); dashed = ray", fontsize=8)
            a1.tick_params(labelsize=6)
            M = np.array([q[0] for q in p["rows"]])
            for q in range(len(M)):                      # per-row contrast stretch (p2-p98)
                v = M[q][np.isfinite(M[q])]
                if len(v) > 4:
                    l_, h_ = np.percentile(v, [2, 98])
                    M[q] = (M[q] - l_) / max(h_ - l_, 1.0)
            a2.imshow(M, cmap="gray", aspect="auto", interpolation="nearest",
                      extent=(p["W0"], p["W1"], len(M) - 0.5, -0.5))
            for q, row_ in enumerate(p["rows"]):            # m7 runs: green bars under each profile
                mk = np.r_[False, row_[2], False].astype(int)
                st, en = np.nonzero(np.diff(mk) == 1)[0], np.nonzero(np.diff(mk) == -1)[0]
                for s0, e0 in zip(st, en):
                    a2.plot([row_[3][s0], row_[3][e0 - 1]], [q + 0.36, q + 0.36], color="#1baf7a", lw=1.6)
            a2.plot(np.zeros(len(M)), np.arange(len(M)), "|", color=CK, ms=5)
            a2.plot([q[1] for q in p["rows"]], np.arange(len(M)), "|", color=CK2, ms=5)
            a2.set_xlabel("vox outward from the k crossing")
            a2.set_ylabel("ray (cell, all levels)")
            a2.set_title("CT along every ray of the cell (9-line mean); green = m7 runs", fontsize=8)
            a2.tick_params(labelsize=6)
        for row in ax[len(panels[f0:f0 + per]):]:
            for x in row:
                x.axis("off")
        fig.suptitle(f"PHercParis4 label audit, blind check {a.band}: count the CT sheets crossed going from k to k2 "
                     f"(turns). Verdicts hidden.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fn = fd / f"label_audit_paris4_check_{a.band}_{f0 // per + 1}.png"
        fig.savefig(fn, dpi=110, facecolor=BG)
        plt.close(fig)
        print("wrote", fn)
    kp = AUD / a.band / "visual_check_KEY_do_not_open_before_counting.json"
    kp.write_text(json.dumps({"seed": SEED, "n_flagged_pool": nfl, "n_ok_pool": nok, "panels": key}, indent=1))
    print("key ->", kp)


# --------------------------------------------------------------------------------------------------------- overview
def cmd_overview(a):
    A, D, P = load(a.band)
    rows = [r for r in A["pairs"] if r["sector"] != "all"]
    edges = np.array(A["slab_edges"])
    nsec = int(round(360 / A["sector_deg"]))
    ks = sorted({r["k"] for r in rows if r["k2"] - r["k"] == 1})
    code = {"ok": 0, "duplicate": 1, "overcount": 1, "untraced": 2, "order": 3, "ambiguous": 4}
    # per k, per (slab, sector): worst confident verdict of the adjacent pair (k, k+1)
    M = np.full((len(ks), (len(edges) - 1) * nsec), np.nan)
    for r in rows:
        if r["k2"] - r["k"] != 1 or r["k"] not in ks:
            continue
        s = int(np.searchsorted(edges, r["z"][0], side="right") - 1)
        c = int(round(r["sector"][0] / A["sector_deg"]))
        v = code[r["verdict"]] if r["conf"] >= 0.8 else 4
        M[ks.index(r["k"]), s * nsec + c] = v
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#1baf7a", "#e34948", "#2a78d6", "#6250d6", "#d9d8d4"])
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG)
    ax.imshow(M, cmap=cm, vmin=-0.5, vmax=4.5, aspect="auto", interpolation="nearest")
    for s in range(1, len(edges) - 1):
        ax.axvline(s * nsec - 0.5, color=INK, lw=0.6)
    ax.set_xticks([s * nsec + nsec / 2 - 0.5 for s in range(len(edges) - 1)],
                  [f"z {int(edges[s])}" for s in range(len(edges) - 1)], fontsize=7)
    yt = [i for i, k in enumerate(ks) if k % 5 == 0]
    ax.set_yticks(yt, [f"k{ks[i]}-{ks[i] + 1}" for i in yt], fontsize=6.5)
    ax.set_xlabel("slab (500 slices), 12 x 30 deg sectors each (0 deg = +x, increasing outward)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cm.colors, [
        "CT-consistent", "overcounted (duplicate: fewer CT sheets than labels)", "undercounted (untraced CT sheet between labels)", "order",
        "ambiguous / conf < 0.8"])], loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=7, frameon=False)
    ax.set_title(f"PHercParis4 organiser labels (w0701), {a.band}: combined CT+m7 verdict of each adjacent label pair "
                 f"per (slab, sector) cell", fontsize=9)
    fig.tight_layout()
    fn = figdir(a.band) / f"label_audit_paris4_overview_{a.band}.png"
    fig.savefig(fn, dpi=120, facecolor=BG)
    print("wrote", fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "overview"])
    ap.add_argument("--band", required=True)
    ap.add_argument("--figdir", help="override the output directory (tests)")
    a = ap.parse_args()
    global FIGDIR
    FIGDIR = a.figdir
    {"check": cmd_check, "overview": cmd_overview}[a.cmd](a)


if __name__ == "__main__":
    main()
