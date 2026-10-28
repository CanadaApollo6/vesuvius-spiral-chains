"""Figures of the CT audit of the public PHerc0139 winding labels (eval/winding_audit.py output).

  uv run python release/fig_label_audit.py                       # all figures -> release/figures/label_audit_*.png
  uv run python release/fig_label_audit.py --only overview validation

Inputs: data/scratch/label_audit/full/{audit.json, rays.npz, pairs_rays.npz} and
data/scratch/label_audit/p0139_label_corrections.json. The spot checks stream one CT slice crop (L0, 9.362 um) each and
draw the human meshes' contours at that z (2.399 tifxyz -> published affine).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import winding_audit as wa  # noqa: E402

AUD = REPO / "data/scratch/label_audit/full"
CORR = REPO / "data/scratch/label_audit/p0139_label_corrections.json"
FIG = REPO / "release/figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

# spot checks: (file tag, z level, ray angle, focus labels, title)
SPOTS = [
    ("dup_w041_w042", 3624, 300.0, [40, 41, 42, 43], "w041 and w042 trace the same sheet (z 3250-7250)"),
    ("dup_untraced_w045_w047", 3816, 0.0, [44, 45, 46, 47, 48],
     "w045 = w046 (same sheet), then an untraced sheet before w047"),
    ("sector2167", 3960, 190.0, [33, 34, 36, 37],
     "#2167 sector: w034 -> w036 should be 1 turn here; CT shows 0"),
    ("bridge_w050", 3816, 15.0, [48, 49, 51, 52],
     "w050 is a partial bridge: where absent, w049 -> w051 is 1 turn"),
]


def load():
    A = json.loads((AUD / "audit.json").read_text())
    P = dict(np.load(AUD / "pairs_rays.npz"))
    return A, P


# ---------------------------------------------------------------------------------------------------------- overview
def overview(A, P, corr):
    rows = [r for r in A["pairs"] if r["sector"] == "all"]
    slabs = sorted({tuple(r["z"]) for r in rows})
    keys = sorted({(r["k"], r["k2"]) for r in rows})
    keys = [k for k in keys if sum((r["k"], r["k2"]) == k for r in rows) >= 2 or k[1] - k[0] == 1]
    M = np.full((len(keys), len(slabs)), np.nan)
    V = np.full((len(keys), len(slabs)), "", object)
    sym = {"ok": "", "duplicate": "D", "untraced": "U", "overcount": "O", "order": "X", "ambiguous": "?"}
    for r in rows:
        if (r["k"], r["k2"]) not in keys:
            continue
        i, j = keys.index((r["k"], r["k2"])), slabs.index(tuple(r["z"]))
        M[i, j] = r["m7_mean"] - r["dk"]
        V[i, j] = sym[r["verdict"]] + ("" if r["conf"] >= 0.8 or r["verdict"] == "ok" else "*")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 7.4), facecolor=BG, gridspec_kw={"width_ratios": [1.15, 1]})
    im = a1.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.5, 1.5), aspect="auto", interpolation="nearest")
    for i in range(len(keys)):
        for j in range(len(slabs)):
            if V[i, j]:
                a1.text(j, i, V[i, j], ha="center", va="center", fontsize=6.5, color=INK, fontweight="bold")
    a1.set_yticks(range(len(keys)), [f"w{k:03d}-w{k2:03d}" for k, k2 in keys], fontsize=6)
    a1.set_xticks(range(len(slabs)), [f"{int(z0)}" for z0, _ in slabs], rotation=90, fontsize=6.5)
    a1.set_xlabel("z slab start (500 slices)")
    a1.set_title("Label-adjacent pairs: m7 turns - label difference\nD duplicate, U untraced sheet, O labels skip "
                 "a sheet, ? detectors disagree (* conf < 0.8)", fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=a1, shrink=0.5, pad=0.02)
    cb.set_label("turns - labels (mean over rays)", fontsize=7)
    # pooled shift per label and slab
    cells = {tuple(c["z"]): c for c in corr["cells"] if c["sector"] == "all"}
    labs = sorted({int(k) for c in cells.values() for k in c["map"]})
    S = np.full((len(labs), len(slabs)), np.nan)
    T = np.zeros(S.shape, bool)
    for j, z in enumerate(slabs):
        for k, m in cells.get(z, {"map": {}})["map"].items():
            i = labs.index(int(k))
            S[i, j] = m["shift"]
            T[i, j] = m["conf"] >= 0.8 and m.get("x_gm_int") == m["new"]
    im2 = a2.imshow(S, cmap="PuOr", norm=TwoSlopeNorm(0, -4, 4), aspect="auto", interpolation="nearest")
    for i in range(len(labs)):
        for j in range(len(slabs)):
            if np.isfinite(S[i, j]):
                a2.text(j, i, f"{int(S[i, j]):+d}" if S[i, j] else "0", ha="center", va="center", fontsize=5.5,
                        color=INK if T[i, j] else "#9a9993")
    a2.set_yticks(range(len(labs)), [f"w{k:03d}" for k in labs], fontsize=6)
    a2.set_xticks(range(len(slabs)), [f"{int(z0)}" for z0, _ in slabs], rotation=90, fontsize=6.5)
    a2.set_xlabel("z slab start (500 slices)")
    a2.set_title(f"CT-consistent label - public label (anchor w{corr['anchor_label']:03d}, all angles,\nseams "
                 "excluded; grey = not trusted)", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "label_audit_overview.png", dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- validation
def validation(A, P):
    v = A["validation"]
    ctrl = set()
    for r in v["rows"]:
        ctrl.add((r["z"][0], r["k"]))
    edges = np.array(A["slab_edges"])
    zs = edges[P["slab"]]
    m = np.array([(z, k) in ctrl for z, k in zip(zs, P["k"])]) & (P["dk"] == 1) & ~P["seam"] & (P["half"] == 1)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.4), facecolor=BG)
    T, G = P["T"][m], P["Tg"][m]
    x = np.arange(-1, 5)
    a1.bar(x, [(T == t).mean() for t in x], color=C[0], width=0.7)
    a1.set_xticks(x); a1.set_xlabel("m7 turns per ray"); a1.set_ylabel("share of rays")
    a1.set_title(f"Held-out label-clean controls (n={m.sum():,} rays)\nmean {T.mean():.3f}", fontsize=8, color=INK)
    a2.hist(G[np.isfinite(G)], bins=np.arange(-0.5, 3.5, 0.05), color=C[1])
    a2.axvline(1, color=INK2, lw=0.8)
    a2.set_xlabel("lasagna grad_mag turns per ray"); a2.set_yticks([])
    a2.set_title(f"same rays: median {np.nanmedian(G):.3f}", fontsize=8, color=INK)
    # per-control verdict estimates
    rows = v["rows"]
    a3.scatter([r["m7_mean"] for r in rows], [r["gm_med"] for r in rows], s=8, c=[C[7] if r["flagged"] else C[2]
                                                                                   for r in rows], lw=0)
    a3.axvline(1, color=GRID, lw=0.8); a3.axhline(1, color=GRID, lw=0.8)
    a3.set_xlabel("m7 mean turns (held-out half)"); a3.set_ylabel("gm median turns")
    a3.set_title(f"{len(rows)} control (slab, pair) cells: {v['false_flags']} flagged "
                 f"({100 * (v['false_flag_rate'] or 0):.1f} %)", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "label_audit_validation.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ spots
def mesh_contours(zs):
    """{z: [(label, segs (m, 4) y0 x0 y1 x1)]} for the PHerc0139 human meshes."""
    out = {z: [] for z in zs}
    for m in wa.iter_meshes_p0139():
        for z in zs:
            s = wa.contour_segments(m["grid"], np.array([z, z + 1e5]))
            if len(s):
                out[z].append((m["label"], s[:, 1:]))
    return out


def spot(tag, z, ang, focus, title, D, A, P, cont, ct):
    il = int(np.argmin(np.abs(D["levels"] - z)))
    ia = int(np.argmin(np.abs(D["angles"] - ang)))
    z, ang = int(D["levels"][il]), float(D["angles"][ia])
    k = np.nonzero((D["lev"] == il) & (D["ia"] == ia))[0]
    if not len(k):
        print(f"{tag}: ray not stored"); return
    k = int(k[0])
    labels = D["labels"]
    Rr = {int(l): float(D["R"][il, ia, j]) for j, l in enumerate(labels) if np.isfinite(D["R"][il, ia, j])}
    rf = [Rr[f] for f in focus if f in Rr]
    lo, hi = min(rf) - 25, max(rf) + 25
    uy, ux = D["umb"][il]
    half = int(max(70, (hi - lo) / 2 + 40))
    rm = (lo + hi) / 2
    cy, cx = uy + rm * np.sin(np.radians(ang)), ux + rm * np.cos(np.radians(ang))
    y0, x0 = int(cy - half), int(cx - half)
    img = np.asarray(ct[z, y0:y0 + 2 * half, x0:x0 + 2 * half]).astype(np.float32)
    fig = plt.figure(figsize=(10, 4.3), facecolor=BG)
    ax = fig.add_axes([0.01, 0.06, 0.40, 0.84])
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    col = {f: C[i % len(C)] for i, f in enumerate(focus)}
    for lab, s in cont[z]:
        sel = ((s[:, 0] >= y0) & (s[:, 0] < y0 + 2 * half) & (s[:, 1] >= x0) & (s[:, 1] < x0 + 2 * half))
        if not sel.any():
            continue
        q = s[sel]
        segs = np.stack([q[:, [1, 0]] - [x0, y0], q[:, [3, 2]] - [x0, y0]], 1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors=col.get(lab, "#c3c2b7"), linewidths=1.3 if lab in col else 0.6,
                                         alpha=0.9 if lab in col else 0.6, zorder=3 if lab in col else 2))
    for f in focus:
        if f in Rr:
            ax.plot([], [], color=col[f], lw=2, label=f"w{f:03d}")
    t = np.array([lo, hi])
    ax.plot(ux + t * np.cos(np.radians(ang)) - x0, uy + t * np.sin(np.radians(ang)) - y0, color="white", lw=0.8,
            ls="--", zorder=4)
    ax.set_xlim(0, 2 * half); ax.set_ylim(2 * half, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.85)
    ax.set_title(f"CT z {z}, {2 * half} vox crop; ray {ang:.0f} deg (dashed); grey = other meshes",
                 fontsize=7.5, color=INK)
    # profiles
    tt, m7, g = wa.per_ray(D, il, ia, k)
    sel = (tt >= lo) & (tt <= hi)
    tt, m7, g = tt[sel], m7[sel], g[sel]
    py = uy + tt * np.sin(np.radians(ang)) - y0
    px = ux + tt * np.cos(np.radians(ang)) - x0
    c = ndi.map_coordinates(img, [py, px], order=1)
    axs = [fig.add_axes([0.48, 0.555 - 0.235 * i, 0.50, 0.19]) for i in range(3)]
    axs[0].plot(tt, c, color=INK2, lw=0.8); axs[0].set_ylabel("CT")
    axs[1].fill_between(tt, 0, m7.astype(float), step="mid", color=C[0], lw=0)
    axs[1].set_ylabel("m7 sheet"); axs[1].set_yticks([])
    axs[2].plot(tt, g / 1000.0, color=C[1], lw=1.1); axs[2].set_ylabel("grad_mag")
    axs[2].set_xlabel("radius from umbilicus (vox)")
    for a in axs:
        a.set_xlim(lo, hi)
        for lab, r in Rr.items():
            if lo < r < hi:
                a.axvline(r, color=col.get(lab, "#a3a29b"), lw=1.0 if lab in col else 0.6,
                          ls="-" if lab in col else ":")
    for a in axs[:2]:
        a.set_xticklabels([])
    seen = []
    for lab, r in sorted(Rr.items(), key=lambda x: x[1]):
        if lo < r < hi:
            dy = 0 if not seen or r - seen[-1] > 6 else 1
            axs[0].text(r + 0.6, 1.0 - 0.13 * dy, f"w{lab:03d}", ha="left", va="top", fontsize=6,
                        color=col.get(lab, INK2), transform=axs[0].get_xaxis_transform())
            seen.append(r)
    # this ray's counts + the audit verdicts for the focus pairs in this slab
    # verdict of the ray's (slab, 30 deg sector) cell
    rows = [r for r in A["pairs"] if r["sector"] != "all" and r["z"][0] <= z < r["z"][1]
            and r["sector"][0] <= ang < r["sector"][1]]
    lines = []
    for a_, b_ in zip(focus[:-1], focus[1:]):
        r = next((r for r in rows if (r["k"], r["k2"]) == (a_, b_)), None)
        mm = (P["lev"] == il) & (P["ia"] == ia) & (P["k"] == a_) & (P["k2"] == b_)
        if mm.any():
            j = np.nonzero(mm)[0][0]
            off = int(P["E"][j] - P["dk"][j])
            this = f"ray: expected {int(P['E'][j])}, m7 {int(P['T_raw'][j])}, gm {P['Tg'][j] + off:.1f}"
        else:
            this = "not adjacent on this ray"
        if r:
            gm_ex = "-" if r["gm_med"] is None else f"{r['gm_med'] - r['dk']:+.2f}"
            lines.append(f"w{a_:03d}-w{b_:03d}  {this}  |  cell z{int(r['z'][0])}-{int(r['z'][1])} "
                         f"{int(r['sector'][0])}-{int(r['sector'][1])} deg, excess "
                         f"m7 {r['m7_mean'] - r['dk']:+.2f}, gm {gm_ex} -> {r['verdict']} ({r['conf']:.2f})")
    fig.text(0.48, 0.985, title, fontsize=9, color=INK, va="top", fontweight="bold")
    fig.text(0.48, 0.935, "\n".join(lines), fontsize=6.0, color=INK, va="top", linespacing=1.35)
    fig.savefig(FIG / f"label_audit_{tag}.png", dpi=120)
    plt.close(fig)
    print(f"label_audit_{tag}.png: z {z}, ang {ang}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="overview validation spots")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    A, P = load()
    want = set(a.only or ["overview", "validation", "spots"])
    if "overview" in want:
        overview(A, P, json.loads(CORR.read_text()))
    if "validation" in want:
        validation(A, P)
    if "spots" in want:
        D = dict(np.load(AUD / "rays.npz"))
        ct = wa.open_zarr(wa.PRESETS["PHerc0139"]["ct"])
        zs = sorted({int(D["levels"][int(np.argmin(np.abs(D["levels"] - s[1])))]) for s in SPOTS})
        cont = mesh_contours(zs)
        for s in SPOTS:
            spot(*s, D, A, P, cont, ct)


if __name__ == "__main__":
    main()
