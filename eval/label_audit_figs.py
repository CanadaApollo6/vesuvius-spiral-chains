"""Diagnostic figures of the PHerc0139 label CT audit (eval/ct_gap_audit.py outputs) -> data/scratch/label_audit/*.png.

  uv run python eval/label_audit_figs.py
Human meshes, CT (one streamed L0 crop), m7 and grad_mag profiles only.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
import ct_gap_audit as ca  # noqa: E402
import label_subsets as ls  # noqa: E402
import spiral_eval as se  # noqa: E402

OUT = REPO / "data" / "scratch" / "label_audit"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
KEY = {40: (C[0], "w040"), 41: (C[1], "w041 = w042"), 42: (C[1], None), 43: (C[2], "w043"),
       45: (C[4], "w045 = w046"), 46: (C[4], None), 47: (C[6], "w047")}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def slice_and_profiles(z=3636, ang=300.5, half=150):
    import zarr
    umb = se.Umbilicus()
    H = ls.load_band(1)
    m = np.abs(H["pts"][:, 0] - z) <= 1.5
    p, k = H["pts"][m], H["k"][m]
    uy, ux = np.interp(z, umb.z, umb.y), np.interp(z, umb.z, umb.x)
    D = dict(np.load(OUT / "ct_profiles_band1.npz"))
    iz, ia = list(D["zs"]).index(z), int(np.argmin(np.abs(D["angles"] - ang)))
    ang = float(D["angles"][ia])
    R = {int(l): D["R"][iz, ia, i] for i, l in enumerate(D["labels"])}
    rmid = (R[39] + R[48]) / 2
    cy, cx = uy + rmid * np.sin(np.radians(ang)), ux + rmid * np.cos(np.radians(ang))
    y0, x0 = int(cy - half), int(cx - half)
    ct = np.asarray(zarr.open_array(ca.CT, mode="r", storage_options={"anon": True})[z, y0:y0 + 2 * half, x0:x0 + 2 * half])

    fig = plt.figure(figsize=(10.5, 4.6), facecolor="#fcfcfb")
    ax = fig.add_axes([0.02, 0.08, 0.40, 0.84])
    ax.imshow(ct, cmap="gray", vmin=np.percentile(ct, 1), vmax=np.percentile(ct, 99.5))
    sel = (p[:, 1] >= y0) & (p[:, 1] < y0 + 2 * half) & (p[:, 2] >= x0) & (p[:, 2] < x0 + 2 * half)
    for lab in sorted(set(k[sel].tolist())):
        q = p[sel & (k == lab)]
        col, name = KEY.get(lab, ("#c3c2b7", None))
        ax.scatter(q[:, 2] - x0, q[:, 1] - y0, s=1.5 if lab in KEY else 0.6, c=col, lw=0,
                   label=name, zorder=3 if lab in KEY else 2)
    t = np.array([R[39] - 10, R[48] + 10])
    ax.plot(ux + t * np.cos(np.radians(ang)) - x0, uy + t * np.sin(np.radians(ang)) - y0, color="white", lw=0.8,
            ls="--", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"CT z {z}, ray {ang:.1f} deg (dashed); grey dots = other human meshes", color=INK, fontsize=8)
    lg = ax.legend(loc="lower left", fontsize=7, markerscale=5, frameon=True)
    lg.get_frame().set_alpha(0.85)

    lo, hi = R[39] - 8, R[48] + 8
    pf = ca.profile(D, iz, ia, lo, hi)
    tt, c, mk, g = pf
    axs = [fig.add_axes([0.50, 0.62 - 0.27 * i, 0.47, 0.22]) for i in range(3)]
    axs[0].plot(tt, c, color=INK2, lw=0.8)
    axs[0].set_ylabel("CT (9-line mean)")
    axs[1].fill_between(tt, 0, (mk > 0).astype(float), step="mid", color=C[0], lw=0)
    axs[1].set_ylabel("m7 sheet"); axs[1].set_yticks([])
    axs[2].plot(tt, g / 1000.0, color=C[0], lw=1.2)
    axs[2].set_ylabel("grad_mag (turns/vox)")
    axs[2].set_xlabel("radius from umbilicus (vox)")
    for a in axs:
        a.set_xlim(lo, hi)
        for lab, r in R.items():
            if np.isfinite(r) and lo < r < hi:
                col = KEY.get(lab, ("#a3a29b", None))[0]
                a.axvline(r, color=col, lw=1.0 if lab in KEY else 0.6, ls="-" if lab in KEY else ":")
    for a in axs[:2]:
        a.set_xticklabels([])
    ymax = axs[0].get_ylim()[1]
    for lab, r in R.items():
        if np.isfinite(r) and lo < r < hi and lab not in (42, 46):
            axs[0].text(r, ymax, f"{lab}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    rows = json.loads((OUT / "ct_counts_band1.json").read_text())["rows"]
    got = {tuple(r["pair"]): r for r in rows if r["z"] == z and abs(r["ang"] - ang) < 0.01}
    txt = "\n".join(f"w{a:03d}-w{b:03d}: labels {b - a}, m7 sheets between {got[(a, b)]['n_m7']}, "
                    f"gm {got[(a, b)]['gm_turns']:.1f} turns" for a, b in [(40, 43), (46, 47)] if (a, b) in got)
    fig.text(0.50, 0.935, txt, fontsize=7.5, color=INK)
    axs[0].set_title("m7 runs within 4 vox of a mesh count as that mesh's sheet", fontsize=7, color=INK2, loc="right", pad=10)
    fig.savefig(OUT / "fig_ct_gap_ray.png", dpi=150)
    plt.close(fig)


def local_turns(b, test, ctrls):
    rows = json.loads((OUT / f"ct_counts_band{b}.json").read_text())["rows"]
    idx = {}
    for r in rows:
        idx.setdefault(tuple(r["pair"]), {})[(r["z"], round(r["ang"], 3))] = r
    g, m = [], []
    for key, r in idx.get(tuple(test), {}).items():
        cs = [(idx[c].get(key), c[1] - c[0]) for c in ctrls]
        if any(c is None or not c["gm"] for c, _ in cs) or not r["gm"]:
            continue
        n = sum(x for _, x in cs)
        g.append(r["gm"] / (sum(c["gm"] for c, _ in cs) / n))
        m.append((r["n_m7"] + 1) / (sum(c["n_m7"] + 1 for c, _ in cs) / n))
    g, m = np.array(g), np.array(m)
    bs = np.random.default_rng(0).integers(0, len(g), (2000, len(g)))
    ci = lambda x: np.percentile(np.median(x[bs], 1), [2.5, 97.5])
    return np.median(g), ci(g), np.median(m), ci(m), len(g)


def counts_figure():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="#fcfcfb", gridspec_kw={"width_ratios": [1, 1.25]})
    rows = json.loads((OUT / "ct_counts_band1.json").read_text())["rows"]
    groups = [("dk=1 controls (expect 1)", lambda r: r["kind"] == "c1"),
              ("dk=2 controls (expect 2)", lambda r: r["kind"] == "c2"),
              ("dk=3 controls (expect 3)", lambda r: r["kind"] == "c3"),
              ("w040-w043 (labels 3)", lambda r: r["pair"] == [40, 43]),
              ("w046-w047 (labels 1)", lambda r: r["pair"] == [46, 47])]
    cols = [C[0], C[1], C[2], C[3], C[4]]
    w = 0.16
    for i, (name, f) in enumerate(groups):
        v = np.array([r["n_m7"] + 1 for r in rows if f(r)])
        x = np.arange(1, 6)
        y = [(v == t).mean() for t in x]
        a1.bar(x + (i - 2) * w, y, width=w - 0.02, color=cols[i], label=f"{name}, n={len(v)}")
    a1.set_xticks(range(1, 6)); a1.set_xlabel("turns between the two meshes (m7 sheets between + 1)")
    a1.set_ylabel("share of rays"); a1.grid(axis="y", color=GRID, lw=0.5); a1.set_axisbelow(True)
    a1.legend(fontsize=6.5, frameon=False); a1.set_title("Band 1, 24 angles x 6 z, seams avoided", color=INK)

    spans = [(1, (29, 33), [(23, 33), (52, 59)], "b1 w029-w033 (ctrl)"),
             (1, (40, 43), [(23, 33), (52, 59)], "b1 w040-w043"),
             (1, (46, 47), [(23, 33), (52, 59)], "b1 w046-w047"),
             (1, (43, 47), [(23, 33), (52, 59)], "b1 w043-w047"),
             (1, (40, 47), [(23, 33), (52, 59)], "b1 w040-w047"),
             (1, (33, 40), [(23, 33), (52, 59)], "b1 w033-w040"),
             (1, (33, 52), [(23, 33), (52, 59)], "b1 w033-w052"),
             (2, (29, 33), [(23, 33), (51, 59)], "b2 w029-w033 (ctrl)"),
             (2, (34, 51), [(23, 33), (51, 59)], "b2 w034-w051"),
             (2, (33, 51), [(23, 33), (51, 59)], "b2 w033-w051")]
    for i, (b, sp, ct, name) in enumerate(spans):
        g, gci, m, mci, n = local_turns(b, sp, ct)
        lab = sp[1] - sp[0]
        y = len(spans) - i
        a2.plot([gci[0] - lab, gci[1] - lab], [y + 0.12] * 2, color=C[0], lw=2)
        a2.plot(g - lab, y + 0.12, "o", color=C[0], ms=4, label="grad_mag" if i == 0 else None)
        a2.plot([mci[0] - lab, mci[1] - lab], [y - 0.12] * 2, color=C[1], lw=2)
        a2.plot(m - lab, y - 0.12, "s", color=C[1], ms=4, label="m7 count" if i == 0 else None)
        a2.text(-4.35, y, f"{name} (labels {lab}, n={n})", va="center", fontsize=7, color=INK)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlim(-4.4, 2.0); a2.set_yticks([])
    a2.set_xlabel("CT turns - label difference (median, 95 % ray bootstrap; per-ray calibrated)")
    a2.grid(axis="x", color=GRID, lw=0.5); a2.set_axisbelow(True)
    a2.legend(fontsize=7, frameon=False, loc="lower right")
    a2.set_title("Counted vs labelled turns (0 = labels right)", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ct_turn_counts.png", dpi=150)
    plt.close(fig)


def sector_figure():
    """w033-w037 turns by angle (dense 140-270 deg rays + the regular set), band 1."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor="#fcfcfb")
    for tag, mk, col in (("", "o", C[0]), ("_s2167", "s", C[1])):
        rows = json.loads((OUT / f"ct_counts_band1{tag}.json").read_text())["rows"]
        for pair, fill in (((33, 37), True), ((34, 36), False)):
            rr = [r for r in rows if r["pair"] == list(pair) and r["gm_turns"] is not None]
            angs = sorted(set(round(r["ang"], 1) for r in rr))
            med = [np.median([r["gm_turns"] for r in rr if round(r["ang"], 1) == a]) for a in angs]
            ax.plot(angs, med, mk, ms=4, color=col, mfc=col if fill else "white", lw=0,
                    label=f"w0{pair[0]}-w0{pair[1]} (labels {pair[1] - pair[0]}), {'dense 140-270' if tag else 'regular'} rays")
    ax.axvspan(145, 260, color="#f0efec", zorder=0)
    ax.text(202, 4.6, "#2167 sector (w035 missing)", ha="center", fontsize=7, color=INK2)
    ax.axhline(4, color=GRID, lw=0.8); ax.axhline(2, color=GRID, lw=0.8)
    ax.set_xlabel("angle around the umbilicus (deg)"); ax.set_ylabel("grad_mag turns (median over 6 z)")
    ax.set_ylim(0, 5); ax.set_xlim(0, 360)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Band 1: inside 145-260 deg w036 sits on w034 and w033-w037 holds ~1.5 turns, not 4", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ct_sector2167.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    slice_and_profiles()
    counts_figure()
    sector_figure()
    print("figures in", OUT)
