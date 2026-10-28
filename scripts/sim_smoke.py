"""Smoke test for sim/scroll.py: table smoothness stats + preview PNG (data/scratch/sim_smoke.png)."""

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
from scroll import N_THETA, Scroll, ScrollParams  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
t0 = time.time()
S = Scroll(ScrollParams(seed=seed))
print(f"tables {time.time() - t0:.1f}s contact frac {S.contact_area_frac:.3f}")
j = S.nz // 2
g, R = S.g.cpu(), S.R.cpu()
for k in [5, 20, 60]:
    dg = (g[k, 1:N_THETA, j] - g[k, : N_THETA - 1, j]).abs().max()
    dR = (R[k, 1:N_THETA, j] - R[k, : N_THETA - 1, j]).abs().max()
    print(f"k {k}: R {R[k, :, j].min():.0f}-{R[k, :, j].max():.0f} max|dg| {dg:.2f} max|dR| {dR:.2f} (arc step {R[k, 0, j] * 6.283 / N_THETA:.0f} um)")

dev = "cuda"
# centre the zoom on a contact zone at mid radius (and on the theta = 0 seam if --seam)
if "--seam" in sys.argv:
    c = torch.tensor([6000.0, 0.0, 0.0], device=dev)
else:
    cand = torch.stack([torch.linspace(-9000, 9000, 4001, device=dev), torch.full((4001,), -3000.0, device=dev),
                        torch.zeros(4001, device=dev)], -1)
    o = S.sample(cand, want_density=False)
    hit = torch.nonzero((o["instance"] > 0) & (o["gap_min"] == 0))[:, 0]
    c = cand[hit[len(hit) // 2]] if len(hit) else cand[0]
print("zoom centre", c.tolist())
S.set_reference(c)
m = 420
f = torch.arange(m, device=dev) * 2.4 - m * 1.2
X, Y = torch.meshgrid(f, f, indexing="ij")
P2 = torch.stack([X.flatten() + c[0], Y.flatten() + c[1], torch.zeros(m * m, device=dev)], -1)
o2 = S.sample(P2)
d2 = o2["density"].reshape(m, m).cpu().numpy()
i2 = o2["instance"].reshape(m, m).cpu().numpy()
g2 = o2["gap_min"].reshape(m, m).cpu().numpy()
n = 1600
xs = torch.linspace(-16000, 16000, n, device=dev)
X, Y = torch.meshgrid(xs, xs, indexing="ij")
inst = S.sample(torch.stack([X.flatten(), Y.flatten(), torch.zeros(n * n, device=dev)], -1), want_density=False)
inst = inst["instance"].reshape(n, n).cpu().numpy()
fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].imshow(np.where(inst > 0, inst % 7 + 1, 0), cmap="tab10", interpolation="nearest")
ax[1].imshow(d2, cmap="gray")
ax[2].imshow(np.where(i2 > 0, i2 % 7 + 1, 0), cmap="tab10", interpolation="nearest")
ax[3].imshow(np.where(i2 > 0, np.minimum(g2, 100), np.nan), cmap="magma")
for a, t in zip(ax, ["instance mod 7 (32 mm)", "density 2.4 um (1 mm)", "instance zoom", "gap um (0 = contact)"]):
    a.set_title(t)
    a.axis("off")
fig.tight_layout()
out = REPO / "data" / "scratch" / "sim_smoke.png"
fig.savefig(out, dpi=60)
print(out)
