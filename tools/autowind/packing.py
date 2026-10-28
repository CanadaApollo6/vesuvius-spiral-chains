"""Label-free sheet-packing statistic S_pack and the `--raise auto` count rule.

S_pack ("median crossing gap") is the statistic registered in the 2026-10-23 Paris4 draft
(data/scratch/third_region/packing_stat.py, sha 594863ce...), ported here unchanged in logic. Input: the solved crop
checkpoints of one run (the planned crops only). Each holds, per probe ray (z, theta), the mask crossings of
`ray_crossings`: runs of the 1-vox-dilated mask along the radial ray, merged across gaps < 2.5 vox, truncated runs
dropped, grazing segments skipped. No label, patch winding, confidence model or grad_mag is used.
  1. Per ray, deduplicate crossings across crops as `assemble` does (cluster by radius <= 2 vox, keep the copy from
     the crop whose segment centre is nearest).
  2. Gaps = radius differences between consecutive deduplicated crossings that one common crop segment holds.
  3. S_pack = median of all gaps pooled over the band, in voxels of the mask's working frame.

Rule (`--raise auto`, notes/2026-10-23-fourth-test-prereg.md): run the grad_mag raise at threshold 0.263 iff
S_pack < 21.0 vox; otherwise no raise.
  - 21.0 = log-midpoint of the PHerc0139 (18.5) and PHerc1667 (24.0) values, fixed in the Paris4 draft.
  - 0.263 = the PHerc0139 zero-drift threshold against CT-corrected counts (notes/2026-10-22-hitl-budget.md section 3),
    fixed on 2026-10-22 before any Paris4 data existed.
"""

from __future__ import annotations

import numpy as np

SPACK_THR = 21.0
AUTO_GM_THR = 0.263


def ray_gaps(results) -> tuple[np.ndarray, int]:
    """results: [(crop dict, per-z segment results)] as loaded from the crop checkpoints.
    Returns rows (z, theta, r_mid, gap) and the number of probe rays."""
    rays: dict = {}
    for c, res in results:
        if not res:
            continue
        for z, seg in res.items():
            rays.setdefault((int(z), float(c["theta"])), []).append((seg["tc"], [x["t"] for x in seg["cr"]]))
    rows = []
    for (z, th), lst in rays.items():
        allx = sorted([(t, abs(t - tc)) for tc, ts in lst for t in ts])
        own, cl = [], []
        for v in allx + [(1e18, 0)]:
            if cl and v[0] - cl[-1][0] > 2:
                own.append(min(cl, key=lambda q: q[1])[0])
                cl = []
            cl.append(v)
        for a, b in zip(own[:-1], own[1:]):
            if any(any(abs(t - a) <= 2 for t in ts) and any(abs(t - b) <= 2 for t in ts) for _, ts in lst):
                rows.append((z, th, (a + b) / 2, b - a))
    return np.array(rows, float).reshape(-1, 4), len(rays)


def s_pack(results, z0: float, z1: float, nboot: int = 1000) -> dict:
    """S_pack (median gap) with a 30 deg x 250 z block bootstrap (seed 1) and 250-z slab medians, for the record."""
    R, nrays = ray_gaps(results)
    if len(R) == 0:
        return dict(S_pack=None, n=0, rays=nrays)
    g = R[:, 3]
    blk = (np.floor((R[:, 0] - z0) / 250) * 100 + np.floor(R[:, 1] / 30)).astype(int)
    ub = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in ub}
    rng = np.random.default_rng(1)
    bs = [np.median(g[np.concatenate([idx[b] for b in rng.choice(ub, len(ub), replace=True)])]) for _ in range(nboot)]
    slab = {}
    for s in range(int(z0), int(z1), 250):
        m = (R[:, 0] >= s) & (R[:, 0] < s + 250)
        if m.sum() > 20:
            slab[f"{s}-{s + 250}"] = round(float(np.median(g[m])), 2)
    return dict(S_pack=round(float(np.median(g)), 2), n=int(len(g)), rays=nrays,
                boot95=[round(float(x), 2) for x in np.percentile(bs, [2.5, 97.5])], slab_medians=slab)


def decide(sp: float | None, spack_thr: float = SPACK_THR) -> str:
    """'gm' iff S_pack < spack_thr (tight packing: the mask merges contacts, chains under-count), else 'off'."""
    if sp is None:
        raise RuntimeError("--raise auto: no crossing gaps in the solved crops; cannot compute S_pack")
    return "gm" if sp < spack_thr else "off"
