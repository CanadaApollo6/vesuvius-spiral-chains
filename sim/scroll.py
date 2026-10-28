"""Implicit synthetic scroll: exact sheet identity through contacts.

Material frame: cylinder axis z, spiral wraps defined by the recurrence
    R(phi + 2pi, z) = R(phi, z) + t(phi, z) + g(phi, z)
where t is sheet thickness and g >= 0 the gap to the next wrap outward. Wrap k
occupies r in [R_k, R_k + t_k). Contacts are exactly where g == 0, so sheet
identity through a contact is analytic. Continuity across every seam follows
from continuous t, g and a linear first wrap.

Image -> material map: patch voxel -> world (um) -> undo elliptical squash and
rotation -> smooth warp X + w(X) with Lipschitz < 0.5 (invertible, so sheets
never interpenetrate) -> cylindrical (r, theta, z) with a small twist.

All lengths in um. Everything is seeded; ScrollParams fully determines a scroll.
"""

import math
from dataclasses import asdict, dataclass, field

import torch

# RFF phases reach 1e3-1e4 rad; TF32 matmuls would scramble them.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TWO_PI = 2.0 * math.pi
N_THETA = 1440  # table samples per wrap


@dataclass
class ScrollParams:
    seed: int = 0
    r0: float = 1500.0  # core radius
    n_wraps: int = 80
    # sheet thickness and gap, driven by smooth 3D fields over nominal physical position, so neighbouring wraps are
    # correlated (compressed zones span several wraps) and R stays smooth in theta.
    t_mean: float = 85.0
    t_rel_sd: float = 0.25
    g_mean: float = 70.0  # gap at nominal pitch (nominal pitch = t_mean + g_mean)
    noise_corr_um: float = 600.0  # Gaussian correlation length of the thickness field
    # wrap placement: target radius T_k = nominal spiral + D(x); R_{k+1} = max(R_k + t_k, T_{k+1}). Radial compression
    # of D brings wraps into exact contact; its amplitude is bisected so the contact area fraction hits contact_frac.
    disp_corr_radial_um: float = 500.0
    disp_corr_tangential_um: float = 2500.0
    contact_frac: float = 0.15
    # global deformation
    squash: tuple = (1.0, 0.62)  # cross-section scale (x, y) before rotation
    squash_angle: float = 0.0  # radians
    twist_per_mm: float = 0.02  # radians per mm along z
    warp_amp_um: float = 120.0
    warp_corr_um: float = 3000.0
    # fibers
    # all texture length scales below are Gaussian correlation lengths (RFF scale, no 2pi). Strands ~2*l wide.
    fiber_across_um: float = 5.0  # -> strands ~7-15 um wide, as in registered real 2.4 um data
    fiber_along_um: float = 80.0
    fiber_fill: float = 0.0  # threshold on the unit-variance field (higher -> sparser strands)
    void_density: float = 0.35  # density between strands, relative to strand density
    # sheets are looser near their faces: real 2.4 um edge profiles keep rising from ~140 at +10 um to ~180 at +38 um
    face_loose: float = 0.35  # fractional density deficit at a face, decaying inward over face_loose_um
    face_loose_um: float = 15.0
    contact_loose: float = 0.0  # same deficit at touching faces, scaled by this (0 = fused, 1 = like an open face)
    # laminar density variation in sheet coordinates (in-plane / across-sheet scales). Replaces isotropic 50 um blobs,
    # which the eye test (real_vs_sim.png, 2026-10-17) showed are absent in real 7.91 um data.
    porosity_sd: float = 0.15
    porosity_inplane_um: float = 150.0
    porosity_depth_um: float = 20.0
    # delamination: thin voids parallel to the sheet (real sheets split into sub-layers; intra-sheet distractor gaps)
    delam_inplane_um: float = 150.0
    delam_depth_um: float = 4.0
    delam_thresh: float = 1.8  # on a unit-variance field; higher -> fewer cracks
    # sparse bright inclusions inside material (seen in real scans at both resolutions)
    incl_cell_um: float = 200.0
    incl_prob: float = 0.06
    incl_radius_um: tuple = (4.0, 12.0)
    incl_density: float = 1.2  # added density
    # open faces are rough (fibrous): the edge position jitters by this much (um), isotropic correlation length below
    surface_rough_um: float = 4.0
    surface_rough_corr_um: float = 8.0
    # contact interface: 0 = fully fused (no density dip where g == 0), 1 = empty line between touching sheets
    interface_dip: float = 0.0
    interface_width_um: float = 1.5
    recto_inner: bool = True  # recto (horizontal fibers) on the inner face; GT surface = inner face
    z_extent_um: float = 6000.0  # material z range covered by the tables, centred on 0
    z_step_um: float = 20.0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class RFF:
    """Random Fourier features: approx unit-variance Gaussian field with given per-axis length scales."""

    def __init__(self, gen: torch.Generator, dims: int, scales, n: int = 128, device="cuda", out: int = 1):
        scales = torch.as_tensor(scales, dtype=torch.float32)
        self.k = (torch.randn(out, n, dims, generator=gen) / scales).to(device)
        self.b = (torch.rand(out, n, generator=gen) * TWO_PI).to(device)
        self.norm = math.sqrt(2.0 / n)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # x [M, D] -> [M, out]
        return self.norm * torch.cos(torch.einsum("md,ond->mon", x, self.k) + self.b).sum(-1)


class Scroll:
    def __init__(self, p: ScrollParams, device: str = "cuda"):
        self.p, self.dev = p, device
        g = torch.Generator().manual_seed(p.seed)
        self.gen = g
        self._build_tables(g)
        self.th_ref = torch.zeros(1, device=device)
        self.u_ref = torch.zeros(1, dtype=torch.float64, device=device)
        L = p.warp_corr_um
        self.warp = RFF(g, 3, [L / TWO_PI] * 3, n=64, device=device, out=3)
        # keep the warp Lipschitz constant well below 1: amp * |k| * sqrt(2/n) * sqrt(n) ~ amp * sqrt(2) * 2pi/L
        self.warp_amp = min(p.warp_amp_um, 0.35 * L / (math.sqrt(2) * TWO_PI * math.sqrt(3)))
        a = p.fiber_across_um
        al = p.fiber_along_um
        # strand fields: coordinates (along, across1, across2)
        self.fib_recto = RFF(g, 3, [al, a, a], n=256, device=device)
        self.fib_verso = RFF(g, 3, [al, a, a], n=256, device=device)
        # sheet coordinates (u, z, depth)
        pi_, pd_ = p.porosity_inplane_um, p.porosity_depth_um
        self.poro = RFF(g, 3, [pi_, pi_, pd_], n=128, device=device)
        di_, dd_ = p.delam_inplane_um, p.delam_depth_um
        self.delam = RFF(g, 3, [di_, di_, dd_], n=256, device=device)
        self.incl_salt = int(torch.randint(1, 2**30, (1,), generator=g))
        self.rough = RFF(g, 3, [p.surface_rough_corr_um] * 3, n=128, device=device, out=2)

    # ---------------- tables over (phi, z) ----------------
    def _build_tables(self, g: torch.Generator) -> None:
        p = self.p
        K = p.n_wraps
        nz = int(round(p.z_extent_um / p.z_step_um)) + 1
        self.z0 = -p.z_extent_um / 2
        phi = torch.arange(K * N_THETA, dtype=torch.float64) * (TWO_PI / N_THETA)
        z = self.z0 + torch.arange(nz, dtype=torch.float64) * p.z_step_um
        # noise coordinates: nominal physical position of each table sample (Archimedean spiral, mean pitch)
        pitch0 = p.t_mean + p.g_mean
        rn = p.r0 + pitch0 * phi / TWO_PI
        th = torch.remainder(phi, TWO_PI)
        Rn, Z = torch.meshgrid(rn, z, indexing="ij")
        Th = th[:, None].expand_as(Rn)
        xs = torch.stack([Rn * torch.cos(Th), Rn * torch.sin(Th), Z], -1).reshape(-1, 3).float()
        L = p.noise_corr_um
        f_t = RFF(g, 3, [L] * 3, n=128, device=self.dev)

        def ev(f):
            parts = [f(c.to(self.dev))[:, 0].double().cpu() for c in xs.split(1 << 20)]
            return torch.cat(parts).reshape(Rn.shape)

        nt = ev(f_t)
        # displacement field over (r / Lr, x / Lt, y / Lt, z / Lt): short radial, long tangential correlation
        Lr, Lt = p.disp_corr_radial_um, p.disp_corr_tangential_um
        xd = torch.stack([Rn / Lr, Rn * torch.cos(Th) / Lt, Rn * torch.sin(Th) / Lt, Z / Lt], -1).reshape(-1, 4).float()
        f_d = RFF(g, 4, [1.0] * 4, n=256, device=self.dev)
        parts = [f_d(c.to(self.dev))[:, 0].double().cpu() for c in xd.split(1 << 19)]
        D = torch.cat(parts).reshape(K, N_THETA, nz)
        del xs, xd
        t = (p.t_mean * torch.exp(p.t_rel_sd * nt - 0.5 * p.t_rel_sd**2)).reshape(K, N_THETA, nz)
        Tn = Rn.reshape(K, N_THETA, nz)

        def place(amp):
            T = Tn + amp * D
            R = torch.empty_like(T)
            R[0] = T[0]
            for k in range(1, K):
                R[k] = torch.maximum(R[k - 1] + t[k - 1], T[k])
            return R

        def contact(R):
            return float(((R[1:] - R[:-1] - t[:-1]) <= 0).double().mean())

        lo, hi = 0.0, 5.0 * pitch0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if contact(place(mid)) < p.contact_frac else (lo, mid)
        self.disp_amp = hi
        R = place(hi)
        gap = torch.empty_like(R)
        gap[:-1] = (R[1:] - R[:-1] - t[:-1]).clamp(min=0.0)
        gap[-1] = (pitch0 - t[-1]).clamp(min=0.0)
        # extend theta by one column (= next wrap's first column) for interpolation across the seam
        def ext(a, last):
            nxt = torch.cat([a[1:, :1], last[None, None]], 0)
            return torch.cat([a, nxt], 1)

        R_last = R[-1, 0] + t[-1, 0] + gap[-1, 0]
        self.R = ext(R, R_last).float().to(self.dev).contiguous()
        self.t = ext(t, t[-1, -1]).float().to(self.dev).contiguous()
        self.g = ext(gap, gap[-1, -1]).float().to(self.dev).contiguous()
        # arc length along the whole spiral (inner face), continuous across the theta seam; float64 for precision
        dphi = TWO_PI / N_THETA
        Rf = R.reshape(K * N_THETA, nz)
        U = torch.cumsum(Rf * dphi, 0) - Rf * dphi
        U = U.reshape(K, N_THETA, nz)
        self.U = ext(U, U[-1, -1] + R[-1, -1] * dphi).to(self.dev).contiguous()
        self.nz = nz
        self.contact_area_frac = float((gap == 0).double().mean())

    def set_reference(self, xw_center: torch.Tensor) -> None:
        """Anchor a patch: arc length u = 0 here (keeps fiber RFF phases small) and the winding cut for instance
        labels lies on the opposite side of the scroll (theta_ref + pi), so it never crosses the patch."""
        r, th, z = self.world_to_material(xw_center.reshape(1, 3).to(self.dev))
        k, it, iz = self.lookup(r, th, z)
        self.th_ref = th
        self.u_ref = self._gather(self.U, k.clamp(0, self.p.n_wraps - 1), it, iz)

    def _gather(self, table: torch.Tensor, k: torch.Tensor, it: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        """Bilinear in (theta, z) at integer wrap k. it in [0, N_THETA), iz in [0, nz-1]."""
        W = N_THETA + 1
        i0 = it.floor().long().clamp(0, N_THETA - 1)
        j0 = iz.floor().long().clamp(0, self.nz - 2)
        fi = (it - i0).clamp(0, 1)
        fj = (iz - j0).clamp(0, 1)
        base = (k * W + i0) * self.nz + j0
        flat = table.view(-1)
        v00 = flat[base]
        v01 = flat[base + 1]
        v10 = flat[base + self.nz]
        v11 = flat[base + self.nz + 1]
        return (v00 * (1 - fi) + v10 * fi) * (1 - fj) + (v01 * (1 - fi) + v11 * fi) * fj

    # ---------------- maps ----------------
    def world_to_material(self, xw: torch.Tensor):
        """xw [M,3] world um (x, y, z) -> (r, theta, z) material cylindrical."""
        p = self.p
        c, s = math.cos(p.squash_angle), math.sin(p.squash_angle)
        x = c * xw[:, 0] + s * xw[:, 1]
        y = -s * xw[:, 0] + c * xw[:, 1]
        X = torch.stack([x / p.squash[0], y / p.squash[1], xw[:, 2]], -1)
        X = X + self.warp_amp * self.warp(X)
        r = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        th = torch.atan2(X[:, 1], X[:, 0]) + p.twist_per_mm * X[:, 2] / 1000.0
        th = torch.remainder(th, TWO_PI)
        return r, th, X[:, 2]

    def lookup(self, r, th, z):
        """Containing interval k (R_k <= r < R_{k+1}); -1 inside the core, K beyond the last wrap."""
        K = self.p.n_wraps
        it = th * (N_THETA / TWO_PI)
        iz = ((z - self.z0) / self.p.z_step_um).clamp(0, self.nz - 1)
        lo = torch.zeros_like(r, dtype=torch.long)
        hi = torch.full_like(lo, K)
        while True:  # bisect_right over wraps
            active = lo < hi
            if not active.any():
                break
            mid = (lo + hi) // 2
            Rm = self._gather(self.R, mid.clamp(max=K - 1), it, iz)
            go_right = (Rm <= r) & active
            lo = torch.where(go_right, mid + 1, lo)
            hi = torch.where(active & ~go_right, mid, hi)
        k = lo - 1
        return k, it, iz

    def metric_scale(self, xw: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """|grad_world r_material|: material radial distances / this = world normal distances (squash, warp)."""
        r0 = self.world_to_material(xw)[0]
        g2 = torch.zeros_like(r0)
        for ax in range(3):
            d = torch.zeros_like(xw)
            d[:, ax] = eps
            g2 = g2 + ((self.world_to_material(xw + d)[0] - r0) / eps) ** 2
        return g2.sqrt()

    def _seam_shift(self, th: torch.Tensor) -> torch.Tensor:
        """Wrap-index offset (-1/0/+1) that makes labels continuous across the theta = 0 seam near the reference.

        A sheet crossing the seam goes from (k, theta ~ 2pi) to (k + 1, theta ~ 0); seen from the reference side,
        points past the seam get -1, points before it +1."""
        d = th - self.th_ref
        return torch.where(d < -math.pi, -1, torch.where(d > math.pi, 1, 0)).long()

    def _hash01(self, c: torch.Tensor, salt: int) -> torch.Tensor:
        """Deterministic uniform [0,1) per integer cell c [M,3] (int64)."""
        h = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791) ^ (self.incl_salt + salt * 2654435761)
        h = h & 0xFFFFFFFF
        for _ in range(2):  # xorshift-multiply mixing
            h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
        h = h ^ (h >> 16)
        return h.double().div(2.0**32).float()

    def _inclusions(self, xw: torch.Tensor) -> torch.Tensor:
        """Soft spheres, at most one per incl_cell_um world cell, fully inside it (so no neighbour lookup)."""
        P = self.p
        L, (rmin, rmax) = P.incl_cell_um, P.incl_radius_um
        c = torch.floor(xw / L).long()
        on = self._hash01(c, 0) < P.incl_prob
        rad = rmin + (rmax - rmin) * self._hash01(c, 1)
        ctr = torch.stack([self._hash01(c, 2 + i) for i in range(3)], -1) * (L - 2 * rmax) + rmax
        dist = (xw - c.float() * L - ctr).norm(dim=-1)
        return torch.where(on, torch.sigmoid((rad - dist) / 0.8), torch.zeros_like(dist))

    @torch.no_grad()
    def sample(self, xw: torch.Tensor, want_density: bool = True) -> dict:
        """Evaluate labels (and density) at world points xw [M,3] (um)."""
        K = self.p.n_wraps
        r, th, z = self.world_to_material(xw)
        k, it, iz = self.lookup(r, th, z)
        kc = k.clamp(0, K - 1)
        Rk = self._gather(self.R, kc, it, iz)
        tk = self._gather(self.t, kc, it, iz)
        gk = self._gather(self.g, kc, it, iz)
        gprev = self._gather(self.g, (kc - 1).clamp(min=0), it, iz)
        gprev = torch.where(kc > 0, gprev, torch.full_like(gprev, 1e6))
        depth = r - Rk
        valid = (k >= 0) & (k < K)
        # a gap thinner than 0.5 um (round-off at exact contacts) belongs to the sheet, so fused stacks have no holes
        in_sheet = valid & (depth >= 0) & ((depth < tk) | (gk <= 0.5))
        R_next = self._gather(self.R, (kc + 1).clamp(max=K - 1), it, iz)
        out = {
            "k": torch.where(valid, k, torch.where(k < 0, -1, K)),
            # winding number relative to the reference: continuous along the sheet except at theta_ref + pi
            "instance": torch.where(in_sheet, k + 2 + self._seam_shift(th), 0),
            "gap_min": torch.where(in_sheet, torch.minimum(gk, gprev), torch.full_like(gk, 1e6)),
            "depth": depth,
            "thick": tk,
            # material-space distance from r to the inner (recto) face of the next sheet out
            "to_next": torch.where(valid & (k < K - 1), R_next - r, torch.full_like(r, 1e6)),
            # winding coordinate s: wrap index (relative to the reference, seam-continuous like "instance") plus the
            # radial fraction of the way to the next wrap's recto face. Continuous through sheets, gaps and contacts;
            # +1 per wrap; recto faces are its integer level sets (in-sheet floor(s) + 2 == instance). NaN outside
            # wraps 0..K-2.
            "wind": torch.where(valid & (k < K - 1),
                                (k + self._seam_shift(th)).float() + depth / (R_next - Rk).clamp(min=1e-3),
                                torch.full_like(r, float("nan"))),
        }
        if want_density:
            # only points within a few um of a sheet need texture; the rest stay 0
            pad = 3.0 + 2 * self.p.surface_rough_um
            near = valid & (depth > -pad) & (depth < tk + pad)
            ix = torch.nonzero(near)[:, 0]
            dens = torch.zeros_like(r)
            if len(ix):
                kn, itn, izn, dn, tn, zn = kc[ix], it[ix], iz[ix], depth[ix], tk[ix], z[ix]
                # arc length along the spiral relative to the patch reference (keeps RFF phases small)
                u = (self._gather(self.U, kn, itn, izn) - self.u_ref).float()
                frac = (dn / tn).clamp(0, 1)
                recto = frac < 0.5 if self.p.recto_inner else frac >= 0.5
                # recto fibers run along u (horizontal), verso fibers along z (vertical)
                f = torch.empty_like(dn)
                ir, iv = torch.nonzero(recto)[:, 0], torch.nonzero(~recto)[:, 0]
                f[ir] = self.fib_recto(torch.stack([u[ir], zn[ir], dn[ir]], -1))[:, 0]
                f[iv] = self.fib_verso(torch.stack([zn[iv], u[iv], dn[iv]], -1))[:, 0]
                P = self.p
                strands = torch.sigmoid((f - P.fiber_fill) / 0.25)
                sc = torch.stack([u, zn, dn], -1)
                poro = 1.0 + P.porosity_sd * self.poro(sc)[:, 0]
                crack = torch.sigmoid((self.delam(sc)[:, 0] - P.delam_thresh) / 0.1)
                d = (P.void_density + (1 - P.void_density) * strands) * poro * (1 - 0.9 * crack)
                d = d + P.incl_density * self._inclusions(xw[ix])
                w_in = torch.where(gprev[ix] > 0.5, 1.0, P.contact_loose)
                w_out = torch.where(gk[ix] > 0.5, 1.0, P.contact_loose)
                lam = P.face_loose_um
                d = d * (1 - P.face_loose * w_in * torch.exp(-dn.clamp(min=0) / lam)) \
                      * (1 - P.face_loose * w_out * torch.exp(-(tn - dn).clamp(min=0) / lam))
                # soft edges (antialiasing at ~1 um) on open faces; at exact contacts the interface is
                # controlled only by interface_dip, so no accidental dark line appears between touching sheets
                dip, w = self.p.interface_dip, self.p.interface_width_um
                gkn, gpn = gk[ix], gprev[ix]
                # roughness only where a face is open (never more than half the local gap), so contacts stay exact
                ro = self.p.surface_rough_um * self.rough(xw[ix]).clamp(-2, 2)
                r_out = torch.maximum(torch.minimum(ro[:, 0], gkn / 2), -gkn / 2)
                r_in = torch.maximum(torch.minimum(ro[:, 1], gpn / 2), -gpn / 2)
                e_out = torch.where(gkn > 0.5, torch.sigmoid((tn + r_out - dn) / 0.6),
                                    1 - dip * torch.exp(-(((tn - dn) / w) ** 2)))
                e_in = torch.where(gpn > 0.5, torch.sigmoid((dn - r_in) / 0.6), 1 - dip * torch.exp(-((dn / w) ** 2)))
                dens[ix] = d * e_out * e_in
            out["density"] = dens
        return out
