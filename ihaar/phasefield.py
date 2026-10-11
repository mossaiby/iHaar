"""
Energy-stable, mass-conservative integrated-Haar collocation for Allen-Cahn (AC)
and Cahn-Hilliard (CH) with homogeneous Neumann conditions on uniform grids.

Time stepping: scalar auxiliary variable (SAV) with BDF1 (first step) / BDF2.
Every time step costs two tensor transforms (forward of the nonlinear term,
inverse of the update) -> 2*d GEMMs (d = dimension), plus O(N^d) pointwise work.
All arrays live on the device given by `xp` (numpy or cupy).

Energy:  E_h(u) = h^d [ sum F(u) + eps^2/2 * u . (-Delta_h u) ],  F = (u^2-1)^2/4.
Discrete mass: h^d sum u  (= exact integral of the quadratic spline on uniform grids).
"""
import numpy as np
from .core import Grid1D, corrected_D2, Eig1D, TensorFDM, NeumannDCT


def F(u):
    return 0.25 * (u * u - 1.0) ** 2


def f(u):
    return u * u * u - u


class SAVPhaseField:
    def __init__(self, N, dim=2, eps=0.01, dt=1e-4, model="CH", order=4,
                 C=1.0, L=1.0, xp=np, dtype=None, transform="gemm", S=0.0, scheme=None):
        self.xp, self.dim, self.eps, self.dt, self.model = xp, dim, eps, dt, model
        g = Grid1D.uniform(N, 0.0, L, bc="N")
        self.h = g.h[0]
        if transform == "dct":
            self.fdm = NeumannDCT(N, dim, self.h, order=order, xp=xp, dtype=dtype, scheme=scheme)
        else:
            D = corrected_D2(g) if order == 4 else g.D2
            eig = Eig1D(D, symmetric=True)
            self.fdm = TensorFDM([eig] * dim, xp=xp, dtype=dtype)
        lam = self.fdm.lam_sum
        self.mlap = -lam                              # symbol of -Delta_h  (>= 0)
        self.G = xp.ones_like(lam) if model == "AC" else self.mlap  # mobility symbol
        self.S = S   # stabilization S (u^{n+1} - ubar) in mu; S = 0 is plain SAV
        self.S1 = 1.0 + dt * self.G * (eps ** 2 * self.mlap + S)
        self.S2 = 3.0 + 2.0 * dt * self.G * (eps ** 2 * self.mlap + S)
        self.w = self.h ** dim
        self.C = C
        self.nsteps = 0

    # -- helpers -----------------------------------------------------------
    def ip(self, a, b):
        return float(self.w * (a * b).sum())

    def E1(self, u):
        return float(self.w * F(u).sum())

    def energy(self, u, uhat=None):
        if uhat is None:
            uhat = self.fdm.to_hat(u)
        grad = 0.5 * self.eps ** 2 * self.w * float((self.mlap * uhat * uhat).sum())
        return self.E1(u) + grad

    def mass(self, u):
        return float(self.w * u.sum())

    # -- time stepping -----------------------------------------------------
    def initialize(self, u0):
        xp = self.xp
        self.u = xp.asarray(u0)
        self.uhat = self.fdm.to_hat(self.u)
        self.r = np.sqrt(self.E1(self.u) + self.C)
        self.u_old = self.uhat_old = None
        self.r_old = None
        self.nsteps = 0

    def step(self):
        xp, dt = self.xp, self.dt
        if self.u_old is None:          # BDF1 start
            ubar, ghat, S, a, rr = self.u, self.uhat, self.S1, 1.0, self.r
            cq = -dt
        else:                           # BDF2
            ubar = 2.0 * self.u - self.u_old
            ghat = 4.0 * self.uhat - self.uhat_old
            S, a, rr = self.S2, 3.0, 4.0 * self.r - self.r_old
            cq = -2.0 * dt
        b = f(ubar) / np.sqrt(self.E1(ubar) + self.C)
        bhat = self.fdm.to_hat(b)
        if self.S:
            ubhat = self.uhat if self.u_old is None else 2.0 * self.uhat - self.uhat_old
            phat = (ghat - cq * self.S * self.G * ubhat) / S
        else:
            phat = ghat / S
        qhat = cq * self.G * bhat / S
        # inner products in coefficient space (orthonormal eigenbasis)
        bq = self.w * float((bhat * qhat).sum())
        bp = self.w * float((bhat * (a * phat - ghat)).sum())
        r_new = (rr + 0.5 * bp) / (a - 0.5 * a * bq)
        uhat_new = phat + r_new * qhat
        self.u_old, self.uhat_old, self.r_old = self.u, self.uhat, self.r
        self.uhat, self.r = uhat_new, r_new
        self.u = self.fdm.from_hat(uhat_new)
        self.nsteps += 1
        return self.u

    def modified_energy(self):
        """SAV modified energy (BDF1 form after first step, BDF2 form after);
        with stabilization S it includes S/2 ||u^{n+1}-u^n||_h^2 (BDF2)."""
        e2 = self.eps ** 2 * self.w
        nrm = lambda vh: float((self.mlap * vh * vh).sum())
        if self.uhat_old is None or self.nsteps < 2:
            return 0.5 * e2 * nrm(self.uhat) + self.r ** 2
        v = 2 * self.uhat - self.uhat_old
        d = self.uhat - self.uhat_old
        return (0.25 * e2 * (nrm(self.uhat) + nrm(v)) + 0.5 * (self.r ** 2 + (2 * self.r - self.r_old) ** 2)
                + 0.5 * self.S * self.w * float((d * d).sum()))
