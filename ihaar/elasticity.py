"""
Plane-strain linear elastostatics by integrated-Haar tensor collocation on
(adaptively graded) tensor grids, solved by right-preconditioned GMRES with a
block fast-diagonalization preconditioner; automatic anisotropic refinement
driven by Haar detail coefficients (jumps of the cell values of u'').
"""
import numpy as np


def to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


from .core import Grid1D, Eig1D, TensorFDM

# ---------------------------------------------------------------------------
# Exact Mode-I Williams field near a re-entrant corner (opening 3*pi/2)
# ---------------------------------------------------------------------------
OMEGA = 1.5 * np.pi
LAM_I = 0.5444837367824632  # root of sin(l*omega) + l*sin(omega) = 0


class WilliamsField:
    """Mode-I Williams field via Kolosov-Muskhelishvili potentials
    phi = z^l, psi = b z^l (local coordinates, bisector on the real axis):
        2G (u_x + i u_y) = kappa*phi - z*conj(phi') - conj(psi).
    Equilibrium holds identically; b makes both wedge faces traction free.
    """

    def __init__(self, E=1.0, nu=0.3, apex=(1.0, 1.0), bisector=1.25 * np.pi, scale=1.0):
        self.E, self.nu = E, nu
        self.G = E / (2 * (1 + nu))
        self.lmb = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = 3 - 4 * nu
        l = LAM_I
        al = OMEGA / 2
        self.l = l
        self.b = -(l + 1) * np.cos((l - 1) * al) / np.cos((l + 1) * al)
        self.apex, self.bis, self.s = complex(*apex), bisector, scale
        self.rot = np.exp(1j * bisector)

    def _z(self, x, y):
        return (np.asarray(x) + 1j * np.asarray(y) - self.apex) / self.rot

    def disp(self, x, y):
        z = self._z(x, y)
        l, k, b = self.l, self.kappa, self.b
        with np.errstate(all="ignore"):
            zl = np.where(z == 0, 0, z ** l)
            dphi = np.where(z == 0, 0, l * z ** (l - 1))
        W = self.s * (k * zl - z * np.conj(dphi) - np.conj(b * zl)) / (2 * self.G) * self.rot
        return W.real, W.imag

    def grad(self, x, y):
        z = self._z(x, y)
        l, k, b = self.l, self.kappa, self.b
        d1 = l * z ** (l - 1)
        d2 = l * (l - 1) * z ** (l - 2)
        Wz = self.s * (k * d1 - np.conj(d1)) / (2 * self.G)
        Wzb = self.s * (-z * np.conj(d2) - np.conj(b * d1)) / (2 * self.G)
        Wzb = Wzb * self.rot ** 2
        Wx = Wz + Wzb
        Wy = 1j * (Wz - Wzb)
        return Wx.real, Wy.real, Wx.imag, Wy.imag


# ---------------------------------------------------------------------------
# Collocation operator on a tensor grid
# ---------------------------------------------------------------------------
class ElasticityTensor:
    def __init__(self, tx, ty, E=1.0, nu=0.3, xp=np):
        self.xp = xp
        self.gx, self.gy = Grid1D(tx, "D"), Grid1D(ty, "D")
        self.lmb = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        gx, gy = self.gx, self.gy
        to = lambda a: xp.asarray(a)
        self.D2x, self.D2y, self.D1x, self.D1y = map(to, (gx.D2, gy.D2, gx.D1, gy.D1))
        ex, ey = Eig1D(gx.D2), Eig1D(gy.D2)
        self.condV = max(ex.condV, ey.condV)
        self.fdm = TensorFDM([ex, ey], xp=xp)
        lx, ly = self.fdm.lams
        a, m = self.lmb + 2 * self.mu, self.mu
        self.Pu = a * lx[:, None] + m * ly[None, :]
        self.Pv = m * lx[:, None] + a * ly[None, :]
        self.shape = (gx.N, gy.N)

    # boundary data ---------------------------------------------------------
    def boundary_data(self, field):
        gx, gy = self.gx, self.gy
        x0, x1, y0, y1 = gx.t[0], gx.t[-1], gy.t[0], gy.t[-1]
        bd = {}
        for comp in (0, 1):
            ev = lambda X, Y: field.disp(np.asarray(X, float), np.asarray(Y, float))[comp]
            bd[comp] = dict(L=ev(x0 + 0 * gy.x, gy.x), R=ev(x1 + 0 * gy.x, gy.x),
                            B=ev(gx.x, y0 + 0 * gx.x), T=ev(gx.x, y1 + 0 * gx.x),
                            c00=float(ev(x0, y0)), c10=float(ev(x1, y0)),
                            c01=float(ev(x0, y1)), c11=float(ev(x1, y1)))
        return bd

    @staticmethod
    def zero_bd(nx, ny):
        z = dict(L=np.zeros(ny), R=np.zeros(ny), B=np.zeros(nx), T=np.zeros(nx),
                 c00=0.0, c10=0.0, c01=0.0, c11=0.0)
        return {0: z, 1: z}

    def derivs(self, U, b):
        xp, gx, gy = self.xp, self.gx, self.gy
        if not hasattr(U, "get") and xp is not np:   # host diagnostics on a device solver
            xp = np
            o = np.outer; A = np.asarray
            Uxx = gx.D2 @ U + o(gx.e0, b["L"]) + o(gx.e1, b["R"])
            Uyy = U @ gy.D2.T + o(b["B"], gy.e0) + o(b["T"], gy.e1)
            Ux = gx.D1 @ U + o(gx.f0, b["L"]) + o(gx.f1, b["R"])
            Uy = U @ gy.D1.T + o(b["B"], gy.f0) + o(b["T"], gy.f1)
            bx = gx.D1 @ b["B"] + gx.f0 * b["c00"] + gx.f1 * b["c10"]
            tx = gx.D1 @ b["T"] + gx.f0 * b["c01"] + gx.f1 * b["c11"]
            return Uxx, Uyy, Ux @ gy.D1.T + o(bx, gy.f0) + o(tx, gy.f1), Ux, Uy
        o = xp.outer
        A = lambda v: xp.asarray(v)
        Uxx = self.D2x @ U + o(A(gx.e0), A(b["L"])) + o(A(gx.e1), A(b["R"]))
        Uyy = U @ self.D2y.T + o(A(b["B"]), A(gy.e0)) + o(A(b["T"]), A(gy.e1))
        Ux = self.D1x @ U + o(A(gx.f0), A(b["L"])) + o(A(gx.f1), A(b["R"]))
        Uy = U @ self.D1y.T + o(A(b["B"]), A(gy.f0)) + o(A(b["T"]), A(gy.f1))
        bx = gx.D1 @ b["B"] + gx.f0 * b["c00"] + gx.f1 * b["c10"]
        tx = gx.D1 @ b["T"] + gx.f0 * b["c01"] + gx.f1 * b["c11"]
        Uxy = Ux @ self.D1y.T + o(A(bx), A(gy.f0)) + o(A(tx), A(gy.f1))
        return Uxx, Uyy, Uxy, Ux, Uy

    def apply(self, U, V, bd):
        lm, mu = self.lmb, self.mu
        uxx, uyy, uxy, *_ = self.derivs(U, bd[0])
        vxx, vyy, vxy, *_ = self.derivs(V, bd[1])
        Ru = (lm + 2 * mu) * uxx + mu * uyy + (lm + mu) * vxy
        Rv = mu * vxx + (lm + 2 * mu) * vyy + (lm + mu) * uxy
        return Ru, Rv

    def precond(self, Ru, Rv):
        return self.fdm.solve_diag(Ru, self.Pu), self.fdm.solve_diag(Rv, self.Pv)

    def solve(self, field, tol=1e-10, restart=60, maxit=2000, body=None):
        """Solve div sigma + b = 0; `body(X, Y) -> (bx, by)` optional."""
        xp = self.xp
        nx, ny = self.shape
        bd = self.boundary_data(field)
        Z = xp.zeros((nx, ny))
        r0u, r0v = self.apply(Z, Z, bd)               # affine part
        if body is not None:
            X, Y = np.meshgrid(self.gx.x, self.gy.x, indexing="ij")
            bx, by = body(X, Y)
            r0u, r0v = r0u + xp.asarray(bx), r0v + xp.asarray(by)
        z0 = self.zero_bd(nx, ny)
        n = nx * ny

        def Aop(w):
            u, v = w[:n].reshape(nx, ny), w[n:].reshape(nx, ny)
            pu, pv = self.precond(u, v)                # right preconditioning
            au, av = self.apply(pu, pv, z0)
            return xp.concatenate([au.ravel(), av.ravel()])

        rhs = -xp.concatenate([r0u.ravel(), r0v.ravel()])
        y, its, hist = gmres(Aop, rhs, tol=tol, restart=restart, maxit=maxit, xp=xp)
        U, V = self.precond(y[:n].reshape(nx, ny), y[n:].reshape(nx, ny))
        self.bd = bd
        return U, V, its, hist

    def errors(self, U, V, field):
        U, V = to_host(U), to_host(V)
        gx, gy = self.gx, self.gy
        X, Y = np.meshgrid(gx.x, gy.x, indexing="ij")
        ue, ve = field.disp(X, Y)
        g = field.grad(X, Y)
        _, _, _, Ux, Uy = self.derivs(U, self.bd[0])
        _, _, _, Vx, Vy = self.derivs(V, self.bd[1])
        w = np.outer(gx.h, gy.h)
        num = (w * ((Ux - g[0]) ** 2 + (Uy - g[1]) ** 2 + (Vx - g[2]) ** 2 + (Vy - g[3]) ** 2)).sum()
        den = (w * (g[0] ** 2 + g[1] ** 2 + g[2] ** 2 + g[3] ** 2)).sum()
        einf = max(np.abs(U - ue).max(), np.abs(V - ve).max()) / max(np.abs(ue).max(), np.abs(ve).max())
        return np.sqrt(num / den), einf

    def errors_gauss(self, U, V, field, ng=4):
        """Relative H^1-seminorm error of the reconstructed tensor spline,
        integrated with ng x ng Gauss points per cell (true norm)."""
        gx, gy = self.gx, self.gy
        xg, wg = np.polynomial.legendre.leggauss(ng)
        def q(g):
            xq = (0.5 * (g.t[:-1, None] + g.t[1:, None]) + 0.5 * g.h[:, None] * xg[None, :]).ravel()
            wq = (0.5 * g.h[:, None] * wg[None, :]).ravel()
            return xq, wq
        xq, wx = q(gx); yq, wy = q(gy)
        Ex, Edx = gx.eval_matrices(xq); Ey, Edy = gy.eval_matrices(yq)
        X, Y = np.meshgrid(xq, yq, indexing="ij")
        g = field.grad(X, Y)
        W = np.outer(wx, wy)
        num = den = 0.0
        U, V = to_host(U), to_host(V)
        for comp, Uc in ((0, U), (1, V)):
            b = self.bd[comp]
            Ue = np.zeros((gx.N + 2, gy.N + 2))
            Ue[1:-1, 1:-1] = np.asarray(Uc); Ue[0, 1:-1] = b["L"]; Ue[-1, 1:-1] = b["R"]
            Ue[1:-1, 0] = b["B"]; Ue[1:-1, -1] = b["T"]
            Ue[0, 0], Ue[-1, 0], Ue[0, -1], Ue[-1, -1] = b["c00"], b["c10"], b["c01"], b["c11"]
            ux = Edx @ Ue @ Ey.T; uy = Ex @ Ue @ Edy.T
            num += (W * ((ux - g[2 * comp]) ** 2 + (uy - g[2 * comp + 1]) ** 2)).sum()
            den += (W * (g[2 * comp] ** 2 + g[2 * comp + 1] ** 2)).sum()
        return np.sqrt(num / den)

    def indicators(self, U, V):
        """Haar-detail (u'' jump) indicators per 1D cell in x and in y."""
        U, V = to_host(U), to_host(V)
        gx, gy = self.gx, self.gy
        ex = np.zeros(gx.N)
        ey = np.zeros(gy.N)
        for W, b in ((U, self.bd[0]), (V, self.bd[1])):
            Wxx, Wyy, *_ = self.derivs(W, b)
            Jx = np.diff(Wxx, axis=0) ** 2          # (Nx-1, Ny)
            Jy = np.diff(Wyy, axis=1) ** 2          # (Nx, Ny-1)
            jx = np.zeros((gx.N, gy.N)); jx[:-1] += Jx; jx[1:] += Jx
            jy = np.zeros((gx.N, gy.N)); jy[:, :-1] += Jy; jy[:, 1:] += Jy
            ex += gx.h ** 3 * (jx @ gy.h) * 0.5
            ey += gy.h ** 3 * (gx.h @ jy) * 0.5
        return ex, ey


# ---------------------------------------------------------------------------
# Refinement utilities
# ---------------------------------------------------------------------------
def dorfler(ex, ey, theta=0.5):
    allv = np.concatenate([ex, ey])
    idx = np.argsort(allv)[::-1]
    cs = np.cumsum(allv[idx])
    k = np.searchsorted(cs, theta * cs[-1]) + 1
    sel = idx[:k]
    return sel[sel < ex.size], sel[sel >= ex.size] - ex.size


def bisect(t, cells, ratio=2.0):
    t = list(t)
    newt = set(t)
    for c in cells:
        newt.add(0.5 * (t[c] + t[c + 1]))
    t = np.array(sorted(newt))
    # enforce neighbour ratio
    while True:
        h = np.diff(t)
        bad = np.where((h[:-1] > ratio * h[1:] + 1e-15))[0]
        bad2 = np.where((h[1:] > ratio * h[:-1] + 1e-15))[0] + 1
        cells = np.unique(np.concatenate([bad, bad2]))
        if cells.size == 0:
            return t
        t = np.array(sorted(set(t) | set(0.5 * (t[cells] + t[cells + 1]))))


# ---------------------------------------------------------------------------
# GMRES (right-preconditioned operator already folded in), device agnostic
# ---------------------------------------------------------------------------
def gmres(Aop, b, tol=1e-10, restart=60, maxit=2000, xp=np):
    n = b.size
    x = xp.zeros(n)
    bnorm = float(xp.linalg.norm(b))
    hist = []
    total = 0
    while total < maxit:
        r = b - Aop(x)
        beta = float(xp.linalg.norm(r))
        hist.append(beta / bnorm)
        if beta / bnorm < tol:
            break
        Vk = xp.zeros((restart + 1, n))
        Vk[0] = r / beta
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1); g[0] = beta
        cs, sn = np.zeros(restart), np.zeros(restart)
        j_last = 0
        for j in range(restart):
            w = Aop(Vk[j])
            hcol = Vk[: j + 1] @ w                    # classical GS, twice (CGS2)
            w = w - hcol @ Vk[: j + 1]
            h2 = Vk[: j + 1] @ w
            w = w - h2 @ Vk[: j + 1]
            hcol = np.asarray(hcol + h2, dtype=float) if xp is np else (hcol + h2).get()
            H[: j + 1, j] = hcol
            H[j + 1, j] = float(xp.linalg.norm(w))
            if H[j + 1, j] > 0:
                Vk[j + 1] = w / H[j + 1, j]
            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            den = np.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = H[j, j] / den, H[j + 1, j] / den
            H[j, j] = den; H[j + 1, j] = 0
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j]
            total += 1
            j_last = j
            hist.append(abs(g[j + 1]) / bnorm)
            if abs(g[j + 1]) / bnorm < tol or total >= maxit:
                break
        k = j_last + 1
        yk = np.linalg.solve(np.triu(H[:k, :k]), g[:k])
        x = x + xp.asarray(yk) @ Vk[:k]
        if hist[-1] < tol:
            break
    return x, total, hist


class SmoothField:
    """Smooth manufactured displacement with analytic body force."""

    def __init__(self, E=1.0, nu=0.3):
        self.G = E / (2 * (1 + nu)); self.lmb = E * nu / ((1 + nu) * (1 - 2 * nu))

    def disp(self, x, y):
        return np.sin(np.pi * x) * np.cos(2 * y) + x * y, np.exp(x) * np.sin(np.pi * y) + y

    def grad(self, x, y):
        p = np.pi
        return (p * np.cos(p * x) * np.cos(2 * y) + y, -2 * np.sin(p * x) * np.sin(2 * y) + x,
                np.exp(x) * np.sin(p * y), p * np.exp(x) * np.cos(p * y) + 1)

    def body(self, x, y):
        p, l, m = np.pi, self.lmb, self.G
        uxx = -p * p * np.sin(p * x) * np.cos(2 * y); uyy = -4 * np.sin(p * x) * np.cos(2 * y)
        uxy = -2 * p * np.cos(p * x) * np.sin(2 * y) + 1
        vxx = np.exp(x) * np.sin(p * y); vyy = -p * p * np.exp(x) * np.sin(p * y)
        vxy = p * np.exp(x) * np.cos(p * y)
        bx = -((l + 2 * m) * uxx + m * uyy + (l + m) * vxy)
        by = -(m * vxx + (l + 2 * m) * vyy + (l + m) * uxy)
        return bx, by
