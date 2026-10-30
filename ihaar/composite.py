"""
Local refinement by nested composite patches (Poisson model problem).

Tensor refinement extends every refinement line across the domain. Here each
refinement level is instead a separate rectangular patch with its own tensor
grid, P_k = (0, 2^-k)^2, k = 0..L, nested toward a corner singularity at the
origin, and each patch is solved by fast diagonalization (well defined on every
patch by the spectrum theorem).

Coupling (its fixed point defines the composite discretization):
  * patch k+1 takes Dirichlet data on its two interior edges from the spline
    of patch k;
  * in a 'hole' well inside P_{k+1}, the collocation rows of patch k are written
    in their local (compact) form  A_x (D_x W + W D_y^T + boundary) A_y^T  and
    given the residual of the fine-solution interpolant (tau correction), so
    patch k sees the fine solution there.
Block Gauss-Seidel sweeps (fine to coarse and back) converge to the fixed point.
Each point of the domain is owned by the finest patch that contains it.
"""
import numpy as np
from .core import Grid1D, SafeguardedSylvester


class Patch:
    """u_xx + u_yy = f on the rectangle tx[0..-1] x ty[0..-1], values prescribed on all edges."""

    def __init__(self, tx, ty):
        self.gx, self.gy = Grid1D(tx), Grid1D(ty)
        self.x, self.y = self.gx.x, self.gy.x
        self.solver = SafeguardedSylvester(self.gx.D2, self.gy.D2)
        self.U = np.zeros((self.gx.N, self.gy.N))

    def set_boundary(self, g, edges="LRBT"):
        """Edge values from g(x, y) (vectorized): L/R at (t_0|t_N, y_j), B/T at (x_i, t_0|t_N), corners."""
        tx, ty, x, y = self.gx.t, self.gy.t, self.x, self.y
        if "L" in edges: self.bL = g(tx[0] + 0 * y, y)
        if "R" in edges: self.bR = g(tx[-1] + 0 * y, y)
        if "B" in edges: self.bB = g(x, ty[0] + 0 * x)
        if "T" in edges: self.bT = g(x, ty[-1] + 0 * x)
        c = g(np.array([tx[0], tx[-1], tx[0], tx[-1]]), np.array([ty[0], ty[0], ty[-1], ty[-1]]))
        if not hasattr(self, "c"):
            self.c = c
        for i, (ex, ey) in enumerate(("LB", "RB", "LT", "RT")):     # a corner follows its edges
            if ex in edges or ey in edges:
                self.c[i] = c[i]

    def bterms(self):
        gx, gy = self.gx, self.gy
        return (np.outer(gx.e0, self.bL) + np.outer(gx.e1, self.bR)
                + np.outer(self.bB, gy.e0) + np.outer(self.bT, gy.e1))

    def lap(self, W):
        return self.gx.D2 @ W + W @ self.gy.D2.T + self.bterms()

    def solve(self, F):
        self.U = self.solver.solve(F - self.bterms())
        return self.U

    def ext(self):
        """Values extended by the boundary data, ordered as eval_matrices expects."""
        W = np.zeros((self.gx.N + 2, self.gy.N + 2))
        W[1:-1, 1:-1] = self.U
        W[0, 1:-1], W[-1, 1:-1], W[1:-1, 0], W[1:-1, -1] = self.bL, self.bR, self.bB, self.bT
        W[0, 0], W[-1, 0], W[0, -1], W[-1, -1] = self.c
        return W

    def eval(self, xs, ys, grad=False):
        """Spline values (and gradient) on the tensor set xs x ys."""
        Ex, Edx = self.gx.eval_matrices(xs); Ey, Edy = self.gy.eval_matrices(ys)
        W = self.ext()
        if not grad:
            return Ex @ W @ Ey.T
        return Ex @ W @ Ey.T, Edx @ W @ Ey.T, Ex @ W @ Edy.T

    def eval_pts(self, xs, ys):
        """Spline values at the scattered points (xs[i], ys[i])."""
        Ex, _ = self.gx.eval_matrices(np.atleast_1d(xs)); Ey, _ = self.gy.eval_matrices(np.atleast_1d(ys))
        return np.einsum("ia,ab,ib->i", Ex, self.ext(), Ey)


class CompositePoisson:
    """Nested corner patches P_k = (0, 2^-k)^2, k = 0..L, each with an n x n uniform grid.

    g: exact Dirichlet data g(x, y) on the physical boundary; f: source f(x, y) or None (Laplace);
    m: hole margin in coarse cells (m >= 2 keeps the hole stencils inside the fine patch)."""

    def __init__(self, n, L, g, f=None, m=3):
        assert n % 2 == 0 and m >= 2
        self.n, self.L, self.m, self.f = n, L, m, f
        self.P = []
        for k in range(L + 1):
            t = np.linspace(0, 2.0 ** -k, n + 1); p = Patch(t, t)
            p.set_boundary(g)                  # interior edges are overwritten by the coupling
            self.P.append(p)
        self.dofs = (L + 1) * n * n

    def _rhs(self, p):
        return np.zeros_like(p.U) if self.f is None else self.f(p.x[:, None], p.y[None, :])

    def _update(self, k):
        p = self.P[k]
        if k > 0:                              # interior edges R, T from the coarser patch
            q, b = self.P[k - 1], 2.0 ** -k
            p.bR = q.eval_pts(b + 0 * p.y, p.y); p.bT = q.eval_pts(p.x, b + 0 * p.x)
            p.c[1:] = q.eval_pts(np.array([b, 0.0, b]), np.array([0.0, b, b]))
        F = self._rhs(p)
        if k < self.L:                         # tau correction in the hole
            fine, b, h = self.P[k + 1], 2.0 ** -(k + 1), 2.0 ** -k / self.n
            hole = p.x < b - self.m * h
            nb = p.x < b - (self.m - 1) * h    # hole plus one cell: the support of its compact rows
            W = p.U.copy(); W[np.ix_(nb, nb)] = fine.eval(p.x[nb], p.y[nb])
            R = p.gx.A @ F @ p.gy.A.T
            R[np.ix_(hole, hole)] = (p.gx.A @ p.lap(W) @ p.gy.A.T)[np.ix_(hole, hole)]
            F = np.linalg.solve(p.gx.A, np.linalg.solve(p.gy.A, R.T).T)
        old = p.U
        p.solve(F)
        return np.abs(p.U - old).max()

    def solve(self, tol=1e-12, maxsweeps=100):
        """Symmetric block Gauss-Seidel sweeps; returns the largest update per sweep."""
        hist = []
        order = list(range(self.L + 1)) + list(range(self.L - 1, -1, -1))
        for _ in range(maxsweeps):
            hist.append(max(self._update(k) for k in order))
            if hist[-1] < tol:
                break
        return hist

    def h1_error(self, grad, ng=4):
        """Relative H^1-seminorm error by Gauss quadrature, each point owned by the finest patch."""
        return h1_error(self.P, grad, ng)


def h1_error(P, grad, ng=4):
    """Relative H^1-seminorm error of nested patches P (a single patch: [p]) against grad(x, y)."""
    gq, wq = np.polynomial.legendre.leggauss(ng)
    num = den = 0.0
    for k, p in enumerate(P):
        def pts(t):
            c, h = 0.5 * (t[:-1] + t[1:]), np.diff(t)
            return (c[:, None] + 0.5 * h[:, None] * gq).ravel(), (0.5 * h[:, None] * wq).ravel()
        X, WX = pts(p.gx.t); Y, WY = pts(p.gy.t)
        _, ux, uy = p.eval(X, Y, grad=True)
        gx, gy = grad(X[:, None], Y[None, :])
        w = np.outer(WX, WY)
        if k < len(P) - 1:
            b = 2.0 ** -(k + 1); w = w * ~((X[:, None] < b) & (Y[None, :] < b))
        num += (w * ((ux - gx) ** 2 + (uy - gy) ** 2)).sum(); den += (w * (gx ** 2 + gy ** 2)).sum()
    return np.sqrt(num / den)


def poisson_tensor(tx, ty, g, f=None):
    """Single tensor-grid solve (reference for the composite patches)."""
    p = Patch(tx, ty); p.set_boundary(g)
    p.solve(np.zeros_like(p.U) if f is None else f(p.x[:, None], p.y[None, :]))
    return p


def corner_knots(n, L):
    """1D union of the patch grids: the tensor grid with the same local resolution."""
    t = set(np.linspace(0, 1, n + 1))
    for k in range(1, L + 1):
        t |= set(np.linspace(0, 2.0 ** -k, n + 1))
    return np.array(sorted(t))
