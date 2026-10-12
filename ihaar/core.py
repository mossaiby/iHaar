"""
Integrated-Haar wavelet collocation (IHWC) -- core 1D/tensor operators.

The integrated-Haar trial space on a (possibly non-uniform) knot vector
t_0 < t_1 < ... < t_N is the space of C^1 piecewise quadratics (quadratic
splines). Collocation is at cell midpoints m_k. The unknowns are the nodal
values U_k = u(m_k); the Haar coefficients of u'' are an orthogonal transform
of the cell values c_k = u''|_{cell k}.

Key identity (Theorem 1 of the paper): for every such spline,
        L U + b0*g0 + b1*g1 = A c ,
with L, A tridiagonal (closed form below). Hence D2 = A^{-1} L is obtained in
O(N) assembly, never forming the dense integration matrices of classical HWCM.

Backends: all heavy kernels take an array module `xp` (numpy or cupy), so the
same code runs on CPU and GPU. Factorizations are done once on the host in
float64 and moved to the device.
"""
import numpy as np
import scipy.linalg as sla


# --------------------------------------------------------------------------
# 1D operators
# --------------------------------------------------------------------------
class Grid1D:
    """Integrated-Haar (C^1 quadratic spline) collocation on knots t.

    bc = 'D' (Dirichlet, values prescribed at t_0, t_N) or
         'N' (homogeneous Neumann, u'(t_0) = u'(t_N) = 0).
    """

    def __init__(self, t, bc="D"):
        t = np.asarray(t, dtype=float)
        self.t, self.bc = t, bc
        h = np.diff(t)
        N = h.size
        self.h, self.N = h, N
        self.x = 0.5 * (t[:-1] + t[1:])
        d = 0.5 * (h[:-1] + h[1:])  # midpoint spacings, length N-1
        A = np.zeros((N, N))
        L = np.zeros((N, N))
        for k in range(N):
            if k > 0:
                A[k, k - 1] = h[k - 1] ** 2 / (8 * d[k - 1])
                L[k, k - 1] = 1.0 / d[k - 1]
            if k < N - 1:
                A[k, k + 1] = h[k + 1] ** 2 / (8 * d[k])
                L[k, k + 1] = 1.0 / d[k]
            akk = h[k]
            if k > 0:
                akk -= h[k] ** 2 / (8 * d[k - 1])
                L[k, k] -= 1.0 / d[k - 1]
            if k < N - 1:
                akk -= h[k] ** 2 / (8 * d[k])
                L[k, k] -= 1.0 / d[k]
            A[k, k] = akk
        b0 = np.zeros(N)
        b1 = np.zeros(N)
        if bc == "D":
            # boundary rows: value at the knot enters through a half-cell
            A[0, 0] -= h[0] / 4
            A[-1, -1] -= h[-1] / 4
            L[0, 0] -= 2.0 / h[0]
            L[-1, -1] -= 2.0 / h[-1]
            b0[0] = 2.0 / h[0]
            b1[-1] = 2.0 / h[-1]
        elif bc != "N":
            raise ValueError(bc)
        self.A, self.L, self.b0, self.b1 = A, L, b0, b1
        lu = sla.lu_factor(A)
        self.D2 = sla.lu_solve(lu, L)            # u'' at midpoints from U
        self.e0 = sla.lu_solve(lu, b0)           # contribution of g0
        self.e1 = sla.lu_solve(lu, b1)           # contribution of g1
        # first derivative at midpoints: u'(m_k) = s0 + sum_{m<k} c_m h_m + c_k h_k/2
        T = np.tril(np.tile(h, (N, 1)), -1) + np.diag(h / 2)
        if bc == "D":
            Tm = T.copy()
            Tm[:, 0] -= h[0] / 4
            r = np.zeros(N)
            r[0] = 2.0 / h[0]
            self.D1 = np.outer(np.ones(N), r) + Tm @ self.D2
            self.f0 = -2.0 / h[0] * np.ones(N) + Tm @ self.e0
            self.f1 = Tm @ self.e1
        else:
            self.D1 = T @ self.D2
            self.f0 = np.zeros(N)
            self.f1 = np.zeros(N)

    def eval_matrices(self, xq):
        """Matrices (E, Ed) with s(xq) = E @ w, s'(xq) = Ed @ w, where
        w = [g0, U_1..U_N, g1] (Dirichlet).  Exact spline reconstruction:
        s = U_k + s'(m_k)(x-m_k) + c_k (x-m_k)^2/2 on cell k."""
        assert self.bc == "D"
        N = self.N
        k = np.clip(np.searchsorted(self.t, xq, side="right") - 1, 0, N - 1)
        dx = xq - self.x[k]
        D1e = np.hstack([self.f0[:, None], self.D1, self.f1[:, None]])
        D2e = np.hstack([self.e0[:, None], self.D2, self.e1[:, None]])
        I = np.zeros((N, N + 2)); I[np.arange(N), np.arange(1, N + 1)] = 1
        E = I[k] + dx[:, None] * D1e[k] + 0.5 * dx[:, None] ** 2 * D2e[k]
        Ed = D1e[k] + dx[:, None] * D2e[k]
        return E, Ed

    # convenience ---------------------------------------------------------
    @classmethod
    def uniform(cls, N, a=0.0, b=1.0, bc="D"):
        return cls(np.linspace(a, b, N + 1), bc)

    def d2(self, U, g0=0.0, g1=0.0):
        return self.D2 @ U + self.e0 * g0 + self.e1 * g1

    def d1(self, U, g0=0.0, g1=0.0):
        return self.D1 @ U + self.f0 * g0 + self.f1 * g1


def corrected_D2(g):
    """Fourth-order spectral correction D2 + (h^2/24) D2^2 (uniform grids)."""
    h = g.h[0]
    assert np.allclose(g.h, h), "fourth-order correction requires a uniform grid"
    return g.D2 + (h * h / 24.0) * (g.D2 @ g.D2)


# --------------------------------------------------------------------------
# reference construction through explicit integrated-Haar basis (for tests)
# --------------------------------------------------------------------------
def haar_matrix(N):
    """Orthonormal Haar transform on N = 2^J equal cells (rows = functions)."""
    J = int(np.log2(N))
    assert 2 ** J == N
    Hm = [np.ones(N) / np.sqrt(N)]
    for j in range(J):
        m = 2 ** j
        w = N // m
        for k in range(m):
            v = np.zeros(N)
            v[k * w:k * w + w // 2] = 1
            v[k * w + w // 2:(k + 1) * w] = -1
            Hm.append(v / np.sqrt(w))
    return np.array(Hm)


def integrated_haar_dirichlet(t, x):
    """Classical HWCM matrices on knots t evaluated at x (homogeneous Dirichlet).

    Returns (Phi, Hval): u(x) = Phi @ a, u''(x) = Hval @ a, where a are the
    coefficients of u'' in the (cell-indicator == Haar up to an orthogonal
    transform) basis.  Built from the closed-form double integral of a cell
    indicator: 0 left of the cell, (x-t_{j})^2/2 inside, h_j (x - m_j) right.
    """
    t = np.asarray(t)
    tl, tr = t[:-1], t[1:]
    h = tr - tl
    m = 0.5 * (tl + tr)
    X = x[:, None]
    P2 = np.where(X < tl, 0.0, np.where(X < tr, 0.5 * (X - tl) ** 2, h * (X - m)))
    a, b = t[0], t[-1]
    P2b = h * (b - m)
    Phi = P2 - ((X - a) / (b - a)) * P2b
    Hval = ((X >= tl) & (X < tr)).astype(float)
    return Phi, Hval


# --------------------------------------------------------------------------
# Fast diagonalization (tensor) solver
# --------------------------------------------------------------------------
class Eig1D:
    """Eigen-factorization D = V diag(lam) V^{-1} computed on host in float64."""

    def __init__(self, D, symmetric=None, imag_tol=1e-8):
        if symmetric is None:
            symmetric = np.allclose(D, D.T, rtol=0, atol=1e-10 * np.abs(D).max())
        self.symmetric = symmetric
        if symmetric:
            lam, V = np.linalg.eigh(0.5 * (D + D.T))
            Vi = V.T
        else:
            lam, V = np.linalg.eig(D)
            rel = np.abs(lam.imag).max() / np.abs(lam).max()
            if rel > imag_tol:
                raise ValueError(f"complex spectrum (rel imag {rel:.2e}); use Schur/Bartels-Stewart")
            lam, V = lam.real, V.real
            idx = np.argsort(lam)
            lam, V = lam[idx], V[:, idx]
            Vi = np.linalg.inv(V)
        self.lam, self.V, self.Vi = lam, V, Vi
        self.condV = np.linalg.cond(V)


class TensorFDM:
    """Device-side fast diagonalization for operators that are functions of
    (D_x (x) I + I (x) D_y) [2D] or the 3D analogue.

    to_hat(F)  : F -> Vx^{-1} F Vy^{-T}
    from_hat(G): G -> Vx G Vy^T
    lam_sum    : lambda_x[i] + lambda_y[j] (+ lambda_z[k])
    """

    def __init__(self, eigs, xp=np, dtype=None):
        self.xp = xp
        dtype = dtype or xp.float64
        self.d = len(eigs)
        self.V = [xp.asarray(e.V, dtype=dtype) for e in eigs]
        self.Vi = [xp.asarray(e.Vi, dtype=dtype) for e in eigs]
        lams = [xp.asarray(e.lam, dtype=dtype) for e in eigs]
        if self.d == 2:
            self.lam_sum = lams[0][:, None] + lams[1][None, :]
        else:
            self.lam_sum = lams[0][:, None, None] + lams[1][None, :, None] + lams[2][None, None, :]
        self.lams = lams

    def to_hat(self, F):
        if self.d == 2:
            return self.Vi[0] @ F @ self.Vi[1].T
        xp = self.xp
        G = xp.tensordot(self.Vi[0], F, axes=(1, 0))
        G = xp.tensordot(G, self.Vi[1], axes=(1, 1)).transpose(0, 2, 1)
        return xp.tensordot(G, self.Vi[2], axes=(2, 1))

    def from_hat(self, G):
        if self.d == 2:
            return self.V[0] @ G @ self.V[1].T
        xp = self.xp
        F = xp.tensordot(self.V[0], G, axes=(1, 0))
        F = xp.tensordot(F, self.V[1], axes=(1, 1)).transpose(0, 2, 1)
        return xp.tensordot(F, self.V[2], axes=(2, 1))

    def solve_diag(self, F, symbol):
        """Solve S(Delta_h) U = F where symbol = S(lam_sum) (array)."""
        return self.from_hat(self.to_hat(F) / symbol)


def sylvester_bs(Ax, By, F):
    """Reference Bartels--Stewart solve of Ax U + U By^T = F (host)."""
    return sla.solve_sylvester(Ax, By.T, F)


def laplacian_symbol(N, h, scheme, shift=0.0):
    """Eigenvalues of 1D second-derivative operators on the uniform midpoint grid,
    in the cosine (shift=0, DCT-II) or quarter-wave (shift=1/2) basis."""
    k = np.arange(N) + shift
    mu = -4 * np.sin(np.pi * k / (2 * N)) ** 2
    if scheme == "ihwc2":
        return 8 * mu / (h * h * (8 + mu))
    if scheme == "ihwc4":
        lam = 8 * mu / (h * h * (8 + mu)); return lam + h * h * lam ** 2 / 24
    if scheme == "fd2":
        return mu / (h * h)
    if scheme == "pade4":
        return mu / (h * h * (1 + mu / 12))
    if scheme == "spectral":
        return -(np.pi * k / (N * h)) ** 2
    raise ValueError(scheme)


class NeumannDCT:
    """Uniform-grid Neumann fast path: the eigenvectors of the integrated-Haar
    D2 are exactly the orthonormal DCT-II vectors (Proposition in paper), with
    eigenvalues lam_k = 8 mu_k / (h^2 (8 + mu_k)), mu_k = -4 sin^2(pi k / 2N).
    O(N^d log N) per transform.  Works with numpy (scipy.fft) or cupy (cupyx)."""

    def __init__(self, N, dim, h, order=2, xp=np, dtype=None, scheme=None):
        """scheme (optional, for comparisons): 'ihwc2', 'ihwc4' (default via order),
        'fd2' (3-point), 'pade4' (classical compact 4th order), 'spectral' (cosine)."""
        self.xp, self.d = xp, dim
        lam = laplacian_symbol(N, h, scheme or ("ihwc4" if order == 4 else "ihwc2"))
        lam = xp.asarray(lam, dtype=dtype or xp.float64)
        if dim == 2:
            self.lam_sum = lam[:, None] + lam[None, :]
        else:
            self.lam_sum = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
        if xp is np:
            import scipy.fft as fft
        else:
            import cupyx.scipy.fft as fft
            if not hasattr(fft, "dctn"):
                raise RuntimeError("this CuPy version has no cupyx.scipy.fft.dctn; use transform='gemm'")
        self.fft = fft

    def to_hat(self, F):
        return self.fft.dctn(F, type=2, norm="ortho")

    def from_hat(self, G):
        return self.fft.idctn(G, type=2, norm="ortho")

    def solve_diag(self, F, symbol):
        return self.from_hat(self.to_hat(F) / symbol)


# --------------------------------------------------------------------------
# Safeguarded 2D Sylvester solver: fast diagonalization with a fallback
# --------------------------------------------------------------------------
class SafeguardedSylvester:
    """Solve a*Dx U + b*U Dy^T + g*U = F.

    Fast diagonalization (FDM) is used when the eigenvector condition numbers are
    below `kappa_max`; after each FDM solve the normwise backward error
        eta = ||R|| / (|a| ||Dx|| ||U|| + |b| ||Dy|| ||U|| + |g| ||U|| + ||F||)
    is checked (two extra matrix products), and if eta > `eta_max` the solve is
    repeated with the backward-stable Bartels--Stewart algorithm. Theorem
    (spectrum on every grid) guarantees that FDM is always *defined*; this class
    guards its *numerical* reliability. Attributes record which path was taken.
    """

    def __init__(self, Dx, Dy, a=1.0, b=1.0, g=0.0, kappa_max=1e6, eta_max=1e-12):
        self.Dx, self.Dy, self.a, self.b, self.g = Dx, Dy, a, b, g
        self.kappa_max, self.eta_max = kappa_max, eta_max
        self.nDx, self.nDy = np.linalg.norm(Dx, 2), np.linalg.norm(Dy, 2)
        try:
            ex, ey = Eig1D(Dx), Eig1D(Dy)
            self.kappa = max(ex.condV, ey.condV)
        except ValueError:
            ex = ey = None; self.kappa = np.inf
        self.use_fdm = self.kappa <= kappa_max
        if self.use_fdm:
            self.fdm = TensorFDM([ex, ey])
            self.sym = a * self.fdm.lams[0][:, None] + b * self.fdm.lams[1][None, :] + g
        self.n_fdm = self.n_fallback = 0
        self.last_eta = None

    def backward_error(self, U, F):
        R = self.a * self.Dx @ U + self.b * U @ self.Dy.T + self.g * U - F
        nu = np.linalg.norm(U)
        return np.linalg.norm(R) / ((abs(self.a) * self.nDx + abs(self.b) * self.nDy + abs(self.g)) * nu + np.linalg.norm(F))

    def solve(self, F):
        if self.use_fdm:
            U = self.fdm.solve_diag(F, self.sym)
            self.last_eta = self.backward_error(U, F)
            if self.last_eta <= self.eta_max:
                self.n_fdm += 1
                return U
        A = self.a * self.Dx + 0.5 * self.g * np.eye(self.Dx.shape[0])
        B = self.b * self.Dy + 0.5 * self.g * np.eye(self.Dy.shape[0])
        U = sla.solve_sylvester(A, B.T, F)
        self.last_eta = self.backward_error(U, F)
        self.n_fallback += 1
        return U
