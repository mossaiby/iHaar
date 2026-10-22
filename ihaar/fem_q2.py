"""
Reference Q2 (biquadratic Lagrange) Galerkin FEM for plane-strain elasticity on
tensor-product grids (uniform or graded), used for comparisons in the paper.

Global matrices are assembled exactly as Kronecker products of 1D matrices
(stiffness S, mass M, mixed C = int phi_i' phi_j), solved with a sparse direct
solver (SuperLU via scipy.sparse.linalg.spsolve). Dirichlet data are nodal
interpolants of the exact field; traction-free edges are natural.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_xg, _wg = np.polynomial.legendre.leggauss(3)


def _ref_basis(xi):
    """Quadratic Lagrange on [-1,1] with nodes -1, 0, 1: values and d/dxi."""
    xi = np.asarray(xi)
    N = np.stack([0.5 * xi * (xi - 1), 1 - xi ** 2, 0.5 * xi * (xi + 1)], -1)
    dN = np.stack([xi - 0.5, -2 * xi, xi + 0.5], -1)
    return N, dN


class Q2Mesh1D:
    def __init__(self, t):
        self.t = np.asarray(t, float); self.ne = len(t) - 1; self.n = 2 * self.ne + 1
        self.nodes = np.empty(self.n); self.nodes[0::2] = self.t
        self.nodes[1::2] = 0.5 * (self.t[:-1] + self.t[1:])
        h = np.diff(self.t); self.h = h
        N, dN = _ref_basis(_xg)
        Me = np.einsum("q,qi,qj->ij", _wg, N, N)          # times h/2
        Se = np.einsum("q,qi,qj->ij", _wg, dN, dN)        # times 2/h
        Ce = np.einsum("q,qi,qj->ij", _wg, dN, N)         # int phi_i' phi_j (scale-free)
        rows, cols, m, s, c = [], [], [], [], []
        for e in range(self.ne):
            idx = [2 * e, 2 * e + 1, 2 * e + 2]
            for a in range(3):
                for b in range(3):
                    rows.append(idx[a]); cols.append(idx[b])
                    m.append(Me[a, b] * h[e] / 2); s.append(Se[a, b] * 2 / h[e]); c.append(Ce[a, b])
        shape = (self.n, self.n)
        self.M = sp.csr_matrix((m, (rows, cols)), shape)
        self.S = sp.csr_matrix((s, (rows, cols)), shape)
        self.C = sp.csr_matrix((c, (rows, cols)), shape)

    def eval_matrices(self, ng=4):
        """Values / derivatives of all basis functions at ng Gauss points per element."""
        xg, wg = np.polynomial.legendre.leggauss(ng)
        N, dN = _ref_basis(xg)
        nq = self.ne * ng
        E = sp.lil_matrix((nq, self.n)); Ed = sp.lil_matrix((nq, self.n))
        xq = np.empty(nq); wq = np.empty(nq)
        for e in range(self.ne):
            for q in range(ng):
                r = e * ng + q
                xq[r] = 0.5 * (self.t[e] + self.t[e + 1]) + 0.5 * self.h[e] * xg[q]
                wq[r] = 0.5 * self.h[e] * wg[q]
                for a in range(3):
                    E[r, 2 * e + a] = N[q, a]; Ed[r, 2 * e + a] = dN[q, a] * 2 / self.h[e]
        return E.tocsr(), Ed.tocsr(), xq, wq


def solve_q2(tx, ty, field, E=1.0, nu=0.3, dirichlet=("L", "R", "B", "T"), solver="direct",
             tol=1e-10, return_info=False, amg_options=None, maxiter=2000):
    """Solve the plane-strain problem with exact Dirichlet data on the listed edges
    (other edges traction free). Returns (relative H1 error, dofs, solve time).

    solver = 'direct' : sparse direct (SuperLU)
             'pcg'    : conjugate gradients preconditioned by block fast diagonalization of
                        (lambda+2mu) Sx(x)My + mu Mx(x)Sy and mu Sx(x)My + (lambda+2mu) Mx(x)Sy
                        on the free nodes -- the Galerkin analogue of the IHWC preconditioner.
    With return_info=True also returns dict(its=..., setup=...)."""
    import time
    import scipy.linalg as sla
    ta = time.perf_counter()
    mx, my = Q2Mesh1D(tx), Q2Mesh1D(ty)
    lm = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    kron = sp.kron
    Kuu = (lm + 2 * mu) * kron(mx.S, my.M) + mu * kron(mx.M, my.S)
    Kvv = mu * kron(mx.S, my.M) + (lm + 2 * mu) * kron(mx.M, my.S)
    Kuv = lm * kron(mx.C, my.C.T) + mu * kron(mx.C.T, my.C)
    K = sp.bmat([[Kuu, Kuv], [Kuv.T, Kvv]]).tocsr()
    nx, ny = mx.n, my.n; n = nx * ny
    X, Y = np.meshgrid(mx.nodes, my.nodes, indexing="ij")
    ue, ve = field.disp(X, Y)
    bmask = np.zeros((nx, ny), bool)
    if "L" in dirichlet: bmask[0, :] = True
    if "R" in dirichlet: bmask[-1, :] = True
    if "B" in dirichlet: bmask[:, 0] = True
    if "T" in dirichlet: bmask[:, -1] = True
    fixed = np.concatenate([bmask.ravel(), bmask.ravel()])
    g = np.concatenate([np.where(bmask, ue, 0).ravel(), np.where(bmask, ve, 0).ravel()])
    free = ~fixed
    rhs = -K[free][:, fixed] @ g[fixed]
    Kff = K[free][:, free].tocsr()
    info = dict(its=0, setup=0.0, assembly=time.perf_counter() - ta,
                mem=Kff.data.nbytes + Kff.indices.nbytes + Kff.indptr.nbytes)
    if solver == "direct":
        t0 = time.perf_counter()
        sol = spla.spsolve(Kff.tocsc(), rhs)
        tsolve = time.perf_counter() - t0
    elif solver == "amg":
        import pyamg
        # rigid-body modes (near-nullspace) for 2D elasticity on the free nodes
        fx = np.ones(nx, bool); fy = np.ones(ny, bool)
        XN, YN = np.meshgrid(mx.nodes, my.nodes, indexing="ij")
        Bm = np.zeros((2 * n, 3)); Bm[:n, 0] = 1; Bm[n:, 1] = 1
        Bm[:n, 2] = -YN.ravel(); Bm[n:, 2] = XN.ravel()
        Bm = Bm[free]
        t0 = time.perf_counter()
        opts = dict(strength="symmetric", max_coarse=500)
        opts.update(amg_options or {})
        ml = pyamg.smoothed_aggregation_solver(Kff, B=Bm, **opts)
        info["setup"] = time.perf_counter() - t0
        info["mem"] += sum(l.A.data.nbytes + l.A.indices.nbytes for l in ml.levels)
        res = []
        t0 = time.perf_counter()
        sol = ml.solve(rhs, tol=tol, accel="cg", residuals=res, maxiter=maxiter)
        tsolve = time.perf_counter() - t0
        info["its"] = len(res) - 1
        info["converged"] = bool(info["its"] < maxiter)
    else:
        # free node sets are tensor products of 1D free sets
        fx = np.ones(nx, bool); fy = np.ones(ny, bool)
        if "L" in dirichlet: fx[0] = False
        if "R" in dirichlet: fx[-1] = False
        if "B" in dirichlet: fy[0] = False
        if "T" in dirichlet: fy[-1] = False
        assert (bmask.ravel() == ~np.outer(fx, fy).ravel()).all()
        t0 = time.perf_counter()
        lx, Vx = sla.eigh(mx.S.toarray()[np.ix_(fx, fx)], mx.M.toarray()[np.ix_(fx, fx)])
        ly, Vy = sla.eigh(my.S.toarray()[np.ix_(fy, fy)], my.M.toarray()[np.ix_(fy, fy)])
        a, b = lm + 2 * mu, mu
        Du = a * lx[:, None] + b * ly[None, :]; Dv = b * lx[:, None] + a * ly[None, :]
        info["setup"] = time.perf_counter() - t0
        info["mem"] += Vx.nbytes + Vy.nbytes + Du.nbytes + Dv.nbytes + 5 * rhs.nbytes
        mxf, myf = fx.sum(), fy.sum(); m = mxf * myf

        def prec(r):
            ru, rv = r[:m].reshape(mxf, myf), r[m:].reshape(mxf, myf)
            zu = Vx @ ((Vx.T @ ru @ Vy) / Du) @ Vy.T
            zv = Vx @ ((Vx.T @ rv @ Vy) / Dv) @ Vy.T
            return np.concatenate([zu.ravel(), zv.ravel()])
        t0 = time.perf_counter()
        x = np.zeros_like(rhs); r = rhs.copy(); z = prec(r); p = z.copy(); rz = r @ z; bn = np.linalg.norm(rhs)
        for it in range(5000):
            Ap = Kff @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap
            if np.linalg.norm(r) < tol * bn: break
            z = prec(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        info["its"] = it + 1
        sol = x
        tsolve = time.perf_counter() - t0
    w = g.copy(); w[free] = sol
    U, V = w[:n].reshape(nx, ny), w[n:].reshape(nx, ny)
    Ex, Edx, xq, wx = mx.eval_matrices(); Ey, Edy, yq, wy = my.eval_matrices()
    XQ, YQ = np.meshgrid(xq, yq, indexing="ij"); gr = field.grad(XQ, YQ); W = np.outer(wx, wy)
    num = den = 0.0
    for c, A in ((0, U), (1, V)):
        ax_ = Edx @ (Ey @ A.T).T; ay_ = Ex @ (Edy @ A.T).T
        num += (W * ((ax_ - gr[2 * c]) ** 2 + (ay_ - gr[2 * c + 1]) ** 2)).sum()
        den += (W * (gr[2 * c] ** 2 + gr[2 * c + 1] ** 2)).sum()
    if return_info:
        return float(np.sqrt(num / den)), int(free.sum()), tsolve, info
    return float(np.sqrt(num / den)), int(free.sum()), tsolve
