"""
Plane-strain integrated-Haar collocation with mixed Dirichlet / traction edges.

Unknowns are *extended* arrays W~ of shape (Nx+2, Ny+2) per displacement
component: interior midpoint values, edge-midpoint values and corner values.
On Dirichlet edges (and at corners touching a Dirichlet edge) the extended
entries are data; on traction edges they are unknowns, determined by
collocating sigma.n = t at the edge midpoints.  Corners shared by two traction
edges carry two unknowns and are closed by the two normal-stress conditions.

Preconditioner: per component, fast diagonalization with *mixed* 1D operators
(Neumann-type rows at traction ends, Theorem 1), followed by recovery of the
edge values from the eliminated end conditions (exact for the decoupled
operator; the u-v coupling is left to GMRES).
"""
import numpy as np
import scipy.linalg as sla
from .core import Grid1D, Eig1D, TensorFDM
from .elasticity import gmres

EDGES = ("L", "R", "B", "T")


class Ext1D:
    """Extended 1D operators acting on w = [g0, U_1..U_N, g1]."""

    def __init__(self, t):
        g = Grid1D(t, "D")
        self.g, self.N, self.h = g, g.N, g.h
        self.D2 = np.hstack([g.e0[:, None], g.D2, g.e1[:, None]])
        self.D1 = np.hstack([g.f0[:, None], g.D1, g.f1[:, None]])
        h = g.h
        ra = np.zeros(g.N + 2)
        ra[0], ra[1] = -2.0 / h[0], 2.0 / h[0]
        self.ra = ra - (h[0] / 4) * self.D2[0]          # s'(a)
        self.rb = self.ra + h @ self.D2                  # s'(b) = s'(a) + sum c_k h_k

    def mixed(self, left_neu, right_neu):
        """Second-derivative operator with Neumann-type elimination at the
        selected ends: w = P U + S q, q = end slopes, Dirichlet ends = 0.
        Returns D2mix (N x N), F (N x nq), SU (nE x N), Sq (nE x nq), ends."""
        N = self.N
        ends = [i for i, neu in ((0, left_neu), (N + 1, right_neu)) if neu]
        if not ends:
            return self.D2[:, 1:-1].copy(), np.zeros((N, 0)), np.zeros((0, N)), np.zeros((0, 0)), ends
        R = np.vstack([self.ra if e == 0 else self.rb for e in ends])
        Rnn = R[:, ends]
        Rni = R[:, 1:-1]
        Sq = np.linalg.inv(Rnn)
        SU = -Sq @ Rni
        D2mix = self.D2[:, 1:-1] + self.D2[:, ends] @ SU
        F = self.D2[:, ends] @ Sq
        return D2mix, F, SU, Sq, ends


class ElasticityMixed:
    def __init__(self, tx, ty, bc, E=1.0, nu=0.3, xp=np):
        """bc: dict edge -> 'D' or 'T' for edges L (x=a), R (x=b), B (y=a), T (y=b)."""
        self.xp = xp
        self.bc = {e: bc.get(e, "D") for e in EDGES}
        assert not (self.bc["L"] == "T" and self.bc["R"] == "T" and self.bc["B"] == "T" and self.bc["T"] == "T")
        self.ex, self.ey = Ext1D(tx), Ext1D(ty)
        self.Nx, self.Ny = self.ex.N, self.ey.N
        self.lmb = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self._build_masks()
        self._build_precond()

    # ---------------------------------------------------------------------
    def _build_masks(self):
        Nx, Ny, bc = self.Nx, self.Ny, self.bc
        unk = np.zeros((Nx + 2, Ny + 2), bool)
        unk[1:-1, 1:-1] = True
        if bc["L"] == "T": unk[0, 1:-1] = True
        if bc["R"] == "T": unk[-1, 1:-1] = True
        if bc["B"] == "T": unk[1:-1, 0] = True
        if bc["T"] == "T": unk[1:-1, -1] = True
        self.tt_corners = []
        for (i, j, ex_, ey_) in ((0, 0, "L", "B"), (-1, 0, "R", "B"), (0, -1, "L", "T"), (-1, -1, "R", "T")):
            if bc[ex_] == "T" and bc[ey_] == "T":
                unk[i, j] = True
                self.tt_corners.append((i, j, ex_, ey_))
        self.unk = unk
        self.nunk = int(unk.sum())

    def _build_precond(self):
        lm, mu = self.lmb, self.mu
        bc = self.bc
        self.pc = {}
        # component 0 = u, 1 = v ; coefficient of w_xx and w_yy in the Navier row
        coef = {0: (lm + 2 * mu, mu), 1: (mu, lm + 2 * mu)}
        for comp in (0, 1):
            Dx, Fx, SUx, Sqx, endsx = self.ex.mixed(bc["L"] == "T", bc["R"] == "T")
            Dy, Fy, SUy, Sqy, endsy = self.ey.mixed(bc["B"] == "T", bc["T"] == "T")
            eigx, eigy = Eig1D(Dx), Eig1D(Dy)
            fdm = TensorFDM([eigx, eigy])
            ax, ay = coef[comp]
            self.pc[comp] = dict(fdm=fdm, sym=ax * fdm.lams[0][:, None] + ay * fdm.lams[1][None, :],
                                 Fx=Fx, SUx=SUx, Sqx=Sqx, endsx=endsx, Fy=Fy, SUy=SUy, Sqy=Sqy,
                                 endsy=endsy, ax=ax, ay=ay, condV=max(eigx.condV, eigy.condV))
        self.condV = max(p["condV"] for p in self.pc.values())

    # ---------------------------------------------------------------------
    def grads_all(self, W):
        ex, ey = self.ex, self.ey
        d = {}
        d["xx"] = ex.D2 @ W[:, 1:-1]
        d["yy"] = W[1:-1, :] @ ey.D2.T
        Wx_all = ex.D1 @ W                                  # (Nx, Ny+2)
        d["xy"] = Wx_all @ ey.D1.T
        Wy_all = W @ ey.D1.T                                # (Nx+2, Ny)
        d["x_int"], d["y_int"] = Wx_all[:, 1:-1], Wy_all[1:-1, :]
        # edges: (normal derivative, tangential derivative)
        d["T"] = dict(x=Wx_all[:, -1], y=W[1:-1, :] @ ey.rb)
        d["B"] = dict(x=Wx_all[:, 0], y=W[1:-1, :] @ ey.ra)
        d["R"] = dict(x=ex.rb @ W[:, 1:-1], y=Wy_all[-1, :])
        d["L"] = dict(x=ex.ra @ W[:, 1:-1], y=Wy_all[0, :])
        # corners: x-derivative from the horizontal edge trace, y from the vertical one
        d["corner"] = {}
        for (i, j) in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
            rx = ex.ra if i == 0 else ex.rb
            ry = ey.ra if j == 0 else ey.rb
            d["corner"][(i, j)] = dict(x=rx @ W[:, j], y=W[i, :] @ ry)
        return d

    def stresses(self, ux, uy, vx, vy):
        lm, mu = self.lmb, self.mu
        return (lm + 2 * mu) * ux + lm * vy, lm * ux + (lm + 2 * mu) * vy, mu * (uy + vx)

    def residual(self, Ue, Ve, body=None, trac=None):
        """Full residual: interior Navier rows, traction rows, corner rows."""
        lm, mu = self.lmb, self.mu
        du, dv = self.grads_all(Ue), self.grads_all(Ve)
        Ru = (lm + 2 * mu) * du["xx"] + mu * du["yy"] + (lm + mu) * dv["xy"]
        Rv = mu * dv["xx"] + (lm + 2 * mu) * dv["yy"] + (lm + mu) * du["xy"]
        if body is not None:
            Ru, Rv = Ru + body[0], Rv + body[1]
        out = {"int": (Ru, Rv)}
        for e in EDGES:
            if self.bc[e] != "T":
                continue
            sxx, syy, sxy = self.stresses(du[e]["x"], du[e]["y"], dv[e]["x"], dv[e]["y"])
            if e == "T": tx_, ty_ = sxy, syy
            elif e == "B": tx_, ty_ = -sxy, -syy
            elif e == "R": tx_, ty_ = sxx, sxy
            else: tx_, ty_ = -sxx, -sxy
            if trac is not None:
                tx_, ty_ = tx_ - trac[e][0], ty_ - trac[e][1]
            out[e] = (tx_, ty_)
        for (i, j, exn, eyn) in self.tt_corners:
            cu, cv = du["corner"][(i, j)], dv["corner"][(i, j)]
            sxx, syy, sxy = self.stresses(cu["x"], cu["y"], cv["x"], cv["y"])
            rxx, ryy = sxx, syy
            if trac is not None:
                rxx = rxx - trac["corner"][(i, j)][0]
                ryy = ryy - trac["corner"][(i, j)][1]
            out[("c", i, j)] = (rxx, ryy)
        return out

    # ---------------------------------------------------------------------
    def data(self, field, body_fn=None):
        """Extended arrays with Dirichlet data in place; traction data at edges."""
        ex, ey = self.ex, self.ey
        xe = np.r_[ex.g.t[0], ex.g.x, ex.g.t[-1]]
        ye = np.r_[ey.g.t[0], ey.g.x, ey.g.t[-1]]
        X, Y = np.meshgrid(xe, ye, indexing="ij")
        ue, ve = field.disp(X, Y)
        Ue = np.where(self.unk, 0.0, ue)
        Ve = np.where(self.unk, 0.0, ve)
        trac = {"corner": {}}
        lm, mu = self.lmb, self.mu

        def tr(x, y, n):
            a, b, c, d = field.grad(x, y)
            sxx, syy, sxy = (lm + 2 * mu) * a + lm * d, lm * a + (lm + 2 * mu) * d, mu * (b + c)
            return sxx * n[0] + sxy * n[1], sxy * n[0] + syy * n[1], sxx, syy
        if self.bc["T"] == "T": trac["T"] = tr(ex.g.x, 0 * ex.g.x + ey.g.t[-1], (0, 1))[:2]
        if self.bc["B"] == "T": trac["B"] = tr(ex.g.x, 0 * ex.g.x + ey.g.t[0], (0, -1))[:2]
        if self.bc["R"] == "T": trac["R"] = tr(0 * ey.g.x + ex.g.t[-1], ey.g.x, (1, 0))[:2]
        if self.bc["L"] == "T": trac["L"] = tr(0 * ey.g.x + ex.g.t[0], ey.g.x, (-1, 0))[:2]
        for (i, j, _, _) in self.tt_corners:
            x0 = ex.g.t[0] if i == 0 else ex.g.t[-1]
            y0 = ey.g.t[0] if j == 0 else ey.g.t[-1]
            r = tr(np.array(x0), np.array(y0), (1, 0))
            trac["corner"][(i, j)] = (float(r[2]), float(r[3]))
        body = None
        if body_fn is not None:
            Xi, Yi = np.meshgrid(ex.g.x, ey.g.x, indexing="ij")
            body = body_fn(Xi, Yi)
        return Ue, Ve, trac, body

    def pack(self, out):
        parts = [out["int"][0].ravel(), out["int"][1].ravel()]
        for e in EDGES:
            if e in out:
                parts += [np.asarray(out[e][0]).ravel(), np.asarray(out[e][1]).ravel()]
        for (i, j, _, _) in self.tt_corners:
            parts += [np.atleast_1d(out[("c", i, j)][0]), np.atleast_1d(out[("c", i, j)][1])]
        return np.concatenate(parts)

    def unpack(self, r):
        """Inverse of pack: residual vector -> dict."""
        Nx, Ny = self.Nx, self.Ny
        o, k = {}, 0
        Ru = r[k:k + Nx * Ny].reshape(Nx, Ny); k += Nx * Ny
        Rv = r[k:k + Nx * Ny].reshape(Nx, Ny); k += Nx * Ny
        o["int"] = (Ru, Rv)
        for e in EDGES:
            if self.bc[e] == "T":
                n = Nx if e in ("T", "B") else Ny
                o[e] = (r[k:k + n], r[k + n:k + 2 * n]); k += 2 * n
        for (i, j, _, _) in self.tt_corners:
            o[("c", i, j)] = (r[k], r[k + 1]); k += 2
        return o

    def embed(self, x, base_u=None, base_v=None):
        Ue = np.zeros((self.Nx + 2, self.Ny + 2)) if base_u is None else base_u.copy()
        Ve = np.zeros_like(Ue) if base_v is None else base_v.copy()
        n = self.nunk
        Ue[self.unk] = x[:n]
        Ve[self.unk] = x[n:]
        return Ue, Ve

    # ---------------------------------------------------------------------
    def precond(self, r):
        """Approximate inverse: residual vector -> unknown vector."""
        o = self.unpack(r)
        lm, mu = self.lmb, self.mu
        res = []
        for comp in (0, 1):
            p = self.pc[comp]
            Rint = o["int"][comp]
            # end slopes from traction rows (decoupled approximation)
            # u: T,B rows ~ +/- mu u_y ; R,L rows ~ +/- (lm+2mu) u_x
            # v: T,B rows ~ +/- (lm+2mu) v_y ; R,L rows ~ +/- mu v_x
            sx = {0: lm + 2 * mu, 1: mu}[comp]
            sy = {0: mu, 1: lm + 2 * mu}[comp]
            Qx = np.zeros((len(p["endsx"]), self.Ny))
            for k, e in enumerate(p["endsx"]):
                edge = "L" if e == 0 else "R"
                sgn = -1.0 if edge == "L" else 1.0
                Qx[k] = sgn * np.asarray(o[edge][comp]) / sx
            Qy = np.zeros((self.Nx, len(p["endsy"])))
            for k, e in enumerate(p["endsy"]):
                edge = "B" if e == 0 else "T"
                sgn = -1.0 if edge == "B" else 1.0
                Qy[:, k] = sgn * np.asarray(o[edge][comp]) / sy
            rhs = Rint - p["ax"] * (p["Fx"] @ Qx) - p["ay"] * (Qy @ p["Fy"].T)
            dU = p["fdm"].solve_diag(rhs, p["sym"])
            W = np.zeros((self.Nx + 2, self.Ny + 2))
            W[1:-1, 1:-1] = dU
            if p["endsx"]:
                gx = p["SUx"] @ dU + p["Sqx"] @ Qx
                for k, e in enumerate(p["endsx"]):
                    W[0 if e == 0 else -1, 1:-1] = gx[k]
            if p["endsy"]:
                gy = dU @ p["SUy"].T + Qy @ p["Sqy"].T
                for k, e in enumerate(p["endsy"]):
                    W[1:-1, 0 if e == 0 else -1] = gy[:, k]
            for (i, j, _, _) in self.tt_corners:
                hloc = min(self.ex.h[0 if i == 0 else -1], self.ey.h[0 if j == 0 else -1])
                W[i, j] = o[("c", i, j)][comp] * hloc / (lm + 2 * mu)
            res.append(W[self.unk])
        return np.concatenate(res)

    def solve(self, field, body_fn=None, tol=1e-10, restart=80, maxit=3000):
        Ue0, Ve0, trac, body = self.data(field, body_fn)
        self.Ue0, self.Ve0 = Ue0, Ve0
        r0 = self.pack(self.residual(Ue0, Ve0, body, trac))
        zero = {"corner": {c[:2]: (0.0, 0.0) for c in self.tt_corners}}

        def Aop(y):
            x = self.precond(y)
            Ue, Ve = self.embed(x)
            return self.pack(self.residual(Ue, Ve))

        y, its, hist = gmres(Aop, -r0, tol=tol, restart=restart, maxit=maxit)
        x = self.precond(y)
        Ue, Ve = self.embed(x, Ue0, Ve0)
        return Ue, Ve, its, hist

    # ---------------------------------------------------------------------
    def errors_gauss(self, Ue, Ve, field, ng=4):
        ex, ey = self.ex, self.ey
        xg, wg = np.polynomial.legendre.leggauss(ng)

        def q(g):
            xq = (0.5 * (g.t[:-1, None] + g.t[1:, None]) + 0.5 * g.h[:, None] * xg[None, :]).ravel()
            return xq, (0.5 * g.h[:, None] * wg[None, :]).ravel()
        xq, wx = q(ex.g); yq, wy = q(ey.g)
        Ex, Edx = ex.g.eval_matrices(xq); Ey, Edy = ey.g.eval_matrices(yq)
        X, Y = np.meshgrid(xq, yq, indexing="ij")
        g = field.grad(X, Y)
        W = np.outer(wx, wy)
        num = den = 0.0
        for comp, A in ((0, Ue), (1, Ve)):
            ax_ = Edx @ A @ Ey.T; ay_ = Ex @ A @ Edy.T
            num += (W * ((ax_ - g[2 * comp]) ** 2 + (ay_ - g[2 * comp + 1]) ** 2)).sum()
            den += (W * (g[2 * comp] ** 2 + g[2 * comp + 1] ** 2)).sum()
        return float(np.sqrt(num / den))

    def cell_errors(self, Ue, Ve, field, ng=3):
        """Squared absolute H1-seminorm error per 2D cell (Gauss quadrature)."""
        ex, ey = self.ex, self.ey
        xg, wg = np.polynomial.legendre.leggauss(ng)
        def q(g):
            return ((0.5 * (g.t[:-1, None] + g.t[1:, None]) + 0.5 * g.h[:, None] * xg[None, :]).ravel(),
                    (0.5 * g.h[:, None] * wg[None, :]).ravel())
        xq, wx = q(ex.g); yq, wy = q(ey.g)
        Ex, Edx = ex.g.eval_matrices(xq); Ey, Edy = ey.g.eval_matrices(yq)
        X, Y = np.meshgrid(xq, yq, indexing="ij"); gr = field.grad(X, Y); W = np.outer(wx, wy)
        e2 = np.zeros_like(W)
        for comp, A in ((0, Ue), (1, Ve)):
            e2 += W * ((Edx @ A @ Ey.T - gr[2 * comp]) ** 2 + (Ex @ A @ Edy.T - gr[2 * comp + 1]) ** 2)
        return e2.reshape(self.Nx, ng, self.Ny, ng).sum(axis=(1, 3))

    def edge_error(self, Ue, Ve, field):
        """Max displacement error on traction edges (relative)."""
        ex, ey = self.ex, self.ey
        xe = np.r_[ex.g.t[0], ex.g.x, ex.g.t[-1]]; ye = np.r_[ey.g.t[0], ey.g.x, ey.g.t[-1]]
        X, Y = np.meshgrid(xe, ye, indexing="ij")
        ue, ve = field.disp(X, Y)
        scale = max(np.abs(ue).max(), np.abs(ve).max())
        m = self.unk.copy(); m[1:-1, 1:-1] = False
        if not m.any():
            return 0.0
        return float(max(np.abs(Ue - ue)[m].max(), np.abs(Ve - ve)[m].max()) / scale)

    def indicators(self, Ue, Ve):
        ex, ey = self.ex, self.ey
        etx = np.zeros(self.Nx); ety = np.zeros(self.Ny)
        for W in (Ue, Ve):
            Wxx = ex.D2 @ W[:, 1:-1]; Wyy = W[1:-1, :] @ ey.D2.T
            Jx = np.diff(Wxx, axis=0) ** 2; Jy = np.diff(Wyy, axis=1) ** 2
            jx = np.zeros_like(Wxx); jx[:-1] += Jx; jx[1:] += Jx
            jy = np.zeros_like(Wyy); jy[:, :-1] += Jy; jy[:, 1:] += Jy
            etx += ex.h ** 3 * (jx @ ey.h) * 0.5
            ety += ey.h ** 3 * (ex.h @ jy) * 0.5
        return etx, ety
