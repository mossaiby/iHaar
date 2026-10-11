"""Reproduce every table and figure of the manuscript.

    python experiments/run_all.py [--only E1,E2,...]

Results -> results/*.json, figures -> figures/*.pdf.  CPU (NumPy) by default;
pass --gpu to run the phase-field timing study with CuPy.
"""
import json, os, sys, time, argparse
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ihaar.core import Grid1D, corrected_D2, Eig1D, TensorFDM, NeumannDCT
from ihaar.phasefield import SAVPhaseField
from ihaar.elasticity_bc import ElasticityMixed
from ihaar.elasticity import (LAM_I as LAMB, WilliamsField, SmoothField, ElasticityTensor,
                              dorfler, bisect)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as fh:
        json.dump(obj, fh, indent=1)


def rates(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    return [None] + list(np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1]))


# ---------------------------------------------------------------------------
# E1: Poisson-type verification and solver cost
# ---------------------------------------------------------------------------
def E1():
    out = {}
    # (a) Dirichlet: -Lap u = f, u = exp(x) sin(pi y) + x^2 y   (not odd-compatible)
    ue = lambda x, y: np.exp(x) * np.sin(np.pi * y) + x * x * y
    lap = lambda x, y: (1 - np.pi ** 2) * np.exp(x) * np.sin(np.pi * y) + 2 * y
    rows = []
    for N in [8, 16, 32, 64, 128, 256, 512]:
        row = {"N": N}
        for label, mk in [("uniform", lambda N: np.linspace(0, 1, N + 1)),
                          ("graded", lambda N: 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, N + 1))))]:
            t = mk(N); g = Grid1D(t, "D"); X, Y = np.meshgrid(g.x, g.x, indexing="ij")
            bL, bR = ue(0 * g.x, g.x), ue(1 + 0 * g.x, g.x)
            bB, bT = ue(g.x, 0 * g.x), ue(g.x, 1 + 0 * g.x)
            Fm = lap(X, Y) - np.outer(g.e0, bL) - np.outer(g.e1, bR) - np.outer(bB, g.e0) - np.outer(bT, g.e1)
            e = Eig1D(g.D2); fd = TensorFDM([e, e])
            U = fd.solve_diag(Fm, fd.lam_sum)
            row[label] = float(np.abs(U - ue(X, Y)).max())
        rows.append(row)
    out["dirichlet"] = rows
    # (b) Neumann: (I - Lap) u = f with u = cos(pi x)cos(2 pi y) + 0.3 cos(3 pi x)
    ue = lambda x, y: np.cos(np.pi * x) * np.cos(2 * np.pi * y) + 0.3 * np.cos(3 * np.pi * x)
    fn = lambda x, y: (1 + 5 * np.pi ** 2) * np.cos(np.pi * x) * np.cos(2 * np.pi * y) + 0.3 * (1 + 9 * np.pi ** 2) * np.cos(3 * np.pi * x)
    rows = []
    for N in [8, 16, 32, 64, 128, 256, 512]:
        g = Grid1D.uniform(N, bc="N"); X, Y = np.meshgrid(g.x, g.x, indexing="ij")
        row = {"N": N}
        for order in (2, 4):
            fd = NeumannDCT(N, 2, g.h[0], order=order)
            U = fd.solve_diag(fn(X, Y), 1 - fd.lam_sum)
            row[f"order{order}"] = float(np.abs(U - ue(X, Y)).max())
        rows.append(row)
    out["neumann"] = rows
    # (c) Cost of the linear solve: dense flattened (classical 2D HWCM) vs
    #     Bartels-Stewart vs FDM-GEMM vs DCT  for -Lap u = f (Neumann+I shift)
    rows = []
    for N in [16, 32, 64, 128, 256, 512, 1024]:
        g = Grid1D.uniform(N, bc="N"); F = np.random.default_rng(0).standard_normal((N, N))
        row = {"N": N, "dofs": N * N}
        A1 = np.eye(N) - g.D2
        if N <= 64:
            K = np.kron(A1, np.eye(N)) - np.kron(np.eye(N), g.D2)
            t0 = time.perf_counter(); np.linalg.solve(K, F.ravel()); row["dense"] = time.perf_counter() - t0
        t0 = time.perf_counter(); sla.solve_sylvester(A1, -g.D2.T, F); row["bartels_stewart"] = time.perf_counter() - t0
        t0 = time.perf_counter(); e = Eig1D(g.D2, symmetric=True); row["fdm_setup"] = time.perf_counter() - t0
        fd = TensorFDM([e, e])
        t0 = time.perf_counter(); fd.solve_diag(F, 1 - fd.lam_sum); row["fdm_gemm"] = time.perf_counter() - t0
        fdd = NeumannDCT(N, 2, g.h[0])
        t0 = time.perf_counter(); fdd.solve_diag(F, 1 - fdd.lam_sum); row["dct"] = time.perf_counter() - t0
        rows.append(row)
    out["cost"] = rows
    save("E1", out)
    print("E1 dirichlet", [(r["N"], f"{r['uniform']:.2e}", f"{r['graded']:.2e}") for r in out["dirichlet"]])
    print("E1 neumann", [(r["N"], f"{r['order2']:.2e}", f"{r['order4']:.2e}") for r in out["neumann"]])
    print("E1 cost", out["cost"])


# ---------------------------------------------------------------------------
# E2: elasticity
# ---------------------------------------------------------------------------
def E2():
    out = {}
    # (a) smooth manufactured, convergence and GMRES iterations
    rows = []
    for N in [8, 16, 32, 64, 128, 256]:
        f = SmoothField(); t = np.linspace(0, 1, N + 1)
        el = ElasticityTensor(t, t); t0 = time.perf_counter()
        U, V, its, _ = el.solve(f, body=f.body); tt = time.perf_counter() - t0
        _, ei = el.errors(U, V, f); e = el.errors_gauss(U, V, f)
        rows.append(dict(N=N, dofs=2 * N * N, eH1=float(e), einf=float(ei), its=its, time=tt))
    out["smooth"] = rows
    rows = []
    for nu in [0.3, 0.4, 0.45, 0.49, 0.499]:
        r = {"nu": nu}
        for N in [32, 64, 128]:
            f = SmoothField(nu=nu); t = np.linspace(0, 1, N + 1)
            el = ElasticityTensor(t, t, nu=nu); U, V, its, _ = el.solve(f, body=f.body)
            r[f"its{N}"] = its; r[f"e{N}"] = float(el.errors_gauss(U, V, f))
        rows.append(r)
    out["nu"] = rows
    # (b) corner singularity: uniform vs automatic refinement
    fld = WilliamsField()
    rows = []
    for N in [8, 16, 32, 64, 128, 256]:
        t = np.linspace(0, 1, N + 1); el = ElasticityTensor(t, t)
        t0 = time.perf_counter(); U, V, its, _ = el.solve(fld); tt = time.perf_counter() - t0
        _, ei = el.errors(U, V, fld); e = el.errors_gauss(U, V, fld)
        rows.append(dict(N=N, dofs=2 * N * N, eH1=float(e), einf=float(ei), its=its, time=tt))
    out["sing_uniform"] = rows
    rows = []; tx = ty = np.linspace(0, 1, 9); grids = {}
    for cyc in range(200):
        el = ElasticityTensor(tx, ty); t0 = time.perf_counter()
        U, V, its, _ = el.solve(fld); tt = time.perf_counter() - t0
        _, ei = el.errors(U, V, fld); e = el.errors_gauss(U, V, fld); ex, ey = el.indicators(U, V)
        dofs = 2 * el.shape[0] * el.shape[1]
        rows.append(dict(cycle=cyc, nx=el.shape[0], ny=el.shape[1], dofs=dofs, eH1=float(e), einf=float(ei),
                         eta=float(np.sqrt(ex.sum() + ey.sum())), its=its, hmin=float(min(np.diff(tx).min(), np.diff(ty).min())),
                         condV=float(el.condV), time=tt))
        if cyc in (6, 14):
            grids[cyc] = (tx.tolist(), ty.tolist())
        if dofs > 1.2e5:
            break
        mx, my = dorfler(ex, ey, 0.6); tx = bisect(tx, mx); ty = bisect(ty, my)
    grids["final"] = (tx.tolist(), ty.tolist())
    out["sing_adaptive"] = rows; out["grids"] = grids
    save("E2", out)
    print("E2 smooth", [(r["N"], f"{r['eH1']:.2e}", r["its"]) for r in out["smooth"]])
    print("E2 nu", out["nu"])
    print("E2 unif", [(r["dofs"], f"{r['eH1']:.2e}", r["its"]) for r in out["sing_uniform"]])
    print("E2 adapt", [(r["dofs"], f"{r['eH1']:.2e}", r["its"], f"{r['condV']:.1f}") for r in out["sing_adaptive"][::3]])

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    su, sa = out["sing_uniform"], out["sing_adaptive"]
    ax[0].loglog([r["dofs"] for r in su], [r["eH1"] for r in su], "o-", label="uniform")
    ax[0].loglog([r["dofs"] for r in sa], [r["eH1"] for r in sa], ".-", label="adaptive (Haar details)")
    ax[0].loglog([r["dofs"] for r in sa], [r["eta"] * sa[-1]["eH1"] / sa[-1]["eta"] for r in sa], ":", label="indicator (scaled)")
    d = np.array([3e2, 1.2e5])
    ax[0].loglog(d, su[1]["eH1"] * 1.4 * (d / su[1]["dofs"]) ** -0.2722, "k--", lw=0.8, label=r"DOF$^{-\lambda/2}$")
    ax[0].loglog(d, sa[-1]["eH1"] * 0.6 * (d / sa[-1]["dofs"]) ** -1.0, "k-.", lw=0.8, label=r"DOF$^{-1}$")
    ax[0].set_xlabel("DOFs"); ax[0].set_ylabel(r"relative $H^1$ error"); ax[0].legend(fontsize=6)
    gx_, gy_ = grids[14]
    for xv in gx_: ax[1].axvline(xv, lw=0.3, color="k")
    for yv in gy_: ax[1].axhline(yv, lw=0.3, color="k")
    ax[1].set_aspect("equal"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    ax[1].set_title(f"cycle 14: {len(gx_)-1} x {len(gy_)-1} cells"); ax[1].plot([1], [1], "r*", ms=8, clip_on=False)
    for lab, kn in (("x", np.array(grids["final"][0])), ("y", np.array(grids["final"][1]))):
        hh = np.diff(kn); dist = 1 - 0.5 * (kn[:-1] + kn[1:])
        ax[2].loglog(dist, hh, ".", ms=3, label=f"{lab}-cells (final grid)")
    kn = np.array(grids["final"][0]); hh = np.diff(kn); dist = 1 - 0.5 * (kn[:-1] + kn[1:])
    msk = (dist > 1e-5) & (dist < 0.3); pf = np.polyfit(np.log(dist[msk]), np.log(hh[msk]), 1)
    out["grading_slope"] = float(pf[0]); save("E2", out)
    dd = np.logspace(-6, -0.3, 20)
    ax[2].loglog(dd, 2.5 * np.exp(pf[1]) * dd ** pf[0], "k--", lw=0.8, label="fit: $h\\propto d^{%.2f}$" % pf[0])
    ax[2].loglog(dd, 0.6 * np.exp(pf[1]) * dd ** (1 - LAMB / 2), "k:", lw=0.8, label="$d^{1-\\lambda/2}$ (optimal)")
    ax[2].set_xlabel("distance of cell to singular edge"); ax[2].set_ylabel("cell size h"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "elasticity_corner.pdf"))


# ---------------------------------------------------------------------------
# E3: Allen-Cahn shrinking circle
# ---------------------------------------------------------------------------
def E3():
    eps, R0, dt, T = 0.02, 0.3, 0.05, 40.0
    out = {"eps": eps, "R0": R0, "dt": dt, "runs": {}}
    for N, order in [(64, 2), (64, 4), (128, 4), (256, 4)]:
        s = SAVPhaseField(N, 2, eps=eps, dt=dt, model="AC", order=order, transform="dct")
        X, Y = np.meshgrid(s.h * (np.arange(N) + .5), s.h * (np.arange(N) + .5), indexing="ij")
        r = np.hypot(X - .5, Y - .5)
        s.initialize(np.tanh((R0 - r) / (np.sqrt(2) * eps)))
        ts, R2, E, Em = [0.0], [float(s.w * ((1 + s.u) / 2).sum() / np.pi)], [s.energy(s.u)], []
        nst = int(round(T / dt))
        for n in range(nst):
            s.step()
            if (n + 1) % 20 == 0:
                ts.append((n + 1) * dt); R2.append(float(s.w * ((1 + s.u) / 2).sum() / np.pi))
                E.append(s.energy(s.u, s.uhat))
            if n >= 1:
                Em.append(s.modified_energy())
        slope = np.polyfit(ts[len(ts) // 4:], R2[len(ts) // 4:], 1)[0]
        out["runs"][f"N{N}_o{order}"] = dict(t=ts, R2=R2, E=E, slope=float(slope),
                                             max_dEmod=float(np.diff(Em).max()), max_dE=float(np.diff(E).max()))
        print("E3", N, order, "slope", slope, "law", -2 * eps ** 2, "max dEmod", np.diff(Em).max())
    save("E3", out)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    tt = np.linspace(0, T, 50)
    ax[0].plot(tt, R0 ** 2 - 2 * eps ** 2 * tt, "k--", label=r"$R_0^2-2\varepsilon^2 t$")
    for k, v in out["runs"].items():
        ax[0].plot(v["t"], v["R2"], label=k.replace("_o", ", order "))
        ax[1].plot(v["t"], v["E"], label=k)
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$R^2$"); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$E_h$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "allen_cahn_circle.pdf"))


# ---------------------------------------------------------------------------
# E4: Cahn-Hilliard: spatial / temporal convergence, spinodal decomposition
# ---------------------------------------------------------------------------
def ch_smooth_ic(N):
    x = (np.arange(N) + .5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 0.1 + 0.25 * np.cos(2 * np.pi * X) * np.cos(np.pi * Y) + 0.15 * np.cos(3 * np.pi * Y) * np.cos(np.pi * X)


def E4():
    out = {}
    eps, T, dt = 0.05, 0.004, 2e-6
    Ns = [18, 54, 162, 486]
    res = {}
    for order in (2, 4):
        sols = {}
        for N in Ns:
            s = SAVPhaseField(N, 2, eps, dt, "CH", order, transform="dct")
            s.initialize(ch_smooth_ic(N))
            for n in range(int(round(T / dt))): s.step()
            sols[N] = s.u.copy()
        ref = sols[Ns[-1]]
        errs = []
        for N in Ns[:-1]:
            stride = Ns[-1] // N
            sub = ref[stride // 2::stride, stride // 2::stride]
            errs.append(float(np.abs(sols[N] - sub).max()))
        res[f"order{order}"] = errs
        print("E4 spatial order", order, errs, rates(errs, Ns[:-1]))
    out["spatial"] = dict(N=Ns[:-1], **res, T=T, dt=dt, eps=eps)
    # temporal convergence
    N, T = 128, 0.004
    sref = SAVPhaseField(N, 2, eps, T / 6400, "CH", 4, transform="dct"); sref.initialize(ch_smooth_ic(N))
    for n in range(6400): sref.step()
    terr = []; dts = []
    for M in [25, 50, 100, 200, 400]:
        s = SAVPhaseField(N, 2, eps, T / M, "CH", 4, transform="dct"); s.initialize(ch_smooth_ic(N))
        for n in range(M): s.step()
        terr.append(float(np.abs(s.u - sref.u).max())); dts.append(T / M)
    out["temporal"] = dict(dt=dts, err=terr)
    print("E4 temporal", terr, rates(terr, 1 / np.array(dts)))
    # spinodal decomposition
    N, eps, dt, nst = 256, 0.01, 2e-6, 5000
    s = SAVPhaseField(N, 2, eps, dt, "CH", 4, transform="dct")
    rng = np.random.default_rng(7)
    s.initialize(0.1 + 0.05 * rng.uniform(-1, 1, (N, N)))
    m0 = s.mass(s.u); E = [s.energy(s.u)]; Em = []; Em2 = []; mdev = []; xi = []; ts = [0.0]; snaps = {}
    t0 = time.perf_counter()
    for n in range(nst):
        s.step()
        if n >= 1: Em.append(s.modified_energy())
        if (n + 1) % 25 == 0:
            E.append(s.energy(s.u, s.uhat)); ts.append((n + 1) * dt)
            mdev.append(abs(s.mass(s.u) - m0)); xi.append(s.r / np.sqrt(s.E1(s.u) + s.C))
        if (n + 1) in (1000, 5000): snaps[(n + 1) * dt] = s.u.copy()
    tt = time.perf_counter() - t0
    # continuation with a 10x larger step (SAV restart), coarsening stage
    t_now = nst * dt; dt2 = 2e-5; s2 = SAVPhaseField(N, 2, eps, dt2, "CH", 4, transform="dct"); s2.initialize(s.u)
    for n in range(4500):
        s2.step()
        if (n + 1) % 25 == 0:
            E.append(s2.energy(s2.u, s2.uhat)); ts.append(t_now + (n + 1) * dt2)
            mdev.append(abs(s2.mass(s2.u) - m0)); xi.append(s2.r / np.sqrt(s2.E1(s2.u) + s2.C))
        if n >= 1: Em2.append(s2.modified_energy())
    snaps[t_now + 4500 * dt2] = s2.u.copy()
    out["spinodal"] = dict(N=N, eps=eps, dt=dt, steps=nst, t=ts, E=E, max_mass_dev=float(max(mdev)),
                           max_dEmod=float(max(np.diff(Em).max(), np.diff(Em2).max())), max_dE=float(np.diff(E).max()),
                           xi_min=float(min(xi)), xi_max=float(max(xi)), time_per_step=tt / nst)
    print("E4 spinodal mass", max(mdev), "dEmod", max(np.diff(Em).max(), np.diff(Em2).max()), "dE", np.diff(E).max(), "xi", min(xi), max(xi), tt / nst)
    save("E4", out)
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.5))
    for k, (tk, u) in enumerate(sorted(snaps.items())):
        ax[k].imshow(u.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-1, vmax=1)
        ax[k].set_title(f"t = {tk:.4f}"); ax[k].set_xticks([]); ax[k].set_yticks([])
    ax[3].semilogx(ts[1:], E[1:]); ax[3].set_xlabel("t"); ax[3].set_ylabel(r"$E_h$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cahn_hilliard_spinodal.pdf"))
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for o, mk in ((2, "o-"), (4, "s-")):
        ax[0].loglog(Ns[:-1], res[f"order{o}"], mk, label=f"order {o} operator")
    n_ = np.array(Ns[:-1], float)
    ax[0].loglog(n_, res["order2"][0] * (n_ / n_[0]) ** -2, "k--", lw=.7)
    ax[0].loglog(n_, res["order4"][0] * (n_ / n_[0]) ** -4, "k-.", lw=.7)
    ax[0].set_xlabel("N"); ax[0].set_ylabel(r"max error"); ax[0].legend(fontsize=7)
    ax[1].loglog(dts, terr, "o-"); ax[1].loglog(dts, terr[0] * (np.array(dts) / dts[0]) ** 2, "k--", lw=.7)
    ax[1].set_xlabel(r"$\Delta t$"); ax[1].set_ylabel("max error")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cahn_hilliard_convergence.pdf"))


# ---------------------------------------------------------------------------
# E5: 3D Cahn-Hilliard
# ---------------------------------------------------------------------------
def E5():
    N, eps, dt, nst = 96, 0.02, 5e-6, 1500
    s = SAVPhaseField(N, 3, eps, dt, "CH", 4, transform="dct")
    rng = np.random.default_rng(11)
    s.initialize(0.1 + 0.05 * rng.uniform(-1, 1, (N, N, N)))
    m0 = s.mass(s.u); E = [s.energy(s.u)]; Em = []
    t0 = time.perf_counter()
    for n in range(nst):
        s.step()
        if n >= 1: Em.append(s.modified_energy())
        if (n + 1) % 50 == 0: E.append(s.energy(s.u, s.uhat))
    tt = time.perf_counter() - t0
    out = dict(N=N, eps=eps, dt=dt, steps=nst, mass_dev=float(abs(s.mass(s.u) - m0)), mass0=float(m0),
               max_dEmod=float(np.diff(Em).max()), max_dE=float(np.diff(E).max()), time_per_step=tt / nst, E=E)
    save("E5", out)
    print("E5", {k: v for k, v in out.items() if k != "E"})
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for k, idx in enumerate([N // 4, N // 2, 3 * N // 4]):
        ax[k].imshow(s.u[:, :, idx].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax[k].set_title(f"z = {(idx + .5) / N:.2f}"); ax[k].set_xticks([]); ax[k].set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cahn_hilliard_3d.pdf"))


# ---------------------------------------------------------------------------
# E7: traction (Neumann) boundary conditions
# ---------------------------------------------------------------------------
def E7():
    out = {}
    # (a) 1D: plain vs enhanced collocation of a Neumann end, u'' = f
    ue = lambda x: np.exp(np.sin(2 * x)) + x ** 3
    up = lambda x: 2 * np.cos(2 * x) * np.exp(np.sin(2 * x)) + 3 * x ** 2
    upp = lambda x: np.exp(np.sin(2 * x)) * (4 * np.cos(2 * x) ** 2 - 4 * np.sin(2 * x)) + 6 * x
    from ihaar.elasticity_bc import Ext1D
    rows = []
    for alpha in (-1 / 12, 0.0, 0.25, 0.5):
        errs = []
        for N in (32, 64, 128, 256, 512):
            ex = Ext1D(np.linspace(0, 1, N + 1)); h = ex.h
            M = np.vstack([ex.D2[:, 1:], ex.rb[1:] + alpha * h[-1] * ex.D2[-1, 1:]])
            b = np.r_[upp(ex.g.x) - ex.D2[:, 0] * ue(0.),
                      up(1.) - ex.rb[0] * ue(0.) + alpha * h[-1] * (upp(1.) - ex.D2[-1, 0] * ue(0.))]
            sol = np.linalg.solve(M, b); errs.append(float(np.abs(sol[:-1] - ue(ex.g.x)).max()))
        rows.append(dict(alpha=alpha, N=[32, 64, 128, 256, 512], err=errs))
    out["enhanced1d"] = rows
    # (b) smooth manufactured solution, several boundary configurations
    f = SmoothField(); cfgs = {"T": {"T": "T"}, "T+R": {"T": "T", "R": "T"}, "L+B": {"L": "T", "B": "T"}, "T+B": {"T": "T", "B": "T"}}
    sm = {}
    for name, bc in cfgs.items():
        rr = []
        for N in (16, 32, 64, 128, 256):
            t = np.linspace(0, 1, N + 1); el = ElasticityMixed(t, t, bc)
            Ue, Ve, its, _ = el.solve(f, body_fn=f.body)
            rr.append(dict(N=N, eH1=el.errors_gauss(Ue, Ve, f), eedge=el.edge_error(Ue, Ve, f), its=its))
        sm[name] = rr
    out["smooth"] = sm
    rows = []
    for nu in (0.3, 0.4, 0.45, 0.49, 0.499):
        fn = SmoothField(nu=nu); r = {"nu": nu}
        for N in (32, 64, 128):
            t = np.linspace(0, 1, N + 1); el = ElasticityMixed(t, t, {"T": "T", "R": "T"}, nu=nu)
            Ue, Ve, its, _ = el.solve(fn, body_fn=fn.body); r[f"its{N}"] = its; r[f"e{N}"] = el.errors_gauss(Ue, Ve, fn)
        rows.append(r)
    out["nu"] = rows
    # (c) corner singularity with a traction-free face
    fld = WilliamsField(apex=(0.0, 1.0)); bc = {"T": "T"}
    xs = np.linspace(0.01, 1, 50); a, b_, c, d = fld.grad(xs, 1 + 0 * xs)
    out["face_traction_check"] = float(max(np.abs(fld.G * (b_ + c)).max(), np.abs(fld.lmb * a + (fld.lmb + 2 * fld.G) * d).max()))
    rows = []
    for N in (8, 16, 32, 64, 128, 256):
        t = np.linspace(0, 1, N + 1); el = ElasticityMixed(t, t, bc)
        t0 = time.perf_counter(); Ue, Ve, its, _ = el.solve(fld); tt = time.perf_counter() - t0
        rows.append(dict(N=N, dofs=2 * el.nunk, eH1=el.errors_gauss(Ue, Ve, fld), its=its, time=tt))
    out["sing_uniform"] = rows
    rows = []; tx = ty = np.linspace(0, 1, 9); grids = {}
    for cyc in range(200):
        el = ElasticityMixed(tx, ty, bc); t0 = time.perf_counter()
        Ue, Ve, its, _ = el.solve(fld); tt = time.perf_counter() - t0
        e = el.errors_gauss(Ue, Ve, fld); etx, ety = el.indicators(Ue, Ve); dofs = 2 * el.nunk
        rows.append(dict(cycle=cyc, dofs=dofs, eH1=e, eta=float(np.sqrt(etx.sum() + ety.sum())), its=its,
                         hmin=float(min(np.diff(tx).min(), np.diff(ty).min())), condV=float(el.condV), time=tt))
        if cyc == 12: grids[12] = (tx.tolist(), ty.tolist())
        if dofs > 1.2e5: break
        mx, my = dorfler(etx, ety, 0.6); tx = bisect(tx, mx); ty = bisect(ty, my)
    out["sing_adaptive"] = rows; out["grids"] = grids
    sa = rows; x_ = np.array([r["dofs"] for r in sa]); y_ = np.array([r["eH1"] for r in sa]); m = x_ > 3000
    out["adaptive_rate"] = float(np.polyfit(np.log(x_[m]), np.log(y_[m]), 1)[0])
    save("E7", out)
    print("E7 1D enhanced", [(r["alpha"], ["%.1e" % v for v in r["err"]]) for r in out["enhanced1d"]])
    print("E7 smooth", {k: [("%.2e" % r["eH1"], r["its"]) for r in v] for k, v in sm.items()})
    print("E7 nu", out["nu"]); print("E7 sing unif", [(r["dofs"], "%.2e" % r["eH1"], r["its"]) for r in out["sing_uniform"]])
    print("E7 sing adapt", [(r["dofs"], "%.2e" % r["eH1"], r["its"], "%.1f" % r["condV"]) for r in sa[::3]], "rate", out["adaptive_rate"])
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.0))
    su = out["sing_uniform"]
    ax[0].loglog([r["dofs"] for r in su], [r["eH1"] for r in su], "o-", label="uniform")
    ax[0].loglog([r["dofs"] for r in sa], [r["eH1"] for r in sa], ".-", label="adaptive (Haar details)")
    ax[0].loglog([r["dofs"] for r in sa], [r["eta"] * sa[-1]["eH1"] / sa[-1]["eta"] for r in sa], ":", label="indicator (scaled)")
    d = np.array([3e2, 1.2e5])
    ax[0].loglog(d, su[1]["eH1"] * 1.4 * (d / su[1]["dofs"]) ** -0.2722, "k--", lw=.8, label=r"DOF$^{-\lambda/2}$")
    ax[0].loglog(d, sa[-1]["eH1"] * 0.6 * (d / sa[-1]["dofs"]) ** -1.0, "k-.", lw=.8, label=r"DOF$^{-1}$")
    ax[0].set_xlabel("DOFs"); ax[0].set_ylabel(r"relative $H^1$ error"); ax[0].legend(fontsize=6)
    gx_, gy_ = grids[12]
    for xv in gx_: ax[1].axvline(xv, lw=0.3, color="k")
    for yv in gy_: ax[1].axhline(yv, lw=0.3, color="k")
    ax[1].plot([0, 1], [1, 1], color="tab:red", lw=3, clip_on=False, label="traction-free face")
    ax[1].plot([0], [1], "r*", ms=9, clip_on=False)
    ax[1].set_aspect("equal"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=6, loc="lower right")
    ax[1].set_title(f"cycle 12: {len(gx_)-1} x {len(gy_)-1} cells")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "elasticity_traction.pdf"))


# ---------------------------------------------------------------------------
# E8: very sharp interfaces (Allen-Cahn / Cahn-Hilliard)
# ---------------------------------------------------------------------------
def _grad_mag(u, N):
    g = Grid1D.uniform(N, bc="N")
    ux = g.D1 @ u; uy = u @ g.D1.T
    return np.hypot(ux, uy)


def E8a():
    from ihaar.elasticity_bc import Ext1D
    eps = 0.01; rows = []
    for N in (12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        ex = Ext1D(np.linspace(0.5, 1.0, N + 1)); h = ex.h[0]; x = ex.g.x
        D2, F, SU, Sq, ends = ex.mixed(False, True)            # Dirichlet (u=0) at 1/2, Neumann at 1
        d3 = (np.diag(-2 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
        d3[0, 0] = -3; d3[-1, -1] = -1; d3 /= h * h            # odd / even ghost values
        ue = np.tanh((x - 0.5) / (np.sqrt(2) * eps))
        row = {"N": 2 * N, "ppw": float(np.sqrt(2) * eps / h)}
        for name, D in (("fd3", d3), ("ihwc2", D2), ("ihwc4", D2 + h * h / 24 * D2 @ D2)):
            U = ue.copy()
            for it in range(60):
                Fv = eps ** 2 * D @ U - (U ** 3 - U)
                J = eps ** 2 * D - np.diag(3 * U ** 2 - 1)
                dU = np.linalg.solve(J, -Fv); U += dU
                if np.abs(dU).max() < 1e-13: break
            row[name] = float(np.abs(U - ue).max())
        rows.append(row)
    print("E8a", [(r["N"], round(r["ppw"], 2), "%.1e %.1e %.1e" % (r["fd3"], r["ihwc2"], r["ihwc4"])) for r in rows])
    save("E8a", {"resolution": rows})
    return {"resolution": rows}


def _cos_eval(ref, Nc):
    """Evaluate the cosine (DCT-II) interpolant of a fine Neumann grid function at
    the midpoints of a coarse N_c grid (exact, no averaging bias)."""
    import scipy.fft as sfft
    Nf = ref.shape[0]
    C = sfft.dctn(ref, type=2, norm="ortho")
    k = np.arange(Nf); xc = (np.arange(Nc) + .5) / Nc
    B = np.sqrt(2.0 / Nf) * np.cos(np.pi * np.outer(xc, k)); B[:, 0] = np.sqrt(1.0 / Nf)
    return B @ C @ B.T


def _star_ic(N, eps):
    x = (np.arange(N) + .5) / N; X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X - .5, Y - .5); th = np.arctan2(Y - .5, X - .5)
    return np.tanh((0.25 + 0.08 * np.cos(6 * th) - r) / (np.sqrt(2) * eps))


def E8b():
    """Allen-Cahn six-fold star with eps = 0.004."""
    eps, T, dt = 0.004, 120.0, 0.1
    star = {}
    for N, order in ((1024, 4), (256, 2), (256, 4), (512, 2), (512, 4)):
        s = SAVPhaseField(N, 2, eps, dt, "AC", order, transform="dct"); s.initialize(_star_ic(N, eps))
        E = [s.energy(s.u)]; t0 = time.perf_counter()
        for n in range(int(round(T / dt))):
            s.step()
            if (n + 1) % 50 == 0: E.append(s.energy(s.u, s.uhat))
        np.save(os.path.join(RES, f"star_N{N}_o{order}.npy"), s.u)
        star[f"N{N}_o{order}"] = dict(N=N, order=order, time_per_step=(time.perf_counter() - t0) / int(round(T / dt)),
                                      max_dE=float(np.diff(E).max()), gradmax=float(_grad_mag(s.u, N).max()))
        del s
    ref = np.load(os.path.join(RES, "star_N1024_o4.npy"))
    for key, v in star.items():
        if v["N"] == 1024: continue
        u = np.load(os.path.join(RES, f"star_N{v['N']}_o{v['order']}.npy"))
        v["err_vs_ref"] = float(np.abs(u - _cos_eval(ref, v["N"])).max())
        v["area_err"] = float(abs((1 + u).mean() / 2 - (1 + ref).mean() / 2))
    save("E8b", dict(star=star, params=dict(eps=eps, T=T, dt=dt, nodes_per_w_256=float(np.sqrt(2) * eps * 256))))
    print("E8b", star)


CH_SHARP = dict(N=1024, eps=0.0015, S=4.0, stages=[(2e-7, 1500), (2e-6, 1500), (2e-5, 1500)])


def E8c_stage(k):
    """Run stage k of the thin-interface Cahn-Hilliard example (resumable)."""
    N, eps, stages = CH_SHARP["N"], CH_SHARP["eps"], CH_SHARP["stages"]
    part = os.path.join(RES, "E8c_part.json")
    if k == 0:
        u = 0.02 * np.random.default_rng(5).uniform(-1, 1, (N, N)); st = dict(t=[0.0], E=[], mdev=[], dEm=[], m0=None, tps=[], t_now=0.0)
    else:
        u = np.load(os.path.join(RES, f"ch_sharp_{k-1}.npy")); st = json.load(open(part))
    dt_, nst = stages[k]
    s = SAVPhaseField(N, 2, eps, dt_, "CH", 4, transform="dct", S=CH_SHARP["S"]); s.initialize(u)
    st.setdefault("xi", [])
    if st["m0"] is None: st["m0"] = s.mass(s.u); st["E"].append(s.energy(s.u))
    Em_prev, dmax, t0 = None, -np.inf, time.perf_counter()
    for n in range(nst):
        s.step()
        if n >= 1:
            Em = s.modified_energy()
            if Em_prev is not None: dmax = max(dmax, Em - Em_prev)
            Em_prev = Em
        if (n + 1) % 50 == 0:
            st["E"].append(s.energy(s.u, s.uhat)); st["t"].append(st["t_now"] + (n + 1) * dt_); st["mdev"].append(abs(s.mass(s.u) - st["m0"]))
            st["xi"].append(float(s.r / np.sqrt(s.E1(s.u) + s.C)))
    st["tps"].append((time.perf_counter() - t0) / nst); st["dEm"].append(float(dmax)); st["t_now"] += nst * dt_
    np.save(os.path.join(RES, f"ch_sharp_{k}.npy"), s.u)
    json.dump(st, open(part, "w"))
    print("E8c stage", k, "done", st["tps"][-1])


def E8c():
    for k in range(len(CH_SHARP["stages"])):
        E8c_stage(k)
    E8c_finalize()


def E8c_finalize():
    N, eps, stages = CH_SHARP["N"], CH_SHARP["eps"], CH_SHARP["stages"]
    st = json.load(open(os.path.join(RES, "E8c_part.json")))
    u = np.load(os.path.join(RES, f"ch_sharp_{len(stages)-1}.npy")); gm = _grad_mag(u, N)
    out = dict(N=N, eps=eps, nodes_per_w=float(np.sqrt(2) * eps * N), stages=stages, t=st["t"], E=st["E"],
               max_mass_dev=float(max(st["mdev"])), max_dEmod=max(st["dEm"]), max_dE=float(np.diff(st["E"]).max()),
               gradmax=float(gm.max()), grad_theory=float(1 / (np.sqrt(2) * eps)),
               time_per_step=float(np.mean(st["tps"])), S=CH_SHARP["S"],
               xi_min=float(min(st["xi"])), xi_max=float(max(st["xi"])), umax=float(np.abs(u).max()),
               t_snap=[float(sum(d * n for d, n in stages[:k + 1])) for k in range(len(stages))])
    save("E8c", out)
    print("E8c", {k: v for k, v in out.items() if k not in ("t", "E")})


def E8d():
    """Plain vs stabilized SAV for the thin interface, restarted from t = 0.0033."""
    u0 = np.load(os.path.join(RES, "ch_sharp_1.npy")); rows = []
    for S, dt in ((0.0, 2e-5), (0.0, 5e-6), (4.0, 2e-5), (4.0, 5e-6)):
        s = SAVPhaseField(1024, 2, CH_SHARP["eps"], dt, "CH", 4, transform="dct", S=S); s.initialize(u0)
        E0 = s.energy(s.u); Em = []
        for n in range(150):
            s.step()
            if n >= 1: Em.append(s.modified_energy())
        rows.append(dict(S=S, dt=dt, umax=float(np.abs(s.u).max()), E0=float(E0), E=float(s.energy(s.u)),
                         xi=float(s.r / np.sqrt(s.E1(s.u) + s.C)), max_dEmod=float(np.diff(Em).max())))
        del s
    save("E8d", rows); print("E8d", rows)


def E8fig():
    ext = [0, 1, 0, 1]
    # resolution
    r_ = R8("E8a")["resolution"]; p = [r["ppw"] for r in r_]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for key, lab, mk in (("fd3", "3-point finite differences", "s-"), ("ihwc2", r"integrated Haar, $D_2$", "o-"), ("ihwc4", r"integrated Haar, $D_2^{(4)}$", "^-")):
        ax.loglog(p, [r[key] for r in r_], mk, ms=4, label=lab)
    ax.set_xlabel(r"nodes per $\sqrt{2}\,\varepsilon$"); ax.set_ylabel("max nodal error"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "interface_resolution.pdf")); plt.close(fig)
    # star
    P = R8("E8b")["params"]; eps, T = P["eps"], P["T"]
    L = lambda N, o: np.load(os.path.join(RES, f"star_N{N}_o{o}.npy"))
    ref = L(1024, 4); ic = _star_ic(1024, eps); xf = (np.arange(1024) + .5) / 1024
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    ax[0].contour(xf, xf, ic.T, [0], colors="0.6", linestyles=":")
    styles = (((1024, 4), "k", "-"), ((256, 2), "tab:red", "--"), ((256, 4), "tab:blue", "-."))
    for (N, o), col, ls in styles:
        xx = (np.arange(N) + .5) / N; ax[0].contour(xx, xx, L(N, o).T, [0], colors=col, linestyles=ls, linewidths=1)
    ax[0].plot([], [], ":", c="0.6", label="t = 0"); ax[0].plot([], [], "k-", label=r"1024$^2$, $D_2^{(4)}$ (ref.)")
    ax[0].plot([], [], "--", c="tab:red", label=r"256$^2$, $D_2$"); ax[0].plot([], [], "-.", c="tab:blue", label=r"256$^2$, $D_2^{(4)}$")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=6, loc="lower left"); ax[0].set_title(f"zero level set, t = {T:g}")
    ax[0].set_xlim(0.12, 0.88); ax[0].set_ylim(0.12, 0.88)
    xx = (np.arange(256) + .5) / 256
    ax[1].contour(xf, xf, ref.T, [0], colors="k")
    ax[1].contour(xx, xx, L(256, 2).T, [0], colors="tab:red", linestyles="--")
    ax[1].contour(xx, xx, L(256, 4).T, [0], colors="tab:blue", linestyles="-.")
    row = ref[:, 512]; kk = np.where((row[:-1] > 0) & (row[1:] <= 0))[0][-1]; xt = xf[kk]
    GX, GY = np.meshgrid(xx, xx, indexing="ij"); box = (xt - 0.05, xt + 0.03, 0.46, 0.54)
    m = (GX > box[0]) & (GX < box[1]) & (GY > box[2]) & (GY < box[3])
    ax[1].plot(GX[m], GY[m], ".", ms=1.5, c="0.5"); ax[1].set_xlim(box[:2]); ax[1].set_ylim(box[2:]); ax[1].set_aspect("equal")
    ax[1].set_title(r"zoom at a tip (dots: 256$^2$ nodes)")
    im = ax[2].imshow(_grad_mag(ref, 1024).T, origin="lower", extent=ext, cmap="magma")
    ax[2].set_title(r"$|\nabla u_h|$ (1024$^2$)"); plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "allen_cahn_star.pdf")); plt.close(fig)
    # CH sharp
    C = R8("E8c"); N, eps = C["N"], C["eps"]
    snaps = [np.load(os.path.join(RES, f"ch_sharp_{k}.npy")) for k in range(3)]; u = snaps[-1]; gm = _grad_mag(u, N)
    fig = plt.figure(figsize=(10, 5.6)); gs = fig.add_gridspec(2, 4)
    for k, uk in enumerate(snaps):
        a = fig.add_subplot(gs[0, k]); a.imshow(uk.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_title(f"t = {C['t_snap'][k]:.4f}"); a.set_xticks([]); a.set_yticks([])
        if k == 2: a.add_patch(plt.Rectangle((0.45, 0.45), 0.05, 0.05, fill=False, ec="k", lw=1))
    a = fig.add_subplot(gs[0, 3]); a.loglog(C["t"][1:], C["E"][1:]); a.set_xlabel("t"); a.set_ylabel(r"$E_h$"); a.set_title("discrete energy")
    a = fig.add_subplot(gs[1, 0]); i0, i1 = int(0.45 * N), int(0.50 * N); edges = (np.arange(i0, i1 + 1)) / N
    a.pcolormesh(edges, edges, u[i0:i1, i0:i1].T, cmap="RdBu_r", vmin=-1, vmax=1, edgecolors="0.75", linewidth=0.08)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title("zoom: one square = one node")
    a = fig.add_subplot(gs[1, 1]); im = a.imshow(gm.T, origin="lower", extent=ext, cmap="magma"); a.set_xticks([]); a.set_yticks([])
    a.set_title(r"$|\nabla u_h|$"); plt.colorbar(im, ax=a, fraction=0.046)
    a = fig.add_subplot(gs[1, 2:])
    j = int(np.argmax(gm.max(axis=0))); row = u[:, j]; xr = (np.arange(N) + .5) / N
    k = int(np.argmax(np.abs(np.diff(row)))); sl = slice(max(k - 12, 0), k + 14)
    sgn = np.sign(row[k + 1] - row[k])
    # sub-cell interface position from linear interpolation of the zero crossing
    xc = xr[k] - row[k] * (xr[k + 1] - xr[k]) / (row[k + 1] - row[k])
    z = np.linspace(xr[sl][0], xr[sl][-1], 400)
    a.plot(z, sgn * np.tanh((z - xc) / (np.sqrt(2) * eps)), "k-", lw=0.8, label=r"$\tanh(d/\sqrt{2}\varepsilon)$")
    a.plot(xr[sl], row[sl], "o", ms=3.5, label="nodal values (1D cut)")
    a.set_xlabel("x"); a.set_ylabel("u"); a.legend(fontsize=7); a.set_title("profile across an interface")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cahn_hilliard_sharp.pdf")); plt.close(fig)


def R8(name):
    with open(os.path.join(RES, name + ".json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# E9: comparison with 3-point FD, compact (Pade) 4th order and cosine spectral
# ---------------------------------------------------------------------------
SCHEMES = (("fd2", "3-point FD"), ("ihwc2", r"IHWC $D_2$"), ("ihwc4", r"IHWC $D_2^{(4)}$"),
           ("pade4", "compact 4th order (Pad\\'e)"), ("spectral", "cosine pseudo-spectral"))


def E9():
    from ihaar.core import laplacian_symbol
    out = {}
    # (a) planar interface: half domain, quarter-wave basis (odd at 1/2, even at 1)
    eps = 0.01; rows = []
    for N in (16, 24, 32, 48, 64, 96, 128, 192, 256):
        h = 0.5 / N; x = 0.5 + h * (np.arange(N) + .5)
        V = np.sqrt(2 / N) * np.sin(np.pi * np.outer(np.arange(N) + .5, np.arange(N) + .5) / N)
        ue = np.tanh((x - 0.5) / (np.sqrt(2) * eps)); row = {"N": 2 * N, "ppw": float(np.sqrt(2) * eps / h)}
        for sc, _ in SCHEMES:
            D = V @ np.diag(laplacian_symbol(N, h, sc, 0.5)) @ V.T; U = ue.copy()
            for it in range(60):
                Fv = eps ** 2 * D @ U - (U ** 3 - U); dU = np.linalg.solve(eps ** 2 * D - np.diag(3 * U ** 2 - 1), -Fv); U += dU
                if np.abs(dU).max() < 1e-13: break
            row[sc] = float(np.abs(U - ue).max())
        rows.append(row)
    out["resolution"] = rows
    print("E9a", [(r["N"], round(r["ppw"], 2), " ".join("%.1e" % r[s] for s, _ in SCHEMES)) for r in rows])
    # (b) Allen-Cahn star at 256^2 and 512^2 (reference: E8b 1024^2 IHWC D2^(4))
    eps, T, dt = 0.004, 120.0, 0.1
    ref = np.load(os.path.join(RES, "star_N1024_o4.npy")); star = {}
    for N in (256, 512):
        for sc, _ in SCHEMES:
            s = SAVPhaseField(N, 2, eps, dt, "AC", 4, transform="dct", scheme=sc); s.initialize(_star_ic(N, eps))
            for n in range(int(round(T / dt))): s.step()
            star[f"{sc}_{N}"] = dict(err=float(np.abs(s.u - _cos_eval(ref, N)).max()),
                                     area=float(abs((1 + s.u).mean() / 2 - (1 + ref).mean() / 2)),
                                     umax=float(np.abs(s.u).max()))
            del s
    out["star"] = star
    print("E9b", star)
    # (c) Cahn-Hilliard smooth self-convergence against a common spectral reference (N = 486)
    epsc, Tc, dtc = 0.05, 0.004, 2e-6; Ns = [18, 54, 162]
    sref = SAVPhaseField(486, 2, epsc, dtc, "CH", 4, transform="dct", scheme="spectral"); sref.initialize(ch_smooth_ic(486))
    for n in range(int(round(Tc / dtc))): sref.step()
    ref = sref.u.copy(); del sref; ch = {}
    for sc, _ in SCHEMES:
        errs = []
        for N in Ns:
            s = SAVPhaseField(N, 2, epsc, dtc, "CH", 4, transform="dct", scheme=sc); s.initialize(ch_smooth_ic(N))
            for n in range(int(round(Tc / dtc))): s.step()
            k = 486 // N; errs.append(float(np.abs(s.u - ref[k // 2::k, k // 2::k]).max()))
        ch[sc] = errs
    out["ch"] = dict(N=Ns, **ch)
    print("E9c", ch)
    save("E9", out)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for (sc, lab), mk in zip(SCHEMES, ("s-", "o-", "^-", "v-", "d-")):
        ax.loglog([r["ppw"] for r in rows], [r[sc] for r in rows], mk, ms=4, label=lab)
    ax.set_xlabel(r"nodes per $\sqrt{2}\,\varepsilon$"); ax.set_ylabel("max nodal error"); ax.grid(alpha=.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scheme_comparison.pdf")); plt.close(fig)


# ---------------------------------------------------------------------------
# E10: IHWC collocation vs Q2 FEM (same uniform / adaptively graded tensor grids)
# ---------------------------------------------------------------------------
def E10():
    from ihaar.fem_q2 import solve_q2
    fld = WilliamsField(apex=(0.0, 1.0)); bc = {"T": "T"}; out = {"uniform": [], "adaptive": []}
    for N in (8, 16, 32, 64, 128):
        t = np.linspace(0, 1, N + 1); el = ElasticityMixed(t, t, bc)
        t0 = time.perf_counter(); Ue, Ve, its, _ = el.solve(fld); ti = time.perf_counter() - t0
        eq, dq, tq = solve_q2(t, t, fld, dirichlet=("L", "R", "B"))
        out["uniform"].append(dict(N=N, ihwc_dofs=2 * el.nunk, ihwc_e=el.errors_gauss(Ue, Ve, fld), ihwc_t=ti,
                                   q2_dofs=dq, q2_e=eq, q2_t=tq))
    tx = ty = np.linspace(0, 1, 9)
    for cyc in range(200):
        el = ElasticityMixed(tx, ty, bc); t0 = time.perf_counter(); Ue, Ve, its, _ = el.solve(fld); ti = time.perf_counter() - t0
        row = dict(cycle=cyc, cells=int((len(tx) - 1) * (len(ty) - 1)), ihwc_dofs=2 * el.nunk, ihwc_e=el.errors_gauss(Ue, Ve, fld), ihwc_t=ti)
        if 2 * (2 * len(tx) - 1) * (2 * len(ty) - 1) < 2.6e5:
            eq, dq, tq = solve_q2(tx, ty, fld, dirichlet=("L", "R", "B")); row.update(q2_dofs=dq, q2_e=eq, q2_t=tq)
        out["adaptive"].append(row)
        if 2 * el.nunk > 1.2e5: break
        etx, ety = el.indicators(Ue, Ve); mx, my = dorfler(etx, ety, 0.6); tx = bisect(tx, mx); ty = bisect(ty, my)
    save("E10", out)
    print("E10 unif", [(r["ihwc_dofs"], "%.2e" % r["ihwc_e"], r["q2_dofs"], "%.2e" % r["q2_e"]) for r in out["uniform"]])
    print("E10 adapt", [(r["ihwc_dofs"], "%.2e" % r["ihwc_e"], "%.2f" % r["ihwc_t"], r.get("q2_dofs"), "%.2e" % r.get("q2_e", 0), "%.2f" % r.get("q2_t", 0)) for r in out["adaptive"][::3]])
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for key, lab, mk in (("uniform", "uniform", "o"), ("adaptive", "IHWC-adapted grids", ".")):
        R_ = out[key]
        ax[0].loglog([r["ihwc_dofs"] for r in R_], [r["ihwc_e"] for r in R_], mk + "-", label=f"IHWC, {lab}")
        Rq = [r for r in R_ if "q2_e" in r]
        ax[0].loglog([r["q2_dofs"] for r in Rq], [r["q2_e"] for r in Rq], mk + "--", label=f"Q2 FEM, {lab}")
        ax[1].loglog([r["ihwc_e"] for r in R_], [r["ihwc_t"] for r in R_], mk + "-", label=f"IHWC (GMRES+FDM), {lab}")
        ax[1].loglog([r["q2_e"] for r in Rq], [r["q2_t"] for r in Rq], mk + "--", label=f"Q2 FEM (sparse direct), {lab}")
    ax[0].set_xlabel("DOFs"); ax[0].set_ylabel(r"relative $H^1$ error"); ax[0].legend(fontsize=6)
    ax[1].set_xlabel(r"relative $H^1$ error"); ax[1].set_ylabel("solve time (s)"); ax[1].invert_xaxis(); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ihwc_vs_q2.pdf")); plt.close(fig)


def E7b():
    """Traction collocation on graded grids (Chebyshev; geometric toward the traction edge)."""
    f = SmoothField(); out = {}
    def geo(N):
        r = 1.15 ** (32 / N); w = (r ** np.arange(N))[::-1]; return np.r_[0, np.cumsum(w) / w.sum()]
    cheb = lambda N: 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, N + 1)))
    for name, gx, gy in (("chebyshev", cheb, cheb), ("geometric", lambda N: np.linspace(0, 1, N + 1), geo)):
        rows = []
        for N in (16, 32, 64, 128):
            tx, ty = gx(N), gy(N); el = ElasticityMixed(tx, ty, {"T": "T", "R": "T"})
            Ue, Ve, its, _ = el.solve(f, body_fn=f.body)
            rows.append(dict(N=N, eH1=el.errors_gauss(Ue, Ve, f), eedge=el.edge_error(Ue, Ve, f), its=its, condV=float(el.condV),
                             hratio=float(max(np.diff(tx).max() / np.diff(tx).min(), np.diff(ty).max() / np.diff(ty).min()))))
        out[name] = rows
    save("E7b", out); print("E7b", out)


# ---------------------------------------------------------------------------
# E6: time per step (CPU here; --gpu for CuPy)
# ---------------------------------------------------------------------------
def E6(gpu=False):
    if gpu:
        import cupy as xp
        sync = lambda: xp.cuda.Stream.null.synchronize()
    else:
        xp = np; sync = lambda: None
    rows = []
    for dim, Ns in ((2, [128, 256, 512, 1024, 2048]), (3, [32, 64, 128])):
        for N in Ns:
            for tr in ("gemm", "dct"):
                if gpu is False and dim == 2 and N == 2048 and tr == "gemm":
                    continue
                for dtype in ((xp.float64, xp.float32) if gpu else (xp.float64,)):
                    s = SAVPhaseField(N, dim, 0.01, 1e-6, "CH", 4, xp=xp, dtype=dtype, transform=tr)
                    u0 = xp.asarray(0.1 + 0.01 * np.random.default_rng(0).standard_normal((N,) * dim), dtype=dtype)
                    s.initialize(u0); s.step(); s.step(); sync()
                    reps = 20 if N ** dim <= 2 ** 20 else 5
                    t0 = time.perf_counter()
                    for _ in range(reps): s.step()
                    sync(); tps = (time.perf_counter() - t0) / reps
                    rows.append(dict(dim=dim, N=N, dofs=N ** dim, transform=tr, dtype=str(np.dtype(dtype).name),
                                     ms_per_step=1e3 * tps, mdofs_per_s=N ** dim / tps / 1e6))
                    print("E6", rows[-1])
    if gpu:
        props = xp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        rv = xp.cuda.runtime.runtimeGetVersion()
        info = dict(gpu=name, cuda=f"{rv // 1000}.{(rv % 1000) // 10}", cupy=xp.__version__)
        save("E6_gpu", dict(info=info, rows=rows))
    else:
        save("E6_cpu", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E1,E2,E3,E4,E5,E7,E7b,E8a,E8b,E8c,E8d,E8fig,E9,E10,E6")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    for name in a.only.split(","):
        t0 = time.time()
        if name == "E6":
            E6(a.gpu)
        elif name.startswith("E8c_stage"):
            E8c_stage(int(name[-1]))
        elif name == "E8c_finalize":
            E8c_finalize()
        else:
            globals()[name]()
        print(f"[{name} done in {time.time() - t0:.1f}s]", flush=True)
