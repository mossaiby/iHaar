"""Individual figure panels for the JSC manuscript (one PDF per panel), arranged in LaTeX.

    python experiments/make_figures.py

Reads results/*.json and results/*.npy (written by run_all.py) and writes figures/panels/*.pdf.
Panels have no titles (the LaTeX captions describe them), consistent fonts, and tight bounding boxes,
so nothing is cropped or overwritten. The 3D Cahn-Hilliard slices need results/ch3d_state.npy, which
is computed here once (a continuation of the E5 setup; no timing is recorded) if it is missing.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "figures", "panels")
os.makedirs(OUT, exist_ok=True)
R = lambda n: json.load(open(os.path.join(RES, n + ".json")))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "lines.linewidth": 1.1, "lines.markersize": 3.5, "savefig.dpi": 300})
W1, H1 = 2.05, 1.75           # half-width panel (inches); two fit, unscaled, in the 11.9 cm (4.69 in) text block
SQ = 1.2                      # square image panels (three per row, unscaled)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def nodes_axis(ax):
    """Readable ticks for 'nodes per sqrt(2) eps' (avoids overlapping minor log labels)."""
    ax.set_xscale("log")
    t = [0.5, 1, 2, 4, 8]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in t]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"nodes per $\sqrt{2}\,\varepsilon$")


def loglog_dofs(ax, xlabel="unknowns"):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xlabel)
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.grid(alpha=0.25, which="major")


# ----------------------------------------------------------------------------- elasticity, traction
def elasticity_traction():
    E7 = R("E7"); su, sa = E7["sing_uniform"], E7["sing_adaptive"]
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot([r["dofs"] for r in su], [r["eH1"] for r in su], "o-", label="uniform")
    ax.plot([r["dofs"] for r in sa], [r["eH1"] for r in sa], ".-", label="adaptive (Haar details)")
    ax.plot([r["dofs"] for r in sa], [r["eta"] * sa[-1]["eH1"] / sa[-1]["eta"] for r in sa], ":", label="indicator (scaled)")
    d = np.array([3e2, 1.5e5])
    ax.plot(d, su[1]["eH1"] * 1.5 * (d / su[1]["dofs"]) ** -0.2722, "k--", lw=0.7, label=r"$\mathrm{DOF}^{-\lambda/2}$")
    ax.plot(d, sa[-1]["eH1"] * 0.6 * (d / sa[-1]["dofs"]) ** -1.0, "k-.", lw=0.7, label=r"$\mathrm{DOF}^{-1}$")
    loglog_dofs(ax); ax.set_ylabel(r"relative $H^1$ error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
    save(fig, "trac_error")
    tx, ty = map(np.array, E7["grids"]["12"])
    fig, ax = plt.subplots(figsize=(W1, H1))
    for lab, kn, dist in (("$x$-cells", tx, lambda c: c), ("$y$-cells", ty, lambda c: 1 - c)):
        c = 0.5 * (kn[:-1] + kn[1:]); ax.plot(dist(c), np.diff(kn), ".", ms=3, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25)
    ax.set_xlabel("distance to the singular point"); ax.set_ylabel("cell size $h$"); ax.legend(loc="upper left", frameon=False)
    save(fig, "trac_cellsize")
    for name, lim in (("trac_grid", (0, 1, 0, 1)),):
        fig, ax = plt.subplots(figsize=(H1, H1))
        for xv in tx: ax.axvline(xv, lw=0.35, color="0.35")
        for yv in ty: ax.axhline(yv, lw=0.35, color="0.35")
        ax.plot([0, 1], [1, 1], color="tab:red", lw=2.5, solid_capstyle="butt", clip_on=False, zorder=3)
        ax.plot([0], [1], marker="*", color="tab:red", ms=9, clip_on=False, zorder=4, markeredgecolor="k", markeredgewidth=0.4)
        pad = 0.02 * (lim[1] - lim[0])
        ax.set_xlim(lim[0] - pad, lim[1] + pad); ax.set_ylim(lim[2] - pad, lim[3] + pad)
        ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        if name.endswith("zoom"):
            ax.set_xticks([0, 0.02, 0.04]); ax.set_yticks([0.96, 0.98, 1.0])
        save(fig, name)


# ----------------------------------------------------------------------------- IHWC vs Q2
def ihwc_vs_q2():
    E10 = R("E10"); A, U = E10["adaptive"], E10["uniform"]
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot([r["ihwc_dofs"] for r in U], [r["ihwc_e"] for r in U], "o-", c="tab:blue", label="IHWC, uniform")
    ax.plot([r["q2_dofs"] for r in U], [r["q2_e"] for r in U], "s--", c="tab:orange", label="$Q_2$, uniform")
    ax.plot([r["ihwc_dofs"] for r in A], [r["ihwc_e"] for r in A], ".-", c="tab:blue", label="IHWC, adapted grids")
    ax.plot([r["q2_dofs"] for r in A], [r["q2_e"] for r in A], ".--", c="tab:orange", label="$Q_2$, same grids")
    loglog_dofs(ax); ax.set_ylabel(r"relative $H^1$ error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
    save(fig, "q2_error")
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot([r["ihwc_e"] for r in A], [r["ihwc_t"] for r in A], ".-", c="tab:blue", label="IHWC, GMRES + FDM")
    ax.plot([r["q2_e"] for r in A], [r["q2pcg_t"] for r in A], ".--", c="tab:orange", label="$Q_2$, CG + FDM")
    Ad = [r for r in A if "q2direct_t" in r]
    ax.plot([r["q2_e"] for r in Ad], [r["q2direct_t"] for r in Ad], "x:", c="tab:red", label="$Q_2$, sparse direct")
    if os.path.exists(os.path.join(RES, "E13.json")):
        for cfg, st, lab in (("default", "^:", "$Q_2$, CG + AMG (default)"), ("evolution", "v:", "$Q_2$, CG + AMG (evolution)")):
            rows = [r for r in R("E13") if r.get("config", "default") == cfg and r.get("converged", True) and r["amg_its"] > 1]
            if rows:
                ax.plot([r["q2_e"] for r in rows], [r["amg_solve"] for r in rows], st, ms=3, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.grid(alpha=0.25)
    ax.set_xlabel(r"relative $H^1$ error"); ax.set_ylabel("solve time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
    save(fig, "q2_time")


# ----------------------------------------------------------------------------- interface resolution
def interface_resolution():
    r_ = R("E8a")["resolution"]; p = [r["ppw"] for r in r_]
    fig, ax = plt.subplots(figsize=(W1, H1))
    for key, lab, mk in (("fd3", "3-point finite differences", "s-"), ("ihwc2", r"IHWC, $D_2$", "o-"), ("ihwc4", r"IHWC, $D_2^{(4)}$", "^-")):
        ax.plot(p, [r[key] for r in r_], mk, label=lab)
    ax.set_yscale("log"); nodes_axis(ax); ax.set_ylabel("max nodal error"); ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=1, frameon=False)
    save(fig, "resolution")
    E9 = R("E9"); rows = E9["resolution"]
    fig, ax = plt.subplots(figsize=(W1, H1))
    for (sc, lab), mk in zip((("fd2", "3-point FD"), ("ihwc2", r"IHWC, $D_2$"), ("ihwc4", r"IHWC, $D_2^{(4)}$"),
                              ("pade4", "compact (Padé)"), ("spectral", "cosine pseudo-spectral")), ("s-", "o-", "^-", "v-", "d-")):
        ax.plot([r["ppw"] for r in rows], [r[sc] for r in rows], mk, label=lab)
    ax.set_yscale("log"); nodes_axis(ax); ax.set_ylabel("max nodal error"); ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=1, frameon=False)
    save(fig, "scheme_comparison")


# ----------------------------------------------------------------------------- Allen-Cahn star
def _grad_mag(u):
    from ihaar.core import Grid1D
    g = Grid1D.uniform(u.shape[0], bc="N"); return np.hypot(g.D1 @ u, u @ g.D1.T)


def allen_cahn_star():
    P = R("E8b")["params"]; eps = P["eps"]
    L = lambda N, o: np.load(os.path.join(RES, f"star_N{N}_o{o}.npy"))
    ref = L(1024, 4); xf = (np.arange(1024) + .5) / 1024
    x = (np.arange(1024) + .5) / 1024; X, Y = np.meshgrid(x, x, indexing="ij")
    ic = np.tanh((0.25 + 0.08 * np.cos(6 * np.arctan2(Y - .5, X - .5)) - np.hypot(X - .5, Y - .5)) / (np.sqrt(2) * eps))
    styles = (((1024, 4), "k", "-", r"$1024^2$, $D_2^{(4)}$ (reference)"), ((256, 2), "tab:red", "--", r"$256^2$, $D_2$"),
              ((256, 4), "tab:blue", "-.", r"$256^2$, $D_2^{(4)}$"))
    fig, ax = plt.subplots(figsize=(SQ * 1.25, SQ * 1.25))
    ax.contour(xf, xf, ic.T, [0], colors="0.6", linestyles=":", linewidths=1)
    for (N, o), col, ls, lab in styles:
        xx = (np.arange(N) + .5) / N; ax.contour(xx, xx, L(N, o).T, [0], colors=col, linestyles=ls, linewidths=1)
    ax.plot([], [], ":", c="0.6", label="$t=0$")
    for _, col, ls, lab in styles: ax.plot([], [], ls, c=col, label=lab)
    ax.set_aspect("equal"); ax.set_xlim(0.12, 0.88); ax.set_ylim(0.12, 0.88); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    save(fig, "star_levelsets")
    xx = (np.arange(256) + .5) / 256
    row = ref[:, 512]; kk = np.where((row[:-1] > 0) & (row[1:] <= 0))[0][-1]; xt = xf[kk]
    box = (xt - 0.05, xt + 0.03, 0.46, 0.54)
    fig, ax = plt.subplots(figsize=(SQ * 1.25, SQ * 1.25))
    GX, GY = np.meshgrid(xx, xx, indexing="ij"); m = (GX > box[0]) & (GX < box[1]) & (GY > box[2]) & (GY < box[3])
    ax.plot(GX[m], GY[m], ".", ms=1.6, c="0.55")
    ax.contour(xf, xf, ref.T, [0], colors="k", linewidths=1.1)
    ax.contour(xx, xx, L(256, 2).T, [0], colors="tab:red", linestyles="--", linewidths=1.1)
    ax.contour(xx, xx, L(256, 4).T, [0], colors="tab:blue", linestyles="-.", linewidths=1.1)
    ax.set_xlim(box[:2]); ax.set_ylim(box[2:]); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    save(fig, "star_zoom")
    fig, ax = plt.subplots(figsize=(SQ * 1.25, SQ * 1.25))
    im = ax.imshow(_grad_mag(ref).T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$|\nabla u_h|$")
    save(fig, "star_gradient")


# ----------------------------------------------------------------------------- thin-interface Cahn-Hilliard
def cahn_hilliard_sharp():
    C = R("E8c"); N, eps = C["N"], C["eps"]
    snaps = [np.load(os.path.join(RES, f"ch_sharp_{k}.npy")) for k in range(3)]; u = snaps[-1]
    zb = (0.45, 0.50)                      # zoom window
    for k, uk in enumerate(snaps):
        fig, ax = plt.subplots(figsize=(SQ, SQ))
        im = ax.imshow(uk.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if k > 0: ax.set_yticklabels([])
        if k == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("$u_h$")
            ax.add_patch(plt.Rectangle((zb[0], zb[0]), zb[1] - zb[0], zb[1] - zb[0], fill=False, ec="k", lw=1.2))
            ax.add_patch(plt.Rectangle((zb[0], zb[0]), zb[1] - zb[0], zb[1] - zb[0], fill=False, ec="w", lw=0.5))
        save(fig, f"ch_snap{k}")

    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.loglog(C["t"][1:], C["E"][1:]); ax.set_xlabel("$t$"); ax.set_ylabel("discrete energy $E_h$"); ax.grid(alpha=0.25)
    save(fig, "ch_energy")
    i0, i1 = int(zb[0] * N), int(zb[1] * N); edges = np.arange(i0, i1 + 1) / N
    fig, ax = plt.subplots(figsize=(SQ * 1.1, SQ * 1.1))
    ax.pcolormesh(edges, edges, u[i0:i1, i0:i1].T, cmap="RdBu_r", vmin=-1, vmax=1, edgecolors="0.7", linewidth=0.1)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_xticks([0.45, 0.475, 0.5]); ax.set_yticks([0.45, 0.475, 0.5])
    save(fig, "ch_zoom")
    gm = _grad_mag(u)
    fig, ax = plt.subplots(figsize=(SQ * 1.1, SQ * 1.1))
    im = ax.imshow(gm.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$|\nabla u_h|$")
    save(fig, "ch_gradient")
    j = int(np.argmax(gm.max(axis=0))); row = u[:, j]; xr = (np.arange(N) + .5) / N
    k = int(np.argmax(np.abs(np.diff(row)))); sl = slice(max(k - 12, 0), k + 14)
    sgn = np.sign(row[k + 1] - row[k]); xc = xr[k] - row[k] * (xr[k + 1] - xr[k]) / (row[k + 1] - row[k])
    z = np.linspace(xr[sl][0], xr[sl][-1], 400)
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot(z, sgn * np.tanh((z - xc) / (np.sqrt(2) * eps)), "k-", lw=0.8, label=r"$\tanh(d/\sqrt{2}\varepsilon)$")
    ax.plot(xr[sl], row[sl], "o", label="nodal values")
    ax.set_xlabel("$x$"); ax.set_ylabel("$u_h$"); ax.legend(loc="lower right", frameon=False); ax.grid(alpha=0.25)
    save(fig, "ch_profile")


# ----------------------------------------------------------------------------- 3D Cahn-Hilliard slices
def cahn_hilliard_3d():
    p = os.path.join(RES, "ch3d_state.npy")
    if not os.path.exists(p):
        from ihaar.phasefield import SAVPhaseField
        N, eps = 96, 0.02; rng = np.random.default_rng(11)
        s = SAVPhaseField(N, 3, eps, 5e-6, "CH", 4, transform="dct"); s.initialize(0.1 + 0.05 * rng.uniform(-1, 1, (N, N, N)))
        for n in range(1500): s.step()
        s2 = SAVPhaseField(N, 3, eps, 5e-5, "CH", 4, transform="dct", S=2.0); s2.initialize(s.u)
        for n in range(600): s2.step()
        np.save(p, s2.u); json.dump(dict(t=1500 * 5e-6 + 600 * 5e-5), open(os.path.join(RES, "ch3d_state.json"), "w"))
    u = np.load(p); N = u.shape[0]
    for k, idx in enumerate([N // 4, N // 2, 3 * N // 4]):
        fig, ax = plt.subplots(figsize=(SQ, SQ))
        im = ax.imshow(u[:, :, idx].T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if k > 0: ax.set_yticklabels([])
        if k == 2:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("$u_h$")
        save(fig, f"ch3d_slice{k}")


if __name__ == "__main__":
    for f in (elasticity_traction, ihwc_vs_q2, interface_resolution, allen_cahn_star, cahn_hilliard_sharp, cahn_hilliard_3d):
        f(); print("done:", f.__name__, flush=True)
