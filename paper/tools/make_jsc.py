"""Build paper/manuscript_jsc.tex (Journal of Scientific Computing, svjour3 smallextended)
from the lean manuscript plus the JSC front matter, method section and appendix in tools/."""
import json, os, re, sys
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/tmp")
lean = open("manuscript_lean.tex").read()
full = open("manuscript.tex").read()
front = open("tools/jsc_front.tex").read()
method = open("tools/jsc_method.tex").read()
appendix = open("tools/jsc_appendix.tex").read()
E7 = json.load(open("../results/E7.json")); E7b = json.load(open("../results/E7b.json"))
# ---- timing-dependent numbers quoted in the text: always computed from the stored results
import numpy as np
_R = lambda n: json.load(open(f"../results/{n}.json"))
_c = _R("E1")["cost"]; _A = _R("E10")["adaptive"]; _f = _A[-1]
_qe = np.array([r["q2_e"] for r in _A]); _o = np.argsort(_qe)
_it = lambda y: float(np.exp(np.interp(np.log(_f["ihwc_e"]), np.log(_qe[_o]), np.log(np.array(y)[_o]))))
_ihwc_tot = _f["ihwc_setup"] + _f["ihwc_t"]
_r_incl = _it([r["q2_assembly"] + r["q2pcg_setup"] + r["q2pcg_t"] for r in _A]) / _ihwc_tot
_r_excl = _it([r["q2pcg_setup"] + r["q2pcg_t"] for r in _A]) / _ihwc_tot
_r_mem = _it([r["q2pcg_mem"] for r in _A]) / _f["ihwc_mem"]
_d = [r for r in _A if "q2direct_t" in r][-1]
_fb = [r["ratio"] for r in _R("E17")["cost"]]
_md = [r["mdofs_per_s"] for r in _R("E6_cpu") if r["transform"] == "dct"]
def _pow10(x):
    e = int(np.floor(np.log10(x))); m = x / 10 ** e
    return f"10^{{{e}}}" if round(m) == 1 else f"{round(m):d}\\times10^{{{e}}}"
NUM = dict(
    dense64="%.1f" % [r for r in _c if r["N"] == 64][0]["dense"], dct1024="%.2f" % _c[-1]["dct"],
    fb_lo="%d" % round(min(_fb)), fb_hi="%d" % round(max(_fb)),
    r_incl="%.1f" % _r_incl, r_excl="%.1f" % _r_excl, mem_pct="%d" % round(100 * (1 - 1 / _r_mem), -1),
    direct_ratio="%d" % round(_d["q2direct_t"] / (_d["q2pcg_setup"] + _d["q2pcg_t"]), -1), direct_n=f"{_d['q2_dofs']:,}".replace(",", "\\,"),
    direct_t="%.1f" % _d["q2direct_t"], mdps_lo=_pow10(min(_md) * 1e6), mdps_hi=_pow10(max(_md) * 1e6))

i = lean.index(r"\section{Fast solvers in two and three dimensions}")
i = lean.rfind("%=====", 0, i)
j = lean.index(r"\section*{Code and data availability}")
body = lean[i:j]
miss = []
def T(a, b):
    global body
    pat = r"\s+".join(re.escape(t) for t in a.split())
    m = re.search(pat, body)
    if not m: miss.append(a[:70]); return
    body = body[:m.start()] + b + body[m.end():]

# ---- environments and cross-references
body = body.replace(r"\begin{result}[Mass and energy]\label{thm:energy}", r"\begin{theorem}[Mass and modified energy]\label{thm:energy}")
body = body.replace(r"\end{result}", r"\end{theorem}")
for k, v in (("thm:spectrum", "Theorem"), ("thm:identity", "Theorem"), ("thm:fourth", "Proposition"),
             ("prop:mass", "Proposition"), ("thm:energy", "Theorem")):
    body = body.replace(r"Result~\ref{" + k + "}", v + r"~\ref{" + k + "}")
T(r"""conserves the discrete mass exactly, and a modified energy never increases.""",
  r"""conserves the discrete mass exactly, and a modified energy never increases. The proof is given in the
Supplementary Material.""")

# ---- SAV scheme on three lines (narrow text block)
body = body.replace(r"&=-G\mu^{n+1},\qquad"+"\n"+r"\mu^{n+1}=\varepsilon^2Ku^{n+1}+S(u^{n+1}-\bar u)+r^{n+1}\bar b,\label{eq:sav1}\\",
                    r"&=-G\mu^{n+1},\label{eq:sav1}\\"+"\n"+r"\mu^{n+1}&=\varepsilon^2Ku^{n+1}+S(u^{n+1}-\bar u)+r^{n+1}\bar b,\label{eq:sav1b}\\")
# ---- elastostatics: compressibility qualifier and quantitative traction accuracy
sm = E7["smooth"]; tr = sm["T+R"]; g = E7b["geometric"]
def sx(v):
    m, e = ("%.1e" % v).split("e"); return f"{m}\\times10^{{{int(e)}}}"
T(r"""displacement component.""", r"""displacement component. For compressible materials the number of iterations does not grow with the grid;
as Poisson's ratio approaches one half it grows strongly (Tables~S1 and~S2), so the solver is intended for
compressible materials.""")
T(r"""plain collocation stays second-order accurate here, also on graded grids (Tables~S10 and~S11).""",
  rf"""plain collocation stays second-order accurate here. With tractions on two adjacent edges and $\nu=0.3$,
the relative $H^1$ error decreases from ${sx(tr[0]['eH1'])}$ at $N={tr[0]['N']}$ to ${sx(tr[-1]['eH1'])}$ at
$N={tr[-1]['N']}$ (observed rate 2.00), and the largest displacement error on the traction edges is
${sx(tr[-1]['eedge'])}$; on a geometric grid graded toward the traction edge with cell-size ratio up to
{g[-1]['hratio']:.0f}, the rate is again two (Tables~S10 and~S11).""")

# ---- results opening: environment, verification table
T(r"""All computations ran on one core of a virtual x86 CPU in double precision. Every number is produced by the scripts of the repository; the Supplementary Material (Part~B) contains additional tables, including convergence tables for smooth solutions, which confirm second order for $D_2$ and fourth order for $D_2^{(4)}$ (Tables~S8, S9 and~S14).""",
  r"""All computations ran in double precision on the system described in Table~\ref{tab:env}; absolute
timings depend on this hardware and software. Every number is produced by the scripts of the repository,
and the tables are generated from the stored results.

\begin{table}[t]\centering\small
\caption{Computing environment and timing protocol.}\label{tab:env}
\input{tables/t_env.tex}
\end{table}

\subsection{Verification}\label{sec:verify}
Table~\ref{tab:poisson} reports the maximum nodal error for $-\Delta u=f$ with prescribed values and
$u=e^x\sin\pi y+x^2y$, on uniform and on Chebyshev-graded grids, and for $(I-\Delta)u=f$ with zero-slope
conditions and $u=\cos\pi x\cos2\pi y+0.3\cos3\pi x$. The plain operator converges at second order on both
grid families, and the corrected operator at fourth order when its conditions hold. Second-order
convergence was also observed for smooth elastostatic solutions with prescribed displacements and with
tractions, in three dimensions on uniform and graded grids, and for the Cahn--Hilliard equation, where the
corrected operator gives fourth order (Tables~S9, S10, S19 and~S14).

\begin{table}[t]\centering\small
\caption{Maximum nodal error and observed convergence rates for two model problems.}\label{tab:poisson}
\input{tables/t_poisson.tex}
\end{table}""")

# ---- robustness: separate the three notions
i0 = body.index(r"\subsection{Robustness and cost of the fast solver}")
i1 = body.index(r"\begin{table}", i0)
body = body[:i0] + r"""\subsection{Well-definedness, backward stability and forward accuracy}\label{sec:robust}
Three different properties must be distinguished. Theorem~\ref{thm:spectrum} guarantees that fast
diagonalization is \emph{well defined} on every grid. Whether it is \emph{backward stable}, that is, whether
the computed solution exactly solves a slightly perturbed problem, depends on the conditioning $\kappa(V)$ of
the eigenvectors, which the theorem does not bound. Whether the computed solution is \emph{accurate} in the
forward sense depends, in addition, on the conditioning of the discrete problem itself. For the Sylvester
operator $U\mapsto\alpha D_2^xU+\beta U(D_2^y)^{\mathsf T}+\gamma U$ we use the estimate
$\kappa_{\mathrm{Syl}}=\kappa(V_x)\kappa(V_y)\max|\mu_{ij}|/\min|\mu_{ij}|$, where $\mu_{ij}$ are its
eigenvalues; it bounds the normwise sensitivity and may be pessimistic.

We tested backward stability against the Bartels--Stewart algorithm~\cite{bartels1972solution} on 8760 random
graded grids (Table~S4) and on adversarial families: alternating cell sizes, isolated tiny end cells,
clusters of small cells, unbounded random sizes, and geometric grading up to $10^{12}$
(Table~\ref{tab:fdmstab}; random grids in Table~S16). The eigenvalues were real in every case, $\kappa(V)$
stayed below 85, and the backward error of fast diagonalization stayed at round-off level, between
$10^{-30}$ and $6\times10^{-15}$. Forward accuracy is a different matter. On the most extreme grids, the
solutions of the two backward-stable methods differ by up to $2\times10^{10}$ in relative terms, and every
such case has $\kappa_{\mathrm{Syl}}\gtrsim10^{16}$, beyond the reciprocal of the machine precision. These
problems are intrinsically ill conditioned: their eigenvalues span many orders of magnitude because the cell
sizes do, and no solver can compute them accurately in double precision.

The implementation therefore uses two safeguards with different purposes. The first concerns backward
stability: after each solve it evaluates the normwise backward error, at the cost of two matrix products,
and repeats the solve with Bartels--Stewart if $\kappa(V)>10^6$ or the backward error exceeds $10^{-12}$.
This never triggered in our computations, and the fallback costs @@fb_lo@@ to @@fb_hi@@ times more on our hardware (Table~SFALLBACK). The second
concerns forward accuracy: the code computes $\kappa_{\mathrm{Syl}}$ and flags the problem as sensitive when
$\kappa_{\mathrm{Syl}}$ times the machine precision exceeds $10^{-6}$. This flag is a warning, not a remedy,
since neither solver can improve the accuracy of an ill-conditioned problem. On all grids generated by the
refinement algorithm, whose neighbour ratio is at most two, $\kappa(V)$ stayed below 25.

Table~\ref{tab:cost} shows the practical gain over classical HWCM. On our hardware, the dense HWCM solve takes
@@dense64@@~s at $64\times64$ cells and is infeasible beyond that, whereas the cosine-transform solver handles
$10^6$ unknowns in @@dct1024@@~s.

""" + body[i1:]
body = body.replace(r"""\caption{Adversarial grid families ($N=256$): eigenvector conditioning, backward errors of fast
diagonalization (FDM) and Bartels--Stewart (B--S), and relative difference of the two solutions.}\label{tab:fdmstab}""",
r"""\caption{Adversarial grid families ($N=256$): eigenvector conditioning $\kappa(V)$, condition estimate
$\kappa_{\mathrm{Syl}}$ of the Sylvester problem, normwise backward errors of fast diagonalization (FDM) and
Bartels--Stewart (B--S), and relative difference of the two solutions.}\label{tab:fdmstab}""")

# ---- qualify claims
T(r"""with 15 to 20 GMRES iterations (Fig.~\ref{fig:trac}).""", r"""with 15 to 20 GMRES iterations for $\nu=0.3$ (Fig.~\ref{fig:trac}).""")
T(r"""At equal accuracy, the collocation method needs a quarter of the unknowns.""", r"""In this benchmark, at equal accuracy, the collocation method needed about a quarter of the unknowns of $Q_2$.""")

# ---- conclusions and guidelines (review: no superlatives, qualified claims, updated advice)
c0 = body.index(r"\section{Conclusions}"); c0 = body.index("\n", body.index("%====", c0)) + 1
body = body[:c0] + r"""Rewritten in nodal form, Haar wavelet collocation is midpoint quadratic-spline collocation, and its
second-derivative operator has real, simple and nonpositive eigenvalues on every strictly increasing grid
(Theorem~\ref{thm:spectrum}). Fast diagonalization is therefore well defined on any graded or adaptively
refined tensor grid. In floating-point arithmetic it was backward stable on random and adversarial grids,
and an inexpensive backward-error check with a Bartels--Stewart fallback guards against failures it can
detect; forward accuracy remains limited by the conditioning of the discrete problem, which the code
estimates and flags. The structure replaces the dense HWCM systems by fast tensor solves, gives a
fourth-order correction at no cost on uniform grids with zero-flux conditions, and supports
mass-conserving phase-field schemes with a modified energy law that, with stabilization, resolved
interfaces with about four nodes across the transition layer. For compressible elastostatics with
tractions, refinement driven by Haar detail coefficients recovered the optimal convergence rate at the
tested corner singularities, and in a benchmark against biquadratic finite elements on identical grids the
method needed about a quarter of the unknowns. For smooth problems on uniform boxes, pseudo-spectral
methods remain the better choice.

""" + body[body.index(r"\section*", c0) if r"\section*" in body[c0:] else len(body):]
T(r"""Time stepping (AC, CH) & Use SAV-BDF2 with stabilization $S\approx4$ and monitor $\xi=r/\sqrt{E_{1,h}+C}$; if it leaves $[0.999,1.001]$, reduce $\Delta t$. Start with small steps from random initial data, and increase $\Delta t$ as the structure coarsens, restarting with BDF1 at each change.\\""",
  r"""Time stepping (AC, CH) & Use SAV-BDF2 with the smallest stabilization that keeps $\xi=r/\sqrt{E_{1,h}+C}$ near one: $S\approx1$ to 2 for $\Delta t\lesssim\varepsilon^2$, and 2 to 4 for larger steps (Table~S18). Keep $\Delta t$ of order $\varepsilon^2$ when the coarsening dynamics matter, and verify by halving $\Delta t$; $\xi\approx1$ alone does not guarantee accuracy.\\""")
T(r"""Solver & On uniform grids, use cosine or sine transforms. On graded grids, use fast diagonalization with dense matrix products (Theorem~\ref{thm:spectrum} guarantees that it applies). For coupled systems such as elasticity, use GMRES with the block preconditioner.\\""",
  r"""Solver & On uniform grids, use cosine or sine transforms. On graded grids, use fast diagonalization (well defined by Theorem~\ref{thm:spectrum}) with the backward-error check and Bartels--Stewart fallback, and heed the sensitivity flag based on $\kappa_{\mathrm{Syl}}$. For coupled systems such as elasticity, use GMRES with the block preconditioner.\\""")

T(r"""In our implementation and on our hardware this translated into about 2.5 to 3 times less end-to-end time including assembly (about 1.4 times without it) and about 40\% less memory.""",
  r"""In our implementation and on our hardware this translated into about @@r_incl@@ times less end-to-end time
including assembly (about @@r_excl@@ times without it) and about @@mem_pct@@\% less memory.""")
T(r"""A sparse direct solver is one to two orders of magnitude slower than either iterative solver,""",
  r"""A sparse direct solver was about @@direct_ratio@@ times slower than conjugate gradients with the
fast-diagonalization preconditioner (@@direct_t@@~s at @@direct_n@@ unknowns),""")
T(r"""amounts to between $10^7$ and $3\times10^7$ unknowns per second""",
  r"""amounted to between $@@mdps_lo@@$ and $@@mdps_hi@@$ unknowns per second""")

body = body.replace("with $N^2$ unknowns (one CPU core).", "with $N^2$ unknowns, on the CPU of Table~\\ref{tab:env}.")
body = re.sub(r"On one CPU core this\s+amounted", lambda m: r"On the CPU of Table~\ref{tab:env} this amounted", body)

# ---- figures: individual panels (experiments/make_figures.py) arranged here
def _cell(f, w, lab):
    return r"\includegraphics{../figures/panels/" + f + ".pdf}" + (r"\\[-1pt]\small " + lab if lab else "")
def panels(rows, caption, label, star=False):
    out = ["\\begin{figure}[t]\\centering"]
    for row in rows:
        out.append("\\begin{tabular}{@{}" + "c" * len(row) + "@{}}")
        out.append(" & ".join(r"\begin{tabular}{@{}c@{}}" + _cell(f, w, lab) + r"\end{tabular}" for f, w, lab in row) + "\\\\")
        out.append("\\end{tabular}\\par\\smallskip")
    out.append("\\caption{" + caption + "}\\label{" + label + "}\n\\end{figure}")
    return "\n".join(out)
FIGS = {
 "fig:trac": panels([[("trac_error", "0.48", "(a)"), ("trac_grid", "0.40", "(b)")], [("trac_cellsize", "0.48", "(c)")]],
   r"Corner singularity at the end of a traction-free face ($\nu=0.3$). (a) Relative $H^1$ error versus number of unknowns for uniform and automatic refinement, with the scaled indicator. (b) Grid generated after 12 refinement cycles; the red edge is the traction-free face and the star marks the singular point. (c) Cell size versus distance to the singular point on the grid of (b).", "fig:trac"),
 "fig:q2": panels([[("q2_error", "0.48", "(a)"), ("q2_time", "0.48", "(b)")]],
   r"Collocation (IHWC) versus $Q_2$ finite elements on identical grids. (a) Error versus number of unknowns. (b) Solve time versus error for IHWC and for $Q_2$ with CG and the fast-diagonalization preconditioner, a sparse direct solver, and CG with algebraic multigrid (converged solves only).", "fig:q2"),
 "fig:res": panels([[("resolution", "0.6", "")]],
   r"Planar interface ($\varepsilon=0.01$): maximum nodal error versus nodes per $\sqrt2\varepsilon$ (data in Table~S5).", "fig:res"),
 "fig:star": panels([[("star_levelsets", "0.48", "(a)"), ("star_zoom", "0.44", "(b)")], [("star_gradient", "0.48", "(c)")]],
   r"Allen--Cahn star, $\varepsilon=0.004$, at $t=120$. (a) Zero level sets of the reference and of the $256^2$ solutions, and the initial interface. (b) Zoom at a tip, with the $256^2$ grid nodes. (c) Gradient magnitude of the reference solution.", "fig:star"),
 "fig:chsharp": panels([[("ch_snap0", "0.29", "(a)"), ("ch_snap1", "0.29", "(b)"), ("ch_snap2", "0.29", "(c)")],
                        [("ch_zoom", "0.40", "(d)"), ("ch_gradient", "0.44", "(e)")],
                        [("ch_energy", "0.48", "(f)"), ("ch_profile", "0.48", "(g)")]],
   r"Cahn--Hilliard with $\varepsilon=0.0015$ on $1024^2$. (a)--(c) Concentration at $t=0.0003$, $0.0033$ and $0.0333$; the box in (c) marks the region shown in (d). (d) Zoom in which each square is one node. (e) Gradient magnitude at the final time. (f) Discrete energy. (g) Nodal values along a grid line crossing an interface, compared with the $\tanh$ profile.", "fig:chsharp"),
 "fig:compare": panels([[("scheme_comparison", "0.6", "")]],
   r"Planar interface: maximum nodal error versus resolution for five operators.", "fig:compare"),
}
for lab, new in FIGS.items():
    i = body.index(r"\label{" + lab + "}"); a = body.rfind(r"\begin{figure}", 0, i); b = body.index(r"\end{figure}", i) + len(r"\end{figure}")
    body = body[:a] + new + body[b:]

# ---- renumber supplementary tables: lean S8 (t_poisson) moved to the main text
def renum(m):
    n = int(m.group(1)); return "S%d" % (n - 1 if n > 8 else n)
body = re.sub(r"(?<![A-Za-z\\])S(\d{1,2})(?![0-9])", renum, body)
body = body.replace("SFALLBACK", "S19")

# ---- declarations
decl = r"""
\section*{Declarations}
\paragraph{Funding} No funding was received for conducting this study.

\paragraph{Competing interests} The author has no relevant financial or non-financial interests to disclose.

\paragraph{Author contributions} Farshid Mossaiby: Conceptualization, Methodology, Software, Validation,
Formal analysis, Investigation, Resources, Data curation, Writing -- original draft, Writing -- review and
editing, Visualization, Supervision, Project administration.

\paragraph{Code availability} The implementation, the tests and the scripts that produce every table and
figure are openly available at \repo; the version used in this paper is the GitHub release v1.1. Dependencies
are listed in \texttt{requirements.txt} (CPU) and \texttt{requirements-gpu.txt} (GPU).

\paragraph{Data availability} All data in this paper are generated by the scripts in the repository; the
stored results (\texttt{results/*.json}) and the generated tables are part of release v1.1.

"""
# ---- bibliography
k0 = lean.index(r"\begin{thebibliography}"); k1 = lean.index(r"\end{thebibliography}")
bib = lean[k0:k1]
for key in ("fallat2011totally",):
    if r"\bibitem{" + key + "}" not in bib:
        m = re.search(r"\\bibitem\{" + key + r"\}[^\n]*\n", full); bib += m.group(0)
bib += "\\end{thebibliography}\n"
out = front + method + body + decl + appendix + "\n" + bib + "\n\\end{document}\n"
for _k, _v in NUM.items():
    out = out.replace("@@" + _k + "@@", _v)
assert "@@" not in out, re.findall(r"@@\w+@@", out)
# svjour3: no \proofin, no result environment, no Supplementary Part A references
out = out.replace(r"Part~A of the Supplementary Material", "the Supplementary Material")
out = out.replace(r"\begin{table}[t]\centering\small", r"\begin{table}[t]\centering\footnotesize")
out = out.replace(r"\begin{tabular}{p{3.3cm}p{10.6cm}}", r"\begin{tabular}{p{0.22\textwidth}p{0.72\textwidth}}")
open("manuscript_jsc.tex", "w").write(out)
print("missing:", miss)
abs_ = out[out.index(r"\begin{abstract}") + 16: out.index(r"\keywords")]
print("abstract words:", len(re.sub(r"\$[^$]*\$", "X", abs_).split()))
