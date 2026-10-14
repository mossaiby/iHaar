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
This never triggered in our computations, and the fallback costs 13 to 18 times more (Table~SFALLBACK). The second
concerns forward accuracy: the code computes $\kappa_{\mathrm{Syl}}$ and flags the problem as sensitive when
$\kappa_{\mathrm{Syl}}$ times the machine precision exceeds $10^{-6}$. This flag is a warning, not a remedy,
since neither solver can improve the accuracy of an ill-conditioned problem. On all grids generated by the
refinement algorithm, whose neighbour ratio is at most two, $\kappa(V)$ stayed below 25.

Table~\ref{tab:cost} shows the practical gain over classical HWCM. The dense HWCM solve takes 1.2~s at
$64\times64$ cells and is infeasible beyond that, whereas the cosine-transform solver handles $10^6$
unknowns in 0.03~s.

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

# ---- renumber supplementary tables: lean S8 (t_poisson) moved to the main text
def renum(m):
    n = int(m.group(1)); return "S%d" % (n - 1 if n > 8 else n)
body = re.sub(r"(?<![A-Za-z\\])S(\d{1,2})(?![0-9])", renum, body)
body = body.replace("SFALLBACK", "S19")

# ---- declarations
decl = r"""
\section*{Declarations}
\paragraph{Funding} \todo{Please confirm: The authors did not receive support from any organization for the submitted work.}

\paragraph{Competing interests} The authors have no competing interests to declare that are relevant to the
content of this article.

\paragraph{Author contributions} \todo{Please state the contributions of F.~Mossaiby and A.~Shojaei, e.g.\
conceptualization, methodology, software, validation, writing.}

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
# svjour3: no \proofin, no result environment, no Supplementary Part A references
out = out.replace(r"Part~A of the Supplementary Material", "the Supplementary Material")
out = out.replace(r"\begin{table}[t]\centering\small", r"\begin{table}[t]\centering\footnotesize")
out = out.replace(r"\begin{tabular}{p{3.3cm}p{10.6cm}}", r"\begin{tabular}{p{0.22\textwidth}p{0.72\textwidth}}")
open("manuscript_jsc.tex", "w").write(out)
print("missing:", miss)
abs_ = out[out.index(r"\begin{abstract}") + 16: out.index(r"\keywords")]
print("abstract words:", len(re.sub(r"\$[^$]*\$", "X", abs_).split()))
