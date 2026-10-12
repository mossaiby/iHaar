"""Generate LaTeX tables from results/*.json (no hand-copied numbers).

Every table is designed to fit the text width without scaling: wide tables are
emitted as two halves stacked vertically, and long text columns wrap (p{...}).
"""
import json, os
import numpy as np

R = lambda n: json.load(open(os.path.join("results", n + ".json")))
have = lambda n: os.path.exists(os.path.join("results", n + ".json"))
T = "paper/tables"
os.makedirs(T, exist_ok=True)


def sci(v, d=2):
    if v is None: return "--"
    if v == 0: return "0"
    m, x = (f"%.{d}e" % v).split("e"); return f"${m}{{\\times}}10^{{{int(x)}}}$"


def rate(a, b, na, nb):
    return "--" if a is None else "%.2f" % (np.log(a / b) / np.log(nb / na))


def w(name, s):
    open(os.path.join(T, name + ".tex"), "w").write(s)


def tab(spec, head, rows):
    return "\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + "\\\\\n\\midrule\n" + "".join(r + "\\\\\n" for r in rows) + "\\bottomrule\n\\end{tabular}\n"


def stack(*parts):
    return "\n\\medskip\n".join(parts)


def part(label, body):
    return "\\textit{" + label + "}\\\\[2pt]\n" + body


# ---------------------------------------------------------------- E1
E1 = R("E1")
d, n = E1["dirichlet"], E1["neumann"]
rr = lambda L, i, k: "--" if i == 0 else rate(L[i-1][k], L[i][k], L[i-1]["N"], L[i]["N"])
top = tab("r cc cc", "& \\multicolumn{2}{c}{uniform grid} & \\multicolumn{2}{c}{Chebyshev-graded grid}\\\\\n$N$ & error & rate & error & rate",
          [f"{d[i]['N']} & {sci(d[i]['uniform'])} & {rr(d,i,'uniform')} & {sci(d[i]['graded'])} & {rr(d,i,'graded')}" for i in range(len(d))])
bot = tab("r cc cc", "& \\multicolumn{2}{c}{$D_2$} & \\multicolumn{2}{c}{$D_2^{(4)}$}\\\\\n$N$ & error & rate & error & rate",
          [f"{n[i]['N']} & {sci(n[i]['order2'])} & {rr(n,i,'order2')} & {sci(n[i]['order4'])} & {rr(n,i,'order4')}" for i in range(len(n))])
w("t_poisson", stack(part("(a) Prescribed values, $-\\Delta u=f$, operator $D_2$", top),
                     part("(b) Zero-slope conditions, $(I-\\Delta)u=f$, uniform grid", bot)))
f = lambda r, k: "--" if k not in r else "%.3f" % r[k]
w("t_cost", tab("rr ccccc", "$N$ & unknowns & dense & Bartels--Stewart & FDM setup & FDM solve & DCT solve",
                [f"{r['N']} & {r['dofs']} & {f(r,'dense')} & {f(r,'bartels_stewart')} & {f(r,'fdm_setup')} & {f(r,'fdm_gemm')} & {f(r,'dct')}" for r in E1["cost"]]))

# ---------------------------------------------------------------- E2
E2 = R("E2")
L = E2["smooth"]
w("t_el_smooth", tab("rr cc cr", "$N$ & unknowns & $e_{H^1}$ & rate & $e_\\infty$ & GMRES its",
                     [f"{r['N']} & {r['dofs']} & {sci(r['eH1'])} & {rr(L,i,'eH1')} & {sci(r['einf'])} & {r['its']}" for i, r in enumerate(L)]))
w("t_el_nu", tab("l ccc ccc", "& \\multicolumn{3}{c}{GMRES iterations} & \\multicolumn{3}{c}{$e_{H^1}$}\\\\\n$\\nu$ & $N=32$ & $64$ & $128$ & $N=32$ & $64$ & $128$",
                 [f"{r['nu']} & {r['its32']} & {r['its64']} & {r['its128']} & {sci(r['e32'])} & {sci(r['e64'])} & {sci(r['e128'])}" for r in E2["nu"]]))


def sing_table(U, A):
    st = max(1, len(A) // 8); sel = A[::st] + ([A[-1]] if (len(A) - 1) % st else [])
    top = tab("rcc", "unknowns & $e_{H^1}$ & GMRES its", [f"{u['dofs']} & {sci(u['eH1'])} & {u['its']}" for u in U])
    bot = tab("rr ccc r", "cycle & unknowns & $e_{H^1}$ & $h_{\\min}$ & $\\kappa(V)$ & GMRES its",
              [f"{a['cycle']} & {a['dofs']} & {sci(a['eH1'])} & {sci(a['hmin'],1)} & {a['condV']:.1f} & {a['its']}" for a in sel])
    return stack(part("(a) Uniform refinement", top), part("(b) Automatic refinement (Haar-detail indicator)", bot))


w("t_el_sing", sing_table(E2["sing_uniform"], E2["sing_adaptive"]))

# ---------------------------------------------------------------- E3, E4, E5
E3 = R("E3")
rows = []
for k, v in E3["runs"].items():
    N, o = k[1:].split("_o")
    rows.append(f"${N}^2$ & {'$D_2$' if o == '2' else '$D_2^{(4)}$'} & {sci(v['slope'],3)} & {abs(v['slope']/(-2*E3['eps']**2)-1)*100:.2f}\\% & {sci(v['max_dEmod'])} & {sci(v['max_dE'])}")
w("t_ac", tab("cc cccc", "grid & operator & $\\mathrm dR^2/\\mathrm dt$ & deviation & max $\\Delta\\tilde E$ & max $\\Delta E_h$", rows))
E4 = R("E4"); sp = E4["spatial"]
w("t_ch_space", tab("r cc cc", "& \\multicolumn{2}{c}{$D_2$} & \\multicolumn{2}{c}{$D_2^{(4)}$}\\\\\n$N$ & error & rate & error & rate",
                    [f"{N} & {sci(sp['order2'][i])} & {'--' if i==0 else rate(sp['order2'][i-1], sp['order2'][i], sp['N'][i-1], N)} & {sci(sp['order4'][i])} & {'--' if i==0 else rate(sp['order4'][i-1], sp['order4'][i], sp['N'][i-1], N)}" for i, N in enumerate(sp["N"])]))
tp = E4["temporal"]
w("t_ch_time", tab("c cc", "$\\Delta t$ & error & rate",
                   [f"{sci(dt)} & {sci(tp['err'][i])} & {'--' if i==0 else rate(tp['err'][i-1], tp['err'][i], 1/tp['dt'][i-1], 1/dt)}" for i, dt in enumerate(tp["dt"])]))
E5 = R("E5"); spd = E4["spinodal"]
w("t_spinodal", tab("p{6.2cm} cc", "quantity & 2D spinodal & 3D spinodal", [
    f"grid & ${spd['N']}^2$ & ${E5['N']}^3$", f"$\\varepsilon$ & {spd['eps']} & {E5['eps']}",
    f"time steps & {spd['steps']}+4500 & {E5['steps']}",
    f"initial mass $M_h^0$ & 0.1 & {E5.get('mass0', 0):.3g}",
    f"largest mass deviation $|M_h^n-M_h^0|$ & {sci(spd['max_mass_dev'])} & {sci(E5['mass_dev'])}",
    f"largest step change of the modified energy & {sci(spd['max_dEmod'])} & {sci(E5['max_dEmod'])}",
    f"largest change of $E_h$ between samples & {sci(spd['max_dE'])} & {sci(E5['max_dE'])}",
    f"range of $\\xi=r/\\sqrt{{E_1+C}}$ & {spd['xi_min']:.5f}--{spd['xi_max']:.5f} & --",
    f"CPU time per step (one core) & {spd['time_per_step']*1e3:.1f} ms & {E5['time_per_step']*1e3:.0f} ms"]))

# ---------------------------------------------------------------- E6 (performance) + GPU note
E6 = R("E6_cpu")
gpu = R("E6_gpu") if have("E6_gpu") else None
grows = gpu["rows"] if gpu else []
keys = sorted({(r["dim"], r["N"]) for r in E6} | {(r["dim"], r["N"]) for r in grows})
fm = lambda D, k, isg: ("%.2f" % D[k]) if k in D else ("\\textit{pending}" if (isg and not gpu) else "--")
top, bot = [], []
for dim, N in keys:
    c = {r["transform"]: r["ms_per_step"] for r in E6 if r["dim"] == dim and r["N"] == N}
    g64 = {r["transform"]: r["ms_per_step"] for r in grows if r["dim"] == dim and r["N"] == N and r["dtype"] == "float64"}
    g32 = {r["transform"]: r["ms_per_step"] for r in grows if r["dim"] == dim and r["N"] == N and r["dtype"] == "float32"}
    top.append(f"{dim} & {N} & {N**dim} & {fm(c,'gemm',0)} & {fm(c,'dct',0)}")
    bot.append(f"{dim} & {N} & {fm(g64,'gemm',1)} & {fm(g64,'dct',1)} & {fm(g32,'gemm',1)} & {fm(g32,'dct',1)}")
w("t_perf", stack(part("(a) CPU, one core, double precision (ms per step)", tab("rr r cc", "$d$ & $N$ & unknowns & GEMM & DCT", top)),
                  part("(b) GPU (ms per step)", tab("rr cc cc", "& & \\multicolumn{2}{c}{double precision} & \\multicolumn{2}{c}{single precision}\\\\\n$d$ & $N$ & GEMM & DCT & GEMM & DCT", bot))))
if gpu:
    note = (f"GPU timings were measured on an {gpu['info']['gpu']} (CUDA {gpu['info']['cuda']}, CuPy {gpu['info']['cupy']}); "
            "times are per step after warm-up, with device synchronization.")
else:
    note = "\\todo{GPU columns pending; see the GPU section of the repository README.}"
open(os.path.join(T, "gpu_note.tex"), "w").write(note + "\n")

# ---------------------------------------------------------------- E7 traction
E7 = R("E7"); sm = E7["smooth"]; names = list(sm.keys())


def trac_half(ns):
    head = "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{traction on {k}}}" for k in ns) + "\\\\\n$N$ & " + " & ".join(["$e_{H^1}$ & rate"] * len(ns))
    rows = []
    for i in range(len(sm[ns[0]])):
        cells = []
        for k in ns:
            Lk = sm[k]; cells += [sci(Lk[i]["eH1"]), "--" if i == 0 else rate(Lk[i-1]["eH1"], Lk[i]["eH1"], Lk[i-1]["N"], Lk[i]["N"])]
        rows.append(f"{sm[ns[0]][i]['N']} & " + " & ".join(cells))
    rows.append("GMRES its & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{min(r['its'] for r in sm[k])}--{max(r['its'] for r in sm[k])}}}" for k in ns))
    rows.append("edge error, $N{=}256$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{sci(sm[k][-1]['eedge'])}}}" for k in ns))
    return tab("l" + " cc" * len(ns), head, rows)


w("t_trac_smooth", stack(trac_half(names[:2]), trac_half(names[2:])))
w("t_trac_nu", tab("l ccc ccc", "& \\multicolumn{3}{c}{GMRES iterations} & \\multicolumn{3}{c}{$e_{H^1}$}\\\\\n$\\nu$ & $N=32$ & $64$ & $128$ & $N=32$ & $64$ & $128$",
                   [f"{r['nu']} & {r['its32']} & {r['its64']} & {r['its128']} & {sci(r['e32'])} & {sci(r['e64'])} & {sci(r['e128'])}" for r in E7["nu"]]))
lab = lambda a: "$-1/12$" if a < 0 else ("0 (plain)" if a == 0 else ("$1/4$" if a == .25 else "$1/2$"))
w("t_enh1d", tab("l ccccc c", "$\\alpha$ & $N=32$ & $64$ & $128$ & $256$ & $512$ & rate",
                 [lab(r["alpha"]) + " & " + " & ".join(sci(v, 1) for v in r["err"]) + f" & {rate(r['err'][-2], r['err'][-1], 256, 512)}" for r in E7["enhanced1d"]]))
w("t_trac_sing", sing_table(E7["sing_uniform"], E7["sing_adaptive"]))
if have("E7b"):
    G = R("E7b"); C_, Gg = G["chebyshev"], G["geometric"]

    def gh(L_, edge):
        return tab("r cc c r", "$N$ & $e_{H^1}$ & rate & " + ("edge error" if edge else "$h_{\\max}/h_{\\min}$") + " & GMRES its",
                   [f"{L_[i]['N']} & {sci(L_[i]['eH1'])} & {'--' if i==0 else rate(L_[i-1]['eH1'], L_[i]['eH1'], L_[i-1]['N'], L_[i]['N'])} & "
                    + (sci(L_[i]['eedge']) if edge else f"{L_[i]['hratio']:.0f}") + f" & {L_[i]['its']}" for i in range(len(L_))])
    w("t_trac_graded", stack(part("(a) Chebyshev grid", gh(C_, True)), part("(b) Geometric grid graded toward the traction edge", gh(Gg, False))))

# ---------------------------------------------------------------- E8 sharp interfaces
r_ = R("E8a")["resolution"]
w("t_resolution", tab("rr ccc", "$N$ & nodes per $\\sqrt2\\varepsilon$ & 3-point FD & $D_2$ & $D_2^{(4)}$",
                      [f"{r['N']} & {r['ppw']:.2f} & {sci(r['fd3'])} & {sci(r['ihwc2'])} & {sci(r['ihwc4'])}" for r in r_ if r["ppw"] >= 0.6]))
st = R("E8b")["star"]
rows = []
for key in ("N256_o2", "N256_o4", "N512_o2", "N512_o4"):
    v = st[key]
    rows.append(f"${v['N']}^2$ & {'$D_2$' if v['order']==2 else '$D_2^{(4)}$'} & {sci(v['err_vs_ref'])} & {sci(v['area_err'])} & {v['time_per_step']*1e3:.1f}")
rows.append(f"$1024^2$ (reference) & $D_2^{{(4)}}$ & -- & -- & {st['N1024_o4']['time_per_step']*1e3:.0f}")
w("t_star", tab("ll ccc", "grid & operator & max nodal error & area error & ms per step", rows))
c = R("E8c")
w("t_chsharp", tab("p{7.2cm} c", "quantity & value", [
    f"grid and $\\varepsilon$ & ${c['N']}^2$, {c['eps']}", f"nodes per $\\sqrt2\\varepsilon$ & {c['nodes_per_w']:.2f}",
    "time steps & " + ", ".join(f"{nn}$\\times${sci(dd,0)}" for dd, nn in c["stages"]),
    f"largest mass deviation & {sci(c['max_mass_dev'])}", f"largest step change of the modified energy & {sci(c['max_dEmod'])}",
    f"largest change of $E_h$ between samples & {sci(c['max_dE'])}",
    f"largest $|\\nabla u_h|$ / planar-interface value $1/(\\sqrt2\\varepsilon)$ & {c['gradmax']:.0f} / {c['grad_theory']:.0f}",
    f"largest $|u_h|$ & {c.get('umax', float('nan')):.3f}", f"CPU time per step (one core) & {c['time_per_step']*1e3:.0f} ms"]))
w("t_sav_stab", tab("cc cccc", "$S$ & $\\Delta t$ & $\\max|u_h|$ & $E_h$ (start $\\to$ end) & $\\xi$ & max $\\Delta\\tilde E$",
                    [f"{r['S']:g} & {sci(r['dt'],0)} & {r['umax']:.2f} & {r['E0']:.3f} $\\to$ {r['E']:.3f} & {r['xi']:.5f} & {sci(r['max_dEmod'])}" for r in R("E8d")]))

# ---------------------------------------------------------------- E9, E10, E11
NAMES = {"fd2": "3-point FD", "ihwc2": "IHWC, $D_2$", "ihwc4": "IHWC, $D_2^{(4)}$", "pade4": "compact (Pad\\'e)", "spectral": "pseudo-spectral"}
E9 = R("E9"); ch = E9["ch"]; stc = E9["star"]; order = ("fd2", "ihwc2", "ihwc4", "pade4", "spectral")
top = tab("l ccc", "scheme & $N=18$ & $N=54$ & $N=162$", [f"{NAMES[k]} & " + " & ".join(sci(v) for v in ch[k]) for k in order])
bot = tab("l cc cc", "& \\multicolumn{2}{c}{max nodal error} & \\multicolumn{2}{c}{area error}\\\\\nscheme & $256^2$ & $512^2$ & $256^2$ & $512^2$",
          [f"{NAMES[k]} & {sci(stc[k+'_256']['err'])} & {sci(stc[k+'_512']['err'])} & {sci(stc[k+'_256']['area'])} & {sci(stc[k+'_512']['area'])}" for k in order])
w("t_compare", stack(part("(a) Smooth Cahn--Hilliard, max error against a pseudo-spectral reference ($N=486$)", top),
                     part("(b) Allen--Cahn star, $\\varepsilon=0.004$, against the IHWC $D_2^{(4)}$ reference ($1024^2$)", bot)))
E10 = R("E10"); A = E10["adaptive"]; sel = A[::3] + ([A[-1]] if (len(A) - 1) % 3 else [])
mb = lambda x: "%.1f" % (x / 1e6)
top = tab("rr c ccc", "cycle & unknowns & $e_{H^1}$ & setup (s) & solve (s) & memory (MB)",
          [f"{r['cycle']} & {r['ihwc_dofs']} & {sci(r['ihwc_e'])} & {r['ihwc_setup']:.2f} & {r['ihwc_t']:.2f} & {mb(r['ihwc_mem'])}" for r in sel])
bot = tab("rr c cccc", "cycle & unknowns & $e_{H^1}$ & assembly (s) & setup + CG (s) & memory (MB) & direct (s)",
          [f"{r['cycle']} & {r['q2_dofs']} & {sci(r['q2_e'])} & {r['q2_assembly']:.2f} & {r['q2pcg_setup'] + r['q2pcg_t']:.2f} & {mb(r['q2pcg_mem'])} & "
           + ("%.2f" % r["q2direct_t"] if "q2direct_t" in r else "--") for r in sel])
w("t_q2", stack(part("(a) IHWC collocation: GMRES with the FDM preconditioner", top),
                part("(b) $Q_2$ finite elements on the same grids: CG with the FDM preconditioner, and sparse direct solve", bot)))
if have("E12"):
    E12 = R("E12")
    w("t_fdmstab", tab("cr r c ccc", "& & & & \\multicolumn{2}{c}{backward error} & solution\\\\\nratio & $N$ & $h_{\\max}/h_{\\min}$ & $\\kappa(V)$ & FDM & B--S & difference",
                       [f"{r['rmax']:g} & {r['N']} & {sci(r['hratio'],1)} & {r['condV']:.1f} & {sci(r['backward_fdm'],1)} & {sci(r['backward_bs'],1)} & {sci(r['rel_diff'],1)}" for r in E12["fdm_vs_bs"]]))
if have("E11"):
    E11 = R("E11")
    w("t_spectrum", tab("cc r r cc", "ends & ratio bound & $N$ & grids & max $|\\mathrm{Im}\\,\\lambda|/|\\lambda|$ & max $\\kappa(V)$",
                        [f"{'values' if r['bc']=='D' else 'zero slope'} & {r['rmax']:g} & {r['N']} & {r['samples']} & {r['max_rel_imag']:.1e} & {r['max_condV']:.1f}" for r in E11]))
print("all tables written")

# ---------------- revision 3: robustness, 3D, S study, adaptivity, environment -----------
if have("E17"):
    E17 = R("E17")
    lab = {"alternating_10": "alternating, ratio 10", "alternating_100": "alternating, ratio 100", "alternating_1000": "alternating, ratio 1000",
           "tiny_ends_1e-6": "end cells $10^{-6}$", "geometric_1e6": "geometric, $10^6$", "geometric_1e12": "geometric, $10^{12}$",
           "random_unbounded": "random, unbounded", "cluster": "cluster of $10^{-4}$ cells"}
    rows = [f"{lab[r['family']]} & {'values' if r['bc']=='D' else 'slope'} & {sci(r['hratio'],0)} & {r['condV']:.1f} & {sci(r['eta_fdm'],0)} & {sci(r['eta_bs'],0)} & {sci(r['rel_diff'],0)}"
            for r in E17["adversarial"] if r["N"] == 256]
    w("t_adversarial", tab("p{3.3cm}c c c ccc", "& & & & \\multicolumn{2}{c}{backward error} & solution\\\\\ngrid family ($N=256$) & ends & $h_{\\max}/h_{\\min}$ & $\\kappa(V)$ & FDM & B--S & difference", rows))
    w("t_fallback", tab("r ccc", "$N$ & FDM with check (s) & Bartels--Stewart (s) & ratio",
                        [f"{c['N']} & {c['fdm_with_check']:.4f} & {c['bartels_stewart']:.4f} & {c['ratio']:.1f}" for c in E17["cost"]]))
if have("E14"):
    E14 = R("E14")
    def r3(g):
        L_ = [r for r in E14 if r["grid"] == g]
        return [f"{r['N']} & {r['unknowns']} & {sci(r['err'])} & {'--' if i==0 else rate(L_[i-1]['err'], r['err'], L_[i-1]['N'], r['N'])} & {r['time']:.4f}" for i, r in enumerate(L_)]
    w("t_3d", stack(part("(a) Uniform grids", tab("rr cc c", "$N$ & unknowns & max error & rate & solve (s)", r3("uniform"))),
                    part("(b) Chebyshev-graded grids", tab("rr cc c", "$N$ & unknowns & max error & rate & solve (s)", r3("graded")))))
if have("E15"):
    E15 = sorted(R("E15"), key=lambda r: (r["dt_over_eps2"], r["S"]))
    w("t_Ssens", tab("cc ccc", "$\\Delta t/\\varepsilon^2$ & $S$ & $\\max_n|u_h|$ & error vs.\\ reference & $\\xi$ at $T$",
                     [f"{r['dt_over_eps2']} & {r['S']:g} & {r['umax']:.3f} & {sci(r['err'])} & {r['xi']:.5f}" for r in E15]))
if have("E16"):
    E16 = R("E16")
    nm = lambda k: (k.replace("face_omega_", "face, $\\omega=").replace("pi", "\\pi$") if k.startswith("face") else k.replace("_", " "))
    top = tab("l c c cc cc", "case & $\\lambda$ & uniform $128^2$ & \\multicolumn{2}{c}{indicator: error, rate} & \\multicolumn{2}{c}{oracle: error, rate}",
              [f"{nm(k)} & {'--' if v['lam'] is None else '%.3f' % v['lam']} & {sci(v['uniform128']['eH1'])} & {sci(v['indicator']['rows'][-1]['eH1'])} & {v['indicator']['rate']:.2f} & {sci(v['oracle']['rows'][-1]['eH1'])} & {v['oracle']['rate']:.2f}" for k, v in E16.items()])
    bot = tab("l c c c c", "case & effectivity range $\\eta/e$ & cycles & final unknowns & total time (s)",
              [f"{nm(k)} & {v['indicator']['eff_min']:.2f}--{v['indicator']['eff_max']:.2f} & {v['indicator']['cycles']} & {v['indicator']['rows'][-1]['dofs']} & {v['indicator']['time']:.1f}" for k, v in E16.items()])
    w("t_adapt", stack(part("(a) Final relative $H^1$ error and fitted rate (about $6\\times10^4$ unknowns)", top),
                       part("(b) Indicator effectivity, number of refinement cycles, and total setup and solve time", bot)))
if have("environment"):
    env = R("environment")
    w("t_env", tab("p{3.2cm}p{10.4cm}", "item & value", [
        f"CPU & {env['cpu']}, {env['logical_cpus']} logical core(s)", f"operating system & {env['os']}",
        f"software & Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}", f"BLAS/LAPACK & {env['blas']}",
        "threads & one (the virtual machine exposes a single core)",
        "timing protocol & wall-clock time (\\texttt{time.perf\\_counter}); solve times are medians of 3 to 5 repetitions; setup and assembly are reported separately and are not amortized"]))
print("revision-3 tables written")

# ---------------- AMG note (E13 requires pyamg) ------------------------
if have("E13"):
    A13 = R("E13"); r = A13[-1]
    amg = (f"With smoothed-aggregation algebraic multigrid (pyamg) as preconditioner for $Q_2$ on the same grids, "
           f"the finest run ({r['q2_dofs']} unknowns) takes {r['amg_setup']:.2f}~s setup and {r['amg_solve']:.2f}~s solve "
           f"with {r['amg_its']} CG iterations; the complete sequence is in \\url{{results/E13.json}}.")
else:
    amg = "\\todo{AMG baseline pending: \\texttt{pip install pyamg}, then run E13 (see README).}"
open(os.path.join(T, "amg_note.tex"), "w").write(amg + "\n")
