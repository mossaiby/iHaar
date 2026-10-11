"""Generate LaTeX tables from results/*.json (no hand-copied numbers)."""
import json, os, numpy as np
R = lambda n: json.load(open(os.path.join("results", n + ".json")))
T = "paper/tables"
e = lambda v: "--" if v is None else ("%.2e" % v).replace("e-0", "e-").replace("e+0", "e+")
def sci(v):
    if v is None: return "--"
    m, x = ("%.2e" % v).split("e"); return f"${m}\\times10^{{{int(x)}}}$"
def rate(a, b, na, nb): return "--" if a is None else "%.2f" % (np.log(a / b) / np.log(nb / na))
def w(name, s): open(os.path.join(T, name + ".tex"), "w").write(s)

E1 = R("E1")
s = "\\begin{tabular}{r cc cc cc}\n\\toprule\n & \\multicolumn{2}{c}{Dirichlet, uniform} & \\multicolumn{2}{c}{Dirichlet, Chebyshev-graded} & \\multicolumn{2}{c}{}\\\\\n$N$ & error & rate & error & rate \\\\\n\\midrule\n"
s = "\\begin{tabular}{r cc cc | cc cc}\n\\toprule\n & \\multicolumn{4}{c|}{Dirichlet, $-\\Delta u=f$ (2nd-order operator)} & \\multicolumn{4}{c}{Neumann, $(I-\\Delta)u=f$}\\\\\n & \\multicolumn{2}{c}{uniform} & \\multicolumn{2}{c|}{graded} & \\multicolumn{2}{c}{$D_2$ (Eq.~\\eqref{eq:D2})} & \\multicolumn{2}{c}{$D_2^{(4)}$ (Eq.~\\eqref{eq:D24})}\\\\\n$N$ & error & rate & error & rate & error & rate & error & rate\\\\\n\\midrule\n"
d, n = E1["dirichlet"], E1["neumann"]
for i in range(len(d)):
    r = lambda L, key: "--" if i == 0 else rate(L[i-1][key], L[i][key], L[i-1]["N"], L[i]["N"])
    s += f"{d[i]['N']} & {sci(d[i]['uniform'])} & {r(d,'uniform')} & {sci(d[i]['graded'])} & {r(d,'graded')} & {sci(n[i]['order2'])} & {r(n,'order2')} & {sci(n[i]['order4'])} & {r(n,'order4')}\\\\\n"
w("t_poisson", s + "\\bottomrule\n\\end{tabular}\n")

s = "\\begin{tabular}{rr ccccc}\n\\toprule\n$N$ & DOFs & dense (classical) & Bartels--Stewart & FDM setup (once) & FDM (GEMM) & FDM (DCT)\\\\\n\\midrule\n"
for r in E1["cost"]:
    f = lambda k: "--" if k not in r else "%.2e" % r[k]
    s += f"{r['N']} & {r['dofs']} & {f('dense')} & {f('bartels_stewart')} & {f('fdm_setup')} & {f('fdm_gemm')} & {f('dct')}\\\\\n"
w("t_cost", s + "\\bottomrule\n\\end{tabular}\n")

E2 = R("E2")
s = "\\begin{tabular}{rr cc cr}\n\\toprule\n$N$ & DOFs & $e_{H^1}$ & rate (in $N$) & $e_\\infty$ & GMRES its\\\\\n\\midrule\n"
L = E2["smooth"]
for i, r in enumerate(L):
    s += f"{r['N']} & {r['dofs']} & {sci(r['eH1'])} & {'--' if i==0 else rate(L[i-1]['eH1'], r['eH1'], L[i-1]['N'], r['N'])} & {sci(r['einf'])} & {r['its']}\\\\\n"
w("t_el_smooth", s + "\\bottomrule\n\\end{tabular}\n")
s = "\\begin{tabular}{l ccc ccc}\n\\toprule\n & \\multicolumn{3}{c}{GMRES iterations} & \\multicolumn{3}{c}{$e_{H^1}$}\\\\\n$\\nu$ & $N=32$ & $64$ & $128$ & $N=32$ & $64$ & $128$\\\\\n\\midrule\n"
for r in E2["nu"]:
    s += f"{r['nu']} & {r['its32']} & {r['its64']} & {r['its128']} & {sci(r['e32'])} & {sci(r['e64'])} & {sci(r['e128'])}\\\\\n"
w("t_el_nu", s + "\\bottomrule\n\\end{tabular}\n")
s = "\\begin{tabular}{rrc rrcccr}\n\\toprule\n\\multicolumn{3}{c}{uniform} & \\multicolumn{6}{c}{automatic (Haar-detail indicator)}\\\\\n\\cmidrule(lr){1-3}\\cmidrule(lr){4-9}\nDOFs & $e_{H^1}$ & its & cycle & DOFs & $e_{H^1}$ & $h_{\\min}$ & $\\kappa(V)$ & its\\\\\n\\midrule\n"
U = E2["sing_uniform"]; A = E2["sing_adaptive"]; sel = [A[k] for k in range(0, len(A), max(1, len(A)//8))] + ([A[-1]] if (len(A)-1) % max(1, len(A)//8) else [])
for i in range(max(len(U), len(sel))):
    u = U[i] if i < len(U) else None; a = sel[i] if i < len(sel) else None
    s += (f"{u['dofs']} & {sci(u['eH1'])} & {u['its']}" if u else "& &") + " & " + \
         (f"{a['cycle']} & {a['dofs']} & {sci(a['eH1'])} & {sci(a['hmin'])} & {a['condV']:.1f} & {a['its']}" if a else "& & & & &") + "\\\\\n"
w("t_el_sing", s + "\\bottomrule\n\\end{tabular}\n")

E3 = R("E3")
s = "\\begin{tabular}{l cccc}\n\\toprule\nrun & fitted $\\mathrm{d}R^2/\\mathrm{d}t$ & rel.\\ deviation from $-2\\varepsilon^2$ & $\\max_n(\\tilde E^{n+1}-\\tilde E^n)$ & $\\max_n(E_h^{n+1}-E_h^n)$\\\\\n\\midrule\n"
for k, v in E3["runs"].items():
    N, o = k[1:].split("_o")
    s += f"$N={N}$, order {o} & {sci(v['slope'])} & {abs(v['slope']/(-2*E3['eps']**2)-1)*100:.2f}\\% & {sci(v['max_dEmod'])} & {sci(v['max_dE'])}\\\\\n"
w("t_ac", s + "\\bottomrule\n\\end{tabular}\n")

E4 = R("E4"); sp = E4["spatial"]
s = "\\begin{tabular}{r cc cc}\n\\toprule\n & \\multicolumn{2}{c}{$D_2$} & \\multicolumn{2}{c}{$D_2^{(4)}$}\\\\\n$N$ & error & rate & error & rate\\\\\n\\midrule\n"
for i, N in enumerate(sp["N"]):
    r2 = "--" if i == 0 else rate(sp["order2"][i-1], sp["order2"][i], sp["N"][i-1], N)
    r4 = "--" if i == 0 else rate(sp["order4"][i-1], sp["order4"][i], sp["N"][i-1], N)
    s += f"{N} & {sci(sp['order2'][i])} & {r2} & {sci(sp['order4'][i])} & {r4}\\\\\n"
w("t_ch_space", s + "\\bottomrule\n\\end{tabular}\n")
tp = E4["temporal"]
s = "\\begin{tabular}{c cc}\n\\toprule\n$\\Delta t$ & error & rate\\\\\n\\midrule\n"
for i, dt in enumerate(tp["dt"]):
    s += f"{sci(dt)} & {sci(tp['err'][i])} & {'--' if i==0 else rate(tp['err'][i-1], tp['err'][i], 1/tp['dt'][i-1], 1/dt)}\\\\\n"
w("t_ch_time", s + "\\bottomrule\n\\end{tabular}\n")

E5 = R("E5"); spd = E4["spinodal"]
s = "\\begin{tabular}{l cc}\n\\toprule\n & 2D spinodal & 3D spinodal\\\\\n\\midrule\n"
s += f"grid & ${spd['N']}^2$ & ${E5['N']}^3$\\\\\n$\\varepsilon$ & {spd['eps']} & {E5['eps']}\\\\\n"
s += f"steps & {spd['steps']}+4500 & {E5['steps']}\\\\\n"
s += f"$\\max_n|M_h^n-M_h^0|$ (with $M_h^0$) & {sci(spd['max_mass_dev'])} (0.1) & {sci(E5['mass_dev'])} ({E5.get('mass0', 0):.3g})\\\\\n"
s += f"$\\max_n(\\tilde E^{{n+1}}-\\tilde E^n)$ & {sci(spd['max_dEmod'])} & {sci(E5['max_dEmod'])}\\\\\n"
s += f"$\\max_n(E_h^{{n+1}}-E_h^n)$ (sampled) & {sci(spd['max_dE'])} & {sci(E5['max_dE'])}\\\\\n"
s += f"$\\xi=r/\\sqrt{{E_1+C}}$ range & [{spd['xi_min']:.6f}, {spd['xi_max']:.6f}] & --\\\\\n"
s += f"CPU time/step (1 core) & {spd['time_per_step']*1e3:.1f} ms & {E5['time_per_step']*1e3:.0f} ms\\\\\n"
w("t_spinodal", s + "\\bottomrule\n\\end{tabular}\n")

E6 = R("E6_cpu")
gpu = R("E6_gpu") if os.path.exists("results/E6_gpu.json") else None
grows = gpu["rows"] if gpu else []
s = ("\\begin{tabular}{rr r cc cc cc}\n\\toprule\n & & & \\multicolumn{2}{c}{CPU, 1 core, fp64} & \\multicolumn{2}{c}{GPU, fp64} & \\multicolumn{2}{c}{GPU, fp32}\\\\\n"
     "$d$ & $N$ & unknowns & GEMM & DCT & GEMM & DCT & GEMM & DCT\\\\\n\\midrule\n")
keys = sorted({(r["dim"], r["N"]) for r in E6} | {(r["dim"], r["N"]) for r in grows})
pend = "\\textit{pending}"
for dim, N in keys:
    c = {r["transform"]: r["ms_per_step"] for r in E6 if r["dim"] == dim and r["N"] == N}
    g64 = {r["transform"]: r["ms_per_step"] for r in grows if r["dim"] == dim and r["N"] == N and r["dtype"] == "float64"}
    g32 = {r["transform"]: r["ms_per_step"] for r in grows if r["dim"] == dim and r["N"] == N and r["dtype"] == "float32"}
    f = lambda D, k, isg: ("%.2f" % D[k]) if k in D else (pend if (isg and not gpu) else "--")
    s += f"{dim} & {N} & {N**dim} & {f(c,'gemm',0)} & {f(c,'dct',0)} & {f(g64,'gemm',1)} & {f(g64,'dct',1)} & {f(g32,'gemm',1)} & {f(g32,'dct',1)}\\\\\n"
if gpu:
    note = (f"GPU timings were measured on an {gpu['info']['gpu']} (CUDA {gpu['info']['cuda']}, CuPy {gpu['info']['cupy']}); "
            "times are per step after warm-up, with device synchronization.")
else:
    note = "\\todo{GPU columns pending: run \\texttt{python experiments/run\\_all.py --only E6 --gpu} and \\texttt{python experiments/make\\_tables.py} on a CUDA machine.}"
open(os.path.join(T, "gpu_note.tex"), "w").write(note + "\n")
w("t_perf", s + "\\bottomrule\n\\end{tabular}\n")
print("tables written")

# ---------------- E7: traction boundary conditions ----------------------
if os.path.exists("results/E7.json"):
    E7 = R("E7")
    sm = E7["smooth"]; names = list(sm.keys())
    s = "\\begin{tabular}{r " + "cc " * len(names) + "}\n\\toprule\n & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{traction on {n}}}" for n in names) + "\\\\\n$N$ & " + " & ".join(["error & rate"] * len(names)) + "\\\\\n\\midrule\n"
    for i in range(len(sm[names[0]])):
        cells = []
        for n in names:
            L = sm[n]; cells.append(sci(L[i]["eH1"])); cells.append("--" if i == 0 else rate(L[i-1]["eH1"], L[i]["eH1"], L[i-1]["N"], L[i]["N"]))
        s += f"{sm[names[0]][i]['N']} & " + " & ".join(cells) + "\\\\\n"
    s += "\\midrule\nGMRES its & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{min(r['its'] for r in sm[n])}--{max(r['its'] for r in sm[n])}}}" for n in names) + "\\\\\n"
    s += "edge error ($N{=}256$) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{sci(sm[n][-1]['eedge'])}}}" for n in names) + "\\\\\n"
    w("t_trac_smooth", s + "\\bottomrule\n\\end{tabular}\n")
    s = "\\begin{tabular}{l ccc ccc}\n\\toprule\n & \\multicolumn{3}{c}{GMRES iterations} & \\multicolumn{3}{c}{$e_{H^1}$}\\\\\n$\\nu$ & $N=32$ & $64$ & $128$ & $N=32$ & $64$ & $128$\\\\\n\\midrule\n"
    for r in E7["nu"]:
        s += f"{r['nu']} & {r['its32']} & {r['its64']} & {r['its128']} & {sci(r['e32'])} & {sci(r['e64'])} & {sci(r['e128'])}\\\\\n"
    w("t_trac_nu", s + "\\bottomrule\n\\end{tabular}\n")
    s = "\\begin{tabular}{r " + "c" * 5 + " c}\n\\toprule\n$\\alpha$ & $N=32$ & $64$ & $128$ & $256$ & $512$ & rate\\\\\n\\midrule\n"
    for r in E7["enhanced1d"]:
        a = r["alpha"]; lab = "$-1/12$" if a < 0 else ("0 (plain)" if a == 0 else ("1/4" if a == .25 else "1/2"))
        s += lab + " & " + " & ".join(sci(v) for v in r["err"]) + f" & {rate(r['err'][-2], r['err'][-1], 256, 512)}\\\\\n"
    w("t_enh1d", s + "\\bottomrule\n\\end{tabular}\n")
    U = E7["sing_uniform"]; A = E7["sing_adaptive"]; st = max(1, len(A) // 8)
    sel = A[::st] + ([A[-1]] if (len(A) - 1) % st else [])
    s = "\\begin{tabular}{rrc rrcccr}\n\\toprule\n\\multicolumn{3}{c}{uniform} & \\multicolumn{6}{c}{automatic (Haar-detail indicator)}\\\\\n\\cmidrule(lr){1-3}\\cmidrule(lr){4-9}\nDOFs & $e_{H^1}$ & its & cycle & DOFs & $e_{H^1}$ & $h_{\\min}$ & $\\kappa(V)$ & its\\\\\n\\midrule\n"
    for i in range(max(len(U), len(sel))):
        u = U[i] if i < len(U) else None; a = sel[i] if i < len(sel) else None
        s += (f"{u['dofs']} & {sci(u['eH1'])} & {u['its']}" if u else "& &") + " & " + \
             (f"{a['cycle']} & {a['dofs']} & {sci(a['eH1'])} & {sci(a['hmin'])} & {a['condV']:.1f} & {a['its']}" if a else "& & & & &") + "\\\\\n"
    w("t_trac_sing", s + "\\bottomrule\n\\end{tabular}\n")
    print("E7 tables written")

# ---------------- E8: sharp interfaces ---------------------------------
if os.path.exists("results/E8a.json"):
    r_ = R("E8a")["resolution"]
    s = "\\begin{tabular}{rr ccc}\n\\toprule\n$N$ & nodes per $\\sqrt2\\varepsilon$ & 3-point FD & $D_2$ & $D_2^{(4)}$\\\\\n\\midrule\n"
    for r in r_:
        if r["ppw"] < 0.6: continue
        s += f"{r['N']} & {r['ppw']:.2f} & {sci(r['fd3'])} & {sci(r['ihwc2'])} & {sci(r['ihwc4'])}\\\\\n"
    w("t_resolution", s + "\\bottomrule\n\\end{tabular}\n")
if os.path.exists("results/E8b.json"):
    st = R("E8b")["star"]
    s = "\\begin{tabular}{l rcc c}\n\\toprule\nrun & unknowns & max error vs.\\ ref. & area error & CPU time/step\\\\\n\\midrule\n"
    for key in ("N256_o2", "N256_o4", "N512_o2", "N512_o4"):
        if key not in st: continue
        v = st[key]; lab = f"${v['N']}^2$, " + ("$D_2$" if v["order"] == 2 else "$D_2^{(4)}$")
        s += f"{lab} & {v['N']**2} & {sci(v['err_vs_ref'])} & {sci(v['area_err'])} & {v['time_per_step']*1e3:.1f} ms\\\\\n"
    v = st["N1024_o4"]; s += f"$1024^2$, $D_2^{{(4)}}$ (ref.) & {1024**2} & -- & -- & {v['time_per_step']*1e3:.0f} ms\\\\\n"
    w("t_star", s + "\\bottomrule\n\\end{tabular}\n")
if os.path.exists("results/E8c.json"):
    c = R("E8c")
    s = "\\begin{tabular}{l c}\n\\toprule\n"
    s += f"grid / $\\varepsilon$ & ${c['N']}^2$ / {c['eps']}\\\\\n"
    s += f"nodes per $\\sqrt2\\varepsilon$ & {c['nodes_per_w']:.2f}\\\\\n"
    s += "time steps & " + " + ".join(f"{n}$\\times${sci(d)}" for d, n in c["stages"]) + "\\\\\n"
    s += f"$\\max_n|M_h^n-M_h^0|$ & {sci(c['max_mass_dev'])}\\\\\n"
    s += f"$\\max_n(\\tilde E^{{n+1}}-\\tilde E^n)$ & {sci(c['max_dEmod'])}\\\\\n"
    s += f"$\\max|\\nabla u_h|$ / planar-interface value $1/(\\sqrt2\\varepsilon)$ & {c['gradmax']:.0f} / {c['grad_theory']:.0f}\\\\\n"
    s += f"CPU time per step (1 core) & {c['time_per_step']*1e3:.0f} ms\\\\\n"
    w("t_chsharp", s + "\\bottomrule\n\\end{tabular}\n")
print("E8 tables written")

if os.path.exists("results/E8d.json"):
    s = "\\begin{tabular}{cc cccc}\n\\toprule\n$S$ & $\\Delta t$ & $\\max|u_h|$ & $E_h$: start $\\to$ end & $\\xi=r/\\sqrt{E_1+C}$ & $\\max_n(\\tilde E^{n+1}-\\tilde E^n)$\\\\\n\\midrule\n"
    for r in R("E8d"):
        s += f"{r['S']:g} & {sci(r['dt'])} & {r['umax']:.2f} & {r['E0']:.4f} $\\to$ {r['E']:.4f} & {r['xi']:.5f} & {sci(r['max_dEmod'])}\\\\\n"
    w("t_sav_stab", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- E9 / E10 comparisons ---------------------------------
NAMES = {"fd2": "3-point FD", "ihwc2": "IHWC $D_2$", "ihwc4": "IHWC $D_2^{(4)}$", "pade4": "compact 4th order (Pad\\'e)", "spectral": "cosine pseudo-spectral"}
if os.path.exists("results/E9.json"):
    E9 = R("E9"); ch = E9["ch"]; st = E9["star"]
    s = "\\begin{tabular}{l ccc cccc}\n\\toprule\n & \\multicolumn{3}{c}{CH smooth, max error} & \\multicolumn{4}{c}{AC star, $\\varepsilon=0.004$}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
    s += "scheme & $N=18$ & $54$ & $162$ & error $256^2$ & error $512^2$ & area $256^2$ & area $512^2$\\\\\n\\midrule\n"
    for k in ("fd2", "ihwc2", "ihwc4", "pade4", "spectral"):
        s += f"{NAMES[k]} & " + " & ".join(sci(v) for v in ch[k]) + f" & {sci(st[k+'_256']['err'])} & {sci(st[k+'_512']['err'])} & {sci(st[k+'_256']['area'])} & {sci(st[k+'_512']['area'])}\\\\\n"
    w("t_compare", s + "\\bottomrule\n\\end{tabular}\n")
if os.path.exists("results/E10.json"):
    E10 = R("E10")
    s = "\\begin{tabular}{r rcc rcc}\n\\toprule\n & \\multicolumn{3}{c}{IHWC collocation (GMRES + FDM)} & \\multicolumn{3}{c}{Q2 FEM (sparse direct)}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
    s += "cells & DOFs & $e_{H^1}$ & solve (s) & DOFs & $e_{H^1}$ & solve (s)\\\\\n\\midrule\n"
    A = E10["adaptive"]
    for r in A[::3] + ([A[-1]] if (len(A) - 1) % 3 else []):
        q = (f"{r['q2_dofs']} & {sci(r['q2_e'])} & {r['q2_t']:.2f}") if "q2_e" in r else "-- & -- & --"
        s += f"{r['cells']} & {r['ihwc_dofs']} & {sci(r['ihwc_e'])} & {r['ihwc_t']:.2f} & {q}\\\\\n"
    w("t_q2", s + "\\bottomrule\n\\end{tabular}\n")
print("comparison tables written")

if os.path.exists("results/E7b.json"):
    G = R("E7b")
    s = "\\begin{tabular}{r cccc cccc}\n\\toprule\n & \\multicolumn{4}{c}{Chebyshev grid} & \\multicolumn{4}{c}{geometric grid (toward traction edge)}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    s += "$N$ & $e_{H^1}$ & rate & edge err. & its & $e_{H^1}$ & rate & $h_{\\max}/h_{\\min}$ & its\\\\\n\\midrule\n"
    C_, Gg = G["chebyshev"], G["geometric"]
    for i in range(len(C_)):
        rc = "--" if i == 0 else rate(C_[i-1]["eH1"], C_[i]["eH1"], C_[i-1]["N"], C_[i]["N"])
        rg = "--" if i == 0 else rate(Gg[i-1]["eH1"], Gg[i]["eH1"], Gg[i-1]["N"], Gg[i]["N"])
        s += f"{C_[i]['N']} & {sci(C_[i]['eH1'])} & {rc} & {sci(C_[i]['eedge'])} & {C_[i]['its']} & {sci(Gg[i]['eH1'])} & {rg} & {Gg[i]['hratio']:.0f} & {Gg[i]['its']}\\\\\n"
    w("t_trac_graded", s + "\\bottomrule\n\\end{tabular}\n")
