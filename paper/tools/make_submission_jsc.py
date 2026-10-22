"""Create paper/submission_jsc/: self-contained JSC submission (flat files, tables inlined,
figures copied, svjour3 class files included). Requires tools/svjour3/ (Springer macro package)."""
import os, re, shutil
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
out = "submission_jsc"
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
def flatten(src, dst):
    s = open(src).read()
    s = re.sub(r"\\input\{tables/([A-Za-z_0-9]+)\.tex\}", lambda m: open(f"tables/{m.group(1)}.tex").read().strip(), s)
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]+)\}", s):
        shutil.copy(f"../figures/{f}", os.path.join(out, os.path.basename(f)))
    s = s.replace("{../figures/panels/", "{").replace("{../figures/", "{")
    open(os.path.join(out, dst), "w").write(s)
flatten("manuscript_jsc.tex", "manuscript.tex")
flatten("supplementary_jsc.tex", "supplementary.tex")
for f in ("svjour3.cls", "svglov3.clo", "spmpsci.bst"):
    shutil.copy(os.path.join("tools", "svjour3", f), os.path.join(out, f))
print("written:", sorted(os.listdir(out)))
