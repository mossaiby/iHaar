"""Build a flat, self-contained submission folder (Elsevier style):
    python paper/make_submission.py      (from the repository root)
-> paper/submission/manuscript.tex (all tables inlined) + figure PDFs, and submission.zip
"""
import os, re, shutil, zipfile
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "submission")
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
src = open(os.path.join(here, "manuscript.tex")).read()

def inline(m):
    path = os.path.join(here, m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
    return open(path).read().rstrip("\n")
src = re.sub(r"\\input\{(tables/[^}]+)\}", inline, src)
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]+)\}", src)
src = src.replace("../figures/", "")
assert "\\input{" not in src, "un-inlined \\input remains"
open(os.path.join(out, "manuscript.tex"), "w").write(src)
for f in figs:
    shutil.copy(os.path.join(here, "..", "figures", f), out)
with zipfile.ZipFile(os.path.join(here, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(out)):
        z.write(os.path.join(out, f), f)
print(f"submission/: manuscript.tex + {len(figs)} figures; zipped to paper/submission.zip")
