"""Create paper/submission_eabe/: self-contained EABE submission (flat files, tables inlined, figures
copied, highlights and cover letter). elsarticle.cls is provided by Elsevier's system and by TeX Live/MiKTeX."""
import os, re, shutil
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
out = "submission_eabe"
shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)   # the folder itself may be locked on Windows
def flatten(src, dst):
    s = open(src).read()
    s = re.sub(r"\\input\{tables/([A-Za-z_0-9]+)\.tex\}", lambda m: open(f"tables/{m.group(1)}.tex").read().strip(), s)
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]+)\}", s):
        shutil.copy(f"../figures/{f}", os.path.join(out, os.path.basename(f)))
    s = s.replace("{../figures/panels/", "{").replace("{../figures/", "{")
    open(os.path.join(out, dst), "w").write(s)
flatten("manuscript_eabe.tex", "manuscript.tex")
flatten("supplementary_eabe.tex", "supplementary.tex")
for f in ("eabe_highlights.txt", "eabe_cover_letter.md"):
    shutil.copy(os.path.join("tools", f), os.path.join(out, f.replace("eabe_", "")))
hl = [l for l in open("tools/eabe_highlights.txt").read().splitlines() if l.strip()]
assert 3 <= len(hl) <= 5 and all(len(l) <= 85 for l in hl), [(len(l), l) for l in hl]
print("written:", sorted(os.listdir(out)))
