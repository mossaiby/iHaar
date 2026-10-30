"""Create paper/submission_camwa/: self-contained CAMWA submission (flat files, tables inlined, figures
copied, highlights and cover letter) and latex_source.zip with the LaTeX sources and figures for upload.
elsarticle.cls is provided by Elsevier's system and by TeX Live/MiKTeX. A compiled cover_letter.pdf is kept."""
import os, re, shutil, zipfile
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
out = "submission_camwa"
os.makedirs(out, exist_ok=True)   # emptied file by file: the folder itself may be locked on Windows
for f in os.listdir(out):
    if f != "cover_letter.pdf":
        p = os.path.join(out, f)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
def flatten(src, dst):
    s = open(src).read()
    s = re.sub(r"\\input\{tables/([A-Za-z_0-9]+)\.tex\}", lambda m: open(f"tables/{m.group(1)}.tex").read().strip(), s)
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]+)\}", s):
        shutil.copy(f"../figures/{f}", os.path.join(out, os.path.basename(f)))
    s = s.replace("{../figures/panels/", "{").replace("{../figures/", "{")
    open(os.path.join(out, dst), "w").write(s)
flatten("manuscript_camwa.tex", "manuscript.tex")
flatten("supplementary_camwa.tex", "supplementary.tex")
for f in ("highlights.txt", "cover_letter.md"):
    shutil.copy(os.path.join("tools", f"camwa_{f}"), os.path.join(out, f))
hl = [l for l in open("tools/camwa_highlights.txt").read().splitlines() if l.strip()]
assert 3 <= len(hl) <= 5 and all(len(l) <= 85 for l in hl), [(len(l), l) for l in hl]
src = ["manuscript.tex", "supplementary.tex"] + sorted(f for f in os.listdir(out) if f.endswith(".pdf") and f != "cover_letter.pdf")
with zipfile.ZipFile(os.path.join(out, "latex_source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in src:
        z.write(os.path.join(out, f), f)
print("written:", sorted(os.listdir(out)))
print("latex_source.zip:", len(src), "files")
