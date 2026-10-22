# Remove build leftovers and Python caches from the iHaar working tree (run from the repo root).
# Keeps: figures/*.pdf and paper/submission_jsc/*.pdf figure files.
$patterns = @("*.aux","*.log","*.out","*.spl","*.fls","*.fdb_latexmk","*.synctex.gz")
foreach ($p in $patterns) { Get-ChildItem -Path paper -Recurse -Filter $p -File | Remove-Item -Verbose }
Remove-Item -Verbose -ErrorAction SilentlyContinue paper\manuscript.pdf, paper\manuscript_lean.pdf, paper\manuscript_jsc.pdf, paper\supplementary*.pdf
Remove-Item -Verbose -ErrorAction SilentlyContinue paper\submission_jsc\manuscript.pdf, paper\submission_jsc\supplementary.pdf
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -Verbose
