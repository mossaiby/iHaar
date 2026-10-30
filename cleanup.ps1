# Remove build leftovers and Python caches from the iHaar working tree (run from the repo root).
# Keeps: figures/*.pdf and the figure PDFs in paper/submission_camwa/.
$patterns = @("*.aux","*.log","*.out","*.spl","*.fls","*.fdb_latexmk","*.synctex.gz","*.bbl","*.blg")
foreach ($p in $patterns) { Get-ChildItem -Path paper -Recurse -Filter $p -File | Remove-Item -Verbose }
Remove-Item -Verbose -ErrorAction SilentlyContinue paper\*.pdf
Remove-Item -Verbose -ErrorAction SilentlyContinue paper\submission_camwa\manuscript.pdf, paper\submission_camwa\supplementary.pdf
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Where-Object { $_.FullName -notmatch '\\.venv\\' } | Remove-Item -Recurse -Force -Verbose
