# ihaar — integrated-Haar wavelet collocation with fast tensor solvers

Code, tests and reproduction scripts for the manuscript in `paper/manuscript.tex`.

## Layout
```
ihaar/core.py        three-term identity (Theorem 1): D2 = A^{-1} L, D1, boundary vectors;
                     4th-order correction; fast diagonalization (GEMM) and DCT solvers;
                     spline reconstruction for Gauss-quadrature error norms
ihaar/phasefield.py  SAV-BDF1/BDF2 Allen–Cahn and Cahn–Hilliard (2D/3D, Neumann, uniform)
ihaar/elasticity.py  plane-strain collocation on graded tensor grids (Dirichlet), Williams corner field,
                     FDM-block-preconditioned GMRES, Haar-detail indicator, Dörfler marking
ihaar/fem_q2.py      reference Q2 (biquadratic) FEM on tensor grids, for comparison
ihaar/elasticity_bc.py  mixed Dirichlet/traction edges (extended unknowns, traction rows,
                     traction-traction corners, Neumann-closure FDM preconditioner)
tests/               unit/verification tests (identity vs classical HWCM matrices, symmetry,
                     mass, energy decay, exact-field checks, DCT == eigenbasis)
experiments/run_all.py     every table/figure: E1 verification/cost, E2 elasticity (Dirichlet),
                           E3 Allen–Cahn circle, E4/E5 Cahn–Hilliard 2D/3D, E7 traction BCs,
                           E7b traction on graded grids, E8a–E8d sharp interfaces (resolution, AC star,
                           thin-interface CH, plain vs stabilized SAV), E8fig figures,
                           E9 FD/compact/spectral comparison, E10 Q2 FEM comparison, E6 timing
requirements.txt / requirements-gpu.txt   dependencies (GPU file adds CuPy)
experiments/make_tables.py LaTeX tables generated from results/*.json
paper/               manuscript (elsarticle), generated tables, internal referee report
```

## Reproduce (CPU)
```
pip install -r requirements.txt
for t in tests/test_*.py; do python $t; done
python experiments/run_all.py            # ~25 min on one core (E8c alone ~8 min, needs ~3 GB RAM)
# long runs can be split: --only E8c_stage0, E8c_stage1, E8c_stage2, then E8c_finalize and E8fig
python experiments/make_tables.py
cd paper && latexmk -pdf manuscript.tex  # needs elsarticle.cls (TeX Live / CTAN)
```

## GPU (required before submission)
On a machine with an NVIDIA GPU (check with `nvidia-smi`):
```
cd ihaar
pip install -r requirements-gpu.txt          # CUDA 12.x; for CUDA 11.x edit to cupy-cuda11x
python tests/test_gpu.py                     # CPU vs GPU agreement; must print "ALL OK"
python experiments/run_all.py --only E6 --gpu   # writes results/E6_gpu.json (fp64 + fp32, GPU name, CUDA, CuPy)
python experiments/make_tables.py            # fills the GPU columns of Table "perf" and replaces the TODO sentence
cd paper && latexmk -pdf manuscript.tex
```
`make_tables.py` writes `paper/tables/gpu_note.tex`. While `results/E6_gpu.json` is missing it contains
the red TODO; once the file exists it is a sentence naming the GPU model and the CUDA and CuPy versions.
