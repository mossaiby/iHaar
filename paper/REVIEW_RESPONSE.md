# Response to the external assessment (CMAME route)

| Reviewer concern | Action taken | Where |
|---|---|---|
| Code availability placeholder (critical) | Replaced with https://github.com/mossaiby/iHaar in the introduction and in the availability statement. | Sec. 1, Code availability |
| Novelty vs QSC / IGA collocation (critical) | New Table 1 lists each ingredient as *inherited* or *new*; a "Central thesis" paragraph states one methodological claim, with the applications as evidence. | Sec. 1, Table 1 |
| Theory does not cover graded grids | **New theorem:** D2 has real, simple, nonpositive eigenvalues on *every* grid (proof via oscillation matrices, Gantmacher–Krein). Fast diagonalization therefore always applies. Conditioning of the eigenvectors is studied on 8760 random graded grids (kappa(V) <= 100). | Thm. 2, App. A.2, Table 5 |
| Proved vs observed claims mixed | New Section 6 and Table 2 separate proved results (with assumptions) from observed ones. | Sec. 6 |
| Q2 comparison confounds discretization with solver (high) | Q2 FEM now also solved with CG preconditioned by fast diagonalization (same solver class as IHWC). Result: with matched solvers, IHWC needs ~4x fewer unknowns and ~2x less time at equal accuracy; the earlier 100x+ factor is attributed explicitly to the solver. | Sec. 7.4, Table 17, Fig. 3 |
| Broad scope | A single thesis (nodal reformulation, spectral structure) frames the paper; each application section states its link to it; limitations are collected in Sec. 9. | Secs. 1, 6, 9 |
| Long title | Shortened. | Title |
| Presentation | Prose throughout (no lists); every table has its own caption; no minipage or resizebox; wide tables split into stacked halves; consistent abstract and conclusions. | whole paper |

## Not addressed (disregarded as requested, or requires new research)
- Adaptive hierarchical IGA-collocation and AMG-FEM comparisons (stated as limitations).
- Energy analysis on graded grids (the spectrum is now proved there; the energy law is not).
- GPU timings (require the author's hardware; commands in the README).
