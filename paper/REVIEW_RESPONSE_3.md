# Response to the third referee report (CMAME)

## Priority 1 — submission blockers
| Item | Action | Status |
|---|---|---|
| Appendix B factor of two | Proof rewritten: (sav1) is multiplied by 2Δt, the left-hand side expanded to 2(Ẽⁿ⁺¹−Ẽⁿ)+dissipation, the scalar SAV identity written out, and the division by two made explicit. Propagates automatically to the lean supplement. | done |
| Repository 404 | Package is a git repo with commit + tag `v1.0`, MIT `LICENSE`, `CITATION.cff`, and step-by-step publishing/Zenodo instructions (README). Paper cites tag v1.0 and an archive DOI (`\archive` macro). | **you must push + mint DOI** |
| Supplement missing | `supplementary_full.tex` (no proofs; Tables S1–S16, Fig. S1) for the full version; `supplementary.tex` (proofs + S1–S19) for the lean version. Every cited S-item exists. | done |
| Central contribution explicit | New title; abstract rebuilt around the spectral theorem and fits on one page; applications framed as consequences. | done |

## Priority 2 — stronger numerical evidence
| Item | Action | Where (full) |
|---|---|---|
| Adversarial grids | E17: alternating cell sizes (ratio up to 1000), 1e-6 end cells, geometric grading to 1e12, unbounded random sizes, clusters; Dirichlet and Neumann; N = 64 and 256. Real spectra always; κ(V) ≤ 85; backward error ≤ 6e-15. | Sec. 7.1, Table 7 |
| Explicit safeguard | `SafeguardedSylvester`: falls back to Bartels–Stewart if κ(V) > 1e6 or normwise backward error > 1e-12 (check = 2 GEMMs). Never triggered; fallback costs 13–18× FDM. | Sec. 2.3, Table 8 |
| Timing metadata | Table of CPU, OS, BLAS, threads, versions and protocol (medians of 3–5 repetitions; setup separate, not amortized); results/environment.json. | Table 5 |
| Competitive FEM baseline | Q2 with CG + fast-diagonalization preconditioner (same solver class), plus sparse direct. **AMG (pyamg) hook implemented (E13) but could not be run here (no internet); the paper shows a TODO sentence that is replaced automatically once you run E13.** Claims separated: unknowns (¼, discretization property) vs time (2.5–3× incl. assembly, ~1.4× without) vs memory (~40 % less), explicitly tied to implementation and hardware. | Sec. 7.5 |
| 3D accuracy | E14: manufactured (I−Δ)u = f on (0,1)³, uniform and Chebyshev-graded grids up to 96³; second order; 885k unknowns in 0.05 s. | Table 9 |

## Priority 3 — theoretical precision and positioning
| Item | Action |
|---|---|
| Prior spectral analyses | Cited and discussed: Dyksen (1987) and Sun (1999) for Hermite-cubic OSC; Bialecki–Fairweather MDAs (1995, survey 2011); QSC FFT/MDA solvers on uniform grids. Closest-methods table now lists trial space, collocation points, grids, spectrum and fast solver. Novelty stated narrowly: real, simple spectrum for **midpoint QSC / nodal HWCM on arbitrary partitions**, where quadrature-weight symmetrization is unavailable. |
| Conditioning not proven | Stated in the theorem discussion, abstract, scope table and conclusions; wording changed from "applies" to "is well defined", with reliability handled by the safeguard. |
| Adaptive indicator | E16: four corner openings (λ = 0.674 … 0.500, incl. near-crack), two singularities, boundary layer; indicator vs true-error oracle; rates −0.95 … −1.05 (oracle similar); effectivity ranges, cycles and total time reported; stated as observation; "anisotropic tensor refinement, not local refinement". |
| S recommendation | E15: S ∈ {0,1,2,4,8} × Δt ∈ {1,8,32}ε²: S = 0 fails; error grows ∝ S at small Δt; recommendation now "smallest S keeping ξ ≈ 1 (≈1–2 for Δt ≲ ε², 2–4 for larger)", with nondimensionalization (S scales with max f′ = 2, not with ε or h) and the explicit statement that ξ ≈ 1 does not certify accuracy. |

## Priority 4 — editorial
Shorter title; one-page abstract; seven secondary tables moved to the supplement (35 pp. vs 37); repeated proved/observed prose replaced by references to the scope table; performance claims qualified by implementation/hardware; limitations visible in abstract and conclusions.

## Please verify before submission
1. The exact scope of Dyksen (1987), Sun (1999) and Bialecki–Fairweather (1995), in particular whether their real-spectrum results cover non-uniform partitions for OSC. The novelty claim is written to be correct either way (it concerns midpoint QSC), but the closest-methods table says "partitions as in the cited works" for that reason.
2. Bibliographic details of the five newly added references.
3. Run E13 (AMG) and E6 (GPU) and regenerate the tables; push the repository; mint the Zenodo DOI.
