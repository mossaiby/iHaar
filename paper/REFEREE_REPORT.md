# Internal adversarial review — status after revision 4

*Self-review written as a hostile CMAME referee (no independent sub-agent was available).
Each item lists what was done and what remains.*

| # | Referee concern | Status | What was done / what remains |
|---|---|---|---|
| 1 | Novelty vs quadratic spline collocation (QSC) | **Addressed (positioning)** | Christara & Ng (2006a, b) cited after verification. The claims are reduced to: the HWCM-variable identity with end rows, unconditional solvability, the eigen-preserving 4th-order correction (rational form shown to equal Padé), structure-preserving phase-field schemes, and Haar-driven 2D anisotropic refinement for elasticity. *Author: read Christara & Ng (2006b) to confirm no 2D overlap.* |
| 2 | Why not compact FD / spectral on uniform boxes? | **Resolved (E9)** | Exactly fair comparison (same SAV solver, only the Laplacian symbol changes). Pseudo-spectral wins on uniform zero-flux boxes; Padé is 1.4x better than D2^(4); IHWC D2 is 2x better than 3-point FD. The paper now says plainly where IHWC is *not* the right choice. |
| 3 | Elasticity: no traction BCs / no traction-free singularity | **Resolved (E7, E7b)** | Traction BCs; traction-free-face singularity with an optimal adaptive rate; 2nd order also on graded grids (Chebyshev, geometric up to h ratio 85). New honest finding: GMRES iterations grow on strongly graded grids with traction edges. |
| 4 | No comparison with established methods | **Resolved for FEM (E10)** | Q2 FEM on identical grids: Q2 is 1.3x more accurate per grid but has 4x more unknowns; IHWC is more accurate per unknown and its solve is ~165x faster at equal error (vs sparse direct). *Remaining: adaptive hierarchical IGA collocation; AMG-preconditioned FEM timings.* |
| 5 | Indicator heuristic | Acknowledged | Stated as empirical; good effectivity shown. A reliability proof is out of scope. |
| 6 | GPU claims without GPU data | **Needs the author's GPU** | Exact commands in the README; `tests/test_gpu.py` checks CPU/GPU agreement; table and sentence are generated automatically. |
| 7 | Near incompressibility | Acknowledged | Reported (Tables 4, 9); a mixed u-p formulation is future work. |
| m1 | Real spectrum on graded grids unproven | Acknowledged | Observed in every run (kappa(V) < 25). |
| m2 | 3D mass check with zero mean | **Fixed** | Mean 0.1: drift 1.4e-17. |
| m3 | Eigendecomposition cost not reported | **Fixed** | "FDM setup" column in Table 2. |
| m4 | Bibliography | **Partly verified** | Christara & Ng 2006a/b and De Lorenzis et al. 2015 checked online. *Author: verify the rest.* |
| m5 | Plain SAV can fail for thin interfaces | **Fixed** | Stabilized SAV with proof; failure and fix both shown. |

## Remaining risks for CMAME
1. Full L-shaped (multi-patch) benchmark absent: geometry is restricted to rectangles.
2. GPU results pending.
3. On uniform boxes the phase-field part is outperformed by pseudo-spectral methods; the
   paper's case rests on the unified, structure-preserving framework, graded/adaptive grids,
   Dirichlet/traction data and the elasticity results.

**Verdict:** submittable to CMAME once item 6 (GPU) is done, with a realistic outcome of major
revision. The most likely request is a multi-patch L-shape and an IGA-collocation comparison.
