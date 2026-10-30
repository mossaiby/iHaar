[Date]

The Editor
*Computers & Mathematics with Applications*

Dear Editor,

Please consider the enclosed manuscript, **"Fast solvers on arbitrary grids for Haar wavelet collocation:
sharp-interface phase-field computations and adaptive elastostatics"**, for publication in *Computers &
Mathematics with Applications* as a Regular Article with Available Code.

The Haar wavelet collocation method (HWCM) is widely used in engineering because it is simple to derive and
to program. In two and three dimensions it is usually solved as one dense system, which limits it to about
64 × 64 cells, and graded grids must be designed by hand. The manuscript removes these limitations. Its
contributions match the journal's areas of discretization methods and numerical analysis for PDEs, fast
numerical algorithms, adaptivity, and code verification:

- **Discretization and analysis.** Written in nodal values, HWCM is midpoint quadratic-spline collocation.
  An exact three-term identity replaces the dense HWCM integration matrices by two tridiagonal matrices. We
  prove that the resulting second-derivative operator has real, simple and nonpositive eigenvalues on
  *every* strictly increasing grid.
- **Fast algorithms.** Fast diagonalization, which reduces tensor-grid problems to a few dense matrix
  products, is therefore well defined on arbitrary graded and adaptively refined grids. Its floating-point
  behaviour is tested separately on random and adversarial grids, with an inexpensive safeguard. On uniform
  grids, a correction at no extra cost gives fourth-order accuracy for zero-flux conditions.
- **Adaptivity.** Refinement driven by the Haar detail coefficients recovers the optimal convergence rate at
  corner singularities in plane elastostatics with traction boundaries. Truly local refinement is shown to be
  compatible with the fast solver through nested composite patches.
- **Applications to PDEs.** Mass-conserving Allen–Cahn and Cahn–Hilliard schemes that satisfy a
  modified-energy law and resolve interfaces with about four nodes across the transition layer.
- **Comparison with finite elements.** On identical grids, the method needed about a quarter of the unknowns
  of biquadratic finite elements, compared against a sparse direct solver, conjugate gradients with a
  fast-diagonalization preconditioner, and conjugate gradients with algebraic multigrid.

The manuscript states the limits of the method plainly: rectangular domains and compressible materials.

All results are reproducible. The implementation, the tests and the scripts that produce every table and
figure are openly available at https://github.com/mossaiby/iHaar (release v1.2). Proofs and detailed tables
are in the Supplementary Material.

The manuscript is original, has not been published, and is not under consideration elsewhere. I am the sole
author and declare no competing interests.

[Suggested reviewers, if requested by the submission system: name, affiliation, e-mail, expertise.]

Sincerely,

Farshid Mossaiby
Department of Civil Engineering, University of Isfahan, Isfahan 81744-73441, Iran
mossaiby@eng.ui.ac.ir
